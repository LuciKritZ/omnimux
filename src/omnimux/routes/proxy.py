import asyncio
import base64
import json
import logging
import re
import time
import typing
import uuid
from contextlib import suppress

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from omnimux.config import settings
from omnimux.models.schemas import (
    AnthropicMessagesRequest,
    ChatCompletionRequest,
    convert_anthropic_to_chat_completion,
)
from omnimux.services.cache import ImageCache
from omnimux.services.compressor import ContextCompressor
from omnimux.services.db import validate_client_key
from omnimux.services.gatekeeper import Gatekeeper
from omnimux.services.multiplexer import Multiplexer
from omnimux.services.unloader import ModelUnloader

router = APIRouter()
logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def verify_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
) -> None:
    x_api_key = request.headers.get("x-api-key")
    token = credentials.credentials if credentials else x_api_key

    db = getattr(request.app.state, "db", None)
    if db and token:
        is_valid = await validate_client_key(db, token)
        if is_valid:
            return

    if settings.proxy_password:
        if token == settings.proxy_password:
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if db:
        async with db.execute(
            "SELECT COUNT(*) FROM client_keys WHERE revoked_at IS NULL"
        ) as cursor:
            row = await cursor.fetchone()
            key_count = row[0] if row else 0
            if key_count > 0:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing Bearer token",
                    headers={"WWW-Authenticate": "Bearer"},
                )


async def _prepare_request_pipeline(
    payload: ChatCompletionRequest,
    fastapi_req: Request,
) -> tuple[list[str], dict[str, typing.Any], int]:
    unloader: ModelUnloader = fastapi_req.app.state.unloader
    compressor: ContextCompressor = fastapi_req.app.state.compressor
    cache: ImageCache = fastapi_req.app.state.cache
    gatekeeper: Gatekeeper = fastapi_req.app.state.gatekeeper

    platforms_data = getattr(fastapi_req.app.state, "platforms", {})
    fallback_model = platforms_data.get("system_fallback", {}).get("model")

    await unloader.unload_if_needed(payload.model)

    has_images_in_latest = False
    latest_msg_index = len(payload.messages) - 1
    cached_hashes: dict[int, str] = {}

    for i, message in enumerate(payload.messages):
        if isinstance(message.content, list):
            for part in message.content:
                if part.type == "image_url" and part.image_url is not None:
                    if i == latest_msg_index:
                        has_images_in_latest = True
                    url = part.image_url.url
                    if url.startswith("data:"):
                        try:
                            _, b64_data = url.split(",", 1)
                            raw_bytes = base64.b64decode(b64_data)
                            image_hash = cache.hash_image(b64_data)
                            cache.store(image_hash, raw_bytes)
                            cached_hashes[id(part)] = image_hash
                        except Exception:
                            pass
                    elif url.startswith(("http://", "https://")):
                        image_hash = cache.hash_image(url)
                        cached_hashes[id(part)] = image_hash

                    if i != latest_msg_index:
                        part.type = "text"
                        image_hash = cached_hashes.get(id(part), "unknown_image")
                        cache.get_path(image_hash)
                        part.text = f"[Image: {image_hash}]"
                        part.image_url = None
        elif isinstance(message.content, str) and "[Image: " in message.content:
            for match in re.finditer(r"\[Image:\s*([a-f0-9]{64})\]", message.content):
                cache.get_path(match.group(1))

    payload_dict_for_gatekeeper = payload.model_dump(exclude_none=True)
    profile = await gatekeeper.profile_request(payload_dict_for_gatekeeper)
    context_length = profile.get("context_length", 0)

    candidates: list[str] = []

    if payload.model in ("auto", "omnimux-auto", "omni-auto", "nexus-auto", "claude-auto"):
        for key, config in platforms_data.items():
            if not isinstance(config, dict) or key == "system_fallback":
                continue
            max_tokens = config.get("max_input_tokens") or 200000
            if max_tokens < context_length:
                continue
            if has_images_in_latest and not config.get("supports_vision"):
                continue
            candidates.append(key)

        if not candidates:
            logger.warning("No candidates found, falling back to system fallback")
            candidates = ["system_fallback"]

        target_model_for_compression = (
            candidates[0].split("/", 1)[1] if "/" in candidates[0] else candidates[0]
        )
        if candidates[0] == "system_fallback":
            target_model_for_compression = fallback_model
    else:
        # Explicit model requested
        if payload.model == fallback_model:
            candidates = ["system_fallback"]
            target_model_for_compression = fallback_model
        elif "/" in payload.model:
            candidates = [payload.model]
            target_model_for_compression = payload.model.split("/", 1)[1]
        else:
            found = [k for k in platforms_data if k.endswith(f"/{payload.model}")]
            if found:
                candidates = list(found)
                target_model_for_compression = payload.model
            elif payload.model.startswith("claude"):
                candidates = [f"anthropic/{payload.model}"]
                target_model_for_compression = payload.model
            else:
                candidates = [f"ollama/{payload.model}"]
                target_model_for_compression = payload.model

        # Append other configured platforms as fallback candidates for autoswitching
        for key, config in platforms_data.items():
            if not isinstance(config, dict) or key == "system_fallback":
                continue
            if key not in candidates:
                if has_images_in_latest and not config.get("supports_vision"):
                    continue
                candidates.append(key)

    payload.messages = compressor.compress_messages(
        payload.messages, target_model_for_compression, has_images_in_latest
    )

    payload_dict = payload.model_dump(exclude_none=True)
    return candidates, payload_dict, context_length


async def with_sse_keepalive(
    generator: typing.AsyncGenerator[bytes, None],
    interval: float | None = None,
) -> typing.AsyncGenerator[bytes, None]:
    """
    Wraps an SSE byte stream generator to emit periodic ': keepalive\n\n'
    comment frames if upstream processing takes longer than interval seconds.
    """
    keepalive_interval = (
        interval if interval is not None else settings.sse_keepalive_interval_seconds
    )
    if keepalive_interval <= 0:
        async for chunk in generator:
            yield chunk
        return

    anext_task: asyncio.Task[bytes] | None = None
    try:
        while True:
            if anext_task is None:
                anext_task = asyncio.create_task(anext(generator))

            done, _ = await asyncio.wait([anext_task], timeout=keepalive_interval)
            if done:
                try:
                    chunk = anext_task.result()
                    yield chunk
                    anext_task = None
                except StopAsyncIteration:
                    break
            else:
                yield b": keepalive\n\n"
    finally:
        if anext_task and not anext_task.done():
            anext_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await anext_task
        with suppress(Exception):
            await generator.aclose()


@router.post("/v1/chat/completions", dependencies=[Depends(verify_auth)])
async def chat_completions(fastapi_req: Request, payload: ChatCompletionRequest) -> Response:
    multiplexer: Multiplexer = fastapi_req.app.state.multiplexer
    candidates, payload_dict, context_length = await _prepare_request_pipeline(payload, fastapi_req)
    extra_kwargs = {k: v for k, v in payload_dict.items() if k not in ("model", "messages")}

    if not payload.stream:
        content_parts: list[str] = []
        async for chunk in multiplexer.generate_stream(
            candidate_models=candidates,
            messages=payload_dict.get("messages", []),
            **extra_kwargs,
        ):
            content_parts.append(chunk)

        full_content = "".join(content_parts)
        completion_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": max(1, context_length),
                "completion_tokens": max(1, len(full_content) // 4),
                "total_tokens": max(1, context_length) + max(1, len(full_content) // 4),
            },
        }
        return JSONResponse(content=completion_response)

    async def sse_wrapper() -> typing.AsyncGenerator[bytes, None]:
        async for chunk in multiplexer.generate_stream(
            candidate_models=candidates,
            messages=payload_dict.get("messages", []),
            **extra_kwargs,
        ):
            openai_chunk = {
                "id": "chatcmpl-dynamic",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": chunk}}],
            }
            yield f"data: {json.dumps(openai_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        with_sse_keepalive(sse_wrapper()),
        media_type="text/event-stream",
    )


@router.post("/v1/messages", dependencies=[Depends(verify_auth)])
async def anthropic_messages(fastapi_req: Request, payload: AnthropicMessagesRequest) -> Response:
    chat_req = convert_anthropic_to_chat_completion(payload)
    multiplexer: Multiplexer = fastapi_req.app.state.multiplexer
    candidates, payload_dict, context_length = await _prepare_request_pipeline(
        chat_req, fastapi_req
    )
    extra_kwargs = {k: v for k, v in payload_dict.items() if k not in ("model", "messages")}
    if "anthropic-beta" in fastapi_req.headers:
        extra_kwargs["anthropic_beta"] = fastapi_req.headers["anthropic-beta"]
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    if payload.stream:

        async def sse_anthropic_wrapper() -> typing.AsyncGenerator[bytes, None]:
            message_start = {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": payload.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": max(1, context_length), "output_tokens": 1},
                },
            }
            yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode()

            content_block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
            cb_start_data = json.dumps(content_block_start)
            yield f"event: content_block_start\ndata: {cb_start_data}\n\n".encode()

            total_output_tokens = 0
            async for chunk in multiplexer.generate_stream(
                candidate_models=candidates,
                messages=payload_dict.get("messages", []),
                **extra_kwargs,
            ):
                if chunk:
                    total_output_tokens += max(1, len(chunk) // 4)
                    delta_event = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": chunk},
                    }
                    delta_data = json.dumps(delta_event)
                    yield f"event: content_block_delta\ndata: {delta_data}\n\n".encode()

            content_block_stop = {"type": "content_block_stop", "index": 0}
            yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n".encode()

            message_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": max(1, total_output_tokens)},
            }
            yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode()

            message_stop = {"type": "message_stop"}
            yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n".encode()

        return StreamingResponse(
            with_sse_keepalive(sse_anthropic_wrapper()),
            media_type="text/event-stream",
        )

    # Non-streaming
    accumulated_content = []
    async for chunk in multiplexer.generate_stream(
        candidate_models=candidates,
        messages=payload_dict.get("messages", []),
        **extra_kwargs,
    ):
        accumulated_content.append(chunk)

    full_text = "".join(accumulated_content)
    total_output_tokens = max(1, len(full_text) // 4)
    return JSONResponse(
        content={
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": full_text}],
            "model": payload.model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": max(1, context_length),
                "output_tokens": total_output_tokens,
            },
        }
    )


@router.post("/v1/messages/count_tokens", dependencies=[Depends(verify_auth)])
async def count_tokens(fastapi_req: Request, payload: AnthropicMessagesRequest) -> JSONResponse:
    chat_req = convert_anthropic_to_chat_completion(payload)
    profile = await fastapi_req.app.state.gatekeeper.profile_request(
        chat_req.model_dump(exclude_none=True)
    )
    base_length = profile.get("context_length", 0)
    tools_length = sum(len(json.dumps(t)) for t in payload.tools) if payload.tools else 0
    return JSONResponse(content={"input_tokens": base_length + tools_length})


@router.get("/v1/models", dependencies=[Depends(verify_auth)])
async def list_models(fastapi_req: Request) -> JSONResponse:
    platforms_data = getattr(fastapi_req.app.state, "platforms", {})
    models_list = [
        {"id": k, "object": "model", "owned_by": k.split("/")[0]}
        for k in platforms_data
        if k not in ("taken-from", "system_fallback") and isinstance(platforms_data[k], dict)
    ]
    return JSONResponse(content={"object": "list", "data": models_list})
