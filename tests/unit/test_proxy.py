import asyncio
import json
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import aiosqlite
import httpx
import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from omnimux.app import create_app
from omnimux.routes.proxy import verify_auth, with_sse_keepalive
from omnimux.services.db import create_client_key, init_db


@pytest_asyncio.fixture
async def memory_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(":memory:") as db:
        await init_db(db)
        yield db


@pytest.fixture
def test_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    req = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
    mock_post_response = httpx.Response(status_code=200, request=req, json={"status": "success"})
    monkeypatch.setattr(httpx.AsyncClient, "post", AsyncMock(return_value=mock_post_response))

    app = create_app()
    with TestClient(app) as client:
        yield client


def test_chat_completions_stream_false_returns_json(test_client: TestClient) -> None:
    with patch("omnimux.services.multiplexer.Multiplexer.generate_stream") as mock_stream:

        async def mock_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield "Hello"
            yield " world"

        mock_stream.return_value = mock_gen()

        res = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("application/json")
        data = res.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello world"
        assert data["choices"][0]["index"] == 0
        assert data["choices"][0]["finish_reason"] == "stop"


def test_openai_sse_chunk_has_index(test_client: TestClient) -> None:
    with patch("omnimux.services.multiplexer.Multiplexer.generate_stream") as mock_stream:

        async def mock_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield "chunk1"

        mock_stream.return_value = mock_gen()

        res = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert res.status_code == 200
        lines = [
            line
            for line in res.text.split("\n")
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        assert len(lines) > 0
        chunk_data = json.loads(lines[0][6:])
        assert "index" in chunk_data["choices"][0]
        assert chunk_data["choices"][0]["index"] == 0


def test_non_data_image_url_preserved(test_client: TestClient) -> None:
    with patch("omnimux.services.multiplexer.Multiplexer.generate_stream") as mock_stream:
        captured_messages = []

        async def mock_gen(
            candidate_models: list[str],
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> AsyncGenerator[str, None]:
            captured_messages.extend(messages)
            yield "Analysis"

        mock_stream.side_effect = mock_gen

        res = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/photo.png"},
                            },
                            {"type": "text", "text": "What is this?"},
                        ],
                    },
                    {"role": "user", "content": "Next question"},
                ],
                "stream": True,
            },
        )
        assert res.status_code == 200
        first_content = str(captured_messages[0]["content"])
        assert "unknown_image" not in first_content


def test_count_tokens_includes_tools(test_client: TestClient) -> None:
    res_base = test_client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    tokens_base = res_base.json()["input_tokens"]

    res_with_tools = test_client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Lookup database records",
                    "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}},
                }
            ],
        },
    )
    tokens_with_tools = res_with_tools.json()["input_tokens"]
    assert tokens_with_tools > tokens_base


def test_anthropic_messages_endpoint_forwards_beta_header(test_client: TestClient) -> None:
    with patch("omnimux.services.multiplexer.Multiplexer.generate_stream") as mock_stream:
        captured_kwargs: dict[str, Any] = {}

        async def mock_gen(
            candidate_models: list[str],
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> AsyncGenerator[str, None]:
            captured_kwargs.update(kwargs)
            yield "Hello"

        mock_stream.side_effect = mock_gen

        res = test_client.post(
            "/v1/messages",
            headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            json={
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert res.status_code == 200
        assert captured_kwargs.get("anthropic_beta") == "prompt-caching-2024-07-31"


@pytest.mark.asyncio
async def test_auth_rejects_invalid_token_when_no_active_keys(
    memory_db: aiosqlite.Connection,
) -> None:
    scope = {
        "type": "http",
        "headers": [(b"x-api-key", b"invalid-dummy-key")],
        "app": type("App", (), {"state": type("State", (), {"db": memory_db})()})(),
    }
    req = Request(scope)

    with pytest.raises(HTTPException) as exc_info:
        await verify_auth(req, credentials=None)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_auth_rejects_missing_token_when_active_keys_exist(
    memory_db: aiosqlite.Connection,
) -> None:
    await create_client_key(memory_db, "key1")
    scope = {
        "type": "http",
        "headers": [],
        "app": type("App", (), {"state": type("State", (), {"db": memory_db})()})(),
    }
    req = Request(scope)

    with pytest.raises(HTTPException) as exc_info:
        await verify_auth(req, credentials=None)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid or missing Bearer token"


def test_proxy_image_cache_wires_get_path_for_historical_images(test_client: TestClient) -> None:
    with patch("omnimux.services.multiplexer.Multiplexer.generate_stream") as mock_stream:

        async def mock_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield "result"

        mock_stream.return_value = mock_gen()

        with patch("omnimux.services.cache.ImageCache.get_path") as mock_get_path:
            res = test_client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": f"[Image: {'a' * 64}] followup question",
                        },
                    ],
                },
            )
            assert res.status_code == 200
            mock_get_path.assert_called()


@pytest.mark.asyncio
async def test_with_sse_keepalive_yields_comments_during_delay() -> None:
    async def delayed_generator() -> AsyncGenerator[bytes, None]:
        await asyncio.sleep(0.05)
        yield b"chunk1"
        await asyncio.sleep(0.05)
        yield b"chunk2"

    items = [item async for item in with_sse_keepalive(delayed_generator(), interval=0.02)]
    # Verify that keepalives were emitted
    assert b": keepalive\n\n" in items
    # Verify payload chunks are delivered in order
    payload_chunks = [item for item in items if item != b": keepalive\n\n"]
    assert payload_chunks == [b"chunk1", b"chunk2"]


@pytest.mark.asyncio
async def test_with_sse_keepalive_no_delay() -> None:
    async def fast_generator() -> AsyncGenerator[bytes, None]:
        yield b"fast1"
        yield b"fast2"

    items = [item async for item in with_sse_keepalive(fast_generator(), interval=1.0)]
    assert items == [b"fast1", b"fast2"]


@pytest.mark.asyncio
async def test_with_sse_keepalive_disabled() -> None:
    async def fast_generator() -> AsyncGenerator[bytes, None]:
        yield b"direct1"
        yield b"direct2"

    items = [item async for item in with_sse_keepalive(fast_generator(), interval=0.0)]
    assert items == [b"direct1", b"direct2"]


@pytest.mark.asyncio
async def test_with_sse_keepalive_exception_propagation() -> None:
    async def failing_generator() -> AsyncGenerator[bytes, None]:
        await asyncio.sleep(0.01)
        raise RuntimeError("upstream stream crashed")
        yield b"never"  # pragma: no cover

    with pytest.raises(RuntimeError, match="upstream stream crashed"):
        async for _ in with_sse_keepalive(failing_generator(), interval=0.05):
            pass


@pytest.mark.asyncio
async def test_with_sse_keepalive_cancellation_and_cleanup() -> None:
    closed = False

    async def long_generator() -> AsyncGenerator[bytes, None]:
        nonlocal closed
        try:
            yield b"first"
            await asyncio.sleep(1.0)
            yield b"second"
        finally:
            closed = True

    gen = with_sse_keepalive(long_generator(), interval=0.02)
    async for item in gen:
        assert item == b"first"
        break  # Consumer exits early

    await gen.aclose()
    assert closed is True


@pytest.mark.asyncio
async def test_with_sse_keepalive_external_task_cancellation() -> None:
    from contextlib import suppress

    async def infinite_sleep_generator() -> AsyncGenerator[bytes, None]:
        await asyncio.sleep(100.0)
        yield b"chunk"  # pragma: no cover

    async def consumer() -> None:
        async for _ in with_sse_keepalive(infinite_sleep_generator(), interval=0.05):
            pass

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def test_chat_completions_streaming_with_keepalive(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omnimux.config import settings

    monkeypatch.setattr(settings, "sse_keepalive_interval_seconds", 0.02)

    with patch("omnimux.services.multiplexer.Multiplexer.generate_stream") as mock_stream:

        async def mock_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            await asyncio.sleep(0.05)
            yield "thinking token"

        mock_stream.return_value = mock_gen()

        res = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "stream": True,
                "messages": [{"role": "user", "content": "solve a puzzle"}],
            },
        )
        assert res.status_code == 200
        text = res.text
        assert ": keepalive\n\n" in text
        assert "thinking token" in text
        assert "data: [DONE]\n\n" in text


def test_anthropic_messages_streaming_with_keepalive(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omnimux.config import settings

    monkeypatch.setattr(settings, "sse_keepalive_interval_seconds", 0.02)

    with patch("omnimux.services.multiplexer.Multiplexer.generate_stream") as mock_stream:

        async def mock_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            await asyncio.sleep(0.05)
            yield "reasoning token"

        mock_stream.return_value = mock_gen()

        res = test_client.post(
            "/v1/messages",
            json={
                "model": "claude-3-7-sonnet-20250219",
                "stream": True,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "deep thinking prompt"}],
            },
        )
        assert res.status_code == 200
        text = res.text
        assert ": keepalive\n\n" in text
        assert "reasoning token" in text
        assert "event: message_stop" in text


def test_extract_session_id_variants() -> None:
    from omnimux.routes.proxy import extract_session_id

    # 1. Header takes priority
    scope_header = {
        "type": "http",
        "headers": [(b"x-session-id", b"custom-sess-123")],
    }
    req_header = Request(scope_header)
    assert extract_session_id(req_header, []) == "custom-sess-123"

    # 2. Conversation id header
    scope_conv = {
        "type": "http",
        "headers": [(b"x-conversation-id", b"conv-456")],
    }
    assert extract_session_id(Request(scope_conv), []) == "conv-456"

    # 3. Deterministic fallback with message
    scope_none = {
        "type": "http",
        "headers": [(b"authorization", b"Bearer token123")],
    }
    sess_a = extract_session_id(Request(scope_none), [{"content": "initial system prompt"}])
    sess_b = extract_session_id(Request(scope_none), [{"content": "initial system prompt"}])
    assert sess_a.startswith("sess_")
    assert sess_a == sess_b

    # 4. Fallback with empty messages
    sess_empty = extract_session_id(Request(scope_none), [])
    assert sess_empty.startswith("sess_")

    # 5. Fallback with primitive string message
    sess_prim = extract_session_id(Request(scope_none), ["raw string turn"])
    assert sess_prim.startswith("sess_")

    # 6. Fallback with client IP when auth is missing
    scope_ip = {
        "type": "http",
        "headers": [],
        "client": ("192.168.1.50", 12345),
    }
    sess_ip = extract_session_id(Request(scope_ip), [])
    assert sess_ip.startswith("sess_")


def test_proxy_chat_completions_cache_usage(test_client: TestClient) -> None:
    with patch("omnimux.services.multiplexer.Multiplexer.generate_stream") as mock_stream:

        async def mock_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            usage_sink = kwargs.get("usage_sink")
            if isinstance(usage_sink, dict):
                usage_sink["cache_creation_input_tokens"] = 50
                usage_sink["cache_read_input_tokens"] = 150
            yield "Hello"

        mock_stream.side_effect = mock_gen

        res = test_client.post(
            "/v1/chat/completions",
            headers={"x-session-id": "test-chat-session"},
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["usage"]["cache_creation_input_tokens"] == 50
        assert data["usage"]["cache_read_input_tokens"] == 150


def test_proxy_anthropic_messages_cache_usage_streaming_and_non_streaming(
    test_client: TestClient,
) -> None:
    with patch("omnimux.services.multiplexer.Multiplexer.generate_stream") as mock_stream:

        async def mock_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            usage_sink = kwargs.get("usage_sink")
            if isinstance(usage_sink, dict):
                usage_sink["cache_creation_input_tokens"] = 25
                usage_sink["cache_read_input_tokens"] = 75
            yield "Anthropic chunk"

        mock_stream.side_effect = mock_gen

        # 1. Non-streaming
        res_non_stream = test_client.post(
            "/v1/messages",
            headers={"anthropic-session-id": "test-anthropic-session"},
            json={
                "model": "claude-3-7-sonnet-20250219",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
        assert res_non_stream.status_code == 200
        data = res_non_stream.json()
        assert data["usage"]["cache_creation_input_tokens"] == 25
        assert data["usage"]["cache_read_input_tokens"] == 75

        # 2. Streaming
        res_stream = test_client.post(
            "/v1/messages",
            headers={"anthropic-session-id": "test-anthropic-session"},
            json={
                "model": "claude-3-7-sonnet-20250219",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        assert res_stream.status_code == 200
        text = res_stream.text
        assert "event: message_delta" in text
        lines = [
            line
            for line in text.split("\n")
            if (line.startswith("data: ") and "message_delta" in line)
            or ("cache_read_input_tokens" in line)
        ]
        assert any("cache_read_input_tokens" in line for line in lines)
