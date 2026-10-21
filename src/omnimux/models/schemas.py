import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ImageUrl(BaseModel):
    url: str


class ContentBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None
    cache_control: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[ContentBlock]
    name: str | None = None
    cache_control: dict[str, Any] | None = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    keep_alive: int | str | None = None
    tools: list[dict[str, Any]] | None = None
    top_p: float | None = None


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[Any]
    cache_control: dict[str, Any] | None = None


class AnthropicMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int = 4096
    stream: bool = False
    temperature: float | None = None
    tools: list[dict[str, Any]] | None = None
    top_p: float | None = None


def convert_anthropic_to_chat_completion(
    req: AnthropicMessagesRequest,
) -> ChatCompletionRequest:
    chat_messages: list[ChatMessage] = []

    if req.system:
        if isinstance(req.system, str):
            chat_messages.append(ChatMessage(role="system", content=req.system))
        elif isinstance(req.system, list):
            has_cache_control = any(
                isinstance(b, dict) and b.get("cache_control") is not None for b in req.system
            )
            if has_cache_control:
                sys_blocks = [
                    ContentBlock(
                        type="text",
                        text=b.get("text", ""),
                        cache_control=b.get("cache_control"),
                    )
                    for b in req.system
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                if sys_blocks:
                    chat_messages.append(ChatMessage(role="system", content=sys_blocks))
            else:
                sys_text = "\n".join(
                    b.get("text", "")
                    for b in req.system
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                if sys_text:
                    chat_messages.append(ChatMessage(role="system", content=sys_text))

    for msg in req.messages:
        role = msg.role
        content = msg.content
        msg_cc = getattr(msg, "cache_control", None)
        if isinstance(content, str):
            chat_messages.append(ChatMessage(role=role, content=content, cache_control=msg_cc))
        elif isinstance(content, list):
            blocks: list[ContentBlock] = []
            has_image = False
            has_cache_control = msg_cc is not None
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                cc = item.get("cache_control")
                if cc is not None:
                    has_cache_control = True
                if item_type == "text":
                    blocks.append(
                        ContentBlock(type="text", text=item.get("text", ""), cache_control=cc)
                    )
                elif item_type == "image":
                    has_image = True
                    source = item.get("source", {})
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    blocks.append(
                        ContentBlock(
                            type="image_url",
                            image_url=ImageUrl(url=f"data:{media_type};base64,{data}"),
                            cache_control=cc,
                        )
                    )
                elif item_type == "tool_use":
                    name = item.get("name", "")
                    inp = json.dumps(item.get("input", {}))
                    blocks.append(
                        ContentBlock(
                            type="text",
                            text=f"[Tool use: {name} input={inp}]",
                            cache_control=cc,
                        )
                    )
                elif item_type == "tool_result":
                    res_content = item.get("content", "")
                    if isinstance(res_content, list):
                        res_text = "\n".join(
                            b.get("text", "")
                            for b in res_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    else:
                        res_text = str(res_content)
                    blocks.append(
                        ContentBlock(
                            type="text",
                            text=f"[Tool result: {res_text}]",
                            cache_control=cc,
                        )
                    )

            if has_image or has_cache_control:
                chat_messages.append(ChatMessage(role=role, content=blocks, cache_control=msg_cc))
            else:
                combined_text = "\n".join(b.text or "" for b in blocks if b.text)
                chat_messages.append(
                    ChatMessage(role=role, content=combined_text, cache_control=msg_cc)
                )

    return ChatCompletionRequest(
        model=req.model,
        messages=chat_messages,
        stream=req.stream,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        tools=req.tools,
        top_p=req.top_p,
    )
