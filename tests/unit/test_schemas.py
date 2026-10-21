from omnimux.models.schemas import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    ChatCompletionRequest,
    convert_anthropic_to_chat_completion,
)


def test_chat_completion_request_text() -> None:
    payload = {"model": "qwen3.5", "messages": [{"role": "user", "content": "Hello world!"}]}
    request = ChatCompletionRequest.model_validate(payload)
    assert request.model == "qwen3.5"
    assert len(request.messages) == 1
    assert request.messages[0].content == "Hello world!"


def test_chat_completion_request_multimodal() -> None:
    payload = {
        "model": "qwen3.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,12345"}},
                ],
            }
        ],
    }
    request = ChatCompletionRequest.model_validate(payload)
    content = request.messages[0].content
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0].type == "text"
    assert content[1].type == "image_url"
    assert content[1].image_url is not None
    assert content[1].image_url.url == "data:image/jpeg;base64,12345"


def test_chat_completion_extra_fields_preserved() -> None:
    payload = {"model": "qwen3.5", "messages": [], "some_unknown_field": "test_value"}
    request = ChatCompletionRequest.model_validate(payload)
    assert request.model_extra is not None
    assert request.model_extra["some_unknown_field"] == "test_value"


def test_convert_anthropic_to_chat_completion() -> None:
    # 1. System as list of blocks, content with mixed blocks and non-dict items
    req = AnthropicMessagesRequest(
        model="claude-3-5-sonnet",
        system=[{"type": "text", "text": "Be helpful."}, {"type": "other"}],
        messages=[
            AnthropicMessage(
                role="user",
                content=[
                    "not-a-dict",
                    {"type": "text", "text": "hello"},
                    {
                        "type": "tool_result",
                        "content": [{"type": "text", "text": "tool output"}],
                    },
                ],
            ),
            AnthropicMessage(
                role="user",
                content="plain string",
            ),
        ],
    )
    chat_req = convert_anthropic_to_chat_completion(req)
    assert len(chat_req.messages) == 3
    assert chat_req.messages[0].role == "system"
    assert chat_req.messages[0].content == "Be helpful."
    assert "hello" in str(chat_req.messages[1].content)
    assert chat_req.messages[2].content == "plain string"


def test_convert_anthropic_preserves_tools_and_top_p() -> None:
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
        }
    ]
    req = AnthropicMessagesRequest(
        model="claude-3-5-sonnet",
        messages=[AnthropicMessage(role="user", content="What's the weather?")],
        tools=tools,
        top_p=0.9,
    )
    converted = convert_anthropic_to_chat_completion(req)
    assert getattr(converted, "tools", None) == tools
    assert getattr(converted, "top_p", None) == 0.9


def test_convert_anthropic_preserves_cache_control() -> None:
    req = AnthropicMessagesRequest(
        model="claude-3-7-sonnet",
        system=[
            {"type": "text", "text": "System prompt", "cache_control": {"type": "ephemeral"}},
            {"type": "other"},
        ],
        messages=[
            AnthropicMessage(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": "Cached user turn",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "image",
                        "source": {"media_type": "image/png", "data": "abc"},
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "tool_use",
                        "name": "calc",
                        "input": {"x": 1},
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "tool_result",
                        "content": "res",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                cache_control={"type": "ephemeral"},
            ),
            AnthropicMessage(
                role="assistant",
                content="String message with cache control",
                cache_control={"type": "ephemeral"},
            ),
        ],
    )
    converted = convert_anthropic_to_chat_completion(req)
    assert len(converted.messages) == 3

    # System message structured ContentBlocks with cache_control
    sys_msg = converted.messages[0]
    assert sys_msg.role == "system"
    assert isinstance(sys_msg.content, list)
    assert sys_msg.content[0].text == "System prompt"
    assert sys_msg.content[0].cache_control == {"type": "ephemeral"}

    # User message blocks
    user_msg = converted.messages[1]
    assert user_msg.role == "user"
    assert user_msg.cache_control == {"type": "ephemeral"}
    assert isinstance(user_msg.content, list)
    assert len(user_msg.content) == 4
    assert user_msg.content[0].cache_control == {"type": "ephemeral"}
    assert user_msg.content[1].cache_control == {"type": "ephemeral"}
    assert user_msg.content[2].cache_control == {"type": "ephemeral"}
    assert user_msg.content[3].cache_control == {"type": "ephemeral"}

    # Assistant message
    asst_msg = converted.messages[2]
    assert asst_msg.role == "assistant"
    assert asst_msg.content == "String message with cache control"
    assert asst_msg.cache_control == {"type": "ephemeral"}
