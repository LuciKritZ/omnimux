from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from omnimux.exceptions import QuotaExceededError, RateLimitError
from omnimux.services.adapters import (
    AnthropicClient,
    BaseLLMClient,
    OpenAICompatibleClient,
    _safe_read_error_response,
    is_billing_exhaustion_error,
)


@pytest.fixture
def openai_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(api_key="test-key", base_url="https://api.test")


@pytest.mark.asyncio
async def test_openai_client_rate_limit(openai_client: OpenAICompatibleClient) -> None:
    # Simulate a 429 response from an OpenAI-compatible API
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(429, headers={"x-ratelimit-reset": "120"}, request=request)

    with pytest.raises(RateLimitError) as exc_info:
        openai_client.handle_error(response)

    assert exc_info.value.retry_after == 120


@pytest.mark.asyncio
async def test_openai_client_rate_limit_default_reset(
    openai_client: OpenAICompatibleClient,
) -> None:
    # Simulate a 429 response without a reset header (should default to 60)
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(429, request=request)

    with pytest.raises(RateLimitError) as exc_info:
        openai_client.handle_error(response)

    assert exc_info.value.retry_after == 60


@pytest.mark.asyncio
async def test_openai_client_quota_exceeded(openai_client: OpenAICompatibleClient) -> None:
    # Simulate a 402 response (insufficient credits)
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(402, request=request)

    with pytest.raises(QuotaExceededError):
        openai_client.handle_error(response)


@pytest.fixture
def anthropic_client() -> AnthropicClient:
    return AnthropicClient(api_key="test-key", base_url="https://api.anthropic.com")


@pytest.mark.asyncio
async def test_anthropic_client_generate_stream(anthropic_client: AnthropicClient) -> None:
    # Use the mock_stream context manager provided by httpx
    # Here we mock httpx.AsyncClient.stream
    with patch("httpx.AsyncClient.stream") as mock_stream:
        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200

        # Create an async iterator for aiter_lines
        async def mock_aiter_lines() -> AsyncGenerator[str, None]:
            yield (
                'data: {"type": "content_block_delta", '
                '"delta": {"type": "text_delta", "text": "Hello"}}'
            )
            yield "data: "
            yield (
                'data: {"type": "content_block_delta", '
                '"delta": {"type": "text_delta", "text": " World"}}'
            )
            yield 'data: {"invalid": json}'
            yield "not a data line"

        mock_response.aiter_lines.return_value = mock_aiter_lines()

        # The stream returns an async context manager which returns the mock_response
        class AsyncContextManager:
            async def __aenter__(self) -> MagicMock:
                return mock_response

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                pass

        mock_stream.return_value = AsyncContextManager()

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        chunks = [
            chunk
            async for chunk in anthropic_client.generate_stream(
                "claude-3-haiku", messages, max_tokens=100
            )
        ]

        assert chunks == ["Hello", " World"]


@pytest.mark.asyncio
async def test_anthropic_client_rate_limit(anthropic_client: AnthropicClient) -> None:
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(429, headers={"anthropic-retry-after": "42.0"}, request=request)

    with pytest.raises(RateLimitError) as exc_info:
        anthropic_client.handle_error(response)

    assert exc_info.value.retry_after == 42.0


@pytest.mark.asyncio
async def test_anthropic_client_quota_exceeded(anthropic_client: AnthropicClient) -> None:
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(403, request=request)

    with pytest.raises(QuotaExceededError):
        anthropic_client.handle_error(response)


@pytest.mark.asyncio
async def test_anthropic_client_error_fallback(anthropic_client: AnthropicClient) -> None:
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(500, request=request)

    with pytest.raises(httpx.HTTPStatusError):
        anthropic_client.handle_error(response)


@pytest.mark.asyncio
async def test_anthropic_client_rate_limit_invalid(anthropic_client: AnthropicClient) -> None:
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(429, headers={"anthropic-retry-after": "invalid"}, request=request)

    with pytest.raises(RateLimitError) as exc_info:
        anthropic_client.handle_error(response)

    assert exc_info.value.retry_after == 60.0


@pytest.mark.asyncio
async def test_anthropic_client_generate_stream_kwargs(anthropic_client: AnthropicClient) -> None:
    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=httpx.Request("POST", "http://test"), response=mock_response
        )

        class AsyncContextManager:
            async def __aenter__(self) -> MagicMock:
                return mock_response

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                pass

        mock_stream.return_value = AsyncContextManager()

        with pytest.raises(httpx.HTTPStatusError):
            [
                c
                async for c in anthropic_client.generate_stream(
                    "model", [{"role": "user", "content": "hi"}], temperature=0.5
                )
            ]


@pytest.mark.asyncio
async def test_base_llm_client_abstract_methods() -> None:
    # Test that the abstract base class methods don't crash when called generically
    class DummyClient(BaseLLMClient):
        async def generate_stream(
            self, model: str, messages: list[dict[str, Any]], **kwargs: Any
        ) -> AsyncGenerator[str, None]:
            async for chunk in BaseLLMClient.generate_stream(self, model, messages, **kwargs):
                yield chunk

        def handle_error(self, response: httpx.Response) -> None:
            BaseLLMClient.handle_error(self, response)

    dummy = DummyClient("key", "url")
    dummy.handle_error(httpx.Response(200))

    chunks = [c async for c in dummy.generate_stream("m", [])]
    assert chunks == [""]


@pytest.mark.asyncio
async def test_openai_client_rate_limit_invalid_reset(
    openai_client: OpenAICompatibleClient,
) -> None:
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(429, headers={"x-ratelimit-reset": "invalid-float"}, request=request)

    with pytest.raises(RateLimitError) as exc_info:
        openai_client.handle_error(response)

    assert exc_info.value.retry_after == 60.0


@pytest.mark.asyncio
async def test_openai_client_generic_error(openai_client: OpenAICompatibleClient) -> None:
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(500, request=request)

    with pytest.raises(httpx.HTTPStatusError):
        openai_client.handle_error(response)


@pytest.mark.asyncio
async def test_openai_client_generate_stream(
    openai_client: OpenAICompatibleClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # We mock httpx.AsyncClient.stream to simulate the async generator yielding lines
    import contextlib

    class MockStreamResponse:
        def __init__(self, lines: list[str]) -> None:
            self.lines = lines
            self.status_code = 200

        async def aiter_lines(self) -> AsyncGenerator[str, None]:
            for line in self.lines:
                yield line

    @contextlib.asynccontextmanager
    async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[MockStreamResponse, None]:
        lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            "data: [DONE]",
        ]
        yield MockStreamResponse(lines)

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    chunks = []
    async for chunk in openai_client.generate_stream(
        "model-1", [{"role": "user", "content": "hi"}]
    ):
        chunks.append(chunk)

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_openai_client_generate_stream_error(
    openai_client: OpenAICompatibleClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib

    class MockStreamResponse:
        def __init__(self) -> None:
            self.status_code = 402
            self.headers: dict[str, str] = {}
            self.request = httpx.Request("POST", "https://api.test")

        def raise_for_status(self) -> None:
            pass

    @contextlib.asynccontextmanager
    async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[MockStreamResponse, None]:
        yield MockStreamResponse()

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    with pytest.raises(QuotaExceededError):
        async for _ in openai_client.generate_stream("model-1", []):
            pass


@pytest.mark.asyncio
async def test_openai_client_generate_stream_invalid_json(
    openai_client: OpenAICompatibleClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib

    class MockStreamResponse:
        def __init__(self, lines: list[str]) -> None:
            self.lines = lines
            self.status_code = 200

        async def aiter_lines(self) -> AsyncGenerator[str, None]:
            for line in self.lines:
                yield line

    @contextlib.asynccontextmanager
    async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[MockStreamResponse, None]:
        lines = ['data: {"invalid_json":', "data: [DONE]"]
        yield MockStreamResponse(lines)

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    chunks = []
    async for chunk in openai_client.generate_stream("model-1", []):
        chunks.append(chunk)

    assert chunks == []


@pytest.mark.asyncio
async def test_anthropic_client_forwards_anthropic_beta() -> None:
    client = AnthropicClient(api_key="test-key", base_url="https://api.anthropic.com")
    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        async def aiter_lines() -> AsyncGenerator[str, None]:
            yield (
                'data: {"type": "content_block_delta", "delta": '
                '{"type": "text_delta", "text": "ok"}}'
            )

        mock_resp.aiter_lines = aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_resp

        chunks = []
        async for c in client.generate_stream(
            "claude-3-5-sonnet",
            [{"role": "user", "content": "hi"}],
            anthropic_beta="prompt-caching-2024-07-31",
        ):
            chunks.append(c)

        _, kwargs = mock_stream.call_args
        assert "anthropic-beta" in kwargs["headers"]
        assert kwargs["headers"]["anthropic-beta"] == "prompt-caching-2024-07-31"


@pytest.mark.asyncio
async def test_anthropic_client_thinking_and_partial_json() -> None:
    client = AnthropicClient(api_key="test-key", base_url="https://api.anthropic.com")
    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        async def aiter_lines() -> AsyncGenerator[str, None]:
            yield (
                'data: {"type": "content_block_delta", "delta": '
                '{"type": "thinking_delta", "thinking": "Let me think."}}'
            )
            yield (
                'data: {"type": "content_block_delta", "delta": '
                '{"type": "input_json_delta", "partial_json": "{\\"arg\\": 1}"}}'
            )
            yield (
                'data: {"type": "content_block_delta", "delta": '
                '{"type": "text_delta", "text": "Done"}}'
            )

        mock_resp.aiter_lines = aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_resp

        chunks = []
        async for c in client.generate_stream(
            "claude-3-5-sonnet",
            [{"role": "user", "content": "hi"}],
        ):
            chunks.append(c)

        assert "Let me think." in chunks
        assert '{"arg": 1}' in chunks
        assert "Done" in chunks


def test_is_billing_exhaustion_error_patterns() -> None:
    req = httpx.Request("POST", "https://api.test")

    # 402 and 403 are always billing/quota errors
    assert is_billing_exhaustion_error(httpx.Response(402, request=req)) is True
    assert is_billing_exhaustion_error(httpx.Response(403, request=req)) is True

    # 500 or non-400 are not billing exhaustion
    assert is_billing_exhaustion_error(httpx.Response(500, request=req)) is False

    # 400 with billing keywords
    resp_anthropic = httpx.Response(
        400,
        request=req,
        text=(
            '{"type": "error", "error": {"type": "invalid_request_error", '
            '"message": "credit balance is too low"}}'
        ),
    )
    assert is_billing_exhaustion_error(resp_anthropic) is True

    resp_openai = httpx.Response(
        400,
        request=req,
        text=(
            '{"error": {"code": "billing_hard_limit_reached", '
            '"message": "Billing hard limit reached"}}'
        ),
    )
    assert is_billing_exhaustion_error(resp_openai) is True

    # 400 syntax error without billing keywords
    resp_syntax = httpx.Response(
        400,
        request=req,
        text='{"error": {"message": "Invalid parameter: max_tokens must be positive"}}',
    )
    assert is_billing_exhaustion_error(resp_syntax) is False

    # Test error decoding fallback when .text raises exception
    class BrokenResponse:
        status_code = 400

        @property
        def text(self) -> str:
            raise RuntimeError("Unreadable response body")

    assert is_billing_exhaustion_error(BrokenResponse()) is False  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_openai_client_billing_exhaustion_400(
    openai_client: OpenAICompatibleClient,
) -> None:
    req = httpx.Request("POST", "https://api.test")
    resp_billing = httpx.Response(
        400,
        request=req,
        text='{"error": {"message": "You exceeded your current quota, please check your plan."}}',
    )
    with pytest.raises(QuotaExceededError, match="Quota exceeded or unauthorized"):
        openai_client.handle_error(resp_billing)

    resp_syntax = httpx.Response(
        400,
        request=req,
        text='{"error": {"message": "Invalid temperature value."}}',
    )
    with pytest.raises(httpx.HTTPStatusError):
        openai_client.handle_error(resp_syntax)


@pytest.mark.asyncio
async def test_anthropic_client_billing_exhaustion_400(
    anthropic_client: AnthropicClient,
) -> None:
    req = httpx.Request("POST", "https://api.anthropic.com")
    resp_billing = httpx.Response(
        400,
        request=req,
        text='{"error": {"message": "Your credit balance is too low to access the Claude API."}}',
    )
    with pytest.raises(QuotaExceededError, match="Anthropic Quota Exceeded"):
        anthropic_client.handle_error(resp_billing)

    resp_syntax = httpx.Response(
        400,
        request=req,
        text='{"error": {"message": "Messages array cannot be empty."}}',
    )
    with pytest.raises(httpx.HTTPStatusError):
        anthropic_client.handle_error(resp_syntax)


@pytest.mark.asyncio
async def test_safe_read_error_response() -> None:
    # Awaitable aread
    resp_with_async_read = MagicMock()
    resp_with_async_read.aread = AsyncMock()
    await _safe_read_error_response(resp_with_async_read)
    resp_with_async_read.aread.assert_awaited_once()

    # Non-awaitable aread
    resp_sync = MagicMock()
    resp_sync.aread = MagicMock(return_value="not-awaitable")
    await _safe_read_error_response(resp_sync)

    # Missing aread
    await _safe_read_error_response(object())


@pytest.mark.asyncio
async def test_base_llm_client_get_client() -> None:
    mock_http = httpx.AsyncClient()
    client = OpenAICompatibleClient(api_key="k", client=mock_http)
    async with client._get_client() as c:
        assert c is mock_http
    await mock_http.aclose()


@pytest.mark.asyncio
async def test_has_cache_control_helper() -> None:
    from omnimux.services.adapters import _has_cache_control

    assert _has_cache_control({"cache_control": {"type": "ephemeral"}}) is True
    assert _has_cache_control([{"nested": {"cache_control": {"type": "ephemeral"}}}]) is True
    assert _has_cache_control({"text": "no cache control"}) is False
    assert _has_cache_control("plain string") is False
    assert _has_cache_control(None) is False


@pytest.mark.asyncio
async def test_anthropic_client_cache_control_and_usage(
    anthropic_client: AnthropicClient,
) -> None:
    captured_request: dict[str, Any] = {}

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_aiter_lines() -> AsyncGenerator[str, None]:
            yield (
                'data: {"type": "message_start", "message": {"usage": '
                '{"input_tokens": 150, "cache_creation_input_tokens": 80, '
                '"cache_read_input_tokens": 70}}}'
            )
            yield (
                'data: {"type": "content_block_delta", '
                '"delta": {"type": "text_delta", "text": "Cached response"}}'
            )
            yield 'data: {"type": "message_delta", "usage": {"output_tokens": 25}}'

        mock_response.aiter_lines.return_value = mock_aiter_lines()

        class AsyncContextManager:
            async def __aenter__(self) -> MagicMock:
                return mock_response

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                pass

        def side_effect(
            method: str, url: str, headers: dict[str, str], json: dict[str, Any]
        ) -> AsyncContextManager:
            captured_request["method"] = method
            captured_request["url"] = url
            captured_request["headers"] = headers
            captured_request["json"] = json
            return AsyncContextManager()

        mock_stream.side_effect = side_effect

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "System prompt with cache",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "User msg",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ]

        chunks = []
        usage_sink: dict[str, Any] = {}
        async for chunk in anthropic_client.generate_stream(
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            anthropic_beta="interleaved-thinking-2025-02-19",
            usage_sink=usage_sink,
        ):
            chunks.append(chunk)

        assert "".join(chunks) == "Cached response"
        assert usage_sink["cache_creation_input_tokens"] == 80
        assert usage_sink["cache_read_input_tokens"] == 70
        # Beta header appended
        captured_headers: dict[str, str] = captured_request["headers"]
        captured_json: dict[str, Any] = captured_request["json"]
        assert "prompt-caching-2024-07-31" in captured_headers["anthropic-beta"]
        assert "interleaved-thinking-2025-02-19" in captured_headers["anthropic-beta"]
        # System was passed as list of dicts
        assert isinstance(captured_json["system"], list)
        assert captured_json["system"][0]["cache_control"] == {"type": "ephemeral"}
        # Check usage recorded
        assert anthropic_client.last_usage["cache_creation_input_tokens"] == 80
        assert anthropic_client.last_usage["cache_read_input_tokens"] == 70
        assert anthropic_client.last_usage["output_tokens"] == 25


@pytest.mark.asyncio
async def test_anthropic_client_cache_control_auto_inject_header_when_none(
    anthropic_client: AnthropicClient,
) -> None:
    captured_request: dict[str, Any] = {}

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_aiter_lines() -> AsyncGenerator[str, None]:
            yield 'data: {"type": "content_block_delta", "delta": {"text": "ok"}}'

        mock_response.aiter_lines.return_value = mock_aiter_lines()

        class AsyncContextManager:
            async def __aenter__(self) -> MagicMock:
                return mock_response

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                pass

        def side_effect(
            method: str, url: str, headers: dict[str, str], json: dict[str, Any]
        ) -> AsyncContextManager:
            captured_request["headers"] = headers
            return AsyncContextManager()

        mock_stream.side_effect = side_effect

        messages = [
            {
                "role": "user",
                "content": "hello",
                "cache_control": {"type": "ephemeral"},
            }
        ]

        async for _ in anthropic_client.generate_stream(
            model="claude-3-7-sonnet",
            messages=messages,
        ):
            pass

        captured_headers: dict[str, str] = captured_request["headers"]
        assert captured_headers["anthropic-beta"] == "prompt-caching-2024-07-31"


@pytest.mark.asyncio
async def test_anthropic_client_system_kwargs_and_mixed_system(
    anthropic_client: AnthropicClient,
) -> None:
    captured_payloads: list[dict[str, Any]] = []

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_aiter_lines() -> AsyncGenerator[str, None]:
            yield 'data: {"type": "content_block_delta", "delta": {"text": "ok"}}'

        mock_response.aiter_lines.return_value = mock_aiter_lines()

        class AsyncContextManager:
            async def __aenter__(self) -> MagicMock:
                return mock_response

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                pass

        def side_effect(
            method: str, url: str, headers: dict[str, str], json: dict[str, Any]
        ) -> AsyncContextManager:
            captured_payloads.append(json)
            return AsyncContextManager()

        mock_stream.side_effect = side_effect

        # 1. System passed via kwargs
        async for _ in anthropic_client.generate_stream(
            model="claude-3-7-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            system="Explicit system instructions",
        ):
            pass
        assert captured_payloads[0]["system"] == "Explicit system instructions"

        # 2. System messages list then string, plus non-dict item in content list
        mixed_messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": "Part 1"}, "raw-string-block"]},
            {"role": "system", "content": "Part 2"},
            {"role": "user", "content": "hello"},
        ]
        async for _ in anthropic_client.generate_stream(
            model="claude-3-7-sonnet",
            messages=mixed_messages,
        ):
            pass
        assert isinstance(captured_payloads[1]["system"], list)
        assert len(captured_payloads[1]["system"]) == 3


@pytest.mark.asyncio
async def test_openai_client_usage_stream(openai_client: OpenAICompatibleClient) -> None:
    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_aiter_lines() -> AsyncGenerator[str, None]:
            yield (
                'data: {"choices": [{"delta": {"content": "Hello"}}], '
                '"usage": {"prompt_tokens": 10, "completion_tokens": 5, '
                '"prompt_tokens_details": {"cached_tokens": 8}}}'
            )

        mock_response.aiter_lines.return_value = mock_aiter_lines()

        class AsyncContextManager:
            async def __aenter__(self) -> MagicMock:
                return mock_response

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                pass

        mock_stream.return_value = AsyncContextManager()

        sink: dict[str, Any] = {}
        async for chunk in openai_client.generate_stream(
            "gpt-4o", [{"role": "user", "content": "hi"}], usage_sink=sink
        ):
            assert chunk == "Hello"

        assert openai_client.last_usage["prompt_tokens"] == 10
        assert openai_client.last_usage["completion_tokens"] == 5
        assert openai_client.last_usage["prompt_tokens_details_cached_tokens"] == 8
        assert sink["prompt_tokens"] == 10
        assert sink["completion_tokens"] == 5
        assert sink["prompt_tokens_details_cached_tokens"] == 8
