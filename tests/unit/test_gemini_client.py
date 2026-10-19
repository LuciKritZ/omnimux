from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from omnimux.exceptions import GeminiAPIError, QuotaExceededError, RateLimitError
from omnimux.services.gemini_client import GeminiClient


def test_convert_openai_to_gemini() -> None:
    client = GeminiClient(client=httpx.AsyncClient())
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is in this image?"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And this one?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE"},
                    },
                ],
            },
            {"role": "assistant", "content": "It is a pixel."},
        ]
    }

    gemini_format = client._convert_openai_to_gemini(payload)

    expected = {
        "contents": [
            {"role": "user", "parts": [{"text": "What is in this image?"}]},
            {
                "role": "user",
                "parts": [
                    {"text": "And this one?"},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAE",
                        }
                    },
                ],
            },
            {"role": "model", "parts": [{"text": "It is a pixel."}]},
        ],
        "system_instruction": {"parts": [{"text": "You are a helpful assistant."}]},
    }

    assert gemini_format == expected


@pytest.mark.asyncio
async def test_generate_stream_success() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200

    async def mock_aiter_lines() -> AsyncGenerator[str, None]:
        yield 'data: {"candidates": [{"content": {"parts": [{"text": "chunk1"}]}}]}'
        yield 'data: {"candidates": [{"content": {"parts": [{"text": "chunk2"}]}}]}'
        yield "data: "
        yield "data: [DONE]"
        yield "data: invalid_json"

    mock_response.aiter_lines = mock_aiter_lines

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(client=mock_client)

    chunks = []
    async for chunk in client.generate_stream("model-1", []):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0] == "chunk1"
    assert chunks[1] == "chunk2"
    mock_client.stream.assert_called_once()


@pytest.mark.asyncio
async def test_generate_stream_rate_limit() -> None:
    # 429 should now raise RateLimitError immediately, skipping the HTTPStatusError retries
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {"retry-after": "120"}

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(client=mock_client)

    with pytest.raises(RateLimitError) as exc_info:
        async for _ in client.generate_stream("model-1", []):
            pass

    assert exc_info.value.retry_after == 120.0
    mock_client.stream.assert_called_once()


@pytest.mark.asyncio
async def test_generate_stream_rate_limit_invalid_reset() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {"retry-after": "invalid-float"}

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(client=mock_client)

    with pytest.raises(RateLimitError) as exc_info:
        async for _ in client.generate_stream("model-1", []):
            pass

    assert exc_info.value.retry_after == 60.0
    mock_client.stream.assert_called_once()


@pytest.mark.asyncio
async def test_generate_stream_quota_exceeded() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 403

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(client=mock_client)

    with pytest.raises(QuotaExceededError):
        async for _ in client.generate_stream("model-1", []):
            pass

    mock_client.stream.assert_called_once()


@pytest.mark.asyncio
@patch("omnimux.services.gemini_client.asyncio.sleep", new_callable=AsyncMock)
async def test_generate_stream_max_retries(mock_sleep: AsyncMock) -> None:
    mock_response_500 = MagicMock()
    mock_response_500.status_code = 500

    request = httpx.Request("POST", "url")
    error = httpx.HTTPStatusError("500 Server Error", request=request, response=mock_response_500)
    mock_response_500.raise_for_status.side_effect = error

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response_500)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(client=mock_client)

    with pytest.raises(GeminiAPIError, match="Max retries reached"):
        async for _ in client.generate_stream("model-1", []):
            pass

    assert mock_client.stream.call_count == 3
    assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_generate_stream_non_retryable_error() -> None:
    mock_response_400 = MagicMock()
    mock_response_400.status_code = 400
    mock_response_400.headers = {}

    request = httpx.Request("POST", "url")
    error = httpx.HTTPStatusError("400 Bad Request", request=request, response=mock_response_400)
    mock_response_400.raise_for_status.side_effect = error

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response_400)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(client=mock_client)

    # 400 isn't caught by the retry logic in `except httpx.HTTPStatusError as e`,
    # but handle_error raises it using `raise_for_status()`, so it will bubble up
    # However, GeminiClient's `except httpx.HTTPStatusError` raises `Unexpected HTTPStatusError`
    # Let's verify that behavior.
    with pytest.raises(GeminiAPIError, match="Unexpected HTTPStatusError: 400 Bad Request"):
        async for _ in client.generate_stream("model-1", []):
            pass

    assert mock_client.stream.call_count == 1


@pytest.mark.asyncio
@patch("omnimux.services.gemini_client.asyncio.sleep", new_callable=AsyncMock)
async def test_generate_stream_request_error_retries(mock_sleep: AsyncMock) -> None:
    request = httpx.Request("POST", "url")
    error = httpx.RequestError("Network unreachable", request=request)

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(side_effect=error)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(client=mock_client)

    with pytest.raises(GeminiAPIError, match="Network error after 3 attempts"):
        async for _ in client.generate_stream("model-1", []):
            pass

    assert mock_client.stream.call_count == 3
    assert mock_sleep.call_count == 2


def test_convert_openai_to_gemini_empty_parts() -> None:
    client = GeminiClient(client=httpx.AsyncClient())
    payload: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": []},
            {"role": "assistant", "content": [{"type": "unknown"}]},
        ]
    }
    gemini_format = client._convert_openai_to_gemini(payload)
    assert gemini_format == {"contents": []}


def test_gemini_client_billing_exhaustion_400() -> None:
    client = GeminiClient(client=httpx.AsyncClient())
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com")
    resp_billing = httpx.Response(
        400,
        request=req,
        text=(
            '{"error": {"status": "RESOURCE_EXHAUSTED", '
            '"message": "Quota exceeded for quota metric"}}'
        ),
    )
    with pytest.raises(QuotaExceededError, match="Gemini Quota Exceeded"):
        client.handle_error(resp_billing)
