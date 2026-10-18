import asyncio
import typing
from unittest.mock import AsyncMock

import httpx
import pytest

from omnimux.services.unloader import ModelUnloader


@pytest.fixture
def mock_client() -> AsyncMock:
    """Mock HTTP client for Ollama API calls."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


async def test_unloader_skipped_on_first_request(mock_client: AsyncMock) -> None:
    """The first request should skip unloading since no model is currently loaded."""
    engine = ModelUnloader(http_client=mock_client)

    unloaded = await engine.unload_if_needed("qwen3.5:9b-mlx")

    assert not unloaded
    assert engine._active_model == "qwen3.5:9b-mlx"
    assert mock_client.post.call_count == 0


async def test_unloader_skipped_when_same_model_requested(mock_client: AsyncMock) -> None:
    """Requesting the active model should skip unloading."""
    engine = ModelUnloader(http_client=mock_client)
    engine._active_model = "qwen3.5:9b-mlx"

    unloaded = await engine.unload_if_needed("qwen3.5:9b-mlx")

    assert not unloaded
    assert engine._active_model == "qwen3.5:9b-mlx"
    assert mock_client.post.call_count == 0


async def test_unloader_sends_correct_payload(mock_client: AsyncMock) -> None:
    """
    A model switch should trigger unloading of the previously active model with keep_alive: 0.

    Execution Flow:
    1. Sets up an initially active model.
    2. Mocks a successful POST response to the Ollama API.
    3. Requests a different model to trigger unloading.
    4. Asserts that the unloading request was sent with the correct payload.
    """
    engine = ModelUnloader(http_client=mock_client)
    engine._active_model = "qwen3.5:9b-mlx"

    req = httpx.Request("POST", "http://127.0.0.1:11434/api/generate")
    mock_response = httpx.Response(200, json={"status": "success"}, request=req)
    mock_client.post.return_value = mock_response

    unloaded = await engine.unload_if_needed("test-target-model")

    assert unloaded
    assert engine._active_model == "test-target-model"
    mock_client.post.assert_called_once_with(
        "http://127.0.0.1:11434/api/generate",
        json={"model": "qwen3.5:9b-mlx", "keep_alive": 0},
    )


async def test_concurrent_requests_dont_double_unload(mock_client: AsyncMock) -> None:
    """
    Multiple concurrent requests for the same new model should trigger unloading only once.

    Execution Flow:
    1. Mocks a delayed POST request to simulate a network call holding the asyncio lock.
    2. Sends three concurrent requests for the same target model.
    3. Asserts that only one request actually executed the HTTP unloading payload.
    """
    engine = ModelUnloader(http_client=mock_client)
    engine._active_model = "qwen3.5:9b-mlx"

    async def delayed_post(*args: typing.Any, **kwargs: typing.Any) -> httpx.Response:
        await asyncio.sleep(0.05)
        req = httpx.Request("POST", "http://127.0.0.1:11434/api/generate")
        return httpx.Response(200, json={"status": "success"}, request=req)

    mock_client.post.side_effect = delayed_post

    results = await asyncio.gather(
        engine.unload_if_needed("test-target-model"),
        engine.unload_if_needed("test-target-model"),
        engine.unload_if_needed("test-target-model"),
    )

    assert sum(1 for r in results if r) == 1
    assert engine._active_model == "test-target-model"
    assert mock_client.post.call_count == 1


async def test_unloader_error_handling(mock_client: AsyncMock) -> None:
    """If the unloading POST request fails, we should handle it or raise custom exceptions."""
    engine = ModelUnloader(http_client=mock_client)
    engine._active_model = "qwen3.5:9b-mlx"

    mock_client.post.side_effect = httpx.RequestError("Connection failed")

    result = await engine.unload_if_needed("test-target-model")

    assert result is True
    assert engine._active_model == "test-target-model"


@pytest.mark.asyncio
async def test_unloader_skips_cloud_models(mock_client: AsyncMock) -> None:
    engine = ModelUnloader(http_client=mock_client)
    for cloud_model in [
        "anthropic/claude-3-opus",
        "openai/gpt-4o",
        "claude-3-5-sonnet",
        "gpt-4",
        "gemini-1.5-pro",
    ]:
        engine._active_model = cloud_model
        unloaded = await engine.unload_if_needed("new-model")
        assert unloaded is False
        assert engine._active_model == "new-model"
        assert mock_client.post.call_count == 0


@pytest.mark.asyncio
async def test_unloader_platforms_data_and_prefixes(mock_client: AsyncMock) -> None:
    platforms_data = {
        "cloud-custom": {"platform": "openrouter"},
        "ollama-custom": {"platform": "ollama"},
        "non-dict": "string-val",
    }
    engine = ModelUnloader(http_client=mock_client, platforms_data=platforms_data)

    assert engine._is_ollama_model("cloud-custom") is False
    assert engine._is_ollama_model("ollama-custom") is True
    assert engine._is_ollama_model("non-dict") is True
    assert engine._is_ollama_model("ollama/llama3") is True
    assert engine._is_ollama_model("gemini.flash") is False

    req = httpx.Request("POST", "http://127.0.0.1:11434/api/generate")
    mock_client.post.return_value = httpx.Response(200, json={"status": "success"}, request=req)

    engine._active_model = "ollama/llama3"
    unloaded = await engine.unload_if_needed("target-model")
    assert unloaded is True
    mock_client.post.assert_called_once_with(
        "http://127.0.0.1:11434/api/generate",
        json={"model": "llama3", "keep_alive": 0},
    )
