from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimux.exceptions import RateLimitError
from omnimux.services.multiplexer import Multiplexer
from omnimux.services.router_core import NoKeysAvailableError


class MockStreamGen:
    def __init__(self, items: list[str | Exception]) -> None:
        self.items = items
        self.index = 0

    def __aiter__(self) -> "MockStreamGen":
        return self

    async def __anext__(self) -> str:
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def mock_router() -> MagicMock:
    router = MagicMock()

    class MockContextManager:
        async def __aenter__(self) -> None:
            pass

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    router.use_key.return_value = MockContextManager()
    router.get_best_platform_and_key = AsyncMock()
    router.mark_key_exhausted = AsyncMock()
    return router


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_success(mock_get_client: MagicMock, mock_router: MagicMock) -> None:
    mock_router.get_best_platform_and_key.return_value = (
        "openrouter",
        {"key_hash": "h1", "key_value": "v1"},
    )

    mock_client = MagicMock()
    mock_client.generate_stream.return_value = MockStreamGen(["chunk1", "chunk2"])
    mock_get_client.return_value = mock_client

    multiplexer = Multiplexer(mock_router)
    chunks = [c async for c in multiplexer.generate_stream(["openrouter/model"], [])]

    assert chunks == ["chunk1", "chunk2"]
    mock_router.get_best_platform_and_key.assert_called_once_with(["openrouter"])
    mock_router.mark_key_exhausted.assert_not_called()


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_hot_swap_on_error(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    mock_router.get_best_platform_and_key.side_effect = [
        ("openrouter", {"key_hash": "h1", "key_value": "v1"}),
        ("openrouter", {"key_hash": "h2", "key_value": "v2"}),
    ]

    client1 = MagicMock()
    client1.generate_stream.return_value = MockStreamGen([RateLimitError("429", retry_after=30.0)])

    client2 = MagicMock()
    client2.generate_stream.return_value = MockStreamGen(["success_chunk"])

    mock_get_client.side_effect = [client1, client2]

    multiplexer = Multiplexer(mock_router)
    chunks = [c async for c in multiplexer.generate_stream(["openrouter/model"], [])]

    assert chunks == ["success_chunk"]
    assert mock_router.get_best_platform_and_key.call_count == 2
    mock_router.mark_key_exhausted.assert_called_once_with("h1", 30.0)


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_fallback_to_ollama_on_exhaust(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    mock_router.get_best_platform_and_key.side_effect = NoKeysAvailableError("No keys")

    ollama_client = MagicMock()
    ollama_client.generate_stream.return_value = MockStreamGen(["ollama_chunk"])
    mock_get_client.return_value = ollama_client

    multiplexer = Multiplexer(mock_router)
    chunks = [c async for c in multiplexer.generate_stream(["openrouter/model"], [])]

    assert chunks == ["ollama_chunk"]
    mock_get_client.assert_called_once_with("ollama", "", None, "http://127.0.0.1:11434")
    mock_router.mark_key_exhausted.assert_not_called()


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_fallback_to_ollama_after_max_retries(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    mock_router.get_best_platform_and_key.return_value = (
        "openrouter",
        {"key_hash": "h1", "key_value": "v1"},
    )

    # 3 attempts fail
    client_fail = MagicMock()
    client_fail.generate_stream.side_effect = lambda *args, **kwargs: MockStreamGen(
        [Exception("Error")]
    )

    ollama_client = MagicMock()
    ollama_client.generate_stream.side_effect = lambda *args, **kwargs: MockStreamGen(
        ["ollama_fallback"]
    )

    mock_get_client.side_effect = [client_fail, client_fail, client_fail, ollama_client]

    multiplexer = Multiplexer(mock_router)
    chunks = [c async for c in multiplexer.generate_stream(["openrouter/model"], [])]

    assert chunks == ["ollama_fallback"]


def test_get_client_for_platform() -> None:
    from omnimux.services.adapters import OpenAICompatibleClient
    from omnimux.services.gemini_client import GeminiClient
    from omnimux.services.multiplexer import get_client_for_platform

    client1 = get_client_for_platform("gemini", "key1")
    assert isinstance(client1, GeminiClient)

    client2 = get_client_for_platform("openrouter", "key2")
    assert isinstance(client2, OpenAICompatibleClient)

    client3 = get_client_for_platform("ollama", "")
    assert isinstance(client3, OpenAICompatibleClient)

    with pytest.raises(ValueError):
        get_client_for_platform("unknown", "key")


def test_get_client_for_groq() -> None:
    from omnimux.services.adapters import OpenAICompatibleClient
    from omnimux.services.multiplexer import get_client_for_platform

    client = get_client_for_platform("groq", "test_key")
    assert isinstance(client, OpenAICompatibleClient)


def test_get_client_for_anthropic() -> None:
    from omnimux.services.adapters import AnthropicClient
    from omnimux.services.multiplexer import get_client_for_platform

    client = get_client_for_platform("anthropic", "test_key")
    assert isinstance(client, AnthropicClient)


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_empty_stream(mock_get_client: MagicMock, mock_router: MagicMock) -> None:
    mock_router.get_best_platform_and_key.return_value = (
        "openrouter",
        {"key_hash": "h1", "key_value": "v1"},
    )

    mock_client = MagicMock()
    mock_client.generate_stream.return_value = MockStreamGen([])
    mock_get_client.return_value = mock_client

    multiplexer = Multiplexer(mock_router)
    chunks = [c async for c in multiplexer.generate_stream(["openrouter/model"], [])]

    assert chunks == []
    assert mock_router.get_best_platform_and_key.call_count == 1
    assert mock_router.mark_key_exhausted.call_count == 0


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_system_fallback(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_client.generate_stream.return_value = MockStreamGen(["Fallback", " works"])
    mock_get_client.return_value = mock_client

    multiplexer = Multiplexer(
        mock_router,
        platforms_data={
            "system_fallback": {"model": "fallback-model", "apiBase": "http://fallback:11434"}
        },
    )
    chunks = [c async for c in multiplexer.generate_stream(["system_fallback"], [])]

    assert chunks == ["Fallback", " works"]
    mock_get_client.assert_called_once_with("ollama", "", None, "http://fallback:11434")
    assert mock_router.get_best_platform_and_key.call_count == 0


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_ollama_fast_path(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_client.generate_stream.return_value = MockStreamGen(["Ollama", " is", " fast"])
    mock_get_client.return_value = mock_client

    multiplexer = Multiplexer(mock_router)
    chunks = [c async for c in multiplexer.generate_stream(["ollama/qwen"], [])]

    assert chunks == ["Ollama", " is", " fast"]
    assert mock_router.get_best_platform_and_key.call_count == 0


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_dot_notation(mock_get_client: MagicMock, mock_router: MagicMock) -> None:
    mock_router.get_best_platform_and_key.return_value = (
        "anthropic",
        {"key_hash": "h1", "key_value": "v1"},
    )

    mock_client = MagicMock()
    mock_client.generate_stream.return_value = MockStreamGen(["dot", " works"])
    mock_get_client.return_value = mock_client

    multiplexer = Multiplexer(mock_router)
    chunks = [
        c
        async for c in multiplexer.generate_stream(
            ["anthropic.claude-3-5-sonnet-20241022", "unknown_model"], []
        )
    ]

    assert chunks == ["dot", " works"]
    mock_client.generate_stream.assert_called_once_with("claude-3-5-sonnet-20241022", [], **{})


def test_get_client_for_platform_custom_protocols() -> None:
    from omnimux.services.adapters import AnthropicClient, OpenAICompatibleClient
    from omnimux.services.multiplexer import get_client_for_platform

    c1 = get_client_for_platform("tokenharbor_anthropic", "k1", api_base="https://custom.api")
    assert isinstance(c1, AnthropicClient)

    c2 = get_client_for_platform("tokenharbor_openai", "k2", api_base="https://custom.api")
    assert isinstance(c2, OpenAICompatibleClient)

    c3 = get_client_for_platform("custom", "k3", protocol="anthropic")
    assert isinstance(c3, AnthropicClient)


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_tier_alias_mapping(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    import json

    meta = {
        "api_base": "https://api.tokenharbor.com",
        "models": {
            "opus": "mapped-opus",
            "sonnet": "mapped-sonnet",
            "haiku": "mapped-haiku",
            "custom": "mapped-custom",
        },
    }
    mock_router.get_best_platform_and_key.return_value = (
        "anthropic",
        {"key_hash": "h1", "key_value": "v1", "meta": json.dumps(meta)},
    )
    mock_client = MagicMock()
    mock_client.generate_stream.side_effect = lambda *args, **kwargs: MockStreamGen(["ok"])
    mock_get_client.return_value = mock_client

    multiplexer = Multiplexer(mock_router)

    for tier, expected in [
        ("claude-3-opus", "mapped-opus"),
        ("claude-3-sonnet", "mapped-sonnet"),
        ("claude-3-haiku", "mapped-haiku"),
        ("custom", "mapped-custom"),
    ]:
        chunks = [c async for c in multiplexer.generate_stream([f"anthropic/{tier}"], [])]
        assert chunks == ["ok"]
        mock_client.generate_stream.assert_called_with(expected, [], **{})


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_quota_exceeded(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    from omnimux.exceptions import QuotaExceededError

    mock_router.get_best_platform_and_key.side_effect = [
        ("anthropic", {"key_hash": "h1", "key_value": "v1", "meta": "{invalid json"}),
        ("openrouter", {"key_hash": "h2", "key_value": "v2"}),
    ]

    mock_client1 = MagicMock()
    mock_client1.generate_stream.return_value = MockStreamGen(
        [QuotaExceededError("Credit exhausted")]
    )
    mock_client2 = MagicMock()
    mock_client2.generate_stream.return_value = MockStreamGen(["recovered"])

    mock_get_client.side_effect = [mock_client1, mock_client2]

    multiplexer = Multiplexer(mock_router)
    chunks = [
        c
        async for c in multiplexer.generate_stream(
            ["anthropic/claude-3-5-sonnet", "openrouter/free"], []
        )
    ]
    assert chunks == ["recovered"]
    assert mock_router.mark_key_exhausted.call_count >= 1


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_mid_stream_failure_reraises(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    from omnimux.exceptions import QuotaExceededError

    mock_router.get_best_platform_and_key.return_value = (
        "openrouter",
        {"key_hash": "h1", "key_value": "v1"},
    )

    # Yields first chunk, then raises Exception mid-stream
    mock_client = MagicMock()
    mock_client.generate_stream.return_value = MockStreamGen(
        ["chunk1", RuntimeError("Mid stream network drop")]
    )
    mock_get_client.return_value = mock_client

    multiplexer = Multiplexer(mock_router)
    gen = multiplexer.generate_stream(["openrouter/model"], [])

    assert await anext(gen) == "chunk1"
    with pytest.raises(RuntimeError, match="Mid stream network drop"):
        await anext(gen)

    # Should not mark key exhausted or attempt fallback
    mock_router.mark_key_exhausted.assert_not_called()

    # Test mid-stream QuotaExceededError
    mock_client.generate_stream.return_value = MockStreamGen(
        ["chunk1", QuotaExceededError("Mid stream quota")]
    )
    gen2 = multiplexer.generate_stream(["openrouter/model"], [])
    assert await anext(gen2) == "chunk1"
    with pytest.raises(QuotaExceededError, match="Mid stream quota"):
        await anext(gen2)

    # Test mid-stream RateLimitError
    mock_client.generate_stream.return_value = MockStreamGen(
        ["chunk1", RateLimitError("Mid stream rate limit")]
    )
    gen3 = multiplexer.generate_stream(["openrouter/model"], [])
    assert await anext(gen3) == "chunk1"
    with pytest.raises(RateLimitError, match="Mid stream rate limit"):
        await anext(gen3)


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_http_400_not_exhausted(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    import httpx

    mock_router.get_best_platform_and_key.return_value = (
        "openrouter",
        {"key_hash": "h1", "key_value": "v1"},
    )

    req = httpx.Request("POST", "https://api.openrouter.ai")
    resp = httpx.Response(400, request=req)
    err = httpx.HTTPStatusError("Bad Request", request=req, response=resp)

    mock_client = MagicMock()
    mock_client.generate_stream.return_value = MockStreamGen([err])
    mock_get_client.return_value = mock_client

    multiplexer = Multiplexer(mock_router)
    gen = multiplexer.generate_stream(["openrouter/model"], [])

    with pytest.raises(httpx.HTTPStatusError):
        await anext(gen)

    mock_router.mark_key_exhausted.assert_not_called()


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_model_retry_indexing(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    # Two keys for openrouter, candidate models has two openrouter models
    mock_router.get_best_platform_and_key.side_effect = [
        ("openrouter", {"key_hash": "h1", "key_value": "v1"}),
        ("openrouter", {"key_hash": "h2", "key_value": "v2"}),
    ]

    client1 = MagicMock()
    client1.generate_stream.return_value = MockStreamGen([RuntimeError("Transient error")])

    client2 = MagicMock()
    client2.generate_stream.return_value = MockStreamGen(["model2_success"])

    mock_get_client.side_effect = [client1, client2]

    multiplexer = Multiplexer(mock_router)
    chunks = [
        c
        async for c in multiplexer.generate_stream(["openrouter/model-1", "openrouter/model-2"], [])
    ]

    assert chunks == ["model2_success"]
    # First attempt used model-1, second attempt used model-2
    client1.generate_stream.assert_called_with("model-1", [], **{})
    client2.generate_stream.assert_called_with("model-2", [], **{})


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_system_fallback_at_non_zero_index(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    # "system_fallback" is not at index 0, so it shouldn't trigger immediate fallback
    mock_router.get_best_platform_and_key.side_effect = NoKeysAvailableError("No keys")

    ollama_client = MagicMock()
    ollama_client.generate_stream.return_value = MockStreamGen(["fallback_chunk"])
    mock_get_client.return_value = ollama_client

    multiplexer = Multiplexer(mock_router)
    chunks = [
        c async for c in multiplexer.generate_stream(["openrouter/model", "system_fallback"], [])
    ]

    assert chunks == ["fallback_chunk"]


@pytest.mark.asyncio
@patch("omnimux.services.multiplexer.get_client_for_platform")
async def test_multiplexer_explicit_ollama_custom_api_base(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    ollama_client = MagicMock()
    ollama_client.generate_stream.return_value = MockStreamGen(["custom_ollama"])
    mock_get_client.return_value = ollama_client

    multiplexer = Multiplexer(
        mock_router,
        platforms_data={"ollama": {"apiBase": "http://custom-host:11434"}},
    )
    chunks = [c async for c in multiplexer.generate_stream(["ollama/llama3"], [])]

    assert chunks == ["custom_ollama"]
    mock_get_client.assert_called_once_with("ollama", "", None, "http://custom-host:11434")
