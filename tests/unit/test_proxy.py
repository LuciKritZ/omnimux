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
from omnimux.routes.proxy import verify_auth
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
