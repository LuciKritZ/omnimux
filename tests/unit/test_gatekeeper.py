from unittest.mock import MagicMock

import httpx
import pytest

from omnimux.services.gatekeeper import Gatekeeper


@pytest.fixture
def http_client() -> MagicMock:
    return MagicMock(spec=httpx.AsyncClient)


@pytest.mark.asyncio
async def test_profile_empty_payload(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    result = await gatekeeper.profile_request({})
    assert result == {"context_length": 0, "has_image": False}


@pytest.mark.asyncio
async def test_profile_context_length(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    payload = {"messages": [{"role": "user", "content": "Hello world!"}]}
    result = await gatekeeper.profile_request(payload)
    assert result == {"context_length": 12}


@pytest.mark.asyncio
async def test_profile_strips_base64_data_uris(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    huge_b64 = "iVBORw0KGgoAAAANSUhEUg" * 500
    # String content with embedded data URI
    payload1 = {
        "messages": [
            {"role": "user", "content": f"Look at data:image/png;base64,{huge_b64} please."}
        ]
    }
    res1 = await gatekeeper.profile_request(payload1)
    assert res1["context_length"] == len("Look at  please.")

    # Multimodal list content
    payload2 = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{huge_b64}"},
                    },
                    {"type": "image", "source": {"type": "base64", "data": huge_b64}},
                ],
            }
        ]
    }
    res2 = await gatekeeper.profile_request(payload2)
    assert res2["context_length"] == len("Describe this:")


@pytest.mark.asyncio
async def test_profile_includes_system_prompt(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)

    # System as string
    payload_str = {
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    res_str = await gatekeeper.profile_request(payload_str)
    assert res_str["context_length"] == len("You are a helpful assistant.") + len("Hi")

    # System as list of blocks
    payload_list = {
        "system": [{"type": "text", "text": "System block 1"}, "System string 2"],
        "messages": [],
    }
    res_list = await gatekeeper.profile_request(payload_list)
    assert res_list["context_length"] == len("System block 1") + len("System string 2")


@pytest.mark.asyncio
async def test_profile_handles_web_image_url_and_tool_blocks(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "https://example.com/test.png"},
                    {"type": "tool_result", "content": "Tool result data"},
                    "Raw string in list",
                ],
            }
        ]
    }
    res = await gatekeeper.profile_request(payload)
    expected = (
        len("https://example.com/test.png") + len("Tool result data") + len("Raw string in list")
    )
    assert res["context_length"] == expected


@pytest.mark.asyncio
async def test_profile_non_string_non_list_content(http_client: MagicMock) -> None:
    gatekeeper = Gatekeeper(http_client)
    payload = {"messages": [{"role": "user", "content": 12345}]}
    res = await gatekeeper.profile_request(payload)
    assert res["context_length"] == 5
