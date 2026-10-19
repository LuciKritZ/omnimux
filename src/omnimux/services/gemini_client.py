import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from omnimux.exceptions import GeminiAPIError, QuotaExceededError, RateLimitError
from omnimux.services.adapters import (
    BaseLLMClient,
    _safe_read_error_response,
    is_billing_exhaustion_error,
)

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """Client for interacting with the Google Gemini API."""

    def __init__(self, api_key: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(
            api_key=api_key or "",
            base_url="https://generativelanguage.googleapis.com",
        )
        self.client = client or httpx.AsyncClient()

    def _convert_openai_to_gemini(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Converts an OpenAI ChatCompletion payload to Gemini format."""
        contents: list[dict[str, Any]] = []
        system_parts: list[dict[str, Any]] = []

        for message in payload.get("messages", []):
            role = message.get("role", "user")
            parts: list[dict[str, Any]] = []
            content = message.get("content", "")

            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text" and "text" in block:
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image_url" and "image_url" in block:
                        url = block["image_url"].get("url", "")
                        if url.startswith("data:"):
                            header, b64_data = url.split(",", 1)
                            mime_type = header.split(";")[0].removeprefix("data:")
                            parts.append({"inlineData": {"mimeType": mime_type, "data": b64_data}})

            if not parts:
                continue

            if role == "system":
                system_parts.extend(parts)
            else:
                if role == "assistant":
                    role = "model"
                contents.append({"role": role, "parts": parts})

        result: dict[str, Any] = {"contents": contents}
        if system_parts:
            result["system_instruction"] = {"parts": system_parts}
        return result

    async def generate_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Sends the payload to Gemini and yields text content, with retry logic."""
        payload = {"messages": messages, **kwargs}
        gemini_payload = self._convert_openai_to_gemini(payload)
        url = f"{self.base_url}/v1beta/models/{model}:streamGenerateContent"

        headers = {"x-goog-api-key": self.api_key}
        params = {"alt": "sse"}

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            client_to_use = self.client or httpx.AsyncClient()
            try:
                async with client_to_use.stream(
                    "POST",
                    url,
                    json=gemini_payload,
                    headers=headers,
                    params=params,
                    timeout=30.0,
                ) as response:
                    if response.status_code != 200:
                        await _safe_read_error_response(response)
                        self.handle_error(response)

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if not data_str or data_str == "[DONE]":
                                continue

                            try:
                                data_json = json.loads(data_str)
                                text = ""
                                if data_json.get("candidates"):
                                    candidate = data_json["candidates"][0]
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        for part in candidate["content"]["parts"]:
                                            if "text" in part:
                                                text += part["text"]
                                if text:
                                    yield text
                            except (json.JSONDecodeError, IndexError, KeyError):
                                pass

                return

            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504):
                    if attempt == max_retries - 1:
                        raise GeminiAPIError(f"Max retries reached. Last error: {e}") from e
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Gemini API returned {e.response.status_code}, retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    raise GeminiAPIError(f"Unexpected HTTPStatusError: {e}") from e
            except httpx.RequestError as e:
                if attempt == max_retries - 1:
                    raise GeminiAPIError(f"Network error after {max_retries} attempts: {e}") from e
                delay = base_delay * (2**attempt)
                logger.warning(f"Network error {e}, retrying in {delay}s...")
                await asyncio.sleep(delay)

    def handle_error(self, response: httpx.Response) -> None:
        if response.status_code == 429:
            reset_time = response.headers.get("retry-after")
            try:
                retry_after = float(reset_time) if reset_time else 60.0
            except ValueError:
                retry_after = 60.0
            raise RateLimitError("Gemini Rate Limited", retry_after=retry_after)
        if response.status_code in (402, 403) or is_billing_exhaustion_error(response):
            raise QuotaExceededError("Gemini Quota Exceeded")
        response.raise_for_status()
