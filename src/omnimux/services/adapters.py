import abc
import inspect
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from omnimux.exceptions import QuotaExceededError, RateLimitError

BILLING_EXHAUSTION_PATTERNS = (
    "credit balance is too low",
    "credit_balance_too_low",
    "insufficient_credits",
    "billing_hard_limit_reached",
    "insufficient_quota",
    "exceeded your current quota",
    "quota_exceeded",
    "out of credits",
    "account_deactivated",
    "resource_exhausted",
    "insufficient funds",
    "payment required",
    "purchase credits",
)


def is_billing_exhaustion_error(response: httpx.Response) -> bool:
    """Detects whether an HTTP error indicates account or credit exhaustion."""
    if response.status_code in (402, 403):
        return True
    if response.status_code != 400:
        return False

    try:
        body = response.text.lower()
    except Exception:
        return False

    return any(p in body for p in BILLING_EXHAUSTION_PATTERNS)


def _has_cache_control(item: Any) -> bool:
    """Recursively checks if any dict or list in item contains a cache_control key."""
    if isinstance(item, dict):
        if item.get("cache_control") is not None:
            return True
        return any(_has_cache_control(v) for v in item.values())
    if isinstance(item, list):
        return any(_has_cache_control(v) for v in item)
    return False


async def _safe_read_error_response(response: Any) -> None:
    """Safely reads the response body if aread is available and awaitable."""
    aread_fn = getattr(response, "aread", None)
    if callable(aread_fn):
        res = aread_fn()
        if inspect.isawaitable(res):
            await res


class BaseLLMClient(abc.ABC):
    """Abstract base class for all LLM provider adapters."""

    def __init__(
        self, api_key: str, base_url: str = "", client: httpx.AsyncClient | None = None
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.client = client
        self.last_usage: dict[str, int] = {}

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        if self.client:
            yield self.client
        else:
            async with httpx.AsyncClient() as client:
                yield client

    @abc.abstractmethod
    async def generate_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generates a streaming response from the LLM provider."""
        yield ""

    @abc.abstractmethod
    def handle_error(self, response: httpx.Response) -> None:
        """Parses an HTTP error response and raises the appropriate domain exception."""
        pass


class OpenAICompatibleClient(BaseLLMClient):
    """Client adapter for OpenAI-compatible APIs (OpenRouter, Groq, Ollama, vLLM)."""

    async def generate_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        self.last_usage = {}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        async with (
            self._get_client() as client_to_use,
            client_to_use.stream(
                "POST", f"{self.base_url}/v1/chat/completions", headers=headers, json=payload
            ) as response,
        ):
            if response.status_code != 200:
                await _safe_read_error_response(response)
                self.handle_error(response)

            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        if "usage" in data and isinstance(data["usage"], dict):
                            for k, v in data["usage"].items():
                                if isinstance(v, (int, float)):
                                    self.last_usage[k] = int(v)
                                elif isinstance(v, dict):
                                    for sub_k, sub_v in v.items():
                                        if isinstance(sub_v, (int, float)):
                                            self.last_usage[f"{k}_{sub_k}"] = int(sub_v)
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass

    def handle_error(self, response: httpx.Response) -> None:
        if response.status_code == 429:
            # Try to parse standard reset headers
            reset_time = response.headers.get("x-ratelimit-reset") or response.headers.get(
                "retry-after"
            )
            try:
                retry_after = float(reset_time) if reset_time else 60.0
            except ValueError:
                retry_after = 60.0
            raise RateLimitError(f"Rate limited by {self.base_url}", retry_after=retry_after)

        if response.status_code in (402, 403) or is_billing_exhaustion_error(response):
            # 402 is standard for Insufficient Quota; some providers return 403 or 400.
            raise QuotaExceededError(f"Quota exceeded or unauthorized for {self.base_url}")

        # Fallback for generic errors
        response.raise_for_status()


class AnthropicClient(BaseLLMClient):
    """Client adapter for Anthropic APIs."""

    async def generate_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        self.last_usage = {}
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        anthropic_beta = kwargs.get("anthropic_beta") or kwargs.get("anthropic-beta")
        if anthropic_beta:
            headers["anthropic-beta"] = str(anthropic_beta)

        system_val = kwargs.get("system")
        system_payload: str | list[dict[str, Any]] | None = None
        anthropic_messages: list[dict[str, Any]] = []

        if system_val is not None:
            system_payload = system_val

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                if system_val is None:
                    if isinstance(content, list):
                        if not isinstance(system_payload, list):
                            system_payload = []
                        system_payload.extend(
                            b if isinstance(b, dict) else {"type": "text", "text": str(b)}
                            for b in content
                        )
                    elif isinstance(content, str):
                        if isinstance(system_payload, list):
                            system_payload.append({"type": "text", "text": content})
                        else:
                            system_payload = (
                                system_payload + "\n" if system_payload else ""
                            ) + content
            else:
                anthropic_messages.append(msg)

        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "stream": True,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if isinstance(system_payload, str) and system_payload.strip():
            payload["system"] = system_payload.strip()
        elif isinstance(system_payload, list) and system_payload:
            payload["system"] = system_payload

        for k, v in kwargs.items():
            if k not in [
                "max_tokens",
                "stream",
                "model",
                "messages",
                "system",
                "anthropic_beta",
                "anthropic-beta",
            ]:
                payload[k] = v

        has_cc = (
            _has_cache_control(payload.get("system"))
            or _has_cache_control(anthropic_messages)
            or _has_cache_control(payload.get("tools"))
        )
        if has_cc:
            existing_beta = headers.get("anthropic-beta")
            if not existing_beta:
                headers["anthropic-beta"] = "prompt-caching-2024-07-31"
            elif "prompt-caching" not in existing_beta:
                headers["anthropic-beta"] = f"{existing_beta},prompt-caching-2024-07-31"

        async with (
            self._get_client() as client_to_use,
            client_to_use.stream(
                "POST", f"{self.base_url}/v1/messages", headers=headers, json=payload
            ) as response,
        ):
            if response.status_code != 200:
                await _safe_read_error_response(response)
                self.handle_error(response)

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if not data_str:
                        continue
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "message_start":
                            usage = data.get("message", {}).get("usage", {})
                            if isinstance(usage, dict):
                                for uk, uv in usage.items():
                                    if isinstance(uv, (int, float)):
                                        self.last_usage[uk] = int(uv)
                        elif data.get("type") == "message_delta":
                            usage = data.get("usage", {})
                            if isinstance(usage, dict):
                                for uk, uv in usage.items():
                                    if isinstance(uv, (int, float)):
                                        self.last_usage[uk] = int(uv)
                        elif data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            content = (
                                delta.get("text")
                                or delta.get("thinking")
                                or delta.get("partial_json")
                            )
                            if content:
                                yield content
                    except (json.JSONDecodeError, KeyError):
                        pass

    def handle_error(self, response: httpx.Response) -> None:
        if response.status_code == 429:
            reset_time = response.headers.get("anthropic-retry-after") or response.headers.get(
                "retry-after"
            )
            try:
                retry_after = float(reset_time) if reset_time else 60.0
            except ValueError:
                retry_after = 60.0
            raise RateLimitError("Anthropic Rate Limited", retry_after=retry_after)

        if response.status_code in (401, 403) or is_billing_exhaustion_error(response):
            raise QuotaExceededError("Anthropic Quota Exceeded or Unauthorized")

        response.raise_for_status()
