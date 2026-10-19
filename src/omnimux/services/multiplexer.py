import asyncio
import json
import logging
import random
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from omnimux.config import settings
from omnimux.exceptions import QuotaExceededError, RateLimitError
from omnimux.services.adapters import AnthropicClient, OpenAICompatibleClient
from omnimux.services.gemini_client import GeminiClient
from omnimux.services.router_core import NoKeysAvailableError, RouterCore

logger = logging.getLogger(__name__)


def get_client_for_platform(
    platform: str,
    api_key: str,
    http_client: httpx.AsyncClient | None = None,
    api_base: str | None = None,
    protocol: str | None = None,
) -> Any:
    """Factory to get the right client for a platform."""
    if platform == "gemini":
        return GeminiClient(api_key=api_key, client=http_client)
    elif platform == "openrouter":
        return OpenAICompatibleClient(
            api_key=api_key, base_url=api_base or "https://openrouter.ai/api", client=http_client
        )
    elif platform == "ollama":
        return OpenAICompatibleClient(
            api_key="ollama", base_url=api_base or "http://127.0.0.1:11434", client=http_client
        )
    elif platform == "groq":
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=api_base or "https://api.groq.com/openai",
            client=http_client,
        )
    elif platform == "anthropic" or protocol == "anthropic":
        return AnthropicClient(
            api_key=api_key, base_url=api_base or "https://api.anthropic.com", client=http_client
        )
    elif api_base:
        if "anthropic" in platform.lower():
            return AnthropicClient(api_key=api_key, base_url=api_base, client=http_client)
        return OpenAICompatibleClient(api_key=api_key, base_url=api_base, client=http_client)
    else:
        raise ValueError(f"Unknown platform: {platform}")


class Multiplexer:
    def __init__(
        self,
        router_core: RouterCore,
        http_client: httpx.AsyncClient | None = None,
        platforms_data: dict[str, Any] | None = None,
        enable_jitter: bool = True,
    ) -> None:
        self.router_core = router_core
        self.http_client = http_client
        self.platforms_data = platforms_data or {}
        self.enable_jitter = enable_jitter

    async def _sleep_with_jitter(self) -> None:
        """
        Applies randomized millisecond jitter backoff before retrying
        to disperse failover stampedes.
        """
        if not self.enable_jitter:
            return
        min_ms = max(0, settings.retry_jitter_min_ms)
        max_ms = max(min_ms, settings.retry_jitter_max_ms)
        if max_ms > 0:
            jitter_sec = random.uniform(min_ms, max_ms) / 1000.0
            await asyncio.sleep(jitter_sec)

    async def generate_stream(
        self,
        candidate_models: list[str],
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Attempts to generate a stream across multiple candidate models dynamically.
        """
        if candidate_models and candidate_models[0] == "system_fallback":
            logger.info("Using system fallback model")
            fallback_info = self.platforms_data.get("system_fallback", {})
            fallback_model = fallback_info.get("model", "llama3")
            fallback_url = fallback_info.get("apiBase", "http://127.0.0.1:11434")

            ollama_client = get_client_for_platform("ollama", "", self.http_client, fallback_url)
            async for chunk in ollama_client.generate_stream(fallback_model, messages, **kwargs):
                yield chunk
            return

        # Map platforms to their corresponding target models from the candidates list
        platform_to_models: dict[str, list[tuple[str, str]]] = {}
        supported_platforms = {"anthropic", "openai", "gemini", "groq", "openrouter", "ollama"}

        for c in candidate_models:
            if c == "system_fallback":
                continue
            if "/" in c:
                parts = c.split("/", 1)
            elif "." in c and c.split(".", 1)[0] in supported_platforms:
                parts = c.split(".", 1)
            else:
                parts = [c]

            p = parts[0]
            m = parts[1] if len(parts) > 1 else parts[0]

            # Special case for explicit ollama
            if p == "ollama":
                logger.info(f"Using explicit ollama model {m}")
                ollama_api_base = (
                    self.platforms_data.get("ollama", {}).get("apiBase")
                    or self.platforms_data.get(c, {}).get("apiBase")
                    or self.platforms_data.get(f"ollama/{m}", {}).get("apiBase")
                )
                ollama_client = get_client_for_platform(
                    "ollama", "", self.http_client, ollama_api_base
                )
                async for chunk in ollama_client.generate_stream(m, messages, **kwargs):
                    yield chunk
                return

            if p not in platform_to_models:
                platform_to_models[p] = []
            platform_to_models[p].append((c, m))

        candidate_platforms = list(platform_to_models.keys())
        platform_model_index: dict[str, int] = {p: 0 for p in platform_to_models}
        max_retries = max(3, len(candidate_platforms) * 2) if candidate_platforms else 0
        attempts = 0

        while attempts < max_retries and candidate_platforms:
            try:
                platform, key_info = await self.router_core.get_best_platform_and_key(
                    candidate_platforms
                )
            except NoKeysAvailableError:
                logger.warning(f"No keys available for any of {candidate_platforms}. Falling back.")
                break

            key_hash = key_info["key_hash"]
            key_value = key_info["key_value"]
            meta_val = key_info.get("meta") or {}
            if isinstance(meta_val, str):
                try:
                    meta_val = json.loads(meta_val)
                except Exception:
                    meta_val = {}

            # Pick candidate model matched for this platform based on retry index
            models_for_platform = platform_to_models[platform]
            current_model_idx = platform_model_index.get(platform, 0)
            full_model_key, target_model = models_for_platform[
                current_model_idx % len(models_for_platform)
            ]
            api_base = (
                meta_val.get("api_base")
                or meta_val.get("apiBase")
                or meta_val.get("base_url")
                or self.platforms_data.get(full_model_key, {}).get("apiBase")
            )

            # Resolve tier alias if defined in meta["models"]
            tier_models = meta_val.get("models")
            if isinstance(tier_models, dict):
                model_lower = target_model.lower()
                if "opus" in model_lower and (
                    "opus" in tier_models or "ANTHROPIC_DEFAULT_OPUS_MODEL" in tier_models
                ):
                    target_model = tier_models.get("opus") or tier_models.get(
                        "ANTHROPIC_DEFAULT_OPUS_MODEL", target_model
                    )
                elif "sonnet" in model_lower and (
                    "sonnet" in tier_models or "ANTHROPIC_DEFAULT_SONNET_MODEL" in tier_models
                ):
                    target_model = tier_models.get("sonnet") or tier_models.get(
                        "ANTHROPIC_DEFAULT_SONNET_MODEL", target_model
                    )
                elif "haiku" in model_lower and (
                    "haiku" in tier_models or "ANTHROPIC_DEFAULT_HAIKU_MODEL" in tier_models
                ):
                    target_model = tier_models.get("haiku") or tier_models.get(
                        "ANTHROPIC_DEFAULT_HAIKU_MODEL", target_model
                    )
                elif target_model in tier_models:
                    target_model = tier_models[target_model]

            protocol = meta_val.get("protocol") or meta_val.get("type")
            client = get_client_for_platform(
                platform, key_value, self.http_client, api_base, protocol
            )

            first_chunk_emitted = False
            try:
                async with self.router_core.use_key(key_hash):
                    stream_gen = client.generate_stream(target_model, messages, **kwargs)

                    try:
                        first_chunk = await anext(stream_gen)
                    except StopAsyncIteration:
                        return

                    first_chunk_emitted = True
                    yield first_chunk

                    async for chunk in stream_gen:
                        yield chunk

                    return

            except QuotaExceededError as e:
                if first_chunk_emitted:
                    raise
                platform_model_index[platform] = current_model_idx + 1
                cooldown_sec = settings.default_cooldown_minutes * 60.0
                logger.warning(
                    f"Quota exceeded on {platform} key {key_hash}: {e}. Swapping platform..."
                )
                await self.router_core.mark_key_exhausted(key_hash, cooldown_sec)
                await self._sleep_with_jitter()
                attempts += 1
            except RateLimitError as e:
                if first_chunk_emitted:
                    raise
                platform_model_index[platform] = current_model_idx + 1
                logger.warning(f"Rate limit on {platform} key {key_hash}: {e}. Swapping...")
                await self.router_core.mark_key_exhausted(key_hash, e.retry_after or 60.0)
                await self._sleep_with_jitter()
                attempts += 1
            except Exception as e:
                if first_chunk_emitted:
                    raise
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 400:
                    raise
                platform_model_index[platform] = current_model_idx + 1
                logger.warning(f"Error on {platform} key {key_hash}: {e}. Swapping...")
                await self.router_core.mark_key_exhausted(key_hash, 60.0)
                await self._sleep_with_jitter()
                attempts += 1

        # Fallback to Ollama if all attempts exhausted or no keys
        logger.info("Falling back to system fallback model")
        fallback_info = self.platforms_data.get("system_fallback", {})
        fallback_model = fallback_info.get("model", "llama3")
        fallback_url = fallback_info.get("apiBase", "http://127.0.0.1:11434")

        ollama_client = get_client_for_platform("ollama", "", self.http_client, fallback_url)
        async for chunk in ollama_client.generate_stream(fallback_model, messages, **kwargs):
            yield chunk
