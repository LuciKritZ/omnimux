import asyncio
import sys
from typing import Any

import httpx


class ModelUnloader:
    """Tracks the active model in Ollama VRAM and handles unloading on model switch."""

    def __init__(
        self, http_client: httpx.AsyncClient, platforms_data: dict[str, Any] | None = None
    ) -> None:
        self.platforms_data = platforms_data or {}
        self._http_client = http_client
        self._active_model: str | None = None
        self._lock = asyncio.Lock()

    def _is_ollama_model(self, model: str) -> bool:
        """Determines if a model belongs to Ollama rather than a remote cloud provider."""
        lower_model = model.lower()
        cloud_prefixes = ("anthropic", "openai", "gemini", "groq", "openrouter")
        prefix = lower_model.split("/")[0] if "/" in lower_model else lower_model.split(".")[0]
        if prefix in cloud_prefixes:
            return False

        if model in self.platforms_data:
            info = self.platforms_data[model]
            if isinstance(info, dict):
                platform = str(info.get("platform", "")).lower()
                if platform and platform != "ollama":
                    return False
                if platform == "ollama":
                    return True

        if lower_model.startswith("ollama/") or lower_model.startswith("ollama."):
            return True

        cloud_name_starts = ("claude", "gpt", "gemini", "o1-", "o3-", "chatgpt")
        return not any(lower_model.startswith(c) for c in cloud_name_starts)

    async def unload_if_needed(self, target_model: str) -> bool:
        """Unloads the active model if it differs from the target model.

        Execution Flow:
        1. Performs a quick check outside the lock to return early if no change is needed.
        2. Rechecks inside a lock (double-checked locking).
        3. If no active model exists (cold start), sets it and returns.
        4. Triggers the unloading POST request to the Ollama API with keep_alive: 0.
        5. Updates the active model state.

        Args:
            target_model: The model name requested for the upcoming generation.

        Returns:
            True if unloading was triggered, False otherwise.
        """
        if self._active_model == target_model:
            return False

        async with self._lock:
            if self._active_model == target_model:
                return False

            if self._active_model is None:
                self._active_model = target_model
                return False

            if not self._is_ollama_model(self._active_model):
                self._active_model = target_model
                return False

            ollama_info = self.platforms_data.get("system_fallback", {})
            ollama_url = ollama_info.get("apiBase", "http://127.0.0.1:11434").rstrip("/")
            url = f"{ollama_url}/api/generate"
            unload_model = self._active_model.removeprefix("ollama/")
            payload = {
                "model": unload_model,
                "keep_alive": 0,
            }

            print(
                f"Unloading model '{self._active_model}' to load '{target_model}'",
                file=sys.stderr,
            )

            try:
                response = await self._http_client.post(url, json=payload)
                response.raise_for_status()
            except httpx.HTTPError as e:
                print(
                    f"Warning: Failed to unload model '{self._active_model}': {e}", file=sys.stderr
                )

            self._active_model = target_model
            return True
