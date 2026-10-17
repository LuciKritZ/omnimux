import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE64_DATA_URI_PATTERN = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")


def _clean_content_length(content: Any) -> int:
    if isinstance(content, str):
        return len(_BASE64_DATA_URI_PATTERN.sub("", content))
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, str):
                total += len(_BASE64_DATA_URI_PATTERN.sub("", item))
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    total += len(str(item.get("text", "")))
                elif item_type in ("image_url", "image"):
                    url = ""
                    if item_type == "image_url":
                        img_val = item.get("image_url")
                        if isinstance(img_val, dict):
                            url = img_val.get("url", "")
                        elif isinstance(img_val, str):
                            url = img_val
                    if url and not url.startswith("data:"):
                        total += len(url)
                else:
                    val = str(item.get("text", "") or item.get("content", "") or "")
                    total += len(_BASE64_DATA_URI_PATTERN.sub("", val))
        return total
    return len(str(content))


class Gatekeeper:
    """Evaluates payload complexity and routes to simple or complex model tiers."""

    def __init__(
        self, http_client: httpx.AsyncClient, platforms_data: dict[str, Any] | None = None
    ) -> None:
        self.http_client = http_client
        self.platforms_data = platforms_data or {}

    async def profile_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Profiles the request for context length and vision requirements.
        """
        messages = payload.get("messages", [])
        system = payload.get("system", "")

        if not messages and not system:
            return {"context_length": 0, "has_image": False}

        system_length = 0
        if isinstance(system, str):
            system_length = len(system)
        elif isinstance(system, list):
            for part in system:
                if isinstance(part, str):
                    system_length += len(part)
                elif isinstance(part, dict) and "text" in part:
                    system_length += len(str(part["text"]))

        messages_length = sum(_clean_content_length(m.get("content", "")) for m in messages)
        total_length = system_length + messages_length

        return {"context_length": total_length}
