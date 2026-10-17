import json
from pathlib import Path
from typing import Any

from omnimux.config import settings
from omnimux.models.schemas import ChatMessage
from omnimux.utils.html import compress_html_to_markdown


class ContextCompressor:
    """Intercepts and compresses raw HTML payloads and message history to protect
    the local LLM context window."""

    def __init__(
        self,
        char_threshold: int = 16384,
        min_reduction_ratio: float = 0.65,
        model_limits: dict[str, int] | None = None,
    ) -> None:
        self.char_threshold = char_threshold
        self.min_reduction_ratio = min_reduction_ratio

        self.model_limits: dict[str, int] = dict(model_limits) if model_limits is not None else {}
        if not self.model_limits:
            paths_to_try = [Path("model_prices_and_context_window.json"), Path("platforms.json")]
            for p in paths_to_try:
                if p.exists():
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        for model_name, info in data.items():
                            if isinstance(info, dict) and "max_input_tokens" in info:
                                clean_model = (
                                    model_name.split("/")[-1] if "/" in model_name else model_name
                                )
                                self.model_limits[model_name] = info["max_input_tokens"]
                                self.model_limits[clean_model] = info["max_input_tokens"]
                    except Exception:  # pragma: no cover
                        pass  # pragma: no cover
                    break

    def compress_if_needed(self, text: str) -> str:
        """
        Evaluates the text and compresses it if it exceeds the character threshold
        and successfully reduces the size by the minimum reduction ratio.
        """
        original_length = len(text)

        if original_length <= self.char_threshold:
            return text

        html_indicators = ("<html", "<body", "<div", "<table")
        if not any(indicator in text.lower() for indicator in html_indicators) and "</" not in text:
            return text

        compressed_text = compress_html_to_markdown(text)
        compressed_length = len(compressed_text)

        if compressed_length == 0:
            return text

        reduction = 1.0 - (compressed_length / original_length)

        if reduction >= self.min_reduction_ratio:
            return compressed_text

        return text

    def _estimate_tokens(self, text: str) -> int:
        """Roughly estimates tokens from text length."""
        return len(text) // 4

    def compress_messages(
        self, messages: list[Any], target_model: str, has_images: bool
    ) -> list[Any]:
        """
        Performs Safe Rolling Summarization on a list of messages.
        It calculates the token budget for the model, reserves tokens for images,
        and truncates/summarizes older messages to fit the context window.
        """
        max_tokens = self.model_limits.get(target_model, 8192)

        # Reserve tokens for images if they exist
        budget = max_tokens
        if has_images:
            budget -= settings.image_token_reserve

        # Keep system messages at the start and the most recent user/assistant messages at the end
        if budget <= 0:
            budget = 1024  # pragma: no cover

        # First pass: compress individual HTML blobs
        for message in messages:
            if isinstance(message.content, str):
                message.content = self.compress_if_needed(message.content)
            elif isinstance(message.content, list):
                for part in message.content:
                    if getattr(part, "type", None) == "text" and getattr(part, "text", None):
                        part.text = self.compress_if_needed(part.text)
                    elif isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        part["text"] = self.compress_if_needed(part["text"])  # pragma: no cover

        # Second pass: calculate tokens
        total_tokens = 0
        message_tokens = []
        for message in messages:
            toks = 0
            if isinstance(message.content, str):
                toks = self._estimate_tokens(message.content)
            elif isinstance(message.content, list):
                for part in message.content:
                    if getattr(part, "type", None) == "text" and getattr(part, "text", None):
                        toks += self._estimate_tokens(part.text)
                    elif isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        toks += self._estimate_tokens(part["text"])
            message_tokens.append(toks)
            total_tokens += toks

        if total_tokens <= budget:
            return messages

        # We need to compress/drop older messages.
        # Strategy: Keep all system messages. Keep the newest messages.
        # Drop the oldest non-system messages.
        final_messages = []
        for i, msg in enumerate(messages):
            role = (
                getattr(msg, "role", None)
                if hasattr(msg, "role")
                else (msg.get("role", "") if isinstance(msg, dict) else "")
            )
            if role == "system":
                final_messages.append(msg)
                budget -= message_tokens[i]

        # Add messages from the end until budget is exhausted
        tail_messages: list[Any] = []
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            role = (
                getattr(msg, "role", None)
                if hasattr(msg, "role")
                else (msg.get("role", "") if isinstance(msg, dict) else "")
            )
            if role == "system":
                continue  # pragma: no cover

            toks = message_tokens[i]
            if budget >= toks:
                tail_messages.insert(0, msg)
                budget -= toks
            else:
                # We can't fit this message entirely. We could truncate it or just stop here.
                # For safety, let's stop including older messages.
                summary_msg = ChatMessage(
                    role="system",
                    content="[System: Older context compressed to fit memory limits]",
                )
                tail_messages.insert(0, summary_msg)
                break

        return final_messages + tail_messages
