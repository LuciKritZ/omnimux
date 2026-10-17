from typing import Any

from omnimux.services.compressor import ContextCompressor


def test_compressor_skips_short_text() -> None:
    """Should return exact original text since it's under 16,384 chars."""
    compressor = ContextCompressor()
    text = "<html><body>" + "a" * 100 + "</body></html>"
    result = compressor.compress_if_needed(text)
    assert result == text


def test_compressor_compresses_large_html() -> None:
    """
    Strips nav and footer from a large HTML string and returns the main content in markdown.
    Must reduce the character count by at least 65%.
    """
    compressor = ContextCompressor()
    html = (
        "<html><body><nav>"
        + "nav_content " * 1000
        + "</nav><main>Main content</main><footer>"
        + "footer_content " * 1000
        + "</footer></body></html>"
    )

    result = compressor.compress_if_needed(html)

    assert "Main content" in result
    assert "nav_content" not in result
    assert "footer_content" not in result

    reduction = 1.0 - (len(result) / len(html))
    assert reduction >= 0.65


def test_compressor_returns_original_if_not_compressible() -> None:
    """
    If the reduction isn't at least 65% (e.g. string has no HTML boilerplate),
    it should return the original text to prevent data loss.
    """
    compressor = ContextCompressor()
    # Include basic HTML to bypass the early exit check (line 28)
    text = "<html><body>" + "Just some very long text. " * 1000 + "</body></html>"
    assert len(text) > 16384

    result = compressor.compress_if_needed(text)

    assert result == text


def test_compressor_skips_non_html_text() -> None:
    """
    If the string is over 16,384 characters but has no HTML indicators,
    it should return the original text immediately.
    """
    compressor = ContextCompressor()
    text = "Just some very long plain text. " * 1000
    assert len(text) > 16384

    result = compressor.compress_if_needed(text)

    assert result == text


def test_compressor_handles_fully_stripped_payload() -> None:
    """
    If the HTML is completely stripped (e.g. only script tags),
    it should fallback to the original to prevent complete data loss.
    """
    compressor = ContextCompressor()
    # Over 16384 chars of pure script tags
    text = "<html><body><script>" + "var x = 1;\n" * 2000 + "</script></body></html>"
    assert len(text) > 16384

    result = compressor.compress_if_needed(text)

    assert result == text


def test_compress_messages_no_truncation() -> None:
    compressor = ContextCompressor()
    compressor.model_limits = {"test-model": 1000}

    class MockMsg:
        def __init__(self, role: str, content: str | list[Any]) -> None:
            self.role = role
            self.content = content

    messages = [MockMsg("system", "System prompt"), MockMsg("user", "Hello")]

    result = compressor.compress_messages(messages, "test-model", False)
    assert len(result) == 2
    assert result[0].content == "System prompt"


def test_compress_messages_with_truncation() -> None:
    compressor = ContextCompressor()
    compressor.model_limits = {"test-model": 500}  # 500 tokens = ~2000 chars

    class MockMsg:
        def __init__(self, role: str, content: str | list[Any]) -> None:
            self.role = role
            self.content = content

    messages = [
        MockMsg("system", "System prompt"),
        MockMsg("user", "x" * 2500),  # 625 tokens, this should be truncated
        MockMsg("assistant", "Hi"),
        MockMsg("user", "y" * 100),  # 25 tokens
    ]

    result = compressor.compress_messages(messages, "test-model", False)
    assert len(result) == 4
    if isinstance(result[1], dict):
        assert "compressed to fit memory" in result[1]["content"]
    else:
        assert "compressed to fit memory" in result[1].content
    assert result[0].role == "system"


def test_compress_messages_with_image_reserve() -> None:
    compressor = ContextCompressor()
    # 5000 tokens - 4096 reserve = 904 tokens (~3600 chars)
    compressor.model_limits = {"test-model": 5000}

    class MockMsg:
        def __init__(self, role: str, content: str | list[Any]) -> None:
            self.role = role
            self.content = content

    messages = [
        MockMsg("system", "System prompt"),
        MockMsg("user", "x" * 4000),  # 1000 tokens, truncated due to image reserve
        MockMsg("assistant", "Hi"),
    ]

    # has_images = True
    result = compressor.compress_messages(messages, "test-model", True)
    assert len(result) == 3
    if isinstance(result[1], dict):
        assert "compressed to fit memory" in result[1]["content"]
    else:
        assert "compressed to fit memory" in result[1].content


def test_compress_messages_compresses_html_first() -> None:
    compressor = ContextCompressor(char_threshold=100, min_reduction_ratio=0.1)
    compressor.model_limits = {"test-model": 10000}

    class MockMsg:
        def __init__(self, role: str, content: str | list[Any]) -> None:
            self.role = role
            self.content = content

    # html blob > 100 chars
    html = "<html><body><nav>" + "nav " * 50 + "</nav><main>Main content</main></body></html>"
    messages = [MockMsg("user", html)]

    result = compressor.compress_messages(messages, "test-model", False)
    assert "Main content" in result[0].content
    assert "<html" not in result[0].content


def test_compress_messages_dict_parts() -> None:
    compressor = ContextCompressor(char_threshold=100, min_reduction_ratio=0.1)
    compressor.model_limits = {"test-model": 1000}

    class MockMsg:
        def __init__(self, role: str, content: str | list[Any]) -> None:
            self.role = role
            self.content = content

    html = "<html><body><nav>" + "nav " * 50 + "</nav><main>Main content</main></body></html>"
    messages = [MockMsg("user", [{"type": "text", "text": html}])]

    result = compressor.compress_messages(messages, "test-model", False)
    assert "Main content" in result[0].content[0]["text"]
    assert "<html" not in result[0].content[0]["text"]


def test_compress_messages_load_json(monkeypatch: Any) -> None:
    from pathlib import Path

    def mock_exists(self: Path) -> bool:
        return True

    def mock_read_text(self: Path, encoding: str = "utf-8") -> str:
        return '{"test/model": {"max_input_tokens": 12345}}'

    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr(Path, "read_text", mock_read_text)

    compressor = ContextCompressor()
    assert compressor.model_limits["test/model"] == 12345
    assert compressor.model_limits["model"] == 12345


def test_compress_messages_fallback_dict() -> None:
    compressor = ContextCompressor()
    compressor.model_limits = {"test-model": 500}

    class MockMsgWithoutModelFields:
        def __init__(self, role: str, content: str | list[Any]) -> None:
            self.role = role
            self.content = content

    # Send a mock message without model_fields to trigger the fallback dict replacement
    messages = [
        MockMsgWithoutModelFields("system", "System"),
        MockMsgWithoutModelFields("user", "x" * 2500),
        MockMsgWithoutModelFields("assistant", "Hi"),
    ]

    result = compressor.compress_messages(messages, "test-model", False)
    assert len(result) == 3
    assert result[1]["role"] == "system"
    assert "compressed to fit memory limits" in result[1]["content"]


def test_compressor_custom_model_limits_passed_directly() -> None:
    limits = {"custom-model": 64000, "provider/custom-model": 64000}
    compressor = ContextCompressor(model_limits=limits)
    assert compressor.model_limits == limits


def test_compress_messages_summary_msg_has_model_dump() -> None:
    from omnimux.models.schemas import ChatMessage

    compressor = ContextCompressor(model_limits={"test-model": 200})
    messages = [
        ChatMessage(role="user", content="x" * 2000),
        ChatMessage(role="assistant", content="Response"),
    ]
    result = compressor.compress_messages(messages, "test-model", False)
    assert len(result) == 2
    assert hasattr(result[0], "model_dump")
    dumped = result[0].model_dump()
    assert dumped["role"] == "system"
    assert "compressed to fit memory limits" in dumped["content"]
