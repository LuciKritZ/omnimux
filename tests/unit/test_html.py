from omnimux.utils.html import compress_html_to_markdown


def test_compress_html_removes_scripts_and_styles() -> None:
    html = """
    <html>
        <head>
            <script>alert('test');</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <h1>Hello World</h1>
            <script>console.log('test2');</script>
        </body>
    </html>
    """
    result = compress_html_to_markdown(html)
    assert "Hello World" in result
    assert "alert('test')" not in result
    assert "body { color: red; }" not in result
    assert "console.log" not in result


def test_compress_html_removes_nav_and_footer() -> None:
    html = """
    <body>
        <nav><ul><li>Link 1</li></ul></nav>
        <header><h1>Site Title</h1></header>
        <main>
            <article>Actual Content</article>
        </main>
        <aside>Sidebar content</aside>
        <footer>Copyright 2026</footer>
    </body>
    """
    result = compress_html_to_markdown(html)
    assert "Actual Content" in result
    assert "Link 1" not in result
    assert "Site Title" not in result
    assert "Sidebar content" not in result
    assert "Copyright 2026" not in result


def test_compress_html_extracts_links() -> None:
    html = '<body>Click <a href="https://example.com">here</a>.</body>'
    result = compress_html_to_markdown(html)
    assert "[here](https://example.com)" in result


def test_compress_html_handles_empty_string() -> None:
    assert compress_html_to_markdown("") == ""
    assert compress_html_to_markdown("   ") == ""


def test_compress_html_preserves_script_and_style_in_markdown_blocks() -> None:
    content = """
    <div>Explanation of script tags:</div>
    ```html
    <script>console.log('keep me');</script>
    <style>body { background: black; }</style>
    ```
    <script>alert('remove me');</script>
    """
    result = compress_html_to_markdown(content)
    assert "<script>console.log('keep me');</script>" in result
    assert "<style>body { background: black; }</style>" in result
    assert "alert('remove me')" not in result


def test_compress_html_preserves_script_and_style_in_pre_code_tags() -> None:
    content = """
    <div>Here is code in HTML:</div>
    <pre><code>
    <script>console.log('html code');</script>
    </code></pre>
    <style>body { font-size: 14px; }</style>
    """
    result = compress_html_to_markdown(content)
    assert "<script>console.log('html code');</script>" in result
    assert "font-size: 14px" not in result


def test_compress_html_preserves_inline_code() -> None:
    content = "<div>Please use the `<script>` and `<style>` tags properly.</div>"
    result = compress_html_to_markdown(content)
    assert "`<script>`" in result
    assert "`<style>`" in result
