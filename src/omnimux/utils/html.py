import re

from bs4 import BeautifulSoup
from markdownify import markdownify


def compress_html_to_markdown(html_string: str) -> str:
    """
    Strips layout boilerplate and converts HTML to Markdown to reduce context bloat.

    Execution Flow:
    1. Returns empty string if input is empty.
    2. Protects markdown code blocks from HTML decomposition.
    3. Uses BeautifulSoup to remove script, style, and structural tags (nav, header, footer, aside)
       outside of code blocks.
    4. Converts the cleaned HTML to Markdown using markdownify.
    5. Restores protected code blocks.
    """
    if not html_string or not html_string.strip():
        return ""

    code_blocks: list[str] = []

    def save_block(match: re.Match[str]) -> str:
        code_blocks.append(match.group(0))
        return f"OMNIMUXCODEBLOCK{len(code_blocks) - 1}X"

    protected = re.sub(r"(```[\s\S]*?```|`[^`\n]+`)", save_block, html_string)

    soup = BeautifulSoup(protected, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        if tag.find_parent(["pre", "code"]):
            tag.replace_with(soup.new_string(str(tag)))
            continue
        tag.decompose()

    cleaned_html = str(soup)

    md_text: str = markdownify(cleaned_html, heading_style="ATX", strip=["img"])

    lines = [line.strip() for line in md_text.splitlines() if line.strip()]
    result = "\n".join(lines)

    for i, block in enumerate(code_blocks):
        result = result.replace(f"OMNIMUXCODEBLOCK{i}X", block)

    return result
