"""Download HTML pages and convert their semantic structure into blocks."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from .structured_corpus import (
    SavedStructuredCorpus,
    clean_text,
    make_table_block,
    make_text_block,
    render_block,
    save_structured_corpus,
)


Progress = Callable[[str], None]
_USER_AGENT = "ket-rag-comparison-prototype/1.0"
_REMOVED_TAGS = ["script", "style", "noscript", "template", "svg", "canvas", "form"]
_CONTENT_TAGS = [
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "ul",
    "ol",
    "blockquote",
    "pre",
    "table",
    "div",
    "section",
]


@dataclass(frozen=True)
class WebPage:
    """Structured content and provenance for one downloaded HTML page."""

    requested_url: str
    final_url: str
    title: str
    text: str
    blocks: tuple[dict, ...] = ()


def parse_url_list(value: str, max_pages: int) -> list[str]:
    """
    Parse and validate one HTTP(S) URL per non-empty input line.

    Inputs:
        value: Multiline URL textbox value.
        max_pages: Maximum permitted number of distinct URLs.

    Returns:
        Distinct URLs in their original order.
    """
    urls: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(value.splitlines(), start=1):
        url = line.strip()
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError(
                f"URL line {line_number} must be a complete http:// or https:// URL"
            )
        if parsed.username or parsed.password:
            raise ValueError(f"URL line {line_number} must not contain credentials")
        if url not in seen:
            urls.append(url)
            seen.add(url)

    if not urls:
        raise ValueError("Enter at least one web-page URL, one URL per line.")
    if len(urls) > max_pages:
        raise ValueError(f"At most {max_pages} distinct URLs can be indexed at once.")
    return urls


def url_request_key(urls: Sequence[str]) -> str:
    """
    Identify an ordered URL selection without downloading it.

    Inputs:
        urls: Validated URLs in user-supplied order.

    Returns:
        A deterministic hexadecimal identifier for that URL list.
    """
    encoded = json.dumps(list(urls), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _positive_span(cell: Tag, attribute: str) -> int:
    """
    Read a positive HTML row/column span.

    Inputs:
        cell: Table header or data-cell element.
        attribute: Either the ``rowspan`` or ``colspan`` attribute name.

    Returns:
        The positive span value, or one for an absent/invalid value.
    """
    try:
        return max(1, int(cell.get(attribute, 1)))
    except (TypeError, ValueError):
        return 1


def _extract_html_table(table: Tag) -> tuple[list[str], list[list[str]]]:
    """
    Expand one HTML table into unique headers and rectangular data rows.

    Inputs:
        table: Beautiful Soup ``table`` element.

    Returns:
        A pair containing normalized headers and rows with spans expanded.
    """
    grid: dict[tuple[int, int], str] = {}
    header_flags: list[bool] = []
    table_rows = table.find_all("tr")
    for row_index, row in enumerate(table_rows):
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        header_flags.append(
            row.find_parent("thead") is not None
            or all(cell.name.casefold() == "th" for cell in cells)
        )
        column = 0
        for cell in cells:
            while (row_index, column) in grid:
                column += 1
            value = clean_text(cell.get_text(" ", strip=True))
            rowspan = _positive_span(cell, "rowspan")
            colspan = _positive_span(cell, "colspan")
            for row_offset in range(rowspan):
                for column_offset in range(colspan):
                    grid[(row_index + row_offset, column + column_offset)] = value
            column += colspan

    if not grid:
        return [], []
    height = max(row for row, _ in grid) + 1
    width = max(column for _, column in grid) + 1
    rows = [
        [grid.get((row, column), "") for column in range(width)]
        for row in range(height)
    ]

    header_count = 0
    for flag in header_flags:
        if not flag:
            break
        header_count += 1
    if header_count:
        headers = []
        for column in range(width):
            levels: list[str] = []
            for row in range(header_count):
                value = rows[row][column]
                if value and (not levels or levels[-1] != value):
                    levels.append(value)
            headers.append(" > ".join(levels))
        data_rows = rows[header_count:]
    else:
        headers = [f"Column {column + 1}" for column in range(width)]
        data_rows = rows
    return headers, data_rows


def extract_html_blocks(
    html: str,
    source_name: str,
    source_url: str,
) -> tuple[str, list[dict]]:
    """
    Extract semantic text, list, heading, and table blocks from static HTML.

    Inputs:
        html: Decoded HTML source.
        source_name: Fallback name for the page.
        source_url: Final page URL stored as provenance.

    Returns:
        The HTML title and ordered common-representation blocks.
    """
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(_REMOVED_TAGS):
        element.decompose()
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    document_name = title or source_name
    root = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[dict] = []
    headings: list[str] = []
    table_number = 0

    if title:
        blocks.append(
            make_text_block(
                title,
                document_name,
                source_url,
                block_type="heading",
            )
        )

    for element in root.find_all(_CONTENT_TAGS):
        if element.name != "table" and element.find_parent("table") is not None:
            continue
        if element.name in {"ul", "ol"} and element.find_parent(["ul", "ol"]):
            continue
        if element.name in {"div", "section"} and element.find(
            _CONTENT_TAGS, recursive=True
        ):
            continue

        if element.name in {f"h{level}" for level in range(1, 7)}:
            text = clean_text(element.get_text(" ", strip=True))
            if not text:
                continue
            level = int(element.name[1])
            headings = headings[: level - 1]
            headings.append(text)
            blocks.append(
                make_text_block(
                    text,
                    document_name,
                    source_url,
                    heading_path=headings[:-1],
                    block_type="heading",
                )
            )
            continue

        if element.name == "table":
            headers, rows = _extract_html_table(element)
            if not rows:
                continue
            table_number += 1
            caption_element = element.find("caption")
            caption = (
                clean_text(caption_element.get_text(" ", strip=True))
                if caption_element
                else ""
            )
            blocks.append(
                make_table_block(
                    headers,
                    rows,
                    document_name,
                    table_id=f"html-table-{table_number}",
                    caption=caption,
                    source_url=source_url,
                    heading_path=headings,
                )
            )
            continue

        if element.name in {"ul", "ol"}:
            items = [
                clean_text(item.get_text(" ", strip=True))
                for item in element.find_all("li", recursive=False)
            ]
            marker = "1." if element.name == "ol" else "-"
            text = " ".join(
                f"{number}. {item}" if marker == "1." else f"- {item}"
                for number, item in enumerate(items, start=1)
                if item
            )
            block_type = "list"
        else:
            text = clean_text(element.get_text(" ", strip=True))
            block_type = "paragraph"
        if text:
            blocks.append(
                make_text_block(
                    text,
                    document_name,
                    source_url,
                    heading_path=headings,
                    block_type=block_type,
                )
            )

    if not blocks:
        fallback = clean_text(root.get_text(" ", strip=True))
        if fallback:
            blocks.append(make_text_block(fallback, document_name, source_url))
    return title, blocks


def extract_html_text(html: str) -> tuple[str, str]:
    """
    Extract a title and readable structured serialization from HTML.

    Inputs:
        html: Decoded HTML source.

    Returns:
        A title and text pair retained for simple callers and tests.
    """
    title, blocks = extract_html_blocks(html, "HTML page", "")
    return title, "\n".join(render_block(block) for block in blocks)


def fetch_html_pages(
    urls: Sequence[str],
    timeout_seconds: float,
    max_page_bytes: int,
    progress: Progress | None = None,
) -> list[WebPage]:
    """
    Download and structurally parse validated static HTML pages.

    Inputs:
        urls: HTTP(S) page URLs to download.
        timeout_seconds: Per-request network timeout.
        max_page_bytes: Maximum response bytes accepted for one page.
        progress: Optional callback accepting status messages.

    Returns:
        Parsed pages in the same order as ``urls``.
    """
    pages: list[WebPage] = []
    total = len(urls)
    for number, url in enumerate(urls, start=1):
        if progress:
            progress(f"Downloading web pages: {number - 1}/{total} completed")
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                final_url = response.geturl()
                if urlparse(final_url).scheme.lower() not in {"http", "https"}:
                    raise ValueError("redirected to a non-HTTP(S) location")
                content_type = response.headers.get_content_type().lower()
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    raise ValueError(f"returned content type {content_type!r}, not HTML")
                payload = response.read(max_page_bytes + 1)
                if len(payload) > max_page_bytes:
                    raise ValueError(
                        f"exceeds the {max_page_bytes:,}-byte per-page limit"
                    )
                encoding = response.headers.get_content_charset() or "utf-8"
        except Exception as exc:
            raise RuntimeError(f"Could not read {url}: {exc}") from exc

        try:
            decoded = payload.decode(encoding, errors="replace")
        except LookupError:
            decoded = payload.decode("utf-8", errors="replace")
        title, blocks = extract_html_blocks(decoded, url, final_url)
        if not blocks:
            raise ValueError(
                f"No readable static content was found at {url}. "
                "The page may require JavaScript."
            )
        text = "\n".join(render_block(block) for block in blocks)
        pages.append(WebPage(url, final_url, title, text, tuple(blocks)))
        if progress:
            progress(f"Downloaded web pages: {number}/{total} completed")
    return pages


def save_web_corpus(
    pages: Sequence[WebPage], root: Path
) -> SavedStructuredCorpus:
    """
    Materialize parsed web pages as a common structured corpus.

    Inputs:
        pages: Downloaded pages in corpus order.
        root: Parent directory for content-addressed URL corpora.

    Returns:
        Paths and identifiers for the saved structured corpus.
    """
    if not pages:
        raise ValueError("Cannot save an empty web corpus.")
    blocks: list[dict] = []
    sources: list[dict] = []
    for page_number, page in enumerate(pages, start=1):
        page_blocks = list(page.blocks) or [
            make_text_block(
                " ".join(part for part in (page.title, page.text) if part),
                page.title or page.final_url,
                page.final_url,
            )
        ]
        blocks.extend(page_blocks)
        sources.append(
            {
                "source": page_number,
                "kind": "html",
                "requested_url": page.requested_url,
                "final_url": page.final_url,
                "title": page.title,
                "characters": len(page.text),
            }
        )
    return save_structured_corpus(
        blocks,
        sources,
        root,
        url_request_key([page.requested_url for page in pages]),
    )


def prepare_web_corpus(
    value: str,
    root: Path,
    max_pages: int,
    timeout_seconds: float,
    max_page_bytes: int,
    progress: Progress | None = None,
) -> SavedStructuredCorpus:
    """
    Validate, download, parse, and persist a structured URL corpus.

    Inputs:
        value: Multiline URL textbox value.
        root: Parent directory for saved URL corpora.
        max_pages: Maximum number of distinct URLs.
        timeout_seconds: Per-request network timeout.
        max_page_bytes: Maximum response bytes accepted for one page.
        progress: Optional callback accepting status messages.

    Returns:
        Description of the saved content-addressed structured corpus.
    """
    urls = parse_url_list(value, max_pages)
    pages = fetch_html_pages(urls, timeout_seconds, max_page_bytes, progress)
    return save_web_corpus(pages, root)
