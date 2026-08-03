"""Download HTML pages and persist their readable text as one corpus."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

Progress = Callable[[str], None]
_USER_AGENT = "ket-rag-comparison-prototype/1.0"
_BLOCK_ELEMENTS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "main",
    "nav",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
}
_IGNORED_ELEMENTS = {
    "canvas",
    "form",
    "noscript",
    "script",
    "style",
    "svg",
    "template",
}


class _ReadableHtmlParser(HTMLParser):
    """Collect text from body, article, and main regions while parsing HTML."""

    def __init__(self) -> None:
        """
        Initialize empty buffers and region-depth counters.

        Inputs:
            None.

        Returns:
            None. Parser state is initialized on this instance.
        """
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.body_parts: list[str] = []
        self.article_parts: list[str] = []
        self.main_parts: list[str] = []
        self._depths = {"title": 0, "body": 0, "article": 0, "main": 0}
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        """
        Track content regions and insert boundaries around block elements.

        Inputs:
            tag: Lower- or mixed-case HTML element name.
            attrs: Attribute pairs supplied by ``HTMLParser``; not needed here.

        Returns:
            None. Parser state and text buffers are updated in place.
        """
        tag = tag.casefold()
        if tag in _IGNORED_ELEMENTS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag in self._depths:
            self._depths[tag] += 1
        if tag in _BLOCK_ELEMENTS:
            self._append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        """
        Handle self-closing elements such as ``<br/>``.

        Inputs:
            tag: Self-closing HTML element name.
            attrs: Attribute pairs supplied by ``HTMLParser``; not needed here.

        Returns:
            None. A block boundary may be appended to active buffers.
        """
        if tag.casefold() in _BLOCK_ELEMENTS and not self._ignored_depth:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        """
        Close ignored/content regions and separate completed blocks.

        Inputs:
            tag: Closing HTML element name.

        Returns:
            None. Parser depths and text buffers are updated in place.
        """
        tag = tag.casefold()
        if self._ignored_depth:
            if tag in _IGNORED_ELEMENTS:
                self._ignored_depth -= 1
            return
        if tag in _BLOCK_ELEMENTS:
            self._append("\n")
        if tag in self._depths and self._depths[tag] > 0:
            self._depths[tag] -= 1

    def handle_data(self, data: str) -> None:
        """
        Collect visible character data for every currently active region.

        Inputs:
            data: Decoded character data supplied by ``HTMLParser``.

        Returns:
            None. Visible data is appended to active buffers.
        """
        if not self._ignored_depth:
            self._append(data)

    def _append(self, value: str) -> None:
        """
        Append one value to all active region buffers.

        Inputs:
            value: Visible text or a block-boundary newline.

        Returns:
            None. Active buffers are updated in place.
        """
        if self._depths["title"]:
            self.title_parts.append(value)
        if self._depths["body"]:
            self.body_parts.append(value)
        if self._depths["article"]:
            self.article_parts.append(value)
        if self._depths["main"]:
            self.main_parts.append(value)


def _clean_html_parts(parts: Sequence[str]) -> str:
    """
    Normalize collected HTML text while preserving useful block boundaries.

    Inputs:
        parts: Text fragments and newlines emitted by the HTML parser.

    Returns:
        Cleaned readable text with one non-empty block per line.
    """
    joined = "".join(parts)
    lines = [re.sub(r"\s+", " ", line).strip() for line in joined.splitlines()]
    return "\n".join(line for line in lines if line)


@dataclass(frozen=True)
class WebPage:
    """Readable content and provenance for one downloaded HTML page."""

    requested_url: str
    final_url: str
    title: str
    text: str


@dataclass(frozen=True)
class SavedWebCorpus:
    """Paths and identifiers for one materialized URL corpus."""

    corpus_path: Path
    sources_path: Path
    request_key: str
    content_key: str
    page_count: int


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


def extract_html_text(html: str) -> tuple[str, str]:
    """
    Extract a page title and readable static text from HTML.

    Inputs:
        html: Decoded HTML source.

    Returns:
        A pair containing the cleaned title and visible page text.
    """
    parser = _ReadableHtmlParser()
    parser.feed(html)
    parser.close()
    title = re.sub(r"\s+", " ", "".join(parser.title_parts)).strip()
    # Prefer explicitly marked main content, then article content, then body.
    selected = parser.main_parts or parser.article_parts or parser.body_parts
    return title, _clean_html_parts(selected)


def fetch_html_pages(
    urls: Sequence[str],
    timeout_seconds: float,
    max_page_bytes: int,
    progress: Progress | None = None,
) -> list[WebPage]:
    """
    Download and parse a validated collection of static HTML pages.

    Inputs:
        urls: HTTP(S) page URLs to download.
        timeout_seconds: Per-request network timeout.
        max_page_bytes: Maximum response bytes accepted for one page.
        progress: Optional callback accepting human-readable status messages.

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

        title, text = extract_html_text(payload.decode(encoding, errors="replace"))
        if not text:
            raise ValueError(
                f"No readable static text was found at {url}. "
                "The page may require JavaScript."
            )
        pages.append(WebPage(url, final_url, title, text))
        if progress:
            progress(f"Downloaded web pages: {number}/{total} completed")
    return pages


def save_web_corpus(pages: Sequence[WebPage], root: Path) -> SavedWebCorpus:
    """
    Materialize parsed pages as one text corpus plus source metadata.

    Inputs:
        pages: Downloaded pages in corpus order.
        root: Parent directory for content-addressed URL corpora.

    Returns:
        Corpus paths, URL-selection key, content key, and page count.
    """
    if not pages:
        raise ValueError("Cannot save an empty web corpus.")

    urls = [page.requested_url for page in pages]
    source_records = [
        {
            "page": number,
            "requested_url": page.requested_url,
            "final_url": page.final_url,
            "title": page.title,
            "characters": len(page.text),
        }
        for number, page in enumerate(pages, start=1)
    ]
    # Put each page title before its body without artificial labels such as
    # "WEB PAGE" or "TITLE"; repeated labels would pollute lexical retrieval.
    corpus_sections = [
        "\n".join(part for part in (page.title, page.text) if part)
        for page in pages
    ]
    corpus_text = "\n\n".join(corpus_sections).strip() + "\n"
    content_key = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()[:16]
    request_key = url_request_key(urls)
    corpus_id = hashlib.sha256(
        f"{request_key}|{content_key}".encode("utf-8")
    ).hexdigest()[:16]
    corpus_dir = root / corpus_id
    corpus_path = corpus_dir / "corpus.txt"
    sources_path = corpus_dir / "sources.json"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    temporary_corpus = corpus_path.with_suffix(".txt.tmp")
    temporary_sources = sources_path.with_suffix(".json.tmp")
    temporary_corpus.write_text(corpus_text, encoding="utf-8")
    temporary_sources.write_text(
        json.dumps(source_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_corpus.replace(corpus_path)
    temporary_sources.replace(sources_path)
    return SavedWebCorpus(
        corpus_path=corpus_path,
        sources_path=sources_path,
        request_key=request_key,
        content_key=content_key,
        page_count=len(pages),
    )


def prepare_web_corpus(
    value: str,
    root: Path,
    max_pages: int,
    timeout_seconds: float,
    max_page_bytes: int,
    progress: Progress | None = None,
) -> SavedWebCorpus:
    """
    Validate, download, parse, and persist a URL corpus.

    Inputs:
        value: Multiline URL textbox value.
        root: Parent directory for saved URL corpora.
        max_pages: Maximum number of distinct URLs.
        timeout_seconds: Per-request network timeout.
        max_page_bytes: Maximum response bytes accepted for one page.
        progress: Optional callback accepting status messages.

    Returns:
        Description of the saved content-addressed corpus.
    """
    urls = parse_url_list(value, max_pages)
    pages = fetch_html_pages(urls, timeout_seconds, max_page_bytes, progress)
    return save_web_corpus(pages, root)
