"""Internal, RAG-independent data-extraction package.

The package exposes a versioned block model shared by the HTML and PDF
adapters.  Import format helpers from here and adapter-specific operations from
``html_extractor`` or ``pdf_extractor``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import urlparse

from .blocks import (
    BLOCKS_SCHEMA_VERSION,
    STRUCTURED_CORPUS_VERSION,
    BlockValidationError,
    SavedStructuredCorpus,
    clean_text,
    load_structured_blocks,
    make_table_block,
    make_text_block,
    render_block,
    render_table_row,
    save_structured_corpus,
    validate_blocks,
)
from .chunking import (
    CHUNKS_SCHEMA_VERSION,
    build_chunks,
    chunk_structured_blocks,
    chunk_words,
    tokenize_words,
)


Progress = Callable[[str], None]


def extract_corpus(
    sources: str | Path | Sequence[str | Path] | None,
    output_directory: str | Path,
    *,
    source_type: str = "auto",
    max_pdf_files: int = 10,
    max_pdf_file_bytes: int = 50_000_000,
    max_url_pages: int = 20,
    url_timeout_seconds: float = 20.0,
    max_url_page_bytes: int = 5_000_000,
    progress: Progress | None = None,
) -> SavedStructuredCorpus:
    """
    Extract PDF files or HTML URLs and persist the common corpus files.

    Inputs:
        sources: One source or an ordered source sequence. All sources must be
            PDF paths or all must be HTTP(S) URLs.
        output_directory: Root directory for content-addressed corpus folders.
        source_type: ``"auto"``, ``"pdf"``, or ``"html"``.
        max_pdf_files: Maximum PDF count for one corpus.
        max_pdf_file_bytes: Maximum permitted size of each PDF.
        max_url_pages: Maximum URL count for one corpus.
        url_timeout_seconds: Download timeout for each HTML page.
        max_url_page_bytes: Maximum downloaded size of each HTML page.
        progress: Optional callback accepting status messages.

    Returns:
        Paths and stable identifiers for ``blocks.json``, ``corpus.txt``, and
        ``sources.json``. Uploaded PDFs are copied into the same corpus
        directory under ``sources/``.
    """
    if sources is None:
        raw_sources: list[str | Path] = []
    elif isinstance(sources, (str, Path)):
        raw_sources = [sources]
    else:
        raw_sources = list(sources)
    if not raw_sources:
        raise ValueError("Provide at least one PDF path or HTML URL.")

    values = [str(source) for source in raw_sources]
    kind = source_type.casefold().strip()
    if kind == "auto":
        are_urls = [
            urlparse(value).scheme.casefold() in {"http", "https"}
            for value in values
        ]
        are_pdfs = [Path(value).suffix.casefold() == ".pdf" for value in values]
        if all(are_urls):
            kind = "html"
        elif all(are_pdfs):
            kind = "pdf"
        else:
            raise ValueError(
                "Automatic source detection requires all HTTP(S) URLs or all PDF paths."
            )
    if kind == "pdf":
        from .pdf_extractor import prepare_pdf_corpus

        return prepare_pdf_corpus(
            raw_sources,
            Path(output_directory),
            max_pdf_files,
            max_pdf_file_bytes,
            progress,
        )
    if kind == "html":
        from .html_extractor import prepare_web_corpus

        return prepare_web_corpus(
            "\n".join(values),
            Path(output_directory),
            max_url_pages,
            url_timeout_seconds,
            max_url_page_bytes,
            progress,
        )
    raise ValueError("source_type must be 'auto', 'pdf', or 'html'.")


__all__ = [
    "BLOCKS_SCHEMA_VERSION",
    "CHUNKS_SCHEMA_VERSION",
    "STRUCTURED_CORPUS_VERSION",
    "BlockValidationError",
    "SavedStructuredCorpus",
    "clean_text",
    "build_chunks",
    "chunk_structured_blocks",
    "chunk_words",
    "extract_corpus",
    "load_structured_blocks",
    "make_table_block",
    "make_text_block",
    "render_block",
    "render_table_row",
    "save_structured_corpus",
    "tokenize_words",
    "validate_blocks",
]
