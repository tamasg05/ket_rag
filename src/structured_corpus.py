"""Common structured blocks, persistence, and structure-aware chunking."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .corpus_processing import chunk_words, tokenize_words


STRUCTURED_CORPUS_VERSION = "v1"


@dataclass(frozen=True)
class SavedStructuredCorpus:
    """Paths and stable identifiers for one materialized structured corpus."""

    corpus_path: Path
    blocks_path: Path
    sources_path: Path
    request_key: str
    content_key: str
    source_count: int


def clean_text(value: object) -> str:
    """
    Normalize whitespace in extracted text.

    Inputs:
        value: Value returned by an HTML or PDF extractor.

    Returns:
        A stripped string containing single internal spaces.
    """
    return re.sub(r"\s+", " ", str(value or "")).strip()


def make_text_block(
    text: str,
    source_name: str,
    source_url: str = "",
    page: int | None = None,
    heading_path: Sequence[str] = (),
    block_type: str = "paragraph",
) -> dict:
    """
    Create a normalized non-table block.

    Inputs:
        text: Visible block text.
        source_name: Human-readable document or page name.
        source_url: Optional originating URL.
        page: Optional one-based PDF page number.
        heading_path: Surrounding section headings from broad to specific.
        block_type: Semantic type such as paragraph, heading, or list.

    Returns:
        A JSON-serializable structured block dictionary.
    """
    return {
        "type": block_type,
        "text": clean_text(text),
        "source_name": clean_text(source_name),
        "source_url": source_url,
        "page": page,
        "heading_path": [clean_text(item) for item in heading_path if clean_text(item)],
    }


def make_table_block(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    source_name: str,
    table_id: str,
    caption: str = "",
    source_url: str = "",
    page: int | None = None,
    heading_path: Sequence[str] = (),
) -> dict:
    """
    Create a normalized table block.

    Inputs:
        headers: Column labels in display order.
        rows: Rectangular or ragged table rows.
        source_name: Human-readable document or page name.
        table_id: Identifier unique within the structured corpus.
        caption: Optional table caption.
        source_url: Optional originating URL.
        page: Optional one-based PDF page number.
        heading_path: Surrounding section headings.

    Returns:
        A JSON-serializable table block with padded rows and unique headers.
    """
    width = max(
        len(headers),
        max((len(row) for row in rows), default=0),
    )
    normalized_headers = [
        clean_text(headers[column]) if column < len(headers) else ""
        for column in range(width)
    ]
    seen: dict[str, int] = {}
    for column, header in enumerate(normalized_headers):
        base = header or f"Column {column + 1}"
        count = seen.get(base.casefold(), 0) + 1
        seen[base.casefold()] = count
        normalized_headers[column] = base if count == 1 else f"{base} ({count})"

    normalized_rows = [
        [clean_text(row[column]) if column < len(row) else "" for column in range(width)]
        for row in rows
        if any(clean_text(cell) for cell in row)
    ]
    return {
        "type": "table",
        "source_name": clean_text(source_name),
        "source_url": source_url,
        "page": page,
        "heading_path": [clean_text(item) for item in heading_path if clean_text(item)],
        "table_id": table_id,
        "caption": clean_text(caption),
        "headers": normalized_headers,
        "rows": normalized_rows,
    }


def render_table_row(block: dict, row: Sequence[str]) -> str:
    """
    Serialize one table row as explicit header-value pairs.

    Inputs:
        block: Table block providing ordered headers.
        row: Cell values in column order.

    Returns:
        Semicolon-separated header-value relationships.
    """
    headers = block.get("headers", [])
    text = "; ".join(
        f"{headers[column]} = {clean_text(value)}"
        for column, value in enumerate(row)
        if column < len(headers) and clean_text(value)
    )
    # A terminal period lets the existing sentence tokenizer treat each table
    # row as a separate keyword-description context.
    return text + "." if text else ""


def render_block(block: dict, rows: Sequence[Sequence[str]] | None = None) -> str:
    """
    Serialize one structured block into embedding- and LLM-friendly text.

    Inputs:
        block: Structured text or table block.
        rows: Optional table-row subset; defaults to every row in the block.

    Returns:
        Readable text retaining headings and table header-value relationships.
    """
    heading_path = [clean_text(item) for item in block.get("heading_path", [])]
    prefix = " > ".join(item for item in heading_path if item)
    if block.get("type") != "table":
        text = clean_text(block.get("text", ""))
        return "\n".join(part for part in (prefix, text) if part)

    selected_rows = block.get("rows", []) if rows is None else rows
    lines = [part for part in (prefix, clean_text(block.get("caption", ""))) if part]
    for row in selected_rows:
        row_text = render_table_row(block, row)
        if row_text:
            lines.append(row_text)
    return "\n".join(lines)


def save_structured_corpus(
    blocks: Sequence[dict],
    sources: Sequence[dict],
    root: Path,
    request_key: str,
) -> SavedStructuredCorpus:
    """
    Persist structured blocks, readable text, and provenance metadata.

    Inputs:
        blocks: Ordered normalized blocks from one or more sources.
        sources: Source-level provenance records.
        root: Parent directory for content-addressed corpora.
        request_key: Stable identifier for the selected URLs or PDF files.

    Returns:
        Paths and identifiers for the saved structured corpus.
    """
    usable_blocks = [block for block in blocks if render_block(block).strip()]
    if not usable_blocks:
        raise ValueError("No readable text or table rows were extracted.")
    encoded_blocks = json.dumps(
        usable_blocks,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    content_key = hashlib.sha256(encoded_blocks.encode("utf-8")).hexdigest()[:16]
    corpus_id = hashlib.sha256(
        f"{STRUCTURED_CORPUS_VERSION}|{request_key}|{content_key}".encode("utf-8")
    ).hexdigest()[:16]
    corpus_dir = root / corpus_id
    corpus_path = corpus_dir / "corpus.txt"
    blocks_path = corpus_dir / "blocks.json"
    sources_path = corpus_dir / "sources.json"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    readable = "\n\n".join(render_block(block) for block in usable_blocks).strip() + "\n"
    values = {
        corpus_path: readable,
        blocks_path: json.dumps(usable_blocks, indent=2, ensure_ascii=False),
        sources_path: json.dumps(list(sources), indent=2, ensure_ascii=False),
    }
    for path, value in values.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    return SavedStructuredCorpus(
        corpus_path=corpus_path,
        blocks_path=blocks_path,
        sources_path=sources_path,
        request_key=request_key,
        content_key=content_key,
        source_count=len(sources),
    )


def load_structured_blocks(path: Path) -> list[dict]:
    """
    Load persisted structured blocks.

    Inputs:
        path: UTF-8 JSON block file.

    Returns:
        Ordered structured block dictionaries.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Structured block file must contain a JSON list.")
    return value


def _copy_metadata(block: dict) -> dict:
    """
    Copy source metadata shared by chunks derived from one block.

    Inputs:
        block: Common-representation source block.

    Returns:
        Present, non-empty source and section metadata fields.
    """
    return {
        key: block[key]
        for key in ("source_name", "source_url", "page", "heading_path")
        if block.get(key) not in (None, "", [])
    }


def _table_word_count(block: dict, rows: Sequence[Sequence[str]]) -> int:
    """
    Count serialized word tokens for a table-row selection.

    Inputs:
        block: Table block or column-group variant.
        rows: Rows to include in the estimate.

    Returns:
        Number of prototype word tokens in the serialized table text.
    """
    return len(tokenize_words(render_block(block, rows)))


def _table_column_groups(block: dict, size: int) -> list[dict]:
    """
    Split an exceptionally wide table into key-preserving column groups.

    The first column is treated as a row identifier and repeated in every
    group. Ordinary tables remain unchanged.

    Inputs:
        block: Normalized table block.
        size: Target maximum word-token count for one row plus its context.

    Returns:
        One or more table blocks containing subsets of the original columns.
    """
    headers = block.get("headers", [])
    rows = block.get("rows", [])
    width = len(headers)
    if width <= 1 or max((_table_word_count(block, [row]) for row in rows), default=0) <= size:
        return [block]

    groups: list[list[int]] = []
    current: list[int] = []
    for column in range(1, width):
        candidate = [0, *current, column]
        candidate_block = {
            **block,
            "headers": [headers[index] for index in candidate],
            "rows": [[row[index] for index in candidate] for row in rows],
        }
        largest_row = max(
            (_table_word_count(candidate_block, [row]) for row in candidate_block["rows"]),
            default=0,
        )
        if current and largest_row > size:
            groups.append([0, *current])
            current = [column]
        else:
            current.append(column)
    if current:
        groups.append([0, *current])

    variants: list[dict] = []
    for group_number, columns in enumerate(groups, start=1):
        variants.append(
            {
                **block,
                "table_id": f"{block.get('table_id', 'table')}-columns-{group_number}",
                "headers": [headers[index] for index in columns],
                "rows": [[row[index] for index in columns] for row in rows],
                "column_group": [headers[index] for index in columns],
            }
        )
    return variants


def chunk_structured_blocks(blocks: Sequence[dict], size: int, overlap: int) -> list[dict]:
    """
    Chunk text by section and tables by complete rows.

    Inputs:
        blocks: Ordered common-representation blocks.
        size: Target maximum word-token count per chunk.
        overlap: Word overlap for ordinary text chunks; tables do not split rows.

    Returns:
        Chunk dictionaries compatible with all three existing RAG indexes.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    chunks: list[dict] = []
    pending: list[dict] = []
    pending_key: tuple | None = None

    def append_chunk(text: str, source_text: str, metadata: dict) -> None:
        """
        Append one globally numbered chunk with source metadata.

        Inputs:
            text: Token-normalized text used by retrieval indexes.
            source_text: Structure-preserving text supplied to answer context.
            metadata: Source, page, heading, or table metadata.

        Returns:
            None. The enclosing chunk list is updated in place.
        """
        item = {
            "id": len(chunks),
            "text": text,
            "source_text": source_text,
            **metadata,
        }
        chunks.append(item)

    def flush_text() -> None:
        """
        Chunk accumulated non-table blocks sharing one source section.

        Inputs:
            None; pending blocks are captured from the enclosing function.

        Returns:
            None. New chunks are appended and the pending list is cleared.
        """
        nonlocal pending
        if not pending:
            return
        source_text = "\n\n".join(render_block(block) for block in pending)
        metadata = _copy_metadata(pending[0])
        metadata["block_type"] = "text"
        for local in chunk_words(source_text, size, overlap):
            append_chunk(local["text"], local["source_text"], metadata)
        pending = []

    def append_table_rows(block: dict) -> None:
        """
        Pack complete rows from one table or column group into chunks.

        Inputs:
            block: Table block whose rows and columns must remain intact.

        Returns:
            None. One or more table chunks are appended to the result.
        """
        rows = block.get("rows", [])
        selected: list[list[str]] = []
        start_row = 0

        def flush_rows(end_row: int) -> None:
            """
            Append the selected rows ending at a zero-based boundary.

            Inputs:
                end_row: Exclusive zero-based row boundary, also equal to the
                    inclusive one-based row number stored as provenance.

            Returns:
                None. A table chunk is appended and selected rows are cleared.
            """
            nonlocal selected
            if not selected:
                return
            table_text = render_block(block, selected)
            metadata = {
                **_copy_metadata(block),
                "block_type": "table",
                "table_id": block.get("table_id", ""),
                "row_start": start_row + 1,
                "row_end": end_row,
                "column_group": block.get("column_group", block.get("headers", [])),
                "table_prefix": "\n".join(
                    part
                    for part in (
                        " > ".join(block.get("heading_path", [])),
                        block.get("caption", ""),
                    )
                    if part
                ),
                "table_row_texts": [
                    render_table_row(block, selected_row) for selected_row in selected
                ],
            }
            append_chunk(" ".join(tokenize_words(table_text)), table_text, metadata)
            selected = []

        for row_index, row in enumerate(rows):
            candidate = selected + [row]
            if selected and _table_word_count(block, candidate) > size:
                flush_rows(row_index)
                start_row = row_index
            selected.append(row)
        flush_rows(len(rows))

    for block in blocks:
        if block.get("type") == "table":
            flush_text()
            if not block.get("rows"):
                continue
            for table_group in _table_column_groups(block, size):
                append_table_rows(table_group)
            pending_key = None
            continue

        key = (
            block.get("source_name", ""),
            block.get("source_url", ""),
            block.get("page"),
            tuple(block.get("heading_path", [])),
        )
        if pending and key != pending_key:
            flush_text()
        pending_key = key
        pending.append(block)
    flush_text()
    return chunks
