"""Versioned block model, validation, rendering, and JSON persistence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


# ``blocks.json`` is currently a JSON list of block objects.  This independent
# schema version will later allow Python and Java extractors to advertise and
# test compatibility without tying the format to a RAG index version.
BLOCKS_SCHEMA_VERSION = "1.0"
# Keep the prototype's existing cache-format identifier stable.  It is
# intentionally separate from the portable block-schema version above.
STRUCTURED_CORPUS_VERSION = "v1"


class BlockValidationError(ValueError):
    """Raised when an extracted block violates the common block contract."""


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
    width = max(len(headers), max((len(row) for row in rows), default=0))
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
    return text + "." if text else ""


def render_block(block: dict, rows: Sequence[Sequence[str]] | None = None) -> str:
    """
    Serialize one block into embedding- and LLM-friendly readable text.

    Inputs:
        block: Structured text or table block.
        rows: Optional table-row subset; defaults to every row in the block.

    Returns:
        Text retaining headings and explicit table header-value relationships.
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


def _validation_error(position: int, message: str) -> BlockValidationError:
    """Create a validation error identifying one zero-based block position."""
    return BlockValidationError(f"Block {position}: {message}")


def validate_blocks(blocks: object) -> list[dict]:
    """
    Validate and return a ``blocks.json`` value.

    Inputs:
        blocks: Decoded JSON value expected to be an ordered block list.

    Returns:
        The validated list, unchanged.

    Raises:
        BlockValidationError: If common metadata or table shape is invalid.
    """
    if not isinstance(blocks, list):
        raise BlockValidationError("blocks.json must contain a JSON list")

    for position, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise _validation_error(position, "must be a JSON object")
        block_type = block.get("type")
        if not isinstance(block_type, str) or not block_type.strip():
            raise _validation_error(position, "requires a non-empty string type")
        if not isinstance(block.get("source_name"), str):
            raise _validation_error(position, "requires a string source_name")
        if not isinstance(block.get("source_url", ""), str):
            raise _validation_error(position, "source_url must be a string")
        page = block.get("page")
        if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page < 1):
            raise _validation_error(position, "page must be a positive integer or null")
        headings = block.get("heading_path", [])
        if not isinstance(headings, list) or not all(isinstance(item, str) for item in headings):
            raise _validation_error(position, "heading_path must be a string list")

        bbox = block.get("bbox")
        if bbox is not None:
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in bbox
                )
            ):
                raise _validation_error(position, "bbox must contain four finite numbers")
            if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                raise _validation_error(position, "bbox boundaries are reversed")

        if block_type == "table":
            headers = block.get("headers")
            rows = block.get("rows")
            if not isinstance(block.get("table_id"), str) or not block["table_id"].strip():
                raise _validation_error(position, "table requires a non-empty table_id")
            if not isinstance(headers, list) or not headers or not all(
                isinstance(header, str) and header.strip() for header in headers
            ):
                raise _validation_error(position, "table headers must be non-empty strings")
            if not isinstance(rows, list):
                raise _validation_error(position, "table rows must be a list")
            for row_number, row in enumerate(rows, start=1):
                if not isinstance(row, list) or len(row) != len(headers):
                    raise _validation_error(
                        position,
                        f"table row {row_number} must have {len(headers)} cells",
                    )
                if not all(isinstance(cell, str) for cell in row):
                    raise _validation_error(position, f"table row {row_number} cells must be strings")
        elif not isinstance(block.get("text"), str):
            raise _validation_error(position, "non-table block requires string text")
    return blocks


def save_structured_corpus(
    blocks: Sequence[dict],
    sources: Sequence[dict],
    root: Path,
    request_key: str,
    format_version: str = STRUCTURED_CORPUS_VERSION,
) -> SavedStructuredCorpus:
    """Persist validated blocks, readable text, and provenance metadata."""
    usable_blocks = [block for block in blocks if render_block(block).strip()]
    validate_blocks(usable_blocks)
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
        f"{format_version}|{request_key}|{content_key}".encode("utf-8")
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
    """Load and validate a persisted UTF-8 ``blocks.json`` file."""
    return validate_blocks(json.loads(path.read_text(encoding="utf-8")))
