"""Create word-based, structure-aware chunks from extracted data blocks."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from .blocks import (
    load_structured_blocks,
    render_block,
    render_table_row,
    validate_blocks,
)


CHUNKS_SCHEMA_VERSION = "1.0"

# Keep letters, numbers, and internal apostrophes, but discard punctuation
# marks around words. Case is preserved so names remain readable.
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)


def tokenize_words(text: str) -> list[str]:
    """
    Return word tokens without surrounding punctuation.

    Inputs:
        text: Text to tokenize.

    Returns:
        Word strings in source order, with case and internal apostrophes kept.
    """
    return _WORD_PATTERN.findall(text)


def _text_span(text: str, matches: list[re.Match], start: int, end: int) -> str:
    """Return the original text covered by a half-open word-token range."""
    if start >= end or start >= len(matches):
        return ""
    character_start = matches[start].start()
    character_end = matches[end].start() if end < len(matches) else len(text)
    return text[character_start:character_end].strip()


def chunk_words(text: str, size: int, overlap: int) -> list[dict]:
    """
    Divide text into overlapping word-based chunks.

    Inputs:
        text: Complete source text.
        size: Maximum number of word tokens in each chunk.
        overlap: Number of tokens repeated between consecutive chunks.

    Returns:
        Chunk dictionaries containing normalized ``text`` and the
        punctuation-preserving ``source_text``.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    matches = list(_WORD_PATTERN.finditer(text))
    words = [match.group(0) for match in matches]
    chunks: list[dict] = []
    step = size - overlap
    for chunk_id, start in enumerate(range(0, len(words), step)):
        end = min(start + size, len(words))
        part = words[start:end]
        if not part:
            break
        chunks.append(
            {
                "id": chunk_id,
                "text": " ".join(part),
                "source_text": _text_span(text, matches, start, end),
            }
        )
        if start + size >= len(words):
            break
    return chunks


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


def build_chunks(
    blocks: Sequence[dict] | str | Path,
    output_path: str | Path | None = None,
    *,
    strategy: str = "words",
    chunk_size: int = 450,
    chunk_overlap: int = 60,
) -> list[dict]:
    """
    Build and optionally persist chunks from extracted blocks.

    Inputs:
        blocks: A validated block list or path to ``blocks.json``.
        output_path: Optional destination for a UTF-8 ``chunks.json`` file.
        strategy: Chunking strategy. Currently only ``"words"`` is supported.
        chunk_size: Target maximum word count for one chunk.
        chunk_overlap: Repeated words between ordinary text chunks. Complete
            table rows are never overlapped or split.

    Returns:
        Structure-aware chunk dictionaries in retrieval order.
    """
    if strategy != "words":
        raise ValueError(
            f"Unsupported chunking strategy {strategy!r}; currently use 'words'."
        )
    if isinstance(blocks, (str, Path)):
        source_blocks = load_structured_blocks(Path(blocks))
    else:
        source_blocks = validate_blocks(list(blocks))

    chunks = chunk_structured_blocks(source_blocks, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("The extracted blocks did not produce any chunks.")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    return chunks
