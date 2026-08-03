"""Extract text and tabular structure from uploaded PDF documents."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from .structured_corpus import (
    SavedStructuredCorpus,
    clean_text,
    make_table_block,
    make_text_block,
    save_structured_corpus,
)


Progress = Callable[[str], None]


def parse_pdf_paths(value, max_files: int, max_file_bytes: int) -> list[Path]:
    """
    Validate uploaded PDF paths and configured size/count limits.

    Inputs:
        value: Gradio single path, path list, or empty value.
        max_files: Maximum number of PDFs in one corpus.
        max_file_bytes: Maximum permitted size of each PDF.

    Returns:
        Distinct resolved PDF paths in UI order.
    """
    raw_values = value if isinstance(value, (list, tuple)) else [value]
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_values:
        if not raw:
            continue
        path = Path(str(raw)).resolve()
        if path in seen:
            continue
        if not path.is_file():
            raise ValueError(f"Uploaded PDF is unavailable: {path}")
        with path.open("rb") as stream:
            signature = stream.read(5)
        if path.suffix.casefold() != ".pdf" or signature != b"%PDF-":
            raise ValueError(f"Uploaded file is not a valid PDF: {path.name}")
        if path.stat().st_size > max_file_bytes:
            raise ValueError(
                f"{path.name} exceeds the {max_file_bytes:,}-byte per-file limit"
            )
        paths.append(path)
        seen.add(path)
    if not paths:
        raise ValueError("Upload at least one PDF document.")
    if len(paths) > max_files:
        raise ValueError(f"At most {max_files} PDF documents can be indexed at once.")
    return paths


def pdf_request_key(paths: Sequence[Path]) -> str:
    """
    Derive a stable identifier from ordered PDF names and bytes.

    Inputs:
        paths: Validated PDF paths.

    Returns:
        A deterministic hexadecimal identifier for the uploaded files.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _inside_any_table(word: dict, boxes: Sequence[tuple]) -> bool:
    """
    Test whether the center of one positioned PDF word is in a table.

    Inputs:
        word: ``pdfplumber`` word dictionary with bounding coordinates.
        boxes: Detected table bounding boxes as ``x0, top, x1, bottom``.

    Returns:
        ``True`` when the word center is inside at least one table box.
    """
    center_x = (float(word["x0"]) + float(word["x1"])) / 2
    center_y = (float(word["top"]) + float(word["bottom"])) / 2
    return any(
        x0 <= center_x <= x1 and top <= center_y <= bottom
        for x0, top, x1, bottom in boxes
    )


def _group_pdf_lines(words: Sequence[dict], tolerance: float = 3.0) -> list[tuple[float, str]]:
    """
    Reconstruct approximate text lines from positioned PDF words.

    Inputs:
        words: ``pdfplumber`` word dictionaries outside detected tables.
        tolerance: Maximum vertical difference within one line.

    Returns:
        Pairs of top coordinate and left-to-right line text.
    """
    lines: list[tuple[float, list[dict]]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        top = float(word["top"])
        if lines and abs(top - lines[-1][0]) <= tolerance:
            lines[-1][1].append(word)
        else:
            lines.append((top, [word]))
    return [
        (
            top,
            clean_text(
                " ".join(
                    str(word.get("text", ""))
                    for word in sorted(line_words, key=lambda item: float(item["x0"]))
                )
            ),
        )
        for top, line_words in lines
    ]


def _clean_pdf_table(raw_rows: Sequence[Sequence[object]]) -> tuple[list[str], list[list[str]]]:
    """
    Normalize a PDF table using its first non-empty row as the header.

    Inputs:
        raw_rows: Cell matrix returned by ``pdfplumber``.

    Returns:
        A header list and remaining non-empty data rows.
    """
    rows = [
        [clean_text(cell) for cell in row]
        for row in raw_rows
        if row and any(clean_text(cell) for cell in row)
    ]
    if not rows:
        return [], []
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    if len(padded) == 1:
        return [f"Column {column + 1}" for column in range(width)], padded
    headers = padded[0]
    return headers, padded[1:]


def extract_pdf_blocks(path: Path, progress: Progress | None = None) -> tuple[list[dict], dict]:
    """
    Extract positioned text and detected tables from one text-based PDF.

    Inputs:
        path: Validated PDF document path.
        progress: Optional callback accepting status messages.

    Returns:
        Ordered common-representation blocks and source metadata.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires pdfplumber; run pip install -r requirements.txt"
        ) from exc

    blocks: list[dict] = []
    with pdfplumber.open(path) as document:
        page_count = len(document.pages)
        for page_number, page in enumerate(document.pages, start=1):
            if progress:
                progress(
                    f"Parsing {path.name} PDF pages: "
                    f"{page_number - 1}/{page_count} completed"
                )
            try:
                found_tables = page.find_tables()
            except Exception:
                # Some unusual PDF drawing instructions defeat table detection;
                # retain the page text instead of failing the complete corpus.
                found_tables = []
            boxes = [tuple(table.bbox) for table in found_tables]
            words = page.extract_words(use_text_flow=True) or []
            outside_words = [word for word in words if not _inside_any_table(word, boxes)]
            positioned: list[tuple[float, dict]] = []
            lines = _group_pdf_lines(outside_words)
            for top, text in lines:
                if text:
                    positioned.append(
                        (
                            top,
                            make_text_block(
                                text,
                                path.name,
                                page=page_number,
                                block_type="paragraph",
                            ),
                        )
                    )

            for table_number, table in enumerate(found_tables, start=1):
                raw_rows = table.extract() or []
                headers, rows = _clean_pdf_table(raw_rows)
                if not rows:
                    continue
                preceding = [text for top, text in lines if top < float(table.bbox[1])]
                caption = preceding[-1] if preceding else ""
                block = make_table_block(
                    headers,
                    rows,
                    path.name,
                    table_id=f"{path.name}-page-{page_number}-table-{table_number}",
                    caption=caption,
                    page=page_number,
                )
                block["bbox"] = [round(float(value), 2) for value in table.bbox]
                positioned.append((float(table.bbox[1]), block))
            blocks.extend(block for _, block in sorted(positioned, key=lambda item: item[0]))
            if progress:
                progress(
                    f"Parsed {path.name} PDF pages: "
                    f"{page_number}/{page_count} completed"
                )

    if not blocks:
        raise ValueError(
            f"No readable text was extracted from {path.name}. "
            "The PDF may be scanned and require OCR."
        )
    source = {
        "kind": "pdf",
        "filename": path.name,
        "bytes": path.stat().st_size,
        "pages": page_count,
    }
    return blocks, source


def prepare_pdf_corpus(
    value,
    root: Path,
    max_files: int,
    max_file_bytes: int,
    progress: Progress | None = None,
) -> SavedStructuredCorpus:
    """
    Validate, extract, persist, and copy an uploaded PDF corpus.

    Inputs:
        value: Gradio PDF upload value.
        root: Parent directory for saved PDF corpora.
        max_files: Maximum number of uploaded documents.
        max_file_bytes: Maximum byte size of each document.
        progress: Optional callback accepting status messages.

    Returns:
        Description of the saved content-addressed structured corpus.
    """
    paths = parse_pdf_paths(value, max_files, max_file_bytes)
    request_key = pdf_request_key(paths)
    blocks: list[dict] = []
    sources: list[dict] = []
    for source_number, path in enumerate(paths, start=1):
        extracted, source = extract_pdf_blocks(path, progress)
        source["source"] = source_number
        source["stored_file"] = f"sources/{source_number:02d}_{path.name}"
        blocks.extend(extracted)
        sources.append(source)

    saved = save_structured_corpus(blocks, sources, root, request_key)
    source_dir = saved.corpus_path.parent / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    for source_number, path in enumerate(paths, start=1):
        shutil.copy2(path, source_dir / f"{source_number:02d}_{path.name}")
    return saved
