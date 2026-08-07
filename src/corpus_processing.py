"""Load plain text and create sentence and KET-specific subchunk views."""

from __future__ import annotations

from pathlib import Path

from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

from .data_extraction.chunking import (
    _WORD_PATTERN,
    _text_span,
    chunk_words,
    tokenize_words,
)

# Punkt normally learns abbreviations from a training corpus. Supplying common
# English titles explicitly gives this small demo reliable behaviour without
# downloading a separate NLTK language-data package.
_PUNKT_PARAMETERS = PunktParameters()
_PUNKT_PARAMETERS.abbrev_types.update(
    {
        "dr",
        "e.g",
        "etc",
        "fig",
        "i.e",
        "jr",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "sr",
        "st",
        "vs",
    }
)
_SENTENCE_TOKENIZER = PunktSentenceTokenizer(_PUNKT_PARAMETERS)


def load_text_corpus(path: Path) -> str:
    """
    Load the UTF-8 text corpus.

    Inputs:
        path: Location of the source text file.

    Returns:
        The complete corpus as one string.
    """
    return path.read_text(encoding="utf-8-sig", errors="replace")


def split_sentences(text: str) -> list[str]:
    """
    Split English text while preserving common titles and abbreviations.

    Inputs:
        text: Punctuation-preserving source text.

    Returns:
        Stripped sentences containing at least three word tokens.
    """
    return [
        sentence.strip()
        for sentence in _SENTENCE_TOKENIZER.tokenize(text)
        if len(tokenize_words(sentence)) >= 3
    ]


def split_chunks_by_tau(chunks: list[dict], tau: int) -> list[dict]:
    """
    Split every chunk into 2**tau approximately equal word ranges.

    This directly computes the final subchunks from Algorithm 3 line 10 rather
    than materializing each of the tau intermediate binary-splitting rounds.

    Inputs:
        chunks: Original chunk dictionaries to divide.
        tau: Number of conceptual binary splits; each chunk yields at most
            ``2**tau`` non-empty subchunks.

    Returns:
        Subchunk dictionaries with globally unique IDs, parent chunk IDs,
        part numbers, token text, and punctuation-preserving source text.
    """
    if tau < 0:
        raise ValueError("tau must be non-negative")
    pieces = 2**tau
    result: list[dict] = []
    for chunk in chunks:
        inherited = {
            key: value
            for key, value in chunk.items()
            if key
            not in {
                "id",
                "text",
                "source_text",
                "table_prefix",
                "table_row_texts",
            }
        }
        table_rows = chunk.get("table_row_texts", [])
        if table_rows:
            prefix = chunk.get("table_prefix", "")
            for part in range(pieces):
                start = round(part * len(table_rows) / pieces)
                end = round((part + 1) * len(table_rows) / pieces)
                selected_rows = table_rows[start:end]
                if not selected_rows:
                    continue
                source_text = "\n".join(
                    value for value in (prefix, *selected_rows) if value
                )
                result.append(
                    {
                        "id": len(result),
                        "parent_id": chunk["id"],
                        "part": part,
                        "text": " ".join(tokenize_words(source_text)),
                        "source_text": source_text,
                        **inherited,
                        "row_start": int(chunk.get("row_start", 1)) + start,
                        "row_end": int(chunk.get("row_start", 1)) + end - 1,
                    }
                )
            continue

        words = tokenize_words(chunk["text"])
        source_text = chunk.get("source_text", chunk["text"])
        source_matches = list(_WORD_PATTERN.finditer(source_text))
        for part in range(pieces):
            start = round(part * len(words) / pieces)
            end = round((part + 1) * len(words) / pieces)
            text = " ".join(words[start:end])
            if text:
                result.append(
                    {
                        "id": len(result),
                        "parent_id": chunk["id"],
                        "part": part,
                        "text": text,
                        "source_text": _text_span(
                            source_text, source_matches, start, end
                        ),
                        **inherited,
                    }
                )
    return result
