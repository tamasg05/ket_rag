"""Load a corpus and create word chunks, sentences, and KET subchunks."""

from __future__ import annotations

import re
from pathlib import Path

from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer


# Keep letters, numbers, and internal apostrophes, but discard punctuation
# marks that occur around words. Case is preserved for names and answer context.
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)

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


def load_book(path: Path) -> str:
    """
    Load the UTF-8 text corpus.

    Inputs:
        path: Location of the source text file.

    Returns:
        The complete corpus as one string.
    """
    return path.read_text(encoding="utf-8-sig", errors="replace")


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
    """
    Return the original text covered by a half-open range of word tokens.

    Inputs:
        text: Original punctuation-preserving source text.
        matches: Regex matches locating every word token in ``text``.
        start: Index of the first word token to include.
        end: Index immediately after the last word token to include.

    Returns:
        The corresponding source span, without leading or trailing whitespace.
    """
    if start >= end or start >= len(matches):
        return ""
    character_start = matches[start].start()
    character_end = matches[end].start() if end < len(matches) else len(text)
    return text[character_start:character_end].strip()


def chunk_words(text: str, size: int, overlap: int) -> list[dict]:
    """
    Divide a corpus into overlapping word-based chunks.

    Inputs:
        text: Complete source corpus.
        size: Maximum number of word tokens in each chunk.
        overlap: Number of tokens repeated between consecutive chunks.

    Returns:
        Chunk dictionaries containing IDs, normalized token text, and the
        corresponding punctuation-preserving source text.
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
                # Retain the original punctuation only for sentence splitting.
                "source_text": _text_span(text, matches, start, end),
            }
        )
        if start + size >= len(words):
            break
    return chunks


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
                    }
                )
    return result
