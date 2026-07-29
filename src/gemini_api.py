"""Gemini embedding, graph-extraction, and answer-generation operations."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from .config import Settings


# These per-chunk ceilings keep graph extraction bounded and predictable.
MAX_ENTITIES_PER_CHUNK = 20
MAX_RELATIONS_PER_CHUNK = 30

EXTRACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "entities": {
                        "type": "array",
                        "maxItems": MAX_ENTITIES_PER_CHUNK,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["name", "description"],
                        },
                    },
                    "relations": {
                        "type": "array",
                        "maxItems": MAX_RELATIONS_PER_CHUNK,
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "relation": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": [
                                "source",
                                "target",
                                "relation",
                                "description",
                            ],
                        },
                    },
                },
                "required": ["chunk_id", "entities", "relations"],
            },
        }
    },
    "required": ["chunks"],
}


class GraphExtractionError(ValueError):
    """The graph model did not return a complete batch in the required shape."""


def _normalise(rows: np.ndarray) -> np.ndarray:
    """
    Scale every row vector to unit length.

    Inputs:
        rows: Two-dimensional array with one vector per row.

    Returns:
        An array of the same shape containing normalized row vectors.
    """
    lengths = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(lengths, 1e-12)


def _parse_extraction_response(text: str, expected_ids: set[int]) -> list[dict]:
    """
    Parse one or more JSON objects returned by the extraction model.

    Gemini occasionally returns two valid JSON objects one after another. A
    normal ``json.loads`` call reports this as ``JSONDecodeError: Extra data``.
    Reading each top-level object lets us keep all valid chunk results.

    Inputs:
        text: Raw model response, optionally wrapped in a Markdown code fence.
        expected_ids: Chunk IDs that must all be present in the response.

    Returns:
        Validated extraction dictionaries ordered by chunk ID.
    """
    cleaned = (text or "{}").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    decoder = json.JSONDecoder()
    values: list[object] = []
    position = 0
    while position < len(cleaned):
        while position < len(cleaned) and cleaned[position].isspace():
            position += 1
        if position >= len(cleaned):
            break
        value, position = decoder.raw_decode(cleaned, position)
        values.append(value)

    extracted: dict[int, dict] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for item in value.get("chunks", []):
            if not isinstance(item, dict) or "chunk_id" not in item:
                continue
            try:
                chunk_id = int(item["chunk_id"])
            except (TypeError, ValueError):
                continue
            extracted[chunk_id] = item

    missing = expected_ids - extracted.keys()
    if missing:
        raise ValueError(
            "The extraction model omitted chunk IDs: "
            + ", ".join(str(chunk_id) for chunk_id in sorted(missing))
        )
    return [extracted[chunk_id] for chunk_id in sorted(expected_ids)]


class Gemini:
    def __init__(self, settings: Settings):
        """
        Create the shared Gemini API client.

        Inputs:
            settings: Model names, API key, rate limits, and embedding settings.

        Returns:
            None. The initialized client is stored on this instance.
        """
        self.settings = settings
        self.client = genai.Client(api_key=settings.require_api_key())

    def embed(
        self,
        texts: Sequence[str],
        task_type: str,
        progress: Callable[[str], None] | None = None,
        checkpoint_path: Path | None = None,
    ) -> np.ndarray:
        """
        Embed in paced batches, optionally resuming from a NumPy checkpoint.

        Inputs:
            texts: Text strings to embed.
            task_type: Gemini embedding task type, such as
                ``RETRIEVAL_DOCUMENT`` or ``QUESTION_ANSWERING``.
            progress: Optional callback that receives human-readable status.
            checkpoint_path: Optional NumPy file used to save and resume rows.

        Returns:
            A normalized embedding matrix with one row per input text.
        """
        if not texts:
            return np.empty((0, self.settings.embedding_dimensions), dtype=np.float32)

        rows: list[list[float]] = []
        if checkpoint_path and checkpoint_path.exists():
            saved = np.load(checkpoint_path)
            if saved.ndim == 2 and saved.shape[1] == self.settings.embedding_dimensions:
                rows = saved.tolist()
                if progress:
                    progress(f"Resumed {len(rows)}/{len(texts)} embeddings from checkpoint")
        if len(rows) > len(texts):
            rows = []

        size = self.settings.embed_batch_size
        for start in range(len(rows), len(texts), size):
            batch = list(texts[start : start + size])
            for attempt in range(4):
                try:
                    response = self.client.models.embed_content(
                        model=self.settings.embedding_model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=self.settings.embedding_dimensions,
                        ),
                    )
                    break
                except Exception as exc:
                    is_quota = getattr(exc, "status_code", None) == 429 or "429" in str(exc)
                    if not is_quota or attempt == 3:
                        raise
                    match = re.search(r"retry in ([0-9.]+)s", str(exc), re.IGNORECASE)
                    delay = float(match.group(1)) + 2 if match else 65.0
                    if progress:
                        progress(f"Embedding quota reached; retrying in {delay:.0f}s")
                    time.sleep(delay)
            rows.extend(item.values for item in response.embeddings)
            if progress:
                progress(f"Embedded {min(start + size, len(texts))}/{len(texts)} texts")
            # Frequent small checkpoints make interruptions inexpensive to resume.
            if checkpoint_path and (
                len(rows) == len(texts) or ((start // size) + 1) % 5 == 0
            ):
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(checkpoint_path, np.asarray(rows, dtype=np.float32))
            if start + size < len(texts):
                time.sleep(self.settings.embed_batch_delay)
        result = _normalise(np.asarray(rows, dtype=np.float32))
        if checkpoint_path:
            np.save(checkpoint_path, result)
        return result

    def embed_documents(
        self,
        texts: Sequence[str],
        progress: Callable[[str], None] | None = None,
        checkpoint_path: Path | None = None,
    ) -> np.ndarray:
        """
        Embed document texts for retrieval.

        Inputs:
            texts: Document or chunk texts to embed.
            progress: Optional callback that receives status messages.
            checkpoint_path: Optional NumPy checkpoint location.

        Returns:
            A normalized embedding matrix with one row per document.
        """
        return self.embed(texts, "RETRIEVAL_DOCUMENT", progress, checkpoint_path)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed one user query for question-answer retrieval.

        Inputs:
            query: User question to embed.

        Returns:
            One normalized query vector.
        """
        return self.embed([query], "QUESTION_ANSWERING")[0]

    def extract_graph_batch(self, chunks: list[dict]) -> list[dict]:
        """
        Extract compact entity and relation records for several core chunks.

        Inputs:
            chunks: Dictionaries containing a ``chunk_id`` and its ``text``.

        Returns:
            One validated extraction dictionary for every supplied chunk.
        """
        payload = json.dumps(chunks, ensure_ascii=False)
        prompt = f"""
Extract a small factual knowledge graph from each text chunk below.
Return JSON only, with this exact shape:
{{"chunks":[{{"chunk_id":0,"entities":[{{"name":"...","description":"..."}}],
"relations":[{{"source":"...","target":"...","relation":"...","description":"..."}}]}}]}}

Rules:
- Keep only important named people, places, organisations, works, and events.
- Merge aliases within a chunk by using one canonical name.
- Descriptions must be short and grounded only in the supplied text.
- Return at most {MAX_ENTITIES_PER_CHUNK} entities and
  {MAX_RELATIONS_PER_CHUNK} relations per chunk.
- Preserve every supplied chunk_id, even when its arrays are empty.

INPUT:
{payload}
"""
        expected_ids = {int(chunk["chunk_id"]) for chunk in chunks}
        last_error: Exception | None = None
        # A repeated large-batch request often fails in the same way. Two
        # attempts are enough before the index builder falls back to one chunk
        # per request. A single-chunk request gets one additional attempt.
        attempts = 3 if len(chunks) == 1 else 2
        for attempt in range(attempts):
            response = self.client.models.generate_content(
                model=self.settings.extraction_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_json_schema=EXTRACTION_RESPONSE_SCHEMA,
                    max_output_tokens=16384,
                ),
            )
            try:
                extracted = _parse_extraction_response(
                    response.text or "{}", expected_ids
                )
                # Enforce the limits locally as well as asking the model to
                # respect them. Model-generated output is not always exact.
                for item in extracted:
                    entities = item.get("entities", [])
                    relations = item.get("relations", [])
                    item["entities"] = (
                        entities[:MAX_ENTITIES_PER_CHUNK]
                        if isinstance(entities, list)
                        else []
                    )
                    item["relations"] = (
                        relations[:MAX_RELATIONS_PER_CHUNK]
                        if isinstance(relations, list)
                        else []
                    )
                return extracted
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(1.0)

        detail = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error
            else "no response details"
        )
        raise GraphExtractionError(
            "Gemini returned invalid or incomplete JSON for the extraction batch "
            f"after {attempts} attempts. Last error: {detail}"
        ) from last_error

    def answer(self, query: str, context: str, method: str, temperature: float) -> str:
        """
        Generate one context-grounded answer.

        Inputs:
            query: User question.
            context: Labelled evidence retrieved by one RAG strategy.
            method: RAG method name included in the model instruction.
            temperature: Generation randomness passed to Gemini.

        Returns:
            Generated answer text, or a fallback message for an empty response.
        """
        prompt = f"""
You are answering a question about the supplied data using {method}.
Use only the context below. If it is insufficient, say so plainly.
Give a direct answer, then briefly explain the supporting evidence.
When useful, cite the context labels exactly as supplied, for example
[chunk 12] or [chunk 12, subchunk 24].

QUESTION:
{query}

CONTEXT:
{context}
"""
        response = self.client.models.generate_content(
            model=self.settings.generation_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=float(temperature)),
        )
        return response.text or "(The model returned no text.)"
