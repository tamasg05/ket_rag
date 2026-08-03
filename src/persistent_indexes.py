"""Build and load persistent Text, KNNG, and KET-RAG index artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .config import Settings
from .corpus_processing import (
    chunk_words,
    load_text_corpus,
    split_chunks_by_tau,
    split_sentences,
)
from .gemini_api import Gemini, GraphExtractionError
from .graph_construction import (
    build_hybrid_knn,
    build_keyword_subchunk_graph,
    build_skeleton_graph,
    format_relationship_text,
    select_core_chunks,
)
from .structured_corpus import (
    STRUCTURED_CORPUS_VERSION,
    chunk_structured_blocks,
    load_structured_blocks,
)


Progress = Callable[[str], None]
INDEX_FORMAT_VERSION = "v2"


def _say(progress: Progress | None, message: str) -> None:
    """
    Send a status message when a progress callback is available.

    Inputs:
        progress: Optional callback accepting one status string.
        message: Status text to send.

    Returns:
        None.
    """
    if progress:
        progress(message)


def _write_json(path: Path, value: object) -> None:
    """
    Atomically write a JSON-serializable value.

    Inputs:
        path: Destination JSON file.
        value: JSON-serializable Python value.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path):
    """
    Read and decode one UTF-8 JSON file.

    Inputs:
        path: JSON file to load.

    Returns:
        The decoded Python value.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _source_key(settings: Settings) -> str:
    """
    Derive a cache key from the corpus and base-index settings.

    Inputs:
        settings: Source path, chunking values, and embedding configuration.

    Returns:
        A short deterministic hexadecimal cache key.
    """
    source_digest = hashlib.sha256(settings.data_file.read_bytes())
    if settings.structured_blocks_file:
        source_digest.update(settings.structured_blocks_file.read_bytes())
    digest = source_digest.hexdigest()[:16]
    identity = (
        f"{digest}|{settings.chunk_words}|{settings.chunk_overlap}|"
        f"{settings.embedding_model}|{settings.embedding_dimensions}|"
        f"{INDEX_FORMAT_VERSION}"
    )
    if settings.structured_blocks_file:
        identity += f"|structured:{STRUCTURED_CORPUS_VERSION}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


class IndexStore:
    """Creates missing artifacts, otherwise performs no paid indexing calls."""

    def __init__(self, settings: Settings, gemini: Gemini):
        """
        Configure the persistent index store.

        Inputs:
            settings: Corpus, cache, model, and indexing configuration.
            gemini: Shared Gemini wrapper used for paid model operations.

        Returns:
            None. The corpus-specific cache directory is stored on the instance.
        """
        self.settings = settings
        self.gemini = gemini
        self.root = settings.cache_dir / _source_key(settings)

    def _embed_relationships(
        self,
        relations: list[dict],
        ket_dir: Path,
        progress: Progress | None,
    ) -> np.ndarray:
        """
        Load or create one persistent embedding per relationship record.

        Inputs:
            relations: Skeleton relationship records in stable index order.
            ket_dir: Directory containing this KET index's artifacts.
            progress: Optional callback accepting status messages.

        Returns:
            A normalized embedding matrix aligned with ``relations``.
        """
        relation_texts = [
            format_relationship_text(relation) for relation in relations
        ]
        _say(progress, f"Embedding {len(relation_texts)} skeleton relationships")
        return self.gemini.embed_documents(
            relation_texts,
            progress,
            ket_dir / "relationship_embeddings.npy",
        )

    def ensure_base(self, progress: Progress | None = None) -> tuple[list[dict], np.ndarray]:
        """
        Load or create chunks and their persistent embeddings.

        Inputs:
            progress: Optional callback accepting status messages.

        Returns:
            A pair containing chunk dictionaries and their embedding matrix.
        """
        chunks_path = self.root / "chunks.json"
        vectors_path = self.root / "chunk_embeddings.npy"
        if chunks_path.exists() and vectors_path.exists():
            _say(progress, "Loaded persistent chunk embeddings")
            return _read_json(chunks_path), np.load(vectors_path)

        if self.settings.structured_blocks_file:
            _say(progress, "Chunking structured corpus")
            blocks = load_structured_blocks(self.settings.structured_blocks_file)
            chunks = chunk_structured_blocks(
                blocks,
                self.settings.chunk_words,
                self.settings.chunk_overlap,
            )
        else:
            _say(progress, "Chunking text corpus")
            chunks = chunk_words(
                load_text_corpus(self.settings.data_file),
                self.settings.chunk_words,
                self.settings.chunk_overlap,
            )
        if not chunks:
            raise ValueError("The selected corpus did not produce any text chunks.")
        vectors = self.gemini.embed_documents([c["text"] for c in chunks], progress)
        self.root.mkdir(parents=True, exist_ok=True)
        _write_json(chunks_path, chunks)
        np.save(vectors_path, vectors)
        _write_json(
            self.root / "manifest.json",
            {
                "source": str(self.settings.data_file),
                "structured_blocks": (
                    str(self.settings.structured_blocks_file)
                    if self.settings.structured_blocks_file
                    else None
                ),
                "chunk_words": self.settings.chunk_words,
                "chunk_overlap": self.settings.chunk_overlap,
                "embedding_model": self.settings.embedding_model,
                "embedding_dimensions": self.settings.embedding_dimensions,
            },
        )
        return chunks, vectors

    def ensure_knn(
        self, k: int, progress: Progress | None = None
    ) -> tuple[list[dict], np.ndarray, list[list[int]]]:
        """
        Load or create the persistent hybrid KNN graph for one ``k``.

        Inputs:
            k: Requested outgoing neighbour count per chunk.
            progress: Optional callback accepting status messages.

        Returns:
            A tuple containing chunks, chunk embeddings, and the adjacency list.
        """
        chunks, vectors = self.ensure_base(progress)
        path = self.root / f"knn_k{k}.json"
        if path.exists():
            _say(progress, f"Loaded persistent KNN graph (k={k})")
            return chunks, vectors, _read_json(path)["adjacency"]
        _say(progress, f"Building hybrid KNN graph (k={k})")
        adjacency = build_hybrid_knn([c["text"] for c in chunks], vectors, k)
        _write_json(path, {"k": k, "adjacency": adjacency})
        return chunks, vectors, adjacency

    def ensure_ket(
        self, k: int, beta: float, tau: int, progress: Progress | None = None
    ) -> dict:
        """
        Load or create the prototype adaptation of KET-Index (Algorithm 3).

        Inputs:
            k: Outgoing neighbour count used by the intermediate KNN graph.
            beta: Fraction of PageRank-leading chunks used for the skeleton.
            tau: Number of conceptual binary subchunk splits.
            progress: Optional callback accepting status messages.

        Returns:
            KET metadata, graph mappings, and embedding arrays.
        """
        beta = round(float(beta), 4)
        extraction_key = hashlib.sha256(
            f"{self.settings.extraction_model}|{INDEX_FORMAT_VERSION}".encode()
        ).hexdigest()[:8]
        ket_dir = self.root / f"ket_k{k}_b{beta:g}_t{tau}_x{extraction_key}"
        complete = ket_dir / "complete.json"
        arrays = ket_dir / "arrays.npz"
        if complete.exists() and arrays.exists():
            _say(progress, f"Loaded persistent KET-RAG index (k={k}, beta={beta}, tau={tau})")
            data = _read_json(complete)
            with np.load(arrays) as saved:
                loaded_arrays = {name: saved[name] for name in saved.files}

            # Older prototype indexes did not embed relationships. Upgrade
            # those arrays in place without repeating paid graph extraction.
            relation_vectors = loaded_arrays.get("relation_vectors")
            expected_shape = (
                len(data["relations"]),
                self.settings.embedding_dimensions,
            )
            if relation_vectors is None or relation_vectors.shape != expected_shape:
                _say(progress, "Adding relationship embeddings to the existing index")
                loaded_arrays["relation_vectors"] = self._embed_relationships(
                    data["relations"], ket_dir, progress
                )
                np.savez_compressed(arrays, **loaded_arrays)
                _say(progress, "Relationship embeddings saved")
            data["arrays"] = loaded_arrays
            return data

        chunks, chunk_vectors, adjacency = self.ensure_knn(k, progress)
        core_ids = select_core_chunks(adjacency, beta)
        _say(progress, f"Selected {len(core_ids)} core chunks with PageRank")

        # Checkpoint LLM extraction after every batch.
        checkpoint = ket_dir / "extraction_checkpoint.json"
        records = _read_json(checkpoint) if checkpoint.exists() else {}
        missing = [chunk_id for chunk_id in core_ids if str(chunk_id) not in records]
        completed_extractions = len(core_ids) - len(missing)
        if records:
            _say(
                progress,
                f"Resumed {completed_extractions}/{len(core_ids)} "
                "skeleton chunk extractions from checkpoint",
            )
        batch_size = self.settings.extraction_batch_size
        for start in range(0, len(missing), batch_size):
            ids = missing[start : start + batch_size]
            payload = [
                {
                    "chunk_id": i,
                    "text": (
                        chunks[i].get("source_text", chunks[i]["text"])
                        if self.settings.structured_blocks_file
                        else chunks[i]["text"]
                    ),
                }
                for i in ids
            ]
            _say(
                progress,
                f"Extracting skeleton chunks: "
                f"{completed_extractions}/{len(core_ids)} completed",
            )
            try:
                extracted = self.gemini.extract_graph_batch(payload)
                by_id = {
                    int(item["chunk_id"]): item
                    for item in extracted
                    if "chunk_id" in item
                }
                for chunk_id in ids:
                    records[str(chunk_id)] = by_id[chunk_id]
                _write_json(checkpoint, records)
                completed_extractions += len(ids)
                _say(
                    progress,
                    f"Extracted skeleton chunks: "
                    f"{completed_extractions}/{len(core_ids)} completed",
                )
            except GraphExtractionError:
                # Large structured responses can still be truncated. Retrying
                # one chunk at a time is slower but much more reliable, and
                # checkpointing each result avoids losing successful work.
                _say(
                    progress,
                    "Batch response was incomplete; retrying these chunks "
                    "individually",
                )
                for chunk_id in ids:
                    _say(
                        progress,
                        f"Extracting fallback chunk ID {chunk_id}: "
                        f"{completed_extractions}/{len(core_ids)} completed",
                    )
                    extracted = self.gemini.extract_graph_batch(
                        [
                            {
                                "chunk_id": chunk_id,
                                "text": (
                                    chunks[chunk_id].get(
                                        "source_text", chunks[chunk_id]["text"]
                                    )
                                    if self.settings.structured_blocks_file
                                    else chunks[chunk_id]["text"]
                                ),
                            }
                        ]
                    )
                    records[str(chunk_id)] = extracted[0]
                    _write_json(checkpoint, records)
                    completed_extractions += 1
                    _say(
                        progress,
                        f"Extracted fallback chunk ID {chunk_id}: "
                        f"{completed_extractions}/{len(core_ids)} completed",
                    )

        entity_list, relations = build_skeleton_graph(core_ids, records)
        entity_texts = [
            f"{e['name']}: {' '.join(dict.fromkeys(e['descriptions']))}" for e in entity_list
        ]
        _say(progress, f"Embedding {len(entity_texts)} skeleton entities")
        entity_vectors = self.gemini.embed_documents(
            entity_texts, progress, ket_dir / "entity_embeddings.npy"
        )
        relation_vectors = self._embed_relationships(
            relations, ket_dir, progress
        )

        # Algorithm 3 line 10: create 2**tau fine-grained subchunks.
        subchunks = split_chunks_by_tau(chunks, tau)
        if tau == 0:
            sub_vectors = chunk_vectors.copy()
        else:
            _say(progress, f"Embedding {len(subchunks)} fine-grained sub-chunks")
            sub_vectors = self.gemini.embed_documents(
                [s["text"] for s in subchunks],
                progress,
                ket_dir / "subchunk_embeddings.npy",
            )

        # The sentence embeddings support keyword descriptions in line 13.
        sentences = split_sentences(
            " ".join(s.get("source_text", s["text"]) for s in subchunks)
        )
        _say(progress, f"Embedding {len(sentences)} sentences for keyword descriptions")
        sentence_vectors = self.gemini.embed_documents(
            sentences, progress, ket_dir / "sentence_embeddings.npy"
        )
        keywords, keyword_to_subchunks, keyword_vectors = (
            build_keyword_subchunk_graph(
                subchunks,
                sentences,
                sentence_vectors,
                self.settings.embedding_dimensions,
            )
        )

        ket_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "k": k,
            "beta": beta,
            "tau": tau,
            "extraction_model": self.settings.extraction_model,
            "core_ids": core_ids,
            "entities": entity_list,
            "relations": relations,
            "subchunks": subchunks,
            "keywords": keywords,
            "keyword_to_subchunks": keyword_to_subchunks,
        }
        _write_json(complete, metadata)
        np.savez_compressed(
            arrays,
            entity_vectors=entity_vectors,
            relation_vectors=relation_vectors,
            subchunk_vectors=sub_vectors,
            keyword_vectors=keyword_vectors,
        )
        metadata["arrays"] = {
            "entity_vectors": entity_vectors,
            "relation_vectors": relation_vectors,
            "subchunk_vectors": sub_vectors,
            "keyword_vectors": keyword_vectors,
        }
        _say(progress, "KET-RAG index saved")
        return metadata
