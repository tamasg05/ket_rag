"""Coordinate persistent indexes, retrieval, and three-answer generation."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .config import Settings
from .gemini_api import Gemini
from .persistent_indexes import IndexStore
from .retrieval_strategies import ket_retrieve, knn_retrieve, text_retrieve


class RagComparison:
    def __init__(self, settings: Settings | None = None):
        """
        Create the service coordinating indexes, retrieval, and generation.

        Inputs:
            settings: Optional configuration; defaults are loaded when omitted.

        Returns:
            None. The API client and persistent store are initialized here.
        """
        self.settings = settings or Settings()
        self.gemini = Gemini(self.settings)
        self.store = IndexStore(self.settings, self.gemini)
        self.loaded: dict | None = None

    def build(self, knn_k: int, ket_k: int, beta: float, tau: int, progress=None) -> str:
        """
        Build or load the indexes selected in the UI.

        Inputs:
            knn_k: Outgoing neighbour count for the KNNG-RAG index.
            ket_k: Neighbour count for KET-RAG's intermediate KNN graph.
            beta: Fraction of chunks used for the KET skeleton.
            tau: Number of conceptual binary KET subchunk splits.
            progress: Optional callback accepting status strings.

        Returns:
            A short readiness message containing chunk count and elapsed time.
        """
        started = time.perf_counter()
        chunks, vectors, knn = self.store.ensure_knn(int(knn_k), progress)
        ket = self.store.ensure_ket(int(ket_k), float(beta), int(tau), progress)
        self.loaded = {
            "parameters": (int(knn_k), int(ket_k), round(float(beta), 4), int(tau)),
            "chunks": chunks,
            "vectors": vectors,
            "knn": knn,
            "ket": ket,
        }
        return f"Ready: {len(chunks)} chunks; indexes loaded in {time.perf_counter() - started:.1f}s."

    def compare(
        self,
        query: str,
        top_k: int,
        temperature: float,
        knn_k: int,
        ket_k: int,
        beta: float,
        tau: int,
        theta: float,
    ):
        """
        Retrieve evidence and generate answers with all three RAG variants.

        Inputs:
            query: User question.
            top_k: Maximum retrieved chunk or subchunk count per strategy.
            temperature: Generation randomness used for all three answers.
            knn_k: KNNG-RAG graph parameter expected in the loaded index.
            ket_k: KET-RAG KNN parameter expected in the loaded index.
            beta: KET skeleton fraction expected in the loaded index.
            tau: KET subchunk split count expected in the loaded index.
            theta: Share of KET's retrieval count assigned to the skeleton.

        Returns:
            Four values: Text RAG answer, KNNG-RAG answer, KET-RAG answer, and
            a retrieval diagnostics dictionary.
        """
        if not query.strip():
            raise ValueError("Please enter a query.")
        requested = (int(knn_k), int(ket_k), round(float(beta), 4), int(tau))
        if not self.loaded or self.loaded["parameters"] != requested:
            raise RuntimeError("Build/load indexes for the selected graph parameters first.")

        query_vector = self.gemini.embed_query(query)
        chunks = self.loaded["chunks"]
        vectors = self.loaded["vectors"]
        text_context, text_ids = text_retrieve(chunks, vectors, query_vector, int(top_k))
        knn_context, knn_ids = knn_retrieve(
            chunks, vectors, self.loaded["knn"], query_vector, int(top_k)
        )
        ket_context, ket_details = ket_retrieve(
            self.loaded["ket"], query_vector, int(top_k), float(theta)
        )

        jobs = [
            ("Text RAG", text_context),
            ("KNNG-RAG", knn_context),
            ("KET-RAG", ket_context),
        ]
        with ThreadPoolExecutor(max_workers=3) as pool:
            answers = list(
                pool.map(
                    lambda item: self.gemini.answer(
                        query, item[1], item[0], float(temperature)
                    ),
                    jobs,
                )
            )
        diagnostics = {
            "text_rag_chunk_ids": text_ids,
            "knng_rag_chunk_ids": knn_ids,
            "ket_rag": ket_details,
        }
        ket_sources = ", ".join(
            f"[chunk {source['chunk_id']}, subchunk {source['subchunk_id']}]"
            for source in ket_details["retrieved_sources"]
        ) or "(no text source retrieved)"
        ket_answer = f"Retrieved sources: {ket_sources}\n\n{answers[2]}"
        return answers[0], answers[1], ket_answer, diagnostics
