"""Coordinate persistent indexes, retrieval, and three-answer generation."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from .config import Settings
from .gemini_api import Gemini
from .pdf_corpus import parse_pdf_paths, pdf_request_key, prepare_pdf_corpus
from .persistent_indexes import IndexStore
from .retrieval_strategies import ket_retrieve, knn_retrieve, text_retrieve
from .web_corpus import parse_url_list, prepare_web_corpus, url_request_key


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

    def build(
        self,
        knn_k: int,
        ket_k: int,
        beta: float,
        tau: int,
        progress=None,
        use_url_corpus: bool = False,
        url_text: str = "",
        use_pdf_corpus: bool = False,
        pdf_files=None,
    ) -> str:
        """
        Build or load the indexes selected in the UI.

        Inputs:
            knn_k: Outgoing neighbour count for the KNNG-RAG index.
            ket_k: Neighbour count for KET-RAG's intermediate KNN graph.
            beta: Fraction of chunks used for the KET skeleton.
            tau: Number of conceptual binary KET subchunk splits.
            progress: Optional callback accepting status strings.
            use_url_corpus: Whether to download and index the supplied pages.
            url_text: One HTTP(S) page URL per line when URL mode is enabled.
            use_pdf_corpus: Whether to index the uploaded PDF documents.
            pdf_files: Gradio PDF upload paths when PDF mode is enabled.

        Returns:
            A short readiness message containing chunk count and elapsed time.
        """
        started = time.perf_counter()
        if use_url_corpus and use_pdf_corpus:
            raise ValueError("Select either URL corpus mode or PDF corpus mode, not both.")
        if use_url_corpus:
            saved = prepare_web_corpus(
                url_text,
                self.settings.url_corpus_dir,
                self.settings.max_url_pages,
                self.settings.url_timeout_seconds,
                self.settings.max_url_page_bytes,
                progress,
            )
            active_settings = replace(
                self.settings,
                data_file=saved.corpus_path,
                structured_blocks_file=saved.blocks_path,
            )
            store = IndexStore(active_settings, self.gemini)
            corpus_selection = f"urls:{saved.request_key}"
            corpus_description = (
                f"{saved.source_count} web page(s), saved in {saved.corpus_path.parent}"
            )
        elif use_pdf_corpus:
            saved = prepare_pdf_corpus(
                pdf_files,
                self.settings.pdf_corpus_dir,
                self.settings.max_pdf_files,
                self.settings.max_pdf_file_bytes,
                progress,
            )
            active_settings = replace(
                self.settings,
                data_file=saved.corpus_path,
                structured_blocks_file=saved.blocks_path,
            )
            store = IndexStore(active_settings, self.gemini)
            corpus_selection = f"pdfs:{saved.request_key}"
            corpus_description = (
                f"{saved.source_count} PDF document(s), saved in "
                f"{saved.corpus_path.parent}"
            )
        else:
            store = self.store
            corpus_selection = f"file:{self.settings.data_file.resolve()}"
            corpus_description = f"local file {self.settings.data_file.name}"

        chunks, vectors, knn = store.ensure_knn(int(knn_k), progress)
        ket = store.ensure_ket(int(ket_k), float(beta), int(tau), progress)
        self.loaded = {
            "parameters": (int(knn_k), int(ket_k), round(float(beta), 4), int(tau)),
            "corpus_selection": corpus_selection,
            "chunks": chunks,
            "vectors": vectors,
            "knn": knn,
            "ket": ket,
        }
        return (
            f"Ready: {len(chunks)} chunks from {corpus_description}; "
            f"indexes loaded in {time.perf_counter() - started:.1f}s."
        )

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
        use_url_corpus: bool = False,
        url_text: str = "",
        use_pdf_corpus: bool = False,
        pdf_files=None,
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
            use_url_corpus: Whether the current UI selection uses web pages.
            url_text: Current URL textbox value used to verify the loaded corpus.
            use_pdf_corpus: Whether the current UI selection uses PDFs.
            pdf_files: Current uploaded PDF paths used to verify the corpus.

        Returns:
            Four values: Text RAG answer, KNNG-RAG answer, KET-RAG answer, and
            a retrieval diagnostics dictionary.
        """
        if not query.strip():
            raise ValueError("Please enter a query.")
        requested = (int(knn_k), int(ket_k), round(float(beta), 4), int(tau))
        if use_url_corpus and use_pdf_corpus:
            raise ValueError("Select either URL corpus mode or PDF corpus mode, not both.")
        if use_url_corpus:
            urls = parse_url_list(url_text, self.settings.max_url_pages)
            corpus_selection = f"urls:{url_request_key(urls)}"
        elif use_pdf_corpus:
            paths = parse_pdf_paths(
                pdf_files,
                self.settings.max_pdf_files,
                self.settings.max_pdf_file_bytes,
            )
            corpus_selection = f"pdfs:{pdf_request_key(paths)}"
        else:
            corpus_selection = f"file:{self.settings.data_file.resolve()}"
        if (
            not self.loaded
            or self.loaded["parameters"] != requested
            or self.loaded["corpus_selection"] != corpus_selection
        ):
            raise RuntimeError(
                "Build/load indexes for the selected corpus and graph parameters first."
            )

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
        def source_label(source: dict) -> str:
            """
            Format one KET source with document and table details.

            Inputs:
                source: Retrieved-source diagnostics dictionary.

            Returns:
                One bracketed human-readable source label.
            """
            parts = [
                f"chunk {source['chunk_id']}",
                f"subchunk {source['subchunk_id']}",
            ]
            for key in ("source_name", "page", "table_id"):
                if source.get(key) not in (None, ""):
                    parts.append(f"{key.replace('_', ' ')} {source[key]}")
            if source.get("row_start") is not None:
                parts.append(
                    f"rows {source['row_start']}-{source.get('row_end', source['row_start'])}"
                )
            return "[" + ", ".join(parts) + "]"

        ket_sources = ", ".join(
            source_label(source) for source in ket_details["retrieved_sources"]
        ) or "(no text source retrieved)"
        ket_answer = f"Retrieved sources: {ket_sources}\n\n{answers[2]}"
        return answers[0], answers[1], ket_answer, diagnostics
