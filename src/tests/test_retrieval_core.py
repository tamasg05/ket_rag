import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.corpus_processing import (
    chunk_words,
    split_chunks_by_tau,
    split_sentences,
    tokenize_words,
)
from src.gemini_api import (
    MAX_ENTITIES_PER_CHUNK,
    MAX_RELATIONS_PER_CHUNK,
    _parse_extraction_response,
)
from src.graph_construction import (
    build_hybrid_knn,
    build_keyword_subchunk_graph,
    build_keyword_vocabulary,
    build_skeleton_graph,
    format_relationship_text,
)
from src.pdf_corpus import _clean_pdf_table, parse_pdf_paths, pdf_request_key
from src.retrieval_strategies import (
    ket_retrieve,
    knn_retrieve,
    serialise_chunks,
    serialise_subchunks,
    text_retrieve,
)
from src.web_corpus import (
    WebPage,
    extract_html_blocks,
    extract_html_text,
    fetch_html_pages,
    parse_url_list,
    save_web_corpus,
    url_request_key,
)
from src.structured_corpus import chunk_structured_blocks


class CoreTests(unittest.TestCase):
    def test_html_page_download(self):
        class FakeResponse:
            def __init__(self):
                self.headers = Message()
                self.headers["Content-Type"] = "text/html; charset=utf-8"

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def geturl(self):
                return "https://example.org/final"

            def read(self, limit):
                return b"<html><body><main><p>Downloaded text.</p></main></body></html>"

        with patch(
            "src.web_corpus.urllib.request.urlopen", return_value=FakeResponse()
        ):
            pages = fetch_html_pages(
                ["https://example.org/start"],
                timeout_seconds=1,
                max_page_bytes=1_000,
            )
        self.assertEqual(pages[0].final_url, "https://example.org/final")
        self.assertEqual(pages[0].text, "Downloaded text.")

    def test_url_list_validation_and_identity(self):
        urls = parse_url_list(
            "https://example.org/one\n\nhttps://example.org/two\n"
            "https://example.org/one",
            max_pages=2,
        )
        self.assertEqual(
            urls, ["https://example.org/one", "https://example.org/two"]
        )
        self.assertEqual(url_request_key(urls), url_request_key(list(urls)))
        with self.assertRaises(ValueError):
            parse_url_list("file:///tmp/private.txt", max_pages=2)
        with self.assertRaises(ValueError):
            parse_url_list("https://user:secret@example.org", max_pages=2)

    def test_html_extraction_and_corpus_persistence(self):
        title, text = extract_html_text(
            """
            <html><head><title> Example   article </title>
            <style>hidden style</style></head><body>
            <nav>navigation</nav><main><h1>Visible heading</h1>
            <p>First <strong>useful</strong> paragraph.</p>
            <script>hidden script</script></main></body></html>
            """
        )
        self.assertEqual(title, "Example article")
        self.assertIn("Visible heading", text)
        self.assertIn("First useful paragraph.", text)
        self.assertNotIn("hidden", text)
        self.assertNotIn("navigation", text)

        pages = [
            WebPage(
                requested_url="https://example.org/article",
                final_url="https://example.org/article",
                title=title,
                text=text,
            )
        ]
        with tempfile.TemporaryDirectory() as temporary:
            saved = save_web_corpus(pages, Path(temporary))
            self.assertTrue(saved.corpus_path.exists())
            saved_text = saved.corpus_path.read_text(encoding="utf-8")
            self.assertIn("Example article", saved_text)
            self.assertNotIn("WEB PAGE", saved_text)
            sources = json.loads(saved.sources_path.read_text(encoding="utf-8"))
            self.assertEqual(sources[0]["requested_url"], pages[0].requested_url)
            self.assertEqual(saved.source_count, 1)
            self.assertTrue(saved.blocks_path.exists())

    def test_html_table_structure_and_row_aware_chunking(self):
        _, blocks = extract_html_blocks(
            """
            <html><body><main><h1>Electrical limits</h1>
            <table><caption>Operating range</caption><thead>
            <tr><th rowspan="2">Parameter</th><th colspan="2">Limits</th></tr>
            <tr><th>Minimum</th><th>Maximum</th></tr></thead><tbody>
            <tr><td>Voltage</td><td>3.0 V</td><td>3.6 V</td></tr>
            <tr><td>Current</td><td>1 A</td><td>2 A</td></tr>
            </tbody></table></main></body></html>
            """,
            source_name="specification",
            source_url="https://example.org/specification",
        )
        table = next(block for block in blocks if block["type"] == "table")
        self.assertEqual(
            table["headers"],
            ["Parameter", "Limits > Minimum", "Limits > Maximum"],
        )
        chunks = chunk_structured_blocks([table], size=30, overlap=0)
        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["block_type"], "table")
        self.assertIn("Parameter = Voltage", chunks[0]["source_text"])
        self.assertIn("Limits > Maximum = 3.6 V", chunks[0]["source_text"])

        subchunks = split_chunks_by_tau(chunks, tau=1)
        self.assertEqual(len(subchunks), 2)
        self.assertIn("Parameter = Voltage", subchunks[0]["source_text"])
        self.assertNotIn("Parameter = Current", subchunks[0]["source_text"])
        self.assertIn("Parameter = Current", subchunks[1]["source_text"])
        self.assertEqual(
            [(subchunk["row_start"], subchunk["row_end"]) for subchunk in subchunks],
            [(1, 1), (2, 2)],
        )

        wide_chunks = chunk_structured_blocks([table], size=12, overlap=0)
        column_groups = {tuple(chunk["column_group"]) for chunk in wide_chunks}
        self.assertGreater(len(column_groups), 1)
        self.assertTrue(all(group[0] == "Parameter" for group in column_groups))
        self.assertTrue(
            all("Parameter =" in chunk["source_text"] for chunk in wide_chunks)
        )

    def test_pdf_upload_validation_and_table_normalization(self):
        headers, rows = _clean_pdf_table(
            [
                ["Parameter", "Minimum", "Maximum"],
                ["Voltage", "3.0 V", "3.6 V"],
                ["Current", "1 A", "2 A"],
            ]
        )
        self.assertEqual(headers, ["Parameter", "Minimum", "Maximum"])
        self.assertEqual(rows[0], ["Voltage", "3.0 V", "3.6 V"])

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "specification.pdf"
            path.write_bytes(b"%PDF-minimal-test")
            parsed = parse_pdf_paths([str(path)], max_files=2, max_file_bytes=100)
            self.assertEqual(parsed, [path.resolve()])
            self.assertEqual(pdf_request_key(parsed), pdf_request_key(list(parsed)))

    def test_word_tokenizer_removes_punctuation(self):
        self.assertEqual(
            tokenize_words("Hello, World! Isn't this Dr. Watson?"),
            ["Hello", "World", "Isn't", "this", "Dr", "Watson"],
        )
        chunks = chunk_words("Hello, World!", size=2, overlap=0)
        self.assertEqual(chunks[0]["text"], "Hello World")
        self.assertEqual(chunks[0]["source_text"], "Hello, World!")

    def test_sentence_splitter_handles_common_titles(self):
        sentences = split_sentences(
            "Dr. Watson arrived. Mr. Holmes waited! No. This fragment remains long enough."
        )
        self.assertEqual(
            sentences,
            [
                "Dr. Watson arrived.",
                "Mr. Holmes waited!",
                "No. This fragment remains long enough.",
            ],
        )

    def test_chunking_and_tau_split(self):
        chunks = chunk_words(" ".join(f"w{i}" for i in range(30)), size=10, overlap=2)
        self.assertEqual(len(chunks), 4)
        subchunks = split_chunks_by_tau(chunks[:1], tau=2)
        self.assertEqual(len(subchunks), 4)
        self.assertEqual(" ".join(s["text"] for s in subchunks), chunks[0]["text"])

    def test_keyword_vocabulary_keeps_all_terms_seen_in_two_subchunks(self):
        keywords = build_keyword_vocabulary(
            [
                "The singular clue appears once",
                "Holmes examines common evidence",
                "Watson records common evidence",
                "Holmes consults Watson",
            ]
        )
        self.assertIn("common", keywords)
        self.assertIn("evidence", keywords)
        self.assertIn("holmes", keywords)
        self.assertIn("watson", keywords)
        self.assertNotIn("singular", keywords)
        self.assertNotIn("the", keywords)

    def test_ket_graph_construction_helpers(self):
        entities, relations = build_skeleton_graph(
            [2, 5],
            {
                "2": {
                    "entities": [{"name": "Holmes", "description": "Detective"}],
                    "relations": [],
                },
                "5": {
                    "entities": [{"name": "holmes", "description": "Consultant"}],
                    "relations": [
                        {
                            "source": "Holmes",
                            "target": "Watson",
                            "relation": "friend",
                        }
                    ],
                },
            },
        )
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["chunk_ids"], [2, 5])
        self.assertEqual(relations[0]["chunk_id"], 5)
        self.assertEqual(
            format_relationship_text(relations[0]),
            "Holmes --[friend]--> Watson",
        )

        keywords, memberships, vectors = build_keyword_subchunk_graph(
            [
                {"text": "Holmes examines shared evidence"},
                {"text": "Watson records shared evidence"},
            ],
            [
                "Holmes examines shared evidence.",
                "Watson records shared evidence.",
            ],
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            embedding_dimensions=2,
        )
        shared = keywords.index("shared")
        self.assertEqual(memberships[shared], [0, 1])
        self.assertAlmostEqual(float(np.linalg.norm(vectors[shared])), 1.0, places=6)

    def test_extraction_parser_accepts_concatenated_json_objects(self):
        response = (
            '{"chunks":[{"chunk_id":2,"entities":[],"relations":[]}]}'
            '\n{"chunks":[{"chunk_id":5,"entities":[],"relations":[]}]}'
        )
        chunks = _parse_extraction_response(response, {2, 5})
        self.assertEqual([chunk["chunk_id"] for chunk in chunks], [2, 5])
        self.assertEqual(MAX_ENTITIES_PER_CHUNK, 20)
        self.assertEqual(MAX_RELATIONS_PER_CHUNK, 30)

    def test_ket_context_labels_parent_chunk_and_subchunk(self):
        subchunks = [
            {"id": 0, "parent_id": 7, "text": "first"},
            {"id": 1, "parent_id": 7, "text": "second"},
        ]
        context = serialise_subchunks(subchunks, [1])
        self.assertEqual(context, "[chunk 7, subchunk 1]\nsecond")

    def test_structured_context_label_and_source_text(self):
        context = serialise_chunks(
            [
                {
                    "text": "normalized text",
                    "source_text": "Parameter = Voltage; Maximum = 3.6 V",
                    "source_name": "specification.pdf",
                    "page": 4,
                    "table_id": "limits",
                    "row_start": 2,
                    "row_end": 2,
                }
            ],
            [0],
        )
        self.assertIn("source specification.pdf, page 4, table limits, rows 2-2", context)
        self.assertIn("Parameter = Voltage; Maximum = 3.6 V", context)

    def test_ket_skeleton_ranks_relationship_and_text_adjacency(self):
        entities = [
            {
                "name": f"Entity {i}",
                "descriptions": [f"Description {i}"],
                "chunk_ids": [0] if i in (0, 1) else [],
            }
            for i in range(11)
        ]
        entity_vectors = np.array(
            [[1.0 - i * 0.01, 0.0] for i in range(10)] + [[-1.0, 0.0]],
            dtype=np.float32,
        )
        relations = [
            {
                "source": "Entity 0",
                "target": "Entity 1",
                "relation": "two-seed relation",
                "description": "Touches two selected entity seeds.",
                "chunk_id": 0,
            },
            {
                "source": "Entity 0",
                "target": "Entity 10",
                "relation": "one-seed relation",
                "description": "Touches one selected entity seed.",
                "chunk_id": 1,
            },
        ]
        index = {
            "entities": entities,
            "relations": relations,
            "subchunks": [
                {"id": 0, "parent_id": 0, "text": "structurally stronger"},
                {"id": 1, "parent_id": 1, "text": "semantically stronger"},
            ],
            "keywords": [],
            "keyword_to_subchunks": [],
            "arrays": {
                "entity_vectors": entity_vectors,
                # The one-seed relation is more semantically similar, but the
                # two-seed relationship must rank first by adjacency.
                "relation_vectors": np.array(
                    [[0.0, 1.0], [1.0, 0.0]], dtype=np.float32
                ),
                "subchunk_vectors": np.array(
                    [[0.0, 1.0], [1.0, 0.0]], dtype=np.float32
                ),
                "keyword_vectors": np.empty((0, 2), dtype=np.float32),
            },
        }

        _, details = ket_retrieve(
            index,
            np.array([1.0, 0.0], dtype=np.float32),
            top_k=2,
            theta=1.0,
        )

        self.assertEqual(
            [
                item["adjacency_to_entity_seeds"]
                for item in details["skeleton_relationships"]
            ],
            [2, 1],
        )
        self.assertEqual(details["subchunk_ids"], [0, 1])

    def test_hybrid_graph_and_retrievers(self):
        texts = ["red apple fruit", "green apple fruit", "train railway", "railway station"]
        vectors = np.eye(4, dtype=np.float32)
        graph = build_hybrid_knn(texts, vectors, k=2)
        self.assertEqual(len(graph), 4)
        self.assertTrue(all(len(neighbours) == 2 for neighbours in graph))

        query = vectors[0]
        _, text_ids = text_retrieve(
            [{"text": text} for text in texts], vectors, query, top_k=1
        )
        _, knn_ids = knn_retrieve(
            [{"text": text} for text in texts], vectors, graph, query, top_k=2
        )
        self.assertEqual(text_ids, [0])
        self.assertIn(0, knn_ids)


if __name__ == "__main__":
    unittest.main()
