import unittest

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
)
from src.retrieval_strategies import (
    knn_retrieve,
    serialise_subchunks,
    text_retrieve,
)


class CoreTests(unittest.TestCase):
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
