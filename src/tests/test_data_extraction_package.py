"""Contract and real-document tests for the internal extraction package."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data_extraction import (
    BLOCKS_SCHEMA_VERSION,
    CHUNKS_SCHEMA_VERSION,
    BlockValidationError,
    build_chunks,
    extract_corpus,
    load_structured_blocks,
    render_block,
    validate_blocks,
)


ARTIFACT = Path(__file__).parent / "artifacts" / "audi_a8_2026_price_list.pdf"


class BlockContractTests(unittest.TestCase):
    """Fast tests for the language-independent ``blocks.json`` contract."""

    def test_schema_has_an_independent_version(self):
        self.assertEqual(BLOCKS_SCHEMA_VERSION, "1.0")
        self.assertEqual(CHUNKS_SCHEMA_VERSION, "1.0")

    def test_validation_rejects_a_ragged_table(self):
        with self.assertRaisesRegex(BlockValidationError, "must have 2 cells"):
            validate_blocks(
                [
                    {
                        "type": "table",
                        "source_name": "test.pdf",
                        "page": 1,
                        "heading_path": [],
                        "table_id": "table-1",
                        "headers": ["Name", "Price"],
                        "rows": [["A8"]],
                    }
                ]
            )

    def test_extract_corpus_requires_a_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Provide at least one"):
                extract_corpus(None, temporary)


class AudiPdfRegressionTests(unittest.TestCase):
    """Extract the real Audi fixture once and check its difficult tables."""

    @classmethod
    def setUpClass(cls):
        if not ARTIFACT.is_file():
            raise AssertionError(f"Missing PDF regression artifact: {ARTIFACT}")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.saved = extract_corpus(
            ARTIFACT,
            Path(cls.temporary.name) / "corpora",
            source_type="auto",
        )
        cls.blocks = load_structured_blocks(cls.saved.blocks_path)
        cls.source = json.loads(cls.saved.sources_path.read_text(encoding="utf-8"))[0]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def _tables_on(blocks: list[dict], page: int) -> list[dict]:
        return [
            block
            for block in blocks
            if block.get("type") == "table" and block.get("page") == page
        ]

    def test_source_provenance_describes_the_fixture(self):
        self.assertEqual(self.source["kind"], "pdf")
        self.assertEqual(self.source["filename"], ARTIFACT.name)
        self.assertEqual(self.source["bytes"], ARTIFACT.stat().st_size)
        self.assertGreaterEqual(self.source["pages"], 19)
        self.assertTrue(
            (self.saved.corpus_path.parent / self.source["stored_file"]).is_file()
        )

    def test_model_price_power_and_consumption_stay_in_one_row(self):
        page_two_tables = self._tables_on(self.blocks, 2)
        row = next(
            row
            for table in page_two_tables
            for row in table["rows"]
            if row[0] == "A8 55 TFSI quattro tiptronic"
        )

        self.assertEqual(row[1], "4NC0DA24")
        self.assertEqual(row[3], "250 (340)")
        self.assertEqual(row[4], "9.3 l/100km")
        self.assertEqual(row[7], "41256010")

        containing_table = next(table for table in page_two_tables if row in table["rows"])
        rendered = render_block(containing_table, [row])
        self.assertIn("A modell megnevezése = A8 55 TFSI quattro tiptronic", rendered)
        self.assertIn("Teljesítmény kW (LE) = 250 (340)", rendered)
        self.assertIn("Fogyasztás* = 9.3 l/100km", rendered)
        self.assertIn("41256010", rendered)

    def test_side_by_side_tables_remain_separate(self):
        tables = self._tables_on(self.blocks, 19)

        self.assertEqual(len(tables), 2)
        self.assertTrue(all(len(table["headers"]) == 8 for table in tables))
        self.assertLess(tables[0]["bbox"][2], tables[1]["bbox"][0])
        self.assertTrue(all("Normál tengelytáv" in table["headers"] for table in tables))

    def test_merged_pl4_price_lines_become_two_logical_rows(self):
        tables = self._tables_on(self.blocks, 19)
        pl4_rows = [
            row for table in tables for row in table["rows"] if row[1] == "PL4"
        ]

        self.assertEqual(len(pl4_rows), 2)
        self.assertEqual([row[-1] for row in pl4_rows], ["721360", "1000760"])
        self.assertEqual(pl4_rows[0][2:7], ["–", "–", "–", "–", "o"])
        self.assertEqual(pl4_rows[1][2:7], ["o", "o", "o", "o", "–"])
        self.assertEqual(pl4_rows[0][0], pl4_rows[1][0])

    def test_structure_aware_chunk_contains_complete_model_row(self):
        page_two_tables = self._tables_on(self.blocks, 2)
        chunks = build_chunks(
            page_two_tables,
            strategy="words",
            chunk_size=450,
            chunk_overlap=60,
        )
        chunk = next(
            item
            for item in chunks
            if "A8 55 TFSI quattro tiptronic" in item["source_text"]
        )

        self.assertEqual(chunk["block_type"], "table")
        self.assertEqual(chunk["page"], 2)
        self.assertIn("Teljesítmény kW (LE) = 250 (340)", chunk["source_text"])
        self.assertIn("Fogyasztás* = 9.3 l/100km", chunk["source_text"])
        self.assertIn("41256010", chunk["source_text"])

    def test_build_chunks_can_load_blocks_and_save_chunks_json(self):
        output_path = self.saved.blocks_path.with_name("chunks.json")
        chunks = build_chunks(
            self.saved.blocks_path,
            output_path,
            strategy="words",
            chunk_size=450,
            chunk_overlap=60,
        )

        self.assertTrue(output_path.is_file())
        self.assertEqual(
            json.loads(output_path.read_text(encoding="utf-8")),
            chunks,
        )
        with self.assertRaisesRegex(ValueError, "Unsupported chunking strategy"):
            build_chunks(self.saved.blocks_path, strategy="tokens")


if __name__ == "__main__":
    unittest.main()
