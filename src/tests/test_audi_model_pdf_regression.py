"""Document-level regression tests for the public Audi model PDF fixtures."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.data_extraction import build_chunks, validate_blocks
from src.data_extraction.pdf_extractor import extract_pdf_blocks


ARTIFACTS = Path(__file__).parent / "artifacts"
AUDI_FILES = (
    "A3_Limousine.pdf",
    "A5_Avant.pdf",
    "A6_Limousine.pdf",
    "Q4_Sportback_e-tron.pdf",
    "Q8.pdf",
)
PAGE_COUNTS = {
    "A3_Limousine.pdf": 30,
    "A5_Avant.pdf": 23,
    "A6_Limousine.pdf": 21,
    "Q4_Sportback_e-tron.pdf": 23,
    "Q8.pdf": 40,
}
MODEL_HEADER = "A modell megnevezése"
PRICE_HEADER = (
    "Ajánlott kiskereskedelmi modell ár "
    "(bruttó, regisztrációs adóval, HUF)"
)


class AudiModelPdfRegressionTests(unittest.TestCase):
    """Check visually verified price-table facts from all five Audi fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.blocks_by_file: dict[str, list[dict]] = {}
        cls.sources_by_file: dict[str, dict] = {}
        for filename in AUDI_FILES:
            path = ARTIFACTS / filename
            if not path.is_file():
                raise AssertionError(f"Missing PDF regression artifact: {path}")
            blocks, source = extract_pdf_blocks(path)
            validate_blocks(blocks)
            cls.blocks_by_file[filename] = blocks
            cls.sources_by_file[filename] = source

    def _price_tables(self, filename: str) -> list[dict]:
        """Return model-price tables identified by their two stable headers."""
        return [
            block
            for block in self.blocks_by_file[filename]
            if block.get("type") == "table"
            and MODEL_HEADER in block.get("headers", [])
            and PRICE_HEADER in block.get("headers", [])
        ]

    def _price_row(self, filename: str, model_name: str) -> dict[str, str]:
        """Find one model row and map its extracted cells to column headers."""
        for table in self._price_tables(filename):
            for row in table["rows"]:
                values = dict(zip(table["headers"], row))
                if values.get(MODEL_HEADER) == model_name:
                    return values
        self.fail(f"Expected model row not found in {filename}: {model_name}")

    def test_every_fixture_has_valid_source_metadata(self):
        for filename, pages in PAGE_COUNTS.items():
            with self.subTest(filename=filename):
                source = self.sources_by_file[filename]
                self.assertEqual(source["kind"], "pdf")
                self.assertEqual(source["filename"], filename)
                self.assertEqual(source["pages"], pages)

    def test_price_tables_have_explicit_consistent_headers(self):
        for filename in AUDI_FILES:
            with self.subTest(filename=filename):
                tables = self._price_tables(filename)
                self.assertTrue(tables)
                for table in tables:
                    self.assertEqual(len(table["headers"]), 8)
                    self.assertFalse(
                        any(header.startswith("Column ") for header in table["headers"])
                    )
                    self.assertTrue(
                        all(len(row) == len(table["headers"]) for row in table["rows"])
                    )

    def test_a3_advanced_diesel_row(self):
        row = self._price_row(
            "A3_Limousine.pdf", "A3 Limousine Advanced 35 TDI S tronic"
        )
        self.assertEqual(row["Modellkód"], "8YMBRG24")
        self.assertEqual(row["Teljesítmény kW (LE)"], "110 (150)")
        self.assertEqual(row["Fogyasztás (l/100 km) *"], "4.8 l/100km")
        self.assertEqual(row[PRICE_HEADER], "15093780")

    def test_a5_e_hybrid_row(self):
        row = self._price_row(
            "A5_Avant.pdf", "A5 Avant e-hybrid quattro 270 kW"
        )
        self.assertEqual(row["Modellkód"], "FU5A2Y24")
        self.assertEqual(row["Teljesítmény kW (LE)"], "185 (252)")
        self.assertEqual(row["Fogyasztás (l/100 km) *"], "0.7 l/100km")
        self.assertEqual(row[PRICE_HEADER], "28944060")

    def test_a6_petrol_quattro_row(self):
        row = self._price_row(
            "A6_Limousine.pdf", "A6 Limousine 55 TFSI quattro S tronic"
        )
        self.assertEqual(row["Modellkód"], "FN2A5Y24")
        self.assertEqual(row["Teljesítmény kW (LE)"], "270 (367)")
        self.assertEqual(row["Fogyasztás (l/100 km) *"], "6.9 l/100km")
        self.assertEqual(row[PRICE_HEADER], "33365310")

    def test_q4_electric_lifestyle_row(self):
        row = self._price_row(
            "Q4_Sportback_e-tron.pdf", "Q4 SB e-tron 55 quattro Lifestyle"
        )
        self.assertEqual(row["Modellkód"], "F4NAU3LS")
        self.assertEqual(row["Teljesítmény kW (LE)"], "250 (340)")
        self.assertEqual(row["Fogyasztás*"], "165.6 Wh/km")
        self.assertEqual(row[PRICE_HEADER], "24160480")

    def test_q8_standard_and_second_page_performance_rows(self):
        standard = self._price_row("Q8.pdf", "Q8 55 TFSI quattro tiptronic")
        self.assertEqual(standard["Modellkód"], "4MT0X224")
        self.assertEqual(standard["Teljesítmény kW (LE)"], "250 (340)")
        self.assertEqual(standard["Fogyasztás (l/100 km) *"], "10.4 l/100km")
        self.assertEqual(standard[PRICE_HEADER], "34232340")

        performance = self._price_row(
            "Q8.pdf", "RS Q8 performance quattro tiptronic"
        )
        self.assertEqual(performance["Modellkód"], "4MTRR224")
        self.assertEqual(performance["Teljesítmény kW (LE)"], "471 (640)")
        self.assertEqual(performance[PRICE_HEADER], "63837690")

    def test_price_row_reaches_structure_aware_chunks(self):
        table = next(
            table
            for table in self._price_tables("Q8.pdf")
            if any(row[0] == "Q8 55 TFSI quattro tiptronic" for row in table["rows"])
        )
        chunks = build_chunks(
            [table], strategy="words", chunk_size=450, chunk_overlap=60
        )
        context = "\n".join(chunk["source_text"] for chunk in chunks)

        self.assertIn("A modell megnevezése = Q8 55 TFSI quattro tiptronic", context)
        self.assertIn("Teljesítmény kW (LE) = 250 (340)", context)
        self.assertIn("Fogyasztás (l/100 km) * = 10.4 l/100km", context)
        self.assertIn(f"{PRICE_HEADER} = 34232340", context)


if __name__ == "__main__":
    unittest.main()
