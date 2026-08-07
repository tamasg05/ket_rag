"""Document-level regression tests for the public Opel PDF fixtures."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.data_extraction import build_chunks, validate_blocks
from src.data_extraction.pdf_extractor import extract_pdf_blocks


ARTIFACTS = Path(__file__).parent / "artifacts"
OPEL_FILES = (
    "opel_HU_Astra_Electric.pdf",
    "opel_HU_Combo__Electric_egyteru.pdf",
    "opel_HU_Frontera_Electric.pdf",
    "opel_HU_Mokka_Electric_MY25 1.pdf",
    "opel_HU_Zafira_Electric.pdf",
)
PRICE_HEADERS = [
    "Felszereltség",
    "Motor",
    "Akkumulátor",
    "WLTP Hatótáv",
    "Listaár",
    "Kedvezmény",
    "Kedvezményes ár",
]


class OpelPdfRegressionTests(unittest.TestCase):
    """Check visually verified facts and structures from all Opel fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.blocks_by_file: dict[str, list[dict]] = {}
        cls.sources_by_file: dict[str, dict] = {}
        for filename in OPEL_FILES:
            path = ARTIFACTS / filename
            if not path.is_file():
                raise AssertionError(f"Missing PDF regression artifact: {path}")
            blocks, source = extract_pdf_blocks(path)
            validate_blocks(blocks)
            cls.blocks_by_file[filename] = blocks
            cls.sources_by_file[filename] = source

    def _price_table(self, filename: str) -> dict:
        expected = (
            ["Felszereltség", "Verzió", *PRICE_HEADERS[1:]]
            if filename == "opel_HU_Zafira_Electric.pdf"
            else PRICE_HEADERS
        )
        return next(
            block
            for block in self.blocks_by_file[filename]
            if block.get("type") == "table"
            and block.get("page") == 3
            and block.get("headers") == expected
        )

    def _price_row(self, filename: str, identifiers: dict[str, str]) -> dict[str, str]:
        table = self._price_table(filename)
        for row in table["rows"]:
            values = dict(zip(table["headers"], row))
            if all(values.get(key) == value for key, value in identifiers.items()):
                return values
        self.fail(f"Expected price row not found in {filename}: {identifiers}")

    def test_every_fixture_has_valid_source_metadata(self):
        expected_pages = {
            "opel_HU_Astra_Electric.pdf": 11,
            "opel_HU_Combo__Electric_egyteru.pdf": 11,
            "opel_HU_Frontera_Electric.pdf": 11,
            "opel_HU_Mokka_Electric_MY25 1.pdf": 11,
            "opel_HU_Zafira_Electric.pdf": 12,
        }
        for filename, pages in expected_pages.items():
            with self.subTest(filename=filename):
                source = self.sources_by_file[filename]
                self.assertEqual(source["kind"], "pdf")
                self.assertEqual(source["filename"], filename)
                self.assertEqual(source["pages"], pages)

    def test_price_tables_have_explicit_non_generic_headers(self):
        for filename in OPEL_FILES:
            with self.subTest(filename=filename):
                table = self._price_table(filename)
                expected = (
                    ["Felszereltség", "Verzió", *PRICE_HEADERS[1:]]
                    if filename == "opel_HU_Zafira_Electric.pdf"
                    else PRICE_HEADERS
                )
                self.assertEqual(table["headers"], expected)
                self.assertTrue(
                    all(len(row) == len(expected) for row in table["rows"])
                )

    def test_astra_edition_price_row(self):
        row = self._price_row(
            "opel_HU_Astra_Electric.pdf",
            {"Felszereltség": "Edition"},
        )
        self.assertEqual(row["Motor"], "Elektromos motor (115 kW / 156 LE)")
        self.assertEqual(row["Akkumulátor"], "54 kWh")
        self.assertEqual(row["WLTP Hatótáv"], "~418 km")
        self.assertEqual(row["Listaár"], "17990000")
        self.assertEqual(row["Kedvezményes ár"], "15690000")

        page_text = " ".join(
            block.get("text", "")
            for block in self.blocks_by_file["opel_HU_Astra_Electric.pdf"]
            if block.get("page") == 3
        )
        self.assertIn("Listaárak és kedvezményes árak", page_text)
        self.assertNotIn("LLiissttaa", page_text)

    def test_combo_gs_xl_price_row(self):
        row = self._price_row(
            "opel_HU_Combo__Electric_egyteru.pdf",
            {"Felszereltség": "GS XL"},
        )
        self.assertEqual(row["Akkumulátor"], "50 kWh")
        self.assertEqual(row["WLTP Hatótáv"], "~ 334 km")
        self.assertEqual(row["Listaár"], "18350000")
        self.assertEqual(row["Kedvezményes ár"], "16340000")

    def test_frontera_gs_54_kwh_price_row(self):
        row = self._price_row(
            "opel_HU_Frontera_Electric.pdf",
            {"Felszereltség": "GS", "Akkumulátor": "54 kWh"},
        )
        self.assertEqual(row["Motor"], "Elektromos motor (83 kW / 113 LE)")
        self.assertEqual(row["WLTP Hatótáv"], "~ 408 km")
        self.assertEqual(row["Listaár"], "12590000")
        self.assertEqual(row["Kedvezményes ár"], "12290000")

    def test_mokka_gs_price_row(self):
        row = self._price_row(
            "opel_HU_Mokka_Electric_MY25 1.pdf",
            {"Felszereltség": "GS"},
        )
        self.assertEqual(row["Motor"], "Elektromos motor (115 kW / 156 LE)")
        self.assertEqual(row["WLTP Hatótáv"], "~403 km")
        self.assertEqual(row["Listaár"], "15790000")
        self.assertEqual(row["Kedvezményes ár"], "14490000")

    def test_zafira_business_edition_l_price_row(self):
        row = self._price_row(
            "opel_HU_Zafira_Electric.pdf",
            {
                "Felszereltség": "Edition",
                "Verzió": "Zafira L",
                "Akkumulátor": "50 kWh",
            },
        )
        self.assertEqual(row["WLTP Hatótáv"], "~220 km")
        self.assertEqual(row["Listaár"], "23010000")
        self.assertEqual(row["Kedvezményes ár"], "19560000")

    def test_zafira_side_by_side_dimension_tables(self):
        tables = [
            block
            for block in self.blocks_by_file["opel_HU_Zafira_Electric.pdf"]
            if block.get("type") == "table" and block.get("page") == 10
        ]
        external = next(
            table for table in tables if table["headers"][0].startswith("Küls")
        )
        internal = next(
            table for table in tables if table["headers"][0].startswith("Bels")
        )

        external_rows = {row[0]: row[1:] for row in external["rows"]}
        internal_rows = {row[0]: row[1:] for row in internal["rows"]}
        self.assertEqual(external["headers"][1:], ["Zafira M", "Zafira L"])
        self.assertEqual(external_rows["Hosszúság"], ["4983", "5330"])
        self.assertEqual(internal["headers"][1:], ["Zafira M", "Zafira L"])
        self.assertEqual(
            internal_rows["Raktér hosszúság az első üléssorig"],
            ["2413", "2763"],
        )

    def test_opel_price_row_reaches_structure_aware_chunks(self):
        table = self._price_table("opel_HU_Astra_Electric.pdf")
        chunks = build_chunks(
            [table], strategy="words", chunk_size=450, chunk_overlap=60
        )
        context = "\n".join(chunk["source_text"] for chunk in chunks)

        self.assertIn("Felszereltség = Edition", context)
        self.assertIn("Akkumulátor = 54 kWh", context)
        self.assertIn("Kedvezményes ár = 15690000", context)


if __name__ == "__main__":
    unittest.main()
