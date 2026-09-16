"""Tests for the Phase 1 raw CSV loader."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import DataLoaderError, load_raw_csv

RAW_PATH = ROOT / "data" / "raw" / "elderly_data.csv"


class DataLoaderTests(unittest.TestCase):
    def test_csv_loads_successfully(self):
        frame, meta = load_raw_csv(RAW_PATH)
        self.assertEqual(frame.shape, (77, 30))
        self.assertTrue(all(str(dtype) == "object" for dtype in frame.dtypes))
        self.assertEqual(meta["n_records"], 77)
        self.assertEqual(meta["n_columns"], 30)

    def test_identifier_column_exists(self):
        frame, _ = load_raw_csv(RAW_PATH)
        self.assertIn("Participant_ID", frame.columns)

    def test_target_column_exists(self):
        frame, _ = load_raw_csv(RAW_PATH)
        self.assertIn("HighFallRisk", frame.columns)

    def test_original_values_preserved(self):
        frame, _ = load_raw_csv(RAW_PATH)
        self.assertEqual(frame.loc[0, "Participant_ID"], "CO-001")
        self.assertEqual(frame.loc[0, "Age"], "75.16769336071184")
        self.assertEqual(frame.loc[0, "HighFallRisk"], "0")
        self.assertEqual(frame.loc[0, "hight"], "")
        self.assertEqual(frame.loc[0, "Total x"], "0.9666666666666667")
        self.assertEqual(frame["سابفه سقوط"].iloc[0], "1.0")

    def test_raw_csv_not_modified(self):
        before = RAW_PATH.read_bytes()
        load_raw_csv(RAW_PATH)
        after = RAW_PATH.read_bytes()
        self.assertEqual(before, after)

    def test_missing_file_raises(self):
        missing = ROOT / "data" / "raw" / "does_not_exist.csv"
        with self.assertRaises(DataLoaderError):
            load_raw_csv(missing)

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.csv"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(DataLoaderError):
                load_raw_csv(empty)

    def test_undecodable_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.csv"
            bad.write_bytes(b"a,b\n\x80\x81\x82\n")
            with self.assertRaises(DataLoaderError):
                load_raw_csv(bad)

    def test_inconsistent_delimiter_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ragged = Path(tmp) / "ragged.csv"
            ragged.write_text("a,b,c\n1,2\n3,4,5,6\n", encoding="utf-8")
            with self.assertRaises(DataLoaderError):
                load_raw_csv(ragged)

    def test_meta_reports_hashes(self):
        _, meta = load_raw_csv(RAW_PATH)
        self.assertTrue(len(meta["sha256"]) == 64)
        self.assertTrue(len(meta["md5"]) == 32)


if __name__ == "__main__":
    unittest.main()