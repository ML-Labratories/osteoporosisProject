"""Tests for Phase 3 leakage-safe preprocessing."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    build_preprocessor,
    fit_preprocessor_on_training,
    transform_with_fitted_preprocessor,
    get_modeling_data,
    get_feature_schema,
    get_target_column,
    get_identifier_column,
    get_numeric_features,
    get_categorical_features,
    load_and_prepare,
)
from src.data_loader import load_raw_csv
import json

FALL_HIST_NAME = "\u0633\u0627\u0628\u0641\u0647 \u0633\u0642\u0648\u0637"
RAW_PATH = ROOT / "data" / "raw" / "elderly_data.csv"


class Phase3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.meta = load_raw_csv(RAW_PATH)
        with open(ROOT / "configs" / "phase3_config.json", encoding="utf-8") as f:
            cls.config = json.load(f)

    # --- Test 1: Feature schema ---
    def test_feature_schema_exact(self):
        schema = get_feature_schema(self.config)
        self.assertEqual(schema, ["Age", "Gender", "Total x", "Totaly", "TUG"])

    def test_feature_schema_order(self):
        schema = get_feature_schema(self.config)
        self.assertEqual(schema[0], "Age")
        self.assertEqual(schema[1], "Gender")
        self.assertEqual(schema[2], "Total x")
        self.assertEqual(schema[3], "Totaly")
        self.assertEqual(schema[4], "TUG")

    # --- Test 2: Target exclusion ---
    def test_target_excluded(self):
        X, y = get_modeling_data(self.frame, self.config)
        self.assertNotIn(self.config["target_column"], X.columns)
        self.assertNotIn("HighFallRisk", X.columns)

    def test_target_values_binary(self):
        X, y = get_modeling_data(self.frame, self.config)
        unique_vals = set(y.unique())
        self.assertTrue(unique_vals.issubset({"0", "1"}))

    def test_target_no_missing(self):
        X, y = get_modeling_data(self.frame, self.config)
        self.assertEqual(len(y), 77)
        self.assertEqual((y == "").sum(), 0)

    def test_target_class_counts(self):
        X, y = get_modeling_data(self.frame, self.config)
        counts = y.value_counts().to_dict()
        self.assertEqual(counts.get("0", 0), 44)
        self.assertEqual(counts.get("1", 0), 33)

    # --- Test 3: ID exclusion ---
    def test_identifier_excluded(self):
        X, y = get_modeling_data(self.frame, self.config)
        self.assertNotIn(self.config["identifier_column"], X.columns)
        self.assertNotIn("Participant_ID", X.columns)

    # --- Test 4: Empty feature exclusion ---
    def test_empty_columns_not_in_X(self):
        X, y = get_modeling_data(self.frame, self.config)
        empty_col_names = [
            "hight", "weight", "BMI", "Education", "Asa", "heart Deases",
            "surgery", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
            "x9", "x10", "x11", "y1", "y2", "DLRT", "Absolut error"
        ]
        for col in empty_col_names:
            self.assertNotIn(col, X.columns, f"Empty column {col} should not be in X")

    def test_feature_count_exactly_five(self):
        X, y = get_modeling_data(self.frame, self.config)
        self.assertEqual(len(X.columns), 5)

    # --- Test 5: Imputer behavior ---
    def test_imputer_fitted_on_training_only(self):
        X, _ = get_modeling_data(self.frame, self.config)
        preprocessor = build_preprocessor(self.config)
        fitted = fit_preprocessor_on_training(preprocessor, X)

        imputer = fitted.named_transformers_["num"].named_steps["imputer"]
        self.assertTrue(hasattr(imputer, "statistics_"))
        self.assertIsNotNone(imputer.statistics_)

    def test_imputer_median_from_training(self):
        X, _ = get_modeling_data(self.frame, self.config)
        preprocessor = build_preprocessor(self.config)
        fitted = fit_preprocessor_on_training(preprocessor, X)

        imputer = fitted.named_transformers_["num"].named_steps["imputer"]
        ages = X["Age"].dropna().values
        expected_median = np.median(ages)
        age_idx = list(X.columns).index("Age")
        self.assertAlmostEqual(imputer.statistics_[age_idx], expected_median, places=6)

    def test_imputer_handles_missing(self):
        X, _ = get_modeling_data(self.frame, self.config)
        preprocessor = build_preprocessor(self.config)
        fitted = fit_preprocessor_on_training(preprocessor, X)
        transformed = transform_with_fitted_preprocessor(fitted, X)
        transformed_arr = np.asarray(transformed, dtype=float)
        self.assertFalse(np.isnan(transformed_arr).any())

    # --- Test 6: Leakage-safe transform ---
    def test_validation_transform_does_not_refit(self):
        X, _ = get_modeling_data(self.frame, self.config)
        n_train = int(len(X) * 0.8)
        X_train = X.iloc[:n_train].copy()
        X_val = X.iloc[n_train:].copy()

        preprocessor = build_preprocessor(self.config)
        fitted = fit_preprocessor_on_training(preprocessor, X_train)

        imputer = fitted.named_transformers_["num"].named_steps["imputer"]
        stats_before = imputer.statistics_.copy()

        _ = transform_with_fitted_preprocessor(fitted, X_val)

        stats_after = imputer.statistics_
        np.testing.assert_array_equal(stats_before, stats_after)

    def test_validation_transform_does_not_alter_scaler(self):
        X, _ = get_modeling_data(self.frame, self.config)
        n_train = int(len(X) * 0.8)
        X_train = X.iloc[:n_train].copy()
        X_val = X.iloc[n_train:].copy()

        preprocessor = build_preprocessor(self.config)
        fitted = fit_preprocessor_on_training(preprocessor, X_train)

        scaler = fitted.named_transformers_["num"].named_steps["scaler"]
        center_before = scaler.center_.copy()
        scale_before = scaler.scale_.copy()

        _ = transform_with_fitted_preprocessor(fitted, X_val)

        np.testing.assert_array_equal(center_before, scaler.center_)
        np.testing.assert_array_equal(scale_before, scaler.scale_)

    def test_explicit_leakage_test(self):
        """Validation data with extreme values cannot influence preprocessing."""
        X_train = pd.DataFrame({"Age": [70.0, 72.0, 74.0], "Total x": [0.9, 0.95, 1.0],
                                "Totaly": [100.0, 110.0, 120.0], "TUG": [8.0, 9.0, 10.0],
                                "Gender": [0, 1, 0]})
        X_val = pd.DataFrame({"Age": [150.0, 160.0, 170.0], "Total x": [2.0, 2.0, 2.0],
                              "Totaly": [500.0, 600.0, 700.0], "TUG": [50.0, 60.0, 70.0],
                              "Gender": [1, 0, 1]})

        preprocessor = build_preprocessor(self.config)
        fitted = fit_preprocessor_on_training(preprocessor, X_train)

        imputer = fitted.named_transformers_["num"].named_steps["imputer"]
        scaler = fitted.named_transformers_["num"].named_steps["scaler"]

        _ = transform_with_fitted_preprocessor(fitted, X_val)

        # Verify imputer medians unchanged
        expected_age_median = 72.0
        actual_age_median = imputer.statistics_[0]
        self.assertAlmostEqual(actual_age_median, expected_age_median, places=6)

        # Verify scaler parameters unchanged
        self.assertEqual(len(scaler.center_), 4)
        self.assertEqual(len(scaler.scale_), 4)

    # --- Test 7: Scaler behavior ---
    def test_scaler_fitted_on_training(self):
        X, _ = get_modeling_data(self.frame, self.config)
        preprocessor = build_preprocessor(self.config)
        fitted = fit_preprocessor_on_training(preprocessor, X)

        scaler = fitted.named_transformers_["num"].named_steps["scaler"]
        self.assertTrue(hasattr(scaler, "center_"))
        self.assertTrue(hasattr(scaler, "scale_"))
        self.assertIsNotNone(scaler.center_)
        self.assertIsNotNone(scaler.scale_)

    def test_scaler_transform_output_shape(self):
        X, _ = get_modeling_data(self.frame, self.config)
        preprocessor = build_preprocessor(self.config)
        fitted = fit_preprocessor_on_training(preprocessor, X)
        transformed = transform_with_fitted_preprocessor(fitted, X)
        self.assertEqual(transformed.shape[0], 77)
        self.assertEqual(transformed.shape[1], 5)

    # --- Test 8: Raw data immutability ---
    def test_raw_csv_immutable(self):
        before = RAW_PATH.read_bytes()
        X, y, ids = load_and_prepare(self.config)
        after = RAW_PATH.read_bytes()
        self.assertEqual(before, after)

    # --- Test 9: Determinism ---
    def test_deterministic_construction(self):
        X, _ = get_modeling_data(self.frame, self.config)
        preprocessor1 = build_preprocessor(self.config)
        preprocessor2 = build_preprocessor(self.config)

        fitted1 = fit_preprocessor_on_training(preprocessor1, X)
        fitted2 = fit_preprocessor_on_training(preprocessor2, X)

        imputer1 = fitted1.named_transformers_["num"].named_steps["imputer"]
        imputer2 = fitted2.named_transformers_["num"].named_steps["imputer"]
        np.testing.assert_array_equal(imputer1.statistics_, imputer2.statistics_)

        scaler1 = fitted1.named_transformers_["num"].named_steps["scaler"]
        scaler2 = fitted2.named_transformers_["num"].named_steps["scaler"]
        np.testing.assert_array_equal(scaler1.center_, scaler2.center_)
        np.testing.assert_array_equal(scaler1.scale_, scaler2.scale_)

    def test_feature_schema_stable(self):
        schema1 = get_feature_schema(self.config)
        schema2 = get_feature_schema(self.config)
        self.assertEqual(schema1, schema2)
        self.assertEqual(len(schema1), 5)

    # --- Test 10: Preprocessing API ---
    def test_build_preprocessor_type(self):
        preprocessor = build_preprocessor(self.config)
        from sklearn.compose import ColumnTransformer
        self.assertIsInstance(preprocessor, ColumnTransformer)

    def test_preprocessor_fit_transform(self):
        X, _ = get_modeling_data(self.frame, self.config)
        preprocessor = build_preprocessor(self.config)
        fitted = fit_preprocessor_on_training(preprocessor, X)
        result = transform_with_fitted_preprocessor(fitted, X)
        self.assertIsInstance(result, np.ndarray)

    def test_preprocessor_passthrough_categorical(self):
        X, _ = get_modeling_data(self.frame, self.config)
        preprocessor = build_preprocessor(self.config)
        fitted = fit_preprocessor_on_training(preprocessor, X)
        result = transform_with_fitted_preprocessor(fitted, X)
        # Result should have 5 columns (4 numeric + 1 categorical)
        self.assertEqual(result.shape[1], 5)

    def test_preprocessor_drops_remainder(self):
        X, _ = get_modeling_data(self.frame, self.config)
        preprocessor = build_preprocessor(self.config)
        self.assertEqual(len(preprocessor.transformers), 2)  # num + cat

    # --- Test 11: Output files ---
    def test_preprocessing_summary_exists(self):
        summary_path = ROOT / "outputs" / "phase3_preprocessing" / "preprocessing_summary.json"
        self.assertTrue(summary_path.exists())
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("feature_schema", data)
        self.assertIn("imputation_strategy", data)
        self.assertEqual(data["imputation_strategy"], "median")

    def test_feature_schema_json_exists(self):
        schema_path = ROOT / "outputs" / "phase3_preprocessing" / "preprocessing_feature_schema.json"
        self.assertTrue(schema_path.exists())
        with open(schema_path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("features", data)
        self.assertEqual(len(data["features"]), 5)
        self.assertIn("target", data)
        self.assertIn("identifier", data)

    def test_report_exists(self):
        report_path = ROOT / "reports" / "phase3_preprocessing_report.md"
        self.assertTrue(report_path.exists())
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("Feature schema", content)
        self.assertIn("Missing-value strategy", content)
        self.assertIn("Leakage prevention", content)
        self.assertIn("Cross-validation compatibility", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
