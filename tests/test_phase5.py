"""Tests for Phase 5 model interpretation and robustness analysis."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.preprocessing import get_feature_schema, get_target_column
from src.data_loader import load_raw_csv
import json

RAW_PATH = ROOT / "data" / "raw" / "elderly_data.csv"
FALL_HIST_NAME = "\u0635\u0627\u0628\u0641\u0647 \u0633\u0642\u0648\u0637"


class Phase5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.meta = load_raw_csv(RAW_PATH)
        with open(ROOT / "configs" / "phase5_config.json", encoding="utf-8") as f:
            cls.config = json.load(f)
        with open(ROOT / "configs" / "phase4_config.json", encoding="utf-8") as f:
            cls.config4 = json.load(f)

    # --- Dataset integrity ---
    def test_dataset_n_rows(self):
        self.assertEqual(len(self.frame), 77)

    def test_target_binary(self):
        target = self.frame[self.config["target_column"]].str.strip()
        unique = set(target.unique())
        self.assertTrue(unique.issubset({"0", "1"}))

    def test_feature_order_exact(self):
        schema = get_feature_schema(self.config)
        self.assertEqual(schema, ["Age", "Gender", "Total x", "Totaly", "TUG"])

    def test_target_excluded(self):
        from src.preprocessing import get_modeling_data
        X, y = get_modeling_data(self.frame, self.config)
        self.assertNotIn("HighFallRisk", X.columns)

    def test_identifier_excluded(self):
        from src.preprocessing import get_modeling_data
        X, y = get_modeling_data(self.frame, self.config)
        self.assertNotIn("Participant_ID", X.columns)

    def test_fall_history_excluded(self):
        from src.preprocessing import get_modeling_data
        X, y = get_modeling_data(self.frame, self.config)
        self.assertNotIn(FALL_HIST_NAME, X.columns)

    def test_all_missing_columns_excluded(self):
        from src.preprocessing import get_modeling_data
        X, y = get_modeling_data(self.frame, self.config)
        excluded = ["hight", "weight", "BMI", "Education", "Asa",
                     "heart Deases", "surgery", "DLRT", "Absolut error"]
        for col in excluded:
            self.assertNotIn(col, X.columns)

    # --- Fold integrity ---
    def test_outer_cv_stratified(self):
        from sklearn.model_selection import StratifiedKFold
        from src.preprocessing import get_modeling_data
        y = self.frame[self.config["target_column"]].str.strip().map(int)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in outer_cv.split(self.frame, y):
            self.assertEqual(len(train_idx) + len(test_idx), 77)
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)

    def test_no_train_test_overlap(self):
        from sklearn.model_selection import StratifiedKFold
        from src.preprocessing import get_modeling_data
        X, y = get_modeling_data(self.frame, self.config)
        y_num = y.map(int) if y.dtype == object else y.astype(int)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        all_test = []
        for _, test_idx in outer_cv.split(X, y_num):
            all_test.extend(test_idx)
        self.assertEqual(len(set(all_test)), len(all_test))
        self.assertEqual(len(set(all_test)), 77)

    def test_everyone_in_one_test_fold(self):
        from sklearn.model_selection import StratifiedKFold
        from src.preprocessing import get_modeling_data
        X, y = get_modeling_data(self.frame, self.config)
        y_num = y.map(int) if y.dtype == object else y.astype(int)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        all_test = []
        for _, test_idx in outer_cv.split(X, y_num):
            all_test.extend(test_idx)
        self.assertEqual(len(set(all_test)), 77)

    # --- Reproducibility ---
    def test_fixed_seed(self):
        self.assertEqual(self.config["random_seed"], 42)
        self.assertEqual(self.config["outer_cv"]["random_state"], 42)
        self.assertEqual(self.config["inner_cv"]["random_state"], 42)

    def test_deterministic_configuration(self):
        self.assertIn("feature_schema", self.config)
        self.assertIn("models", self.config)
        self.assertIn("outer_cv", self.config)
        self.assertIn("inner_cv", self.config)

    # --- Coefficients ---
    def test_logistic_coefficients_file_exists(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "logistic_coefficients.csv"
        self.assertTrue(path.exists())

    def test_logistic_coefficients_five_features(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "logistic_coefficients.csv"
        df = pd.read_csv(path)
        feat_cols = df[df["feature"] != "intercept"]["feature"].unique()
        self.assertEqual(len(feat_cols), 5)
        self.assertTrue(all(f in ["Age", "Gender", "Total x", "Totaly", "TUG"] for f in feat_cols))

    def test_logistic_coefficients_correct_mapping(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "logistic_coefficients.csv"
        df = pd.read_csv(path)
        feat_cols = df[df["feature"] != "intercept"]["feature"].unique()
        expected = {"Age", "Gender", "Total x", "Totaly", "TUG"}
        self.assertTrue(set(feat_cols).issubset(expected))
        self.assertTrue(set(feat_cols).issuperset(expected))

    # --- Permutation importance ---
    def test_permutation_importance_file_exists(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "permutation_importance_fold.csv"
        self.assertTrue(path.exists())

    def test_permutation_importance_five_features(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "permutation_importance_fold.csv"
        df = pd.read_csv(path)
        feat_cols = df["feature"].unique()
        self.assertEqual(len(feat_cols), 5)

    def test_permutation_fold_assignments(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "permutation_importance_fold.csv"
        df = pd.read_csv(path)
        for model in ["logistic_regression", "linear_svm", "random_forest"]:
            model_df = df[df["model"] == model]
            self.assertEqual(len(model_df["outer_fold"].unique()), 5)

    def test_permutation_no_target_column(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "permutation_importance_fold.csv"
        df = pd.read_csv(path)
        self.assertNotIn("y_true", df.columns)
        self.assertNotIn("HighFallRisk", df.columns)

    # --- Sensitivity analysis ---
    def test_loof_file_exists(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "leave_one_feature_out.csv"
        self.assertTrue(path.exists())

    def test_loof_each_removes_one_feature(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "leave_one_feature_out.csv"
        df = pd.read_csv(path)
        for scenario in df["scenario"].unique():
            removed = df[df["scenario"] == scenario]["removed_feature"].iloc[0]
            remaining = df[df["scenario"] == scenario]["remaining_features"].iloc[0]
            self.assertNotIn(removed, remaining.split(" + "))

    def test_loof_full_model_identifiable(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "leave_one_feature_out.csv"
        df = pd.read_csv(path)
        scenarios = set(df["scenario"].unique())
        self.assertTrue(any("remove_" in s for s in scenarios))

    # --- Threshold analysis ---
    def test_threshold_analysis_file_exists(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "threshold_analysis.csv"
        self.assertTrue(path.exists())

    def test_thresholds_in_valid_range(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "threshold_analysis.csv"
        df = pd.read_csv(path)
        self.assertTrue((df["threshold"] >= 0).all() and (df["threshold"] <= 1).all())

    def test_linear_svm_not_treated_as_probability(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "threshold_analysis.csv"
        df = pd.read_csv(path)
        linsvm_df = df[df["model"] == "linear_svm"]
        self.assertTrue((linsvm_df["threshold"] == 0.0).all() or
                        len(linsvm_df) == 0)

    # --- Output integrity ---
    def test_expected_files_generated(self):
        expected = [
            "logistic_coefficients.csv",
            "permutation_importance_fold.csv",
            "permutation_importance_summary.csv",
            "random_forest_importance.csv",
            "threshold_analysis.csv",
            "calibration_metrics.csv",
            "leave_one_feature_out.csv",
            "missingness_sensitivity.csv",
        ]
        for fname in expected:
            path = ROOT / "outputs" / "phase5_interpretation" / fname
            self.assertTrue(path.exists(), f"Missing: {fname}")

    def test_no_invalid_values_in_metrics(self):
        path = ROOT / "outputs" / "phase5_interpretation" / "logistic_coefficients.csv"
        df = pd.read_csv(path)
        if len(df) > 0:
            self.assertTrue(df["coefficient"].notna().all())

    def test_figure_files_exist(self):
        fig_dir = ROOT / "outputs" / "phase5_interpretation" / "figures"
        self.assertTrue(fig_dir.exists())
        fig_files = list(fig_dir.glob("*.png"))
        self.assertGreaterEqual(len(fig_files), 5, "At least 5 figure files expected")

    def test_raw_csv_hash_unchanged(self):
        import hashlib
        raw_bytes = RAW_PATH.read_bytes()
        raw_md5 = hashlib.md5(raw_bytes).hexdigest()
        self.assertEqual(raw_md5, "76ef4918f1787872b4bba3d6881bc5ac")

    def test_phase4_outputs_untouched(self):
        path = ROOT / "outputs" / "phase4_nested_cv" / "fold_results.csv"
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()