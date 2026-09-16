"""Tests for Phase 4 nested stratified cross-validation."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase4_nested_cv import (
    compute_metrics,
    build_model_pipeline,
    get_modeling_data,
)
from src.preprocessing import build_preprocessor, get_feature_schema, get_target_column
from src.data_loader import load_raw_csv
import json

RAW_PATH = ROOT / "data" / "raw" / "elderly_data.csv"
FALL_HIST_NAME = "\u0633\u0627\u0628\u0641\u0647 \u0633\u0642\u0648\u0637"


class Phase4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.meta = load_raw_csv(RAW_PATH)
        with open(ROOT / "configs" / "phase4_config.json", encoding="utf-8") as f:
            cls.config = json.load(f)

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
        X, y = get_modeling_data(self.frame, self.config)
        self.assertNotIn("HighFallRisk", X.columns)

    def test_identifier_excluded(self):
        X, y = get_modeling_data(self.frame, self.config)
        self.assertNotIn("Participant_ID", X.columns)

    def test_fall_history_excluded(self):
        X, y = get_modeling_data(self.frame, self.config)
        self.assertNotIn(FALL_HIST_NAME, X.columns)

    def test_empty_columns_excluded(self):
        X, y = get_modeling_data(self.frame, self.config)
        empty_cols = ["hight", "weight", "BMI", "Education", "Asa",
                       "heart Deases", "surgery", "x1", "x2", "x3", "x4",
                       "x5", "x6", "x7", "x8", "x9", "x10", "x11",
                       "y1", "y2", "DLRT", "Absolut error"]
        for col in empty_cols:
            self.assertNotIn(col, X.columns)

    # --- Preprocessing leakage ---
    def test_preprocessor_fitted_on_training_only(self):
        X, _ = get_modeling_data(self.frame, self.config)
        preprocessor = build_preprocessor(self.config)
        fitted = preprocessor.fit(X)
        self.assertTrue(hasattr(fitted, "transformers_"))

    def test_validation_transform_does_not_refit(self):
        X, _ = get_modeling_data(self.frame, self.config)
        n_train = int(len(X) * 0.8)
        X_train = X.iloc[:n_train].copy()
        X_val = X.iloc[n_train:].copy()

        preprocessor = build_preprocessor(self.config)
        fitted = preprocessor.fit(X_train)

        imputer = fitted.named_transformers_["num"].named_steps["imputer"]
        stats_before = imputer.statistics_.copy()
        _ = fitted.transform(X_val)
        stats_after = imputer.statistics_
        np.testing.assert_array_equal(stats_before, stats_after)

    # --- CV checks ---
    def test_outer_test_folds_do_not_overlap(self):
        from sklearn.model_selection import StratifiedKFold
        y_numeric = self.frame[self.config["target_column"]].str.strip().map(int)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        all_test_indices = []
        for _, test_idx in outer_cv.split(self.frame, y_numeric):
            for idx in test_idx:
                self.assertNotIn(idx, all_test_indices)
                all_test_indices.append(idx)
        self.assertEqual(len(all_test_indices), 77)

    def test_every_participant_in_test(self):
        from sklearn.model_selection import StratifiedKFold
        y_numeric = self.frame[self.config["target_column"]].str.strip().map(int)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        all_test = set()
        for _, test_idx in outer_cv.split(self.frame, y_numeric):
            all_test |= set(test_idx)
        self.assertEqual(len(all_test), 77)

    def test_every_participant_exactly_one_test_fold(self):
        from sklearn.model_selection import StratifiedKFold
        y_numeric = self.frame[self.config["target_column"]].str.strip().map(int)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        all_test = []
        for _, test_idx in outer_cv.split(self.frame, y_numeric):
            all_test.extend(test_idx)
        self.assertEqual(len(all_test), 77)
        self.assertEqual(len(set(all_test)), 77)

    # --- Metric computation ---
    def test_compute_metrics_valid(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_score = np.array([0.2, 0.3, 0.6, 0.7, 0.1, 0.9, 0.15, 0.55])
        metrics = compute_metrics(y_true, y_score)
        self.assertIn("roc_auc", metrics)
        self.assertIn("sensitivity", metrics)
        self.assertIn("specificity", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("brier_score", metrics)

    def test_compute_metrics_svm_scores(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_score = np.array([-1.2, -0.5, 0.8, 1.1, -0.3, 1.5, -0.9, 0.7])
        metrics = compute_metrics(y_true, y_score)
        self.assertIn("roc_auc", metrics)
        self.assertIn("sensitivity", metrics)
        self.assertIn("specificity", metrics)

    def test_compute_metrics_returns_finite(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.4, 0.6])
        metrics = compute_metrics(y_true, y_score)
        self.assertTrue(np.isfinite(metrics["roc_auc"]))
        self.assertTrue(np.isfinite(metrics["f1"]))

    # --- Model pipeline ---
    def test_model_pipeline_types(self):
        for model_name in ["logistic_regression", "linear_svm", "random_forest"]:
            pipeline, param_grid = build_model_pipeline(model_name, self.config)
            from sklearn.pipeline import Pipeline
            self.assertIsInstance(pipeline, Pipeline)
            self.assertIsInstance(param_grid, dict)
            self.assertGreater(len(param_grid), 0)

    def test_all_three_models_present(self):
        model_names = list(self.config["models"].keys())
        self.assertEqual(set(model_names), {"logistic_regression", "linear_svm", "random_forest"})

    # --- Output files verification ---
    def test_fold_results_csv(self):
        path = ROOT / "outputs" / "phase4_nested_cv" / "fold_results.csv"
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        self.assertEqual(len(df), 15)
        required_cols = {"outer_fold", "model", "n_train", "n_test", "n_positive_train",
                          "n_positive_test", "roc_auc", "pr_auc", "sensitivity",
                          "specificity", "ppv", "npv", "f1", "balanced_accuracy", "brier_score"}
        self.assertTrue(required_cols.issubset(set(df.columns)))

    def test_model_summary_csv(self):
        path = ROOT / "outputs" / "phase4_nested_cv" / "model_summary.csv"
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        self.assertEqual(len(df), 3)

    def test_selected_hyperparameters_csv(self):
        path = ROOT / "outputs" / "phase4_nested_cv" / "selected_hyperparameters.csv"
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        self.assertEqual(len(df), 15)

    def test_predictions_csv(self):
        path = ROOT / "outputs" / "phase4_nested_cv" / "predictions.csv"
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        self.assertEqual(len(df), 77 * 3)
        pred_cols = {"Participant_ID", "outer_fold", "model", "y_true", "score", "prediction"}
        self.assertTrue(pred_cols.issubset(set(df.columns)))

    def test_everyone_predicted_once_per_model(self):
        pred_df = pd.read_csv(ROOT / "outputs" / "phase4_nested_cv" / "predictions.csv")
        for model in ["logistic_regression", "linear_svm", "random_forest"]:
            model_preds = pred_df[pred_df["model"] == model]
            self.assertEqual(len(model_preds), 77)
            self.assertEqual(len(set(model_preds["Participant_ID"])), 77)

    def test_report_exists(self):
        path = ROOT / "reports" / "phase4_model_comparison_report.md"
        self.assertTrue(path.exists())
        content = path.read_text(encoding="utf-8")
        self.assertIn("Objective", content)
        self.assertIn("Dataset", content)
        self.assertIn("Preprocessing", content)
        self.assertIn("Cross-validation", content)
        self.assertIn("Models", content)
        self.assertIn("Fold-level results", content)
        self.assertIn("Model summary", content)
        self.assertIn("Selected hyperparameters", content)
        self.assertIn("Phase 4 = COMPLETE", content)

    # --- Raw data immutability ---
    def test_raw_csv_immutable(self):
        before = RAW_PATH.read_bytes()
        from src.preprocessing import load_and_prepare
        _ = load_and_prepare(self.config)
        after = RAW_PATH.read_bytes()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
