"""Tests for Phase 2 statistical & data audit."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_raw_csv, DataLoaderError
from src.phase2_statistical_audit import (
    classify_variables,
    missing_data_analysis,
    descriptive_statistics,
    normality_assessment,
    outlier_analysis,
    group_comparison,
    categorical_comparison,
    correlation_analysis,
    univariate_logistic,
    roc_analysis,
    multicollinearity_diagnostic,
    _benjamini_hochberg,
    _md_table,
    missing_mask,
    _as_numeric,
)

RAW_PATH = ROOT / "data" / "raw" / "elderly_data.csv"

FALL_HIST_NAME = "\u0633\u0627\u0628\u0641\u0647 \u0633\u0642\u0648\u0637"
NUMERIC_COLS = ["Age", "Total x", "Totaly", "TUG"]
BINARY_COLS = ["Gender"]
TARGET_COL = "HighFallRisk"


class Phase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.meta = load_raw_csv(RAW_PATH)
        cls.config = {"dataset": {"raw_path": str(RAW_PATH)}}
        import json
        with open(ROOT / "configs" / "phase2_config.json", encoding="utf-8") as f:
            cls.config = json.load(f)

    # --- raw CSV immutability ---
    def test_raw_csv_immutable(self):
        before = RAW_PATH.read_bytes()
        load_raw_csv(RAW_PATH)
        after = RAW_PATH.read_bytes()
        self.assertEqual(before, after)

    # --- missingness calculations ---
    def test_missingness_age(self):
        mask = missing_mask(self.frame)
        n_missing = int(mask["Age"].sum())
        self.assertEqual(n_missing, 1)

    def test_missingness_gender(self):
        mask = missing_mask(self.frame)
        n_missing = int(mask["Gender"].sum())
        self.assertEqual(n_missing, 0)

    def test_missingness_total_x(self):
        mask = missing_mask(self.frame)
        n_missing = int(mask["Total x"].sum())
        self.assertEqual(n_missing, 5)

    def test_missingness_totaly(self):
        mask = missing_mask(self.frame)
        n_missing = int(mask["Totaly"].sum())
        self.assertEqual(n_missing, 3)

    def test_missingness_tug(self):
        mask = missing_mask(self.frame)
        n_missing = int(mask["TUG"].sum())
        self.assertEqual(n_missing, 0)

    def test_missingness_highfallrisk(self):
        mask = missing_mask(self.frame)
        n_missing = int(mask["HighFallRisk"].sum())
        self.assertEqual(n_missing, 0)

    def test_missingness_table_has_all_columns(self):
        miss_df = missing_data_analysis(self.frame)
        self.assertEqual(len(miss_df), 30)
        self.assertIn("variable", miss_df.columns)
        self.assertIn("N", miss_df.columns)
        self.assertIn("n_missing", miss_df.columns)
        self.assertIn("missing_pct", miss_df.columns)
        self.assertIn("n_observed", miss_df.columns)

    def test_missingness_all_empty_count(self):
        miss_df = missing_data_analysis(self.frame)
        all_empty = miss_df[miss_df["n_missing"] == 77]
        self.assertEqual(len(all_empty), 22)

    # --- descriptive statistics ---
    def test_descriptive_statistics_shape(self):
        desc_df = descriptive_statistics(self.frame, NUMERIC_COLS)
        self.assertEqual(len(desc_df), 4)
        required_cols = {"variable", "N", "mean", "std", "median", "Q1", "Q3", "IQR", "min", "max", "skewness", "kurtosis"}
        self.assertTrue(required_cols.issubset(set(desc_df.columns)))

    def test_descriptive_statistics_age(self):
        desc_df = descriptive_statistics(self.frame, NUMERIC_COLS)
        age_row = desc_df[desc_df["variable"] == "Age"].iloc[0]
        self.assertEqual(int(age_row["N"]), 76)
        self.assertGreater(age_row["mean"], 60)
        self.assertLess(age_row["mean"], 90)
        self.assertGreater(age_row["std"], 0)

    def test_descriptive_statistics_no_nan(self):
        desc_df = descriptive_statistics(self.frame, NUMERIC_COLS)
        for col in desc_df.columns:
            if col != "variable":
                self.assertFalse(desc_df[col].isna().any())

    # --- target-group counting ---
    def test_target_group_counts(self):
        target = self.frame[TARGET_COL].str.strip()
        n_controls = int((target == "0").sum())
        n_fallers = int((target == "1").sum())
        self.assertEqual(n_controls, 44)
        self.assertEqual(n_fallers, 33)
        self.assertEqual(n_controls + n_fallers, 77)

    def test_target_no_missing(self):
        mask = missing_mask(self.frame)
        n_missing = int(mask[TARGET_COL].sum())
        self.assertEqual(n_missing, 0)

    # --- IQR outlier calculation ---
    def test_outlier_analysis_shape(self):
        outlier_df = outlier_analysis(self.frame, NUMERIC_COLS)
        self.assertEqual(len(outlier_df), 4)
        required_cols = {"variable", "Q1", "Q3", "IQR", "lower_fence", "upper_fence", "n_outside_fences", "pct_outside_fences"}
        self.assertTrue(required_cols.issubset(set(outlier_df.columns)))

    def test_outlier_queuing_logic(self):
        outlier_df = outlier_analysis(self.frame, NUMERIC_COLS)
        for _, row in outlier_df.iterrows():
            self.assertAlmostEqual(row["IQR"], round(row["Q3"] - row["Q1"], 6), places=5)
            self.assertAlmostEqual(row["lower_fence"], round(row["Q1"] - 1.5 * (row["Q3"] - row["Q1"]), 6), places=5)
            self.assertAlmostEqual(row["upper_fence"], round(row["Q3"] + 1.5 * (row["Q3"] - row["Q1"]), 6), places=5)

    def test_outlier_pct_reasonable(self):
        outlier_df = outlier_analysis(self.frame, NUMERIC_COLS)
        for _, row in outlier_df.iterrows():
            self.assertGreaterEqual(row["pct_outside_fences"], 0)
            self.assertLessEqual(row["pct_outside_fences"], 100)

    # --- categorical frequency calculations ---
    def test_categorical_comparison_returns_dataframes(self):
        cont_df, stat_df = categorical_comparison(self.frame, BINARY_COLS, TARGET_COL)
        self.assertIsInstance(cont_df, pd.DataFrame)
        self.assertIsInstance(stat_df, pd.DataFrame)
        self.assertIn("variable", cont_df.columns)
        self.assertIn("count", cont_df.columns)
        self.assertIn("variable", stat_df.columns)
        self.assertIn("test", stat_df.columns)
        self.assertIn("p_value", stat_df.columns)

    def test_categorical_gender_counts(self):
        cont_df, _ = categorical_comparison(self.frame, BINARY_COLS, TARGET_COL)
        gender_controls = cont_df[(cont_df["variable"] == "Gender") & (cont_df["target_class"] == "0")]
        gender_fallers = cont_df[(cont_df["variable"] == "Gender") & (cont_df["target_class"] == "1")]
        total_controls = int(gender_controls["count"].sum())
        total_fallers = int(gender_fallers["count"].sum())
        self.assertEqual(total_controls, 44)
        self.assertEqual(total_fallers, 33)

    # --- correlation calculation ---
    def test_correlation_analysis_shape(self):
        corr_df = correlation_analysis(self.frame, NUMERIC_COLS)
        n_pairs = len(NUMERIC_COLS) * (len(NUMERIC_COLS) - 1) // 2
        self.assertEqual(len(corr_df), n_pairs)
        required_cols = {"var1", "var2", "spearman_rho", "spearman_p", "pearson_r", "pearson_p", "n"}
        self.assertTrue(required_cols.issubset(set(corr_df.columns)))

    def test_correlation_spearman_values_range(self):
        corr_df = correlation_analysis(self.frame, NUMERIC_COLS)
        for _, row in corr_df.iterrows():
            self.assertGreaterEqual(row["spearman_rho"], -1.0)
            self.assertLessEqual(row["spearman_rho"], 1.0)
            self.assertGreaterEqual(row["spearman_p"], 0.0)
            self.assertLessEqual(row["spearman_p"], 1.0)

    def test_correlation_has_adjusted_p(self):
        corr_df = correlation_analysis(self.frame, NUMERIC_COLS)
        self.assertIn("adjusted_p_value_bh", corr_df.columns)

    # --- reproducibility/determinism ---
    def test_deterministic_results(self):
        desc1 = descriptive_statistics(self.frame, NUMERIC_COLS)
        desc2 = descriptive_statistics(self.frame, NUMERIC_COLS)
        pd.testing.assert_frame_equal(desc1, desc2)

    def test_fdr_monotonicity(self):
        pvals = np.array([0.01, 0.05, 0.1, 0.5, 0.8])
        adj = _benjamini_hochberg(pvals)
        for i in range(len(adj) - 1):
            self.assertLessEqual(adj[i], adj[i + 1])
        self.assertTrue(all(adj <= 1.0))
        self.assertTrue(all(adj >= 0.0))

    def test_fdr_original_ordered(self):
        pvals = np.array([0.01, 0.05, 0.1, 0.5, 0.8])
        adj = _benjamini_hochberg(pvals)
        self.assertAlmostEqual(adj[0], 0.05, places=10)

    # --- variable classification ---
    def test_classification_identifies_22_empty(self):
        class_df = classify_variables(self.frame, self.config)
        empty_count = len(class_df[class_df["classification"] == "Completely empty"])
        self.assertEqual(empty_count, 22)

    def test_classification_identifies_target(self):
        class_df = classify_variables(self.frame, self.config)
        target_rows = class_df[class_df["classification"] == "Target"]
        self.assertEqual(len(target_rows), 1)
        self.assertEqual(target_rows.iloc[0]["variable"], "HighFallRisk")

    def test_classification_identifies_identifier(self):
        class_df = classify_variables(self.frame, self.config)
        id_rows = class_df[class_df["classification"] == "Identifier"]
        self.assertEqual(len(id_rows), 1)
        self.assertEqual(id_rows.iloc[0]["variable"], "Participant_ID")

    def test_classification_continuous_predictors(self):
        class_df = classify_variables(self.frame, self.config)
        cont_rows = class_df[class_df["classification"] == "Candidate continuous/numeric predictor"]
        cont_names = set(cont_rows["variable"].tolist())
        self.assertTrue({"Age", "Total x", "Totaly", "TUG"}.issubset(cont_names))

    def test_classification_binary_predictor(self):
        class_df = classify_variables(self.frame, self.config)
        bin_rows = class_df[class_df["classification"] == "Candidate binary predictor"]
        self.assertEqual(len(bin_rows), 1)
        self.assertEqual(bin_rows.iloc[0]["variable"], "Gender")

    def test_classification_fall_history(self):
        class_df = classify_variables(self.frame, self.config)
        fh_rows = class_df[class_df["classification"] == "Fall-history (leakage review)"]
        self.assertEqual(len(fh_rows), 1)

    # --- univariate logistic regression ---
    def test_univariate_logistic_has_rows(self):
        logistic_df = univariate_logistic(self.frame, NUMERIC_COLS + BINARY_COLS, TARGET_COL)
        self.assertGreater(len(logistic_df), 0)
        self.assertIn("predictor", logistic_df.columns)
        self.assertIn("odds_ratio", logistic_df.columns)
        self.assertIn("p_value", logistic_df.columns)

    def test_univariate_logistic_has_all_predictors(self):
        logistic_df = univariate_logistic(self.frame, NUMERIC_COLS + BINARY_COLS, TARGET_COL)
        predictors = set(logistic_df["predictor"].tolist())
        self.assertTrue({"Age", "Total x", "Totaly", "TUG", "Gender"}.issubset(predictors))

    # --- ROC analysis ---
    def test_roc_analysis_has_rows(self):
        roc_df = roc_analysis(self.frame, NUMERIC_COLS, TARGET_COL)
        self.assertEqual(len(roc_df), 4)
        self.assertIn("auc", roc_df.columns)
        self.assertIn("variable", roc_df.columns)

    def test_roc_auc_values_range(self):
        roc_df = roc_analysis(self.frame, NUMERIC_COLS, TARGET_COL)
        for _, row in roc_df.iterrows():
            if row["auc"] != "":
                self.assertGreaterEqual(float(row["auc"]), 0.0)
                self.assertLessEqual(float(row["auc"]), 1.0)

    def test_roc_tug_auc_best(self):
        roc_df = roc_analysis(self.frame, NUMERIC_COLS, TARGET_COL)
        tug_row = roc_df[roc_df["variable"] == "TUG"].iloc[0]
        age_row = roc_df[roc_df["variable"] == "Age"].iloc[0]
        tug_auc = float(tug_row["auc"])
        age_auc = float(age_row["auc"])
        self.assertGreater(tug_auc, age_auc)

    # --- multicollinearity ---
    def test_vif_diagnostic_has_rows(self):
        corr_mat, vif_df = multicollinearity_diagnostic(self.frame, NUMERIC_COLS)
        self.assertEqual(len(vif_df), 4)
        self.assertIn("variable", vif_df.columns)
        self.assertIn("vif", vif_df.columns)

    def test_correlation_matrix_square(self):
        corr_mat, _ = multicollinearity_diagnostic(self.frame, NUMERIC_COLS)
        self.assertEqual(corr_mat.shape[0], corr_mat.shape[1])
        self.assertEqual(corr_mat.shape[0], 4)

    # --- report exists ---
    def test_report_exists(self):
        report_path = ROOT / "reports" / "phase2_statistical_data_audit_report.md"
        self.assertTrue(report_path.exists())
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("Dataset overview", content)
        self.assertIn("Variable classification", content)
        self.assertIn("Missing-data analysis", content)
        self.assertIn("Leakage audit", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
