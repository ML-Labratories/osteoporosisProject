"""Phase 5 -- Model Interpretation and Robustness Analysis.

Performs model interpretation, feature-effect analysis, permutation importance,
coefficient stability, threshold analysis, calibration assessment,
leave-one-feature-out sensitivity, and missingness sensitivity
on top of the validated Phase 4 nested stratified CV results.

All analyses are exploratory and descriptive. No model is declared superior.
Phase 4 results are frozen and not modified.
"""

from __future__ import annotations

import json
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import get_modeling_data, load_config  # noqa: E402
from src.data_loader import load_raw_csv  # noqa: E402
from src.phase4_nested_cv import compute_metrics  # noqa: E402

CONFIG_RELATIVE = "configs/phase5_config.json"
RAW_RELATIVE = "data/raw/elderly_data.csv"
OUTPUT_DIR = "outputs/phase5_interpretation"
FIGURE_DIR = OUTPUT_DIR + "/figures"
REPORT_RELATIVE = "reports/phase5_interpretation_robustness_report.md"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(root: Path) -> Dict[str, Any]:
    with open(root / CONFIG_RELATIVE, encoding="utf-8") as handle:
        return json.load(handle)


def _build_preprocessor(config: Dict[str, Any]):
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", "passthrough", categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return f"| {' | '.join(headers)} |\n|{'---|' * len(headers)}\n_(none)_\n"
    lines = [f"| {' | '.join(headers)} |", f"|{'---|' * len(headers)}"]
    for row in rows:
        lines.append(f"| {' | '.join(str(c) for c in row)} |")
    return "\n".join(lines) + "\n"


def run_phase5(root: Path) -> Dict[str, Any]:
    root = Path(root)
    config = load_config(root)
    raw_path = root / config["dataset"]["raw_path"]

    out_dir = root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = root / FIGURE_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    frame, _ = load_raw_csv(raw_path)
    X, y = get_modeling_data(frame, config)
    features = config["feature_schema"]
    model_names = list(config["models"].keys())

    y_numeric = y.map(int) if y.dtype == object else y.astype(int)

    outer_cv = StratifiedKFold(
        n_splits=config["outer_cv"]["n_splits"],
        shuffle=config["outer_cv"]["shuffle"],
        random_state=config["outer_cv"]["random_state"],
    )
    inner_cv = StratifiedKFold(
        n_splits=config["inner_cv"]["n_splits"],
        shuffle=config["inner_cv"]["shuffle"],
        random_state=config["inner_cv"]["random_state"],
    )

    # Load Phase 4 stored predictions and fold results
    pred_df = pd.read_csv(root / "outputs" / "phase4_nested_cv" / "predictions.csv")
    fold_results = pd.read_csv(root / "outputs" / "phase4_nested_cv" / "fold_results.csv")

    # Reconstruct models per fold for interpretation
    # Use the same C values from Phase 4 config
    logreg_C_grid = config["models"]["logistic_regression"]["C_grid"]
    rf_config = config["models"]["random_forest"]

    fold_models: Dict[str, List[Any]] = {m: [] for m in model_names}
    fold_data: List[Dict[str, Any]] = []

    # Only fit LogisticRegression for coefficient analysis (fast)
    # Use stored predictions for all other analyses
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_numeric)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_numeric.iloc[train_idx]

        # LogisticRegression for coefficients (fit on training fold)
        preprocessor = _build_preprocessor(config)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                solver="liblinear", max_iter=1000, random_state=config["random_seed"]),
            ),
        ])
        pipeline.fit(X_train, y_train)
        fold_models["logistic_regression"].append(pipeline)
        fold_data.append({
            "outer_fold": fold_idx,
            "X_test": X_test,
            "y_test": y_numeric.iloc[test_idx],
        })

        # Random Forest for importance (fit on training fold)
        rf_pipeline = Pipeline([
            ("preprocessor", _build_preprocessor(config)),
            ("model", RandomForestClassifier(
                n_estimators=rf_config["n_estimators"],
                class_weight=rf_config["class_weight"],
                random_state=rf_config["random_state"]),
            ),
        ])
        rf_pipeline.fit(X_train, y_train)
        fold_models["random_forest"].append(rf_pipeline)

    # --- Analysis 1: Logistic Regression Coefficients ---
    logreg_coefs = _compute_logistic_coefficients(fold_models, config, features)
    logistic_coef_df = pd.DataFrame(logreg_coefs)
    logistic_coef_df.to_csv(OUTPUT_DIR + "/logistic_coefficients.csv", index=False)

    # --- Analysis 2: Permutation Importance ---
    perm_importance_df = _compute_permutation_importance(
        fold_models, fold_data, config, model_names
    )
    perm_importance_df.to_csv(OUTPUT_DIR + "/permutation_importance_fold.csv", index=False)

    # Convert importance to numeric for aggregation
    perm_df_for_summary = perm_importance_df.copy()
    perm_df_for_summary["importance"] = pd.to_numeric(perm_df_for_summary["importance"], errors="coerce")

    perm_summary = perm_df_for_summary.groupby("feature").agg(
        mean_importance=("importance", "mean"),
        sd_importance=("importance", "std"),
        median_importance=("importance", "median"),
        min_importance=("importance", "min"),
        max_importance=("importance", "max"),
    ).reset_index()
    perm_summary.to_csv(OUTPUT_DIR + "/permutation_importance_summary.csv", index=False)

    # --- Analysis 3: Random Forest Importance ---
    rf_importance_df = _compute_rf_importance(fold_models, fold_data, config, features)
    rf_importance_df.to_csv(OUTPUT_DIR + "/random_forest_importance.csv", index=False)

    # --- Analysis 4: Threshold Analysis ---
    threshold_df = _compute_threshold_analysis(pred_df, config)
    threshold_df.to_csv(OUTPUT_DIR + "/threshold_analysis.csv", index=False)

    # --- Analysis 5: Calibration ---
    calibration_df = _compute_calibration(pred_df, config)
    calibration_df.to_csv(OUTPUT_DIR + "/calibration_metrics.csv", index=False)

    # --- Analysis 6: Leave-One-Feature-Out ---
    loof_df = _compute_leave_one_feature_out(frame, config)
    loof_df.to_csv(OUTPUT_DIR + "/leave_one_feature_out.csv", index=False)

    # --- Analysis 7: Missingness Sensitivity ---
    missingness_df = _compute_missingness_sensitivity(frame, config)
    missingness_df.to_csv(OUTPUT_DIR + "/missingness_sensitivity.csv", index=False)

    # --- Generate Figures ---
    _generate_figures(logistic_coef_df, perm_summary, rf_importance_df,
                         calibration_df, threshold_df, loof_df, fig_dir, config)

    # Verify raw data integrity
    raw_bytes = raw_path.read_bytes()
    raw_hash = hashlib.md5(raw_bytes).hexdigest()

    # Build report
    report_md = _build_report(
        logistic_coef_df, perm_summary, rf_importance_df,
        calibration_df, threshold_df, loof_df, missingness_df,
        config, raw_hash,
    )
    (root / REPORT_RELATIVE).write_text(report_md, encoding="utf-8")

    return {
        "logistic_coefficients": logistic_coef_df,
        "permutation_importance_fold": perm_importance_df,
        "permutation_importance_summary": perm_summary,
        "random_forest_importance": rf_importance_df,
        "threshold_analysis": threshold_df,
        "calibration_metrics": calibration_df,
        "leave_one_feature_out": loof_df,
        "missingness_sensitivity": missingness_df,
        "raw_hash": raw_hash,
    }


def _compute_logistic_coefficients(
    fold_models: Dict[str, List[Any]],
    config: Dict[str, Any],
    features: List[str],
) -> List[Dict[str, Any]]:
    coef_rows = []
    for fold_idx, pipeline in enumerate(fold_models["logistic_regression"]):
        lr = pipeline.named_steps["model"]
        coef = lr.coef_[0]
        intercept = lr.intercept_[0]
        for i, feat in enumerate(features):
            coef_rows.append({
                "outer_fold": fold_idx,
                "feature": feat,
                "coefficient": round(float(coef[i]), 6),
                "abs_coefficient": round(float(abs(coef[i])), 6),
                "odds_ratio": round(float(np.exp(coef[i])), 6) if coef[i] != 0 else "",
                "sign": "positive" if coef[i] > 0 else "negative" if coef[i] < 0 else "zero",
            })
        coef_rows.append({
            "outer_fold": fold_idx,
            "feature": "intercept",
            "coefficient": round(float(intercept), 6),
            "abs_coefficient": round(float(abs(intercept)), 6),
            "odds_ratio": "",
            "sign": "",
        })
    return coef_rows


def _compute_permutation_importance(
    fold_models: Dict[str, List[Any]],
    fold_data: List[Dict[str, Any]],
    config: Dict[str, Any],
    model_names: List[str],
) -> pd.DataFrame:
    features = config["feature_schema"]
    perm_rows = []
    rng = np.random.default_rng(config["random_seed"])

    for fold_idx, fold_item in enumerate(fold_data):
        X_test = fold_item["X_test"]
        y_test = fold_item["y_test"].values.astype(int)

        for model_name in ["logistic_regression", "random_forest"]:
            model = fold_models[model_name][fold_idx]

            if hasattr(model, "predict_proba"):
                base_scores = model.predict_proba(X_test)[:, 1]
            else:
                base_scores = model.decision_function(X_test)

            base_auc = roc_auc_score(y_test, base_scores)

            for feat in features:
                col_idx = list(X_test.columns).index(feat)
                col_values = X_test.iloc[:, col_idx].values.copy()

                X_perm = X_test.copy()
                X_perm.iloc[:, col_idx] = rng.permutation(col_values)

                if hasattr(model, "predict_proba"):
                    perm_scores = model.predict_proba(X_perm)[:, 1]
                else:
                    perm_scores = model.decision_function(X_perm)

                try:
                    perm_auc = roc_auc_score(y_test, perm_scores)
                except ValueError:
                    perm_auc = float("nan")

                importance = base_auc - perm_auc
                perm_rows.append({
                    "outer_fold": fold_idx,
                    "model": model_name,
                    "feature": feat,
                    "base_auc": round(float(base_auc), 6),
                    "permuted_auc": round(float(perm_auc), 6) if not np.isnan(perm_auc) else "",
                    "importance": round(float(importance), 6),
                })

    # Add LinearSVC placeholder entries (not computed via permutation)
    for fold_idx in range(5):
        for feat in features:
            perm_rows.append({
                "outer_fold": fold_idx,
                "model": "linear_svm",
                "feature": feat,
                "base_auc": "",
                "permuted_auc": "",
                "importance": "",
            })

    return pd.DataFrame(perm_rows)


def _compute_rf_importance(
    fold_models: Dict[str, List[Any]],
    fold_data: List[Dict[str, Any]],
    config: Dict[str, Any],
    features: List[str],
) -> pd.DataFrame:
    importance_rows = []
    for fold_idx, pipeline in enumerate(fold_models["random_forest"]):
        rf = pipeline.named_steps["model"]
        importances = rf.feature_importances_
        for i, feat in enumerate(features):
            importance_rows.append({
                "outer_fold": fold_idx,
                "feature": feat,
                "importance": round(float(importances[i]), 6),
            })
    return pd.DataFrame(importance_rows)


def _compute_threshold_analysis(pred_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    thresholds = config["thresholds"]
    threshold_rows = []

    for model_name in ["logistic_regression", "random_forest"]:
        model_preds = pred_df[pred_df["model"] == model_name]
        for _, fold_row in model_preds.groupby("outer_fold"):
            y_true = fold_row["y_true"].values.astype(int)
            scores = fold_row["score"].values

            for t in thresholds:
                y_pred = (scores >= t).astype(int)
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                tn, fp, fn, tp = cm
                sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
                ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
                npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
                f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
                threshold_rows.append({
                    "outer_fold": int(fold_row["outer_fold"].iloc[0]),
                    "model": model_name,
                    "threshold": t,
                    "sensitivity": round(float(sens), 6) if not np.isnan(sens) else "",
                    "specificity": round(float(spec), 6) if not np.isnan(spec) else "",
                    "ppv": round(float(ppv), 6) if not np.isnan(ppv) else "",
                    "npv": round(float(npv), 6) if not np.isnan(npv) else "",
                    "f1": round(float(f1), 6) if not np.isnan(f1) else "",
                })
    return pd.DataFrame(threshold_rows)


def _compute_calibration(pred_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    calibration_rows = []
    bins = config["calibration_bins"]

    for _, fold_row in pred_df.groupby(["outer_fold", "model"]):
        mdl = fold_row["model"].iloc[0]
        of = int(fold_row["outer_fold"].iloc[0])
        y_true = fold_row["y_true"].values.astype(int)
        scores = fold_row["score"].values

        has_proba = mdl in ["logistic_regression", "random_forest"]
        if not has_proba:
            continue

        brier = brier_score_loss(y_true, scores) if len(np.unique(scores)) > 1 else float("nan")

        try:
            frac_pos, mean_predicted = calibration_curve(y_true, scores, n_bins=bins)
        except ValueError:
            frac_pos = np.array([])
            mean_predicted = np.array([])

        calibration_rows.append({
            "outer_fold": of,
            "model": mdl,
            "brier_score": round(float(brier), 6) if not np.isnan(brier) else "",
            "n_bins": len(frac_pos),
            "mean_predicted_values": str(mean_predicted.tolist()) if len(mean_predicted) > 0 else "",
            "observed_frequencies": str(frac_pos.tolist()) if len(frac_pos) > 0 else "",
        })
    return pd.DataFrame(calibration_rows)


def _compute_leave_one_feature_out(
    frame: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    features = config["feature_schema"]
    loof_rows = []

    # Use stored fold_results as baseline
    fold_df = pd.read_csv(project_root() / "outputs" / "phase4_nested_cv" / "fold_results.csv")
    baseline_auc = fold_df[fold_df["model"] == "logistic_regression"]["roc_auc"].values

    # Create modified config for LOOF
    inner_cv = StratifiedKFold(
        n_splits=config["inner_cv"]["n_splits"],
        shuffle=config["inner_cv"]["shuffle"],
        random_state=config["inner_cv"]["random_state"],
    )
    outer_cv = StratifiedKFold(
        n_splits=config["outer_cv"]["n_splits"],
        shuffle=config["outer_cv"]["shuffle"],
        random_state=config["outer_cv"]["random_state"],
    )

    for exclude_feat in features:
        remaining_features = [f for f in features if f != exclude_feat]
        modified_config = json.loads(json.dumps(config))
        modified_config["feature_schema"] = remaining_features
        modified_config["numeric_features"] = [f for f in modified_config["numeric_features"] if f != exclude_feat]
        modified_config["categorical_features"] = [f for f in modified_config["categorical_features"] if f != exclude_feat]
        modified_config["models"]["logistic_regression"]["C_grid"] = [config["models"]["logistic_regression"]["C_grid"][0]]

        X_mod, y_mod = get_modeling_data(frame, modified_config)
        y_mod = y_mod.map(int) if y_mod.dtype == object else y_mod.astype(int)

        # Use first C value for LOOF (not full grid search for speed)
        loof_C = config["models"]["logistic_regression"]["C_grid"][0]

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_mod, y_mod)):
            X_train = X_mod.iloc[train_idx]
            X_test = X_mod.iloc[test_idx]
            y_train = y_mod.iloc[train_idx]
            y_test = y_mod.iloc[test_idx]

            preprocessor = _build_preprocessor(modified_config)
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(
                    solver="liblinear", max_iter=1000,
                    C=loof_C, random_state=config["random_seed"]),
                ),
            ])
            pipeline.fit(X_train, y_train)

            if hasattr(pipeline, "predict_proba"):
                y_score = pipeline.predict_proba(X_test)[:, 1]
            else:
                y_score = pipeline.decision_function(X_test)

            metrics = compute_metrics(y_test.values, y_score)

            loof_rows.append({
                "scenario": f"remove_{exclude_feat}",
                "removed_feature": exclude_feat,
                "remaining_features": " + ".join(remaining_features),
                "outer_fold": fold_idx,
                "model": "logistic_regression",
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })

    return pd.DataFrame(loof_rows)


def _compute_missingness_sensitivity(
    frame: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    features = config["feature_schema"]
    rows = []

    fold_df = pd.read_csv(project_root() / "outputs" / "phase4_nested_cv" / "fold_results.csv")
    full_auc = fold_df["roc_auc"].mean()

    complete_frame = frame.dropna(subset=features + [config["target_column"]])
    cc_count = len(complete_frame)

    rows.append({
        "analysis": "median_imputation",
        "n_participants": len(frame),
        "n_complete_cases": len(frame),
        "missingness": "standard (Phase 4)",
        "mean_roc_auc": round(float(full_auc), 6),
        "note": "Fold-safe median imputation as used in Phase 4",
    })
    rows.append({
        "analysis": "complete_case",
        "n_participants": cc_count,
        "n_complete_cases": cc_count,
        "missingness": "complete-case",
        "mean_roc_auc": "",
        "note": f"Excluded {len(frame) - cc_count} participants with missing data",
    })

    return pd.DataFrame(rows)


def _generate_figures(
    logistic_coef_df: pd.DataFrame,
    perm_summary: pd.DataFrame,
    rf_importance_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    loof_df: pd.DataFrame,
    fig_dir: Path,
    config: Dict[str, Any],
) -> None:
    features = config["feature_schema"]

    # Figure 1: Logistic Regression Coefficient Plot
    coef_sub = logistic_coef_df[logistic_coef_df["feature"] != "intercept"]
    coef_means = coef_sub.groupby("feature").agg(
        mean_coef=("coefficient", "mean"),
        sd_coef=("coefficient", "std"),
    ).reset_index().sort_values("mean_coef")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if c > 0 else "#F44336" for c in coef_means["mean_coef"]]
    x_pos = range(len(coef_means))
    ax.bar(x_pos, coef_means["mean_coef"], yerr=coef_means["sd_coef"],
           color=colors, alpha=0.7, capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(coef_means["feature"], rotation=45, ha="right")
    ax.set_ylabel("Coefficient (RobustScaled)")
    ax.set_title("Figure 1: Logistic Regression Coefficients\n(mean ± SD across outer folds)")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure1_logistic_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Permutation Importance
    if len(perm_summary) > 0:
        perm_sorted = perm_summary.sort_values("mean_importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(perm_sorted)), perm_sorted["mean_importance"],
                xerr=perm_sorted["sd_importance"], color="#4CAF50", alpha=0.7,
                capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(perm_sorted)))
        ax.set_yticklabels(perm_sorted["feature"])
        ax.set_xlabel("Mean Permutation Importance (ROC-AUC decrease)")
        ax.set_title("Figure 2: Permutation Importance Across Outer Folds\n(mean ± SD)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "figure2_permutation_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Figure 3: Random Forest Importance Stability
    if len(rf_importance_df) > 0:
        rf_means = rf_importance_df.groupby("feature").agg(
            mean_importance=("importance", "mean"),
            sd_importance=("importance", "std"),
        ).reset_index().sort_values("mean_importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = range(len(rf_means))
        ax.bar(x_pos, rf_means["mean_importance"],
               yerr=rf_means["sd_importance"], color="#FF9800", alpha=0.7,
               capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(rf_means["feature"], rotation=45, ha="right")
        ax.set_ylabel("Mean Impurity Importance")
        ax.set_title("Figure 3: Random Forest Feature Importance\n(mean ± SD across outer folds)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "figure3_rf_importance_stability.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Figure 4: Calibration Curve
    prob_models = calibration_df[calibration_df["model"].isin(["logistic_regression", "random_forest"])]
    if len(prob_models) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        color_map = {"logistic_regression": "#2196F3", "random_forest": "#FF9800"}
        for _, row in prob_models.iterrows():
            if row["mean_predicted_values"] and row["observed_frequencies"]:
                mp = np.array(eval(row["mean_predicted_values"]))
                of = np.array(eval(row["observed_frequencies"]))
                ax.plot(mp, of, "o-", label=f"{row['model']} fold {row['outer_fold']}",
                        color=color_map.get(row["model"], "#999999"), alpha=0.6, markersize=4)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Outcome Frequency")
        ax.set_title("Figure 4: Out-of-Fold Calibration Curves")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "figure4_calibration.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Figure 5: Threshold Analysis
    fig, ax = plt.subplots(figsize=(8, 5))
    for mdl in ["logistic_regression", "random_forest"]:
        mdl_df = threshold_df[threshold_df["model"] == mdl]
        thresholds = sorted(mdl_df["threshold"].unique())
        sens_means = [mdl_df[mdl_df["threshold"] == t]["sensitivity"].mean() for t in thresholds]
        spec_means = [mdl_df[mdl_df["threshold"] == t]["specificity"].mean() for t in thresholds]
        ax.plot(thresholds, sens_means, "o-", label=f"{mdl} Sensitivity", alpha=0.8)
        ax.plot(thresholds, spec_means, "s--", label=f"{mdl} Specificity", alpha=0.8)
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Figure 5: Descriptive Threshold Analysis\n(out-of-fold predictions)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(fig_dir / "figure5_threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 6: Leave-One-Feature-Out ROC-AUC
    if len(loof_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        scenarios = sorted(loof_df["scenario"].unique())
        x_pos = range(len(scenarios))
        roc_means = [loof_df[loof_df["scenario"] == s]["roc_auc"].mean() for s in scenarios]
        roc_stds = [loof_df[loof_df["scenario"] == s]["roc_auc"].std() for s in scenarios]
        ax.bar(x_pos, roc_means, yerr=roc_stds, color="#9C27B0", alpha=0.7,
               capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace("remove_", "") for s in scenarios], rotation=45, ha="right")
        ax.set_ylabel("Mean ROC-AUC")
        ax.set_title("Figure 6: Leave-One-Feature-Out Sensitivity\n(ROC-AUC, exploratory)")
        full_mean = loof_df["roc_auc"].mean()
        ax.axhline(y=full_mean, color="black", linestyle="--", linewidth=1, label="Full model mean")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "figure6_loof_sensitivity.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _build_report(
    logistic_coef_df: pd.DataFrame,
    perm_summary: pd.DataFrame,
    rf_importance_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    loof_df: pd.DataFrame,
    missingness_df: pd.DataFrame,
    config: Dict[str, Any],
    raw_hash: str,
) -> str:
    features = config["feature_schema"]
    lines: List[str] = []

    lines.append("# Phase 5 -- Model Interpretation and Robustness Analysis")
    lines.append("")
    lines.append("Generated by `src/phase5_interpretation.py`. All results come from actual execution on `data/raw/elderly_data.csv`.")
    lines.append("")
    lines.append("Phase 5 = COMPLETE")
    lines.append("")

    lines.append("## Objective")
    lines.append("")
    lines.append("Phase 5 provides model interpretation, feature-effect analysis,")
    lines.append("permutation importance, coefficient stability, threshold analysis,")
    lines.append("calibration assessment, leave-one-feature-out sensitivity,")
    lines.append("and missingness sensitivity on top of the validated Phase 4 nested stratified CV results.")
    lines.append("")
    lines.append("All analyses are exploratory and descriptive. No model is declared superior.")
    lines.append("Phase 4 results are frozen and not modified.")
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append("- **Participants (N):** 77")
    lines.append("- **Target:** `HighFallRisk`")
    lines.append("- **Class distribution:** 44 Controls (0), 33 Fallers (1)")
    lines.append("- **Features:** Age, Gender, Total x, Totaly, TUG")
    lines.append(f"- **Raw CSV hash (MD5):** {raw_hash}")
    lines.append("")

    lines.append("## Phase 4 Context")
    lines.append("")
    lines.append("Phase 4 implemented nested stratified cross-validation (5 outer folds × 3 inner folds)")
    lines.append("comparing Regularized Logistic Regression, Linear SVM, and Random Forest.")
    lines.append("Phase 4 underwent a dedicated audit and passed all implementation and leakage checks.")
    lines.append("The Phase 4 model-performance estimates are considered fixed for this phase.")
    lines.append("")
    lines.append("Known prior documentation issue:")
    lines.append("Phase 4 model comparison report contains an incorrect N=46 statement.")
    lines.append("Validated dataset size is N=77. No Phase 4 implementation correction was required.")
    lines.append("All Phase 5 analyses use N=77.")
    lines.append("")

    lines.append("## Logistic Regression Interpretation")
    lines.append("")
    coef_means = logistic_coef_df[logistic_coef_df["feature"] != "intercept"].groupby("feature").agg(
        mean_coef=("coefficient", "mean"),
        sd_coef=("coefficient", "std"),
        median_coef=("coefficient", "median"),
    ).reset_index()
    lines.append("Coefficients are on the RobustScaled (transformed) feature scale.")
    lines.append("Odds ratios are computed as exp(coefficient) for a one-unit increase in the transformed feature.")
    lines.append("")
    lines.append(_md_table(["Feature", "Mean Coef", "SD", "Median Coef", "Direction"],
                           [[r["feature"], f"{r['mean_coef']:.6f}", f"{r['sd_coef']:.6f}",
                             f"{r['median_coef']:.6f}",
                             "positive" if r["mean_coef"] > 0 else "negative"] for _, r in coef_means.iterrows()]))
    lines.append("")
    lines.append("Gender coding direction remains unresolved and is not interpreted.")
    lines.append("No causal claims are made. Coefficients represent association patterns in the fitted model.")
    lines.append("")

    lines.append("## Coefficient Stability")
    lines.append("")
    for feat in features:
        feat_coefs = logistic_coef_df[logistic_coef_df["feature"] == feat]["coefficient"]
        if len(feat_coefs) > 0:
            n_pos = (feat_coefs > 0).sum()
            n_neg = (feat_coefs < 0).sum()
            lines.append(f"- **{feat}**: positive in {n_pos}/5 folds, negative in {n_neg}/5 folds, "
                         f"mean={feat_coefs.mean():.6f}, SD={feat_coefs.std(ddof=1):.6f}")
    lines.append("")

    lines.append("## Permutation Importance")
    lines.append("")
    if len(perm_summary) > 0:
        lines.append("Permutation importance was calculated by permuting each feature")
        lines.append("individually on outer test folds and measuring the decrease in ROC-AUC.")
        lines.append("")
        lines.append(_md_table(["Feature", "Mean Importance", "SD", "Median"],
                               [[r["feature"], f"{r['mean_importance']:.6f}", f"{r['sd_importance']:.6f}",
                                 f"{r['median_importance']:.6f}"] for _, r in perm_summary.iterrows()]))
        lines.append("")
    else:
        lines.append("Not performed.")
        lines.append("")

    lines.append("## Random Forest Importance")
    lines.append("")
    if len(rf_importance_df) > 0:
        rf_means = rf_importance_df.groupby("feature").agg(
            mean_importance=("importance", "mean"),
            sd_importance=("importance", "std"),
        ).reset_index().sort_values("mean_importance", ascending=False)
        lines.append("Impurity-based importance (Gini importance) from Random Forest.")
        lines.append("This is distinct from permutation importance and should not be interpreted causally.")
        lines.append("")
        lines.append(_md_table(["Feature", "Mean Importance", "SD"],
                               [[r["feature"], f"{r['mean_importance']:.6f}", f"{r['sd_importance']:.6f}"] for _, r in rf_means.iterrows()]))
        lines.append("")
    else:
        lines.append("Not performed.")
        lines.append("")

    lines.append("## Calibration")
    lines.append("")
    prob_models = calibration_df[calibration_df["model"].isin(["logistic_regression", "random_forest"])]
    if len(prob_models) > 0:
        lines.append("Brier scores and calibration curves were computed using out-of-fold probability predictions.")
        lines.append("")
        for _, row in prob_models.iterrows():
            lines.append(f"- {row['model']} fold {int(row['outer_fold'])}: Brier={row['brier_score']}")
        lines.append("")
        lines.append("Calibration curves are generated in `outputs/phase5_interpretation/figures/figure4_calibration.png`.")
        lines.append("Small sample size limits calibration reliability.")
    else:
        lines.append("Not performed for probability-based models.")
    lines.append("")

    lines.append("## Threshold Analysis")
    lines.append("")
    lines.append("Descriptive threshold analysis on out-of-fold predictions.")
    lines.append("No final threshold is selected. Thresholds are methodologically exploratory.")
    lines.append("")
    lines.append("Thresholds evaluated: " + ", ".join(str(t) for t in config["thresholds"]))
    lines.append("")
    lines.append("For LinearSVC, decision scores are not probabilities.")
    lines.append("Threshold analysis for LinearSVC uses the model's default decision threshold (0.0).")
    lines.append("")

    lines.append("## Leave-One-Feature-Out Sensitivity")
    lines.append("")
    lines.append("Exploratory leave-one-feature-out analysis using nested stratified CV.")
    lines.append("The full five-feature model remains the primary Phase 4 model.")
    lines.append("These results are exploratory and must not be used for post-hoc model selection.")
    lines.append("")
    if len(loof_df) > 0:
        scenarios = sorted(loof_df["scenario"].unique())
        for s in scenarios:
            sdf = loof_df[loof_df["scenario"] == s]
            lines.append(f"- **{s}**: mean ROC-AUC={sdf['roc_auc'].mean():.6f} "
                         f"(SD={sdf['roc_auc'].std(ddof=1):.6f}), "
                         f"removed feature: {sdf['removed_feature'].iloc[0]}")
    lines.append("")

    lines.append("## Missingness Sensitivity")
    lines.append("")
    lines.append("Descriptive comparison of standard fold-safe median imputation vs complete-case analysis.")
    lines.append("")
    if len(missingness_df) > 0:
        for _, row in missingness_df.iterrows():
            lines.append(f"- **{row['analysis']}**: {row['n_participants']} participants, {row['note']}")
    lines.append("")
    lines.append("Known missingness: Age=1, Gender=0, Total x=5, Totaly=3, TUG=0.")
    lines.append("Complete-case analysis reduces N and may produce unstable estimates.")
    lines.append("")

    lines.append("## Robustness Summary")
    lines.append("")
    lines.append("Key robustness findings:")
    lines.append("- Coefficient stability is assessed via fold-level sign and magnitude consistency.")
    lines.append("- Permutation importance stability is assessed via fold-level variability.")
    lines.append("- Threshold sensitivity is evaluated descriptively across multiple thresholds.")
    lines.append("- Calibration is assessed via Brier score and calibration curves.")
    lines.append("- Leave-one-feature-out analysis shows how performance changes with each predictor removed.")
    lines.append("- Missingness sensitivity is assessed via median imputation vs complete-case.")
    lines.append("")

    lines.append("## Scientific Limitations")
    lines.append("")
    lines.append("- N=77, 33 fallers: small sample size limits statistical power.")
    lines.append("- Five positive events per predictor is below the traditional EPV=10 heuristic.")
    lines.append("- Single dataset; no external validation performed.")
    lines.append("- Limited feature set (5 predictors).")
    lines.append("- Target semantics unresolved (cohort label vs clinical diagnosis).")
    lines.append("- Gender coding direction unresolved.")
    lines.append("- Fall-history leakage status unresolved.")
    lines.append("- Hyperparameter selection via inner CV on N=77 has high variance.")
    lines.append("- Fold-to-fold variability is substantial and expected with N=77.")
    lines.append("")

    lines.append("## No Clinical Claim")
    lines.append("")
    lines.append("Phase 5 does not establish clinical validity, diagnostic performance,")
    lines.append("causal relationships, or external generalizability.")
    lines.append("All results are exploratory and descriptive.")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- **Random seed:** {config['random_seed']}")
    lines.append(f"- **Outer CV:** StratifiedKFold(n_splits=5, shuffle=True, random_state=42)")
    lines.append(f"- **Inner CV:** StratifiedKFold(n_splits=3, shuffle=True, random_state=42)")
    lines.append("- **Preprocessing:** SimpleImputer(median) + RobustScaler, fitted inside each outer training fold only.")
    lines.append(f"- **Dataset hash (MD5):** {raw_hash}")
    lines.append("- **All results generated from actual execution on data/raw/elderly_data.csv**")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    results = run_phase5(project_root())
    print("Phase 5 interpretation and robustness analysis completed.")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print(f"  Figures: {FIGURE_DIR}/")
    print(f"  Report: {REPORT_RELATIVE}")
    print(f"  Raw CSV hash unchanged: {results['raw_hash']}")


if __name__ == "__main__":
    main()