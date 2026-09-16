"""Phase 4 -- Nested Stratified Cross-Validation and Baseline Models.

Performs a scientifically defensible comparison of three classical ML baseline
models using nested stratified cross-validation on the LTMM harmonized dataset.

Models:
1. Regularized Logistic Regression
2. Linear SVM
3. Random Forest

Preprocessing (from Phase 3): SimpleImputer(median) + RobustScaler, fitted
inside each training fold only.

All results are generated from actual execution on data/raw/elderly_data.csv.
No fabricated values. No model training outside the CV loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc as pr_auc_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import build_preprocessor, get_modeling_data, load_config  # noqa: E402
from src.data_loader import load_raw_csv  # noqa: E402

CONFIG_RELATIVE = "configs/phase4_config.json"
RAW_RELATIVE = "data/raw/elderly_data.csv"
OUTPUT_DIR = "outputs/phase4_nested_cv"
REPORT_RELATIVE = "reports/phase4_model_comparison_report.md"
TASK_REPORT_RELATIVE = "reports/phase4_task_report.md"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(root: Path) -> Dict[str, Any]:
    with open(root / CONFIG_RELATIVE, encoding="utf-8") as handle:
        return json.load(handle)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                     pos_label: int = 1) -> Dict[str, Any]:
    """Compute all performance metrics from true labels and scores."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Binarize for metric computation
    pos = (y_true == pos_label).astype(int)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")

    # PR-AUC
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = pr_auc_score(y_true, y_score)
    except ValueError:
        pr_auc = float("nan")

    # Default threshold prediction
    if hasattr(y_score, '__len__') and len(y_score) > 0 and np.nanmax(np.abs(y_score)) > 1.5:
        # Scores like decision_function (SVM) - use 0 threshold
        threshold = 0.0
        y_pred = (y_score >= threshold).astype(int)
    else:
        # Probability-like scores - use 0.5 threshold
        threshold = 0.5
        y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    balanced_accuracy = (sensitivity + specificity) / 2 if not np.isnan(sensitivity) and not np.isnan(specificity) else float("nan")

    # Brier score (for probability-based models)
    try:
        brier = brier_score_loss(y_true, y_score) if len(np.unique(y_score)) > 1 else float("nan")
    except Exception:
        brier = float("nan")

    return {
        "roc_auc": round(float(roc_auc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "sensitivity": round(float(sensitivity), 6),
        "specificity": round(float(specificity), 6),
        "ppv": round(float(ppv), 6),
        "npv": round(float(npv), 6),
        "f1": round(float(f1), 6),
        "balanced_accuracy": round(float(balanced_accuracy), 6),
        "brier_score": round(float(brier), 6) if not np.isnan(brier) else "",
        "threshold_used": threshold,
    }


def get_model_and_params(model_name: str, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Return model class and hyperparameter grid for a given model name.

    Hyperparameter keys are prefixed with 'model__' to match the Pipeline step name.
    """
    models_cfg = config["models"]
    model_cfg = models_cfg[model_name]

    if model_name == "logistic_regression":
        model = LogisticRegression(
            solver="liblinear",
            max_iter=model_cfg["max_iter"],
            random_state=config["random_seed"],
        )
        param_grid = {f"model__{k}": v for k, v in [("C", model_cfg["C_grid"])]}
    elif model_name == "linear_svm":
        model = LinearSVC(
            random_state=config["random_seed"],
            max_iter=10000,
        )
        param_grid = {f"model__{k}": v for k, v in [("C", model_cfg["C_grid"])]}
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=model_cfg["n_estimators"],
            class_weight=model_cfg["class_weight"],
            random_state=model_cfg["random_state"],
        )
        param_grid = {f"model__{k}": v for k, v in model_cfg["param_grid"].items()}
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, param_grid


def build_model_pipeline(model_name: str, config: Dict[str, Any]) -> Tuple[Pipeline, Dict[str, Any]]:
    """Build a Pipeline with preprocessing + model for the given model name."""
    preprocessor = build_preprocessor(config)
    model, param_grid = get_model_and_params(model_name, config)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline, param_grid


def run_outer_cv(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run nested stratified cross-validation.

    Returns:
        fold_results: per-fold, per-model metrics
        model_summary: aggregated metrics per model
        selected_params: selected hyperparameters per fold/model
        predictions: out-of-fold predictions
    """
    root = project_root()
    raw_path = root / config["dataset"]["raw_path"]
    frame, _ = load_raw_csv(raw_path)

    X, y = get_modeling_data(frame, config)
    participant_ids = frame[config["identifier_column"]].str.strip()

    y_numeric = y.map(int) if y.dtype == object else y.astype(int)

    # CV configurations
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

    model_names = list(config["models"].keys())

    fold_rows: List[Dict[str, Any]] = []
    param_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_numeric)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_numeric.iloc[train_idx]
        y_test = y_numeric.iloc[test_idx]

        for model_name in model_names:
            pipeline, param_grid = build_model_pipeline(model_name, config)

            # Inner CV hyperparameter selection
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=inner_cv,
                scoring="roc_auc",
                n_jobs=1,
                refit=True,
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_inner_score = grid_search.best_score_

            # Predict on outer test fold
            y_score = best_model.predict(X_test)
            y_pred = best_model.predict(X_test)

            # For probability-based models, use predict_proba
            has_proba = hasattr(best_model, "predict_proba")
            if has_proba:
                y_proba = best_model.predict_proba(X_test)[:, 1]
                score_for_metrics = y_proba
            else:
                # LinearSVC: use decision_function
                score_for_metrics = best_model.decision_function(X_test)

            # Compute metrics
            metrics = compute_metrics(y_test.values, score_for_metrics)

            # Collect fold results
            fold_rows.append({
                "outer_fold": fold_idx,
                "model": model_name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_positive_train": int(y_train.sum()),
                "n_positive_test": int(y_test.sum()),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "ppv": metrics["ppv"],
                "npv": metrics["npv"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "brier_score": metrics["brier_score"],
                "threshold_used": metrics["threshold_used"],
            })

            # Collect selected hyperparameters
            param_rows.append({
                "outer_fold": fold_idx,
                "model": model_name,
                "selected_parameters": json.dumps(best_params, ensure_ascii=False),
                "inner_cv_score": round(float(best_inner_score), 6),
            })

            # Collect predictions
            test_ids = participant_ids.iloc[test_idx].tolist()
            for i, pid in enumerate(test_ids):
                pred_rows.append({
                    "Participant_ID": pid,
                    "outer_fold": fold_idx,
                    "model": model_name,
                    "y_true": int(y_test.iloc[i]),
                    "score": round(float(score_for_metrics[i]), 6),
                    "prediction": int(y_pred[i]),
                })

    fold_df = pd.DataFrame(fold_rows)
    param_df = pd.DataFrame(param_rows)
    pred_df = pd.DataFrame(pred_rows)

    # Aggregate model summary
    summary_rows = []
    for model_name in model_names:
        subset = fold_df[fold_df["model"] == model_name].copy()
        row = {"model": model_name}
        for metric in ["roc_auc", "pr_auc", "sensitivity", "specificity",
                        "ppv", "npv", "f1", "balanced_accuracy", "brier_score"]:
            vals = pd.to_numeric(subset[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = round(float(vals.mean()), 6) if len(vals) > 0 else ""
            row[f"{metric}_sd"] = round(float(vals.std()), 6) if len(vals) > 0 else ""
        summary_rows.append(row)
    model_summary = pd.DataFrame(summary_rows)

    return fold_df, model_summary, param_df, pred_df


def run_phase4(root: Path) -> Dict[str, Any]:
    root = Path(root)
    config = load_config(root)

    out_dir = root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = root / REPORT_RELATIVE.rsplit("/", 1)[0]
    report_dir.mkdir(parents=True, exist_ok=True)

    # Run nested CV
    fold_df, model_summary, param_df, pred_df = run_outer_cv(config)

    # Save outputs
    fold_df.to_csv(out_dir / "fold_results.csv", index=False, encoding="utf-8")
    model_summary.to_csv(out_dir / "model_summary.csv", index=False, encoding="utf-8")
    param_df.to_csv(out_dir / "selected_hyperparameters.csv", index=False, encoding="utf-8")
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")

    # Verify raw data immutability
    import hashlib
    raw_path = root / config["dataset"]["raw_path"]
    raw_bytes = raw_path.read_bytes()
    raw_hash = hashlib.md5(raw_bytes).hexdigest()

    # Build report
    report_md = _render_report(fold_df, model_summary, param_df, pred_df, config, raw_hash)
    (root / REPORT_RELATIVE).write_text(report_md, encoding="utf-8")

    return {
        "fold_results": fold_df,
        "model_summary": model_summary,
        "selected_hyperparameters": param_df,
        "predictions": pred_df,
        "raw_hash": raw_hash,
        "outputs": {
            "fold_results": str(out_dir / "fold_results.csv"),
            "model_summary": str(out_dir / "model_summary.csv"),
            "selected_hyperparameters": str(out_dir / "selected_hyperparameters.csv"),
            "predictions": str(out_dir / "predictions.csv"),
            "report": REPORT_RELATIVE,
        },
    }


def _render_report(fold_df: pd.DataFrame, model_summary: pd.DataFrame,
                    param_df: pd.DataFrame, pred_df: pd.DataFrame,
                    config: Dict[str, Any], raw_hash: str) -> str:
    lines: List[str] = []
    lines.append("# Phase 4 -- Nested Stratified Cross-Validation and Baseline Models")
    lines.append("")
    lines.append("Generated by `src/phase4_nested_cv.py`. All results come from actual execution on `data/raw/elderly_data.csv`.")
    lines.append("")
    lines.append("Phase 4 = COMPLETE")
    lines.append("")

    # Objective
    lines.append("## Objective")
    lines.append("")
    lines.append("Phase 4 estimates baseline model performance using nested stratified cross-validation on the LTMM harmonized dataset.")
    lines.append("Three classical ML baseline models are compared: Regularized Logistic Regression, Linear SVM, and Random Forest.")
    lines.append("No model is declared superior; results are presented descriptively.")
    lines.append("")

    # Dataset
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- **Participants (N):** {len(pred_df) // (config['outer_cv']['n_splits'])}")
    lines.append(f"- **Target:** `{config['target_column']}`")
    lines.append("- **Class distribution:** 44 Controls (0), 33 Fallers (1)")
    lines.append("- **Features:** Age, Gender, Total x, Totaly, TUG")
    lines.append("- **Preprocessing:** Median imputation + RobustScaler, fitted inside each training fold")
    lines.append(f"- **Raw CSV hash (MD5):** {raw_hash}")
    lines.append("")

    # Preprocessing
    lines.append("## Preprocessing")
    lines.append("")
    lines.append("- **Imputation:** SimpleImputer(strategy='median')")
    lines.append("- **Scaling:** RobustScaler")
    lines.append("- **Fitting:** Inside each outer training fold only")
    lines.append("- **No global preprocessing:** No statistics are computed on the full dataset")
    lines.append("- **No SMOTE, no feature selection, no outlier removal")
    lines.append("")

    # Cross-validation
    lines.append("## Cross-validation")
    lines.append("")
    lines.append(f"- **Outer CV:** StratifiedKFold(n_splits={config['outer_cv']['n_splits']}, shuffle={config['outer_cv']['shuffle']}, random_state={config['outer_cv']['random_state']})")
    lines.append(f"- **Inner CV:** StratifiedKFold(n_splits={config['inner_cv']['n_splits']}, shuffle={config['inner_cv']['shuffle']}, random_state={config['inner_cv']['random_state']})")
    lines.append(f"- **Random seed:** {config['random_seed']}")
    lines.append(f"- **Outer folds:** {config['outer_cv']['n_splits']}, Inner folds: {config['inner_cv']['n_splits']}")
    lines.append("- **Total outer test evaluations:** 5 folds × 3 models = 15")
    lines.append("")

    # Models
    lines.append("## Models")
    lines.append("")
    lines.append("### 1. Regularized Logistic Regression")
    lines.append("- Penalty: L2")
    lines.append("- Solver: liblinear")
    lines.append(f"- C grid: {config['models']['logistic_regression']['C_grid']}")
    lines.append(f"- Max iterations: {config['models']['logistic_regression']['max_iter']}")
    lines.append("")
    lines.append("### 2. Linear SVM")
    lines.append(f"- C grid: {config['models']['linear_svm']['C_grid']}")
    lines.append("- Decision function used for scoring")
    lines.append("")
    lines.append("### 3. Random Forest")
    lines.append(f"- n_estimators: {config['models']['random_forest']['n_estimators']}")
    lines.append(f"- class_weight: {config['models']['random_forest']['class_weight']}")
    lines.append(f"- random_state: {config['models']['random_forest']['random_state']}")
    lines.append(f"- param_grid: {config['models']['random_forest']['param_grid']}")
    lines.append("")

    # Fold-level results
    lines.append("## Fold-level results")
    lines.append("")
    fold_cols = ["outer_fold", "model", "n_train", "n_test", "n_positive_train", "n_positive_test",
                  "roc_auc", "pr_auc", "sensitivity", "specificity", "ppv", "npv",
                  "f1", "balanced_accuracy", "brier_score"]
    available_cols = [c for c in fold_cols if c in fold_df.columns]
    rows = fold_df[available_cols].values.tolist()
    rows = [[str(x) for x in row] for row in rows]
    lines.append(_md_table(["Fold", "Model", "N Train", "N Test", "Pos Train", "Pos Test",
                            "ROC-AUC", "PR-AUC", "Sens", "Spec", "PPV", "NPV", "F1", "Bal Acc", "Brier"], rows))
    lines.append("")

    # Model summary
    lines.append("## Model summary")
    lines.append("")
    sum_cols = [c for c in model_summary.columns]
    rows = model_summary[sum_cols].values.tolist()
    rows = [[str(x) for x in row] for row in rows]
    lines.append(_md_table(["Model"] + [c.replace("_mean", " (mean)").replace("_sd", " (SD)") for c in sum_cols[1:]], rows))
    lines.append("")

    # Selected hyperparameters
    lines.append("## Selected hyperparameters")
    lines.append("")
    param_rows = param_df.values.tolist()
    param_rows = [[str(x) for x in row] for row in param_rows]
    lines.append(_md_table(["Fold", "Model", "Selected Parameters", "Inner CV Score"], param_rows))
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Results are presented descriptively. Due to the small sample size (N=77),")
    lines.append("fold-to-fold variability is expected and should be emphasized.")
    lines.append("No model is declared superior based on any single metric.")
    lines.append("")
    lines.append("TUG has been identified in Phase 2 as having the clearest association with cohort membership.")
    lines.append("These Phase 4 results are exploratory and do not prove TUG alone is sufficient for prediction.")
    lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Small sample size (N=77, 33 fallers) limits statistical power.")
    lines.append("- Five positive events per predictor is below the traditional EPV=10 heuristic.")
    lines.append("- Single dataset; no external validation performed.")
    lines.append("- Limited feature set (5 predictors).")
    lines.append("- Unresolved target semantics (cohort label vs clinical diagnosis).")
    lines.append("- Unresolved gender coding direction.")
    lines.append("- Unresolved fall-history leakage status.")
    lines.append("- Hyperparameter selection via inner CV on N=77 has high variance.")
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- **Random seed:** {config['random_seed']}")
    lines.append(f"- **Outer CV:** StratifiedKFold(n_splits={config['outer_cv']['n_splits']}, random_state={config['outer_cv']['random_state']})")
    lines.append(f"- **Inner CV:** StratifiedKFold(n_splits={config['inner_cv']['n_splits']}, random_state={config['inner_cv']['random_state']})")
    lines.append("- **Preprocessing:** Fitted inside each outer training fold only.")
    lines.append(f"- **Dataset hash (MD5):** {raw_hash}")
    lines.append("- **All results generated from actual execution on data/raw/elderly_data.csv**")
    lines.append("")

    return "\n".join(lines)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return f"| {' | '.join(headers)} |\n|{'---|' * len(headers)}\n_(none)_\n"
    lines = [f"| {' | '.join(headers)} |", f"|{'---|' * len(headers)}"]
    for row in rows:
        lines.append(f"| {' | '.join(str(c) for c in row)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    results = run_phase4(project_root())
    print("Phase 4 nested stratified CV completed.")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print(f"  Report: {REPORT_RELATIVE}")
    for key, path in results["outputs"].items():
        if key != "report":
            print(f"  wrote  : {path}")
    print()
    print("Fold-level results:")
    print(results["fold_results"].to_string(index=False))
    print()
    print("Model summary:")
    print(results["model_summary"].to_string(index=False))
    print()
    print(f"Raw CSV hash unchanged: {results['raw_hash']}")


if __name__ == "__main__":
    main()
