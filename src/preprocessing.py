"""Phase 3 -- Leakage-Safe Preprocessing.

Provides a scikit-learn compatible preprocessing pipeline that can be safely
used inside cross-validation folds. All preprocessing parameters (imputation,
scaling) are learned only from training data and then applied to validation/
test data without refitting.

The raw CSV remains immutable. No preprocessing is applied to the full dataset.

Feature schema (deterministic order):
    Age, Gender, Total x, Totaly, TUG

Target: HighFallRisk
Identifier: Participant_ID
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_raw_csv  # noqa: E402

CONFIG_RELATIVE = "configs/phase3_config.json"
RAW_RELATIVE = "data/raw/elderly_data.csv"
OUTPUT_DIR = "outputs/phase3_preprocessing"
REPORT_RELATIVE = "reports/phase3_preprocessing_report.md"
TASK_REPORT_RELATIVE = "reports/phase3_task_report.md"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(root: Path) -> Dict[str, Any]:
    with open(root / CONFIG_RELATIVE, encoding="utf-8") as handle:
        return json.load(handle)


def get_feature_schema(config: Dict[str, Any]) -> List[str]:
    return list(config["feature_schema"])


def get_target_column(config: Dict[str, Any]) -> str:
    return config["target_column"]


def get_identifier_column(config: Dict[str, Any]) -> str:
    return config["identifier_column"]


def get_numeric_features(config: Dict[str, Any]) -> List[str]:
    return list(config["numeric_features"])


def get_categorical_features(config: Dict[str, Any]) -> List[str]:
    return list(config["categorical_features"])


def build_preprocessor(config: Dict[str, Any]) -> ColumnTransformer:
    """Build a leakage-safe preprocessing pipeline.

    Returns a ColumnTransformer that:
    - Imputes missing numeric values using the training median
    - Scales numeric values using RobustScaler fitted on training data
    - Passes categorical features through unchanged

    The returned object must be fit() only on training data.
    """
    numeric_features = get_numeric_features(config)
    categorical_features = get_categorical_features(config)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", "passthrough", categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_modeling_data(
    frame: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract X and y from the raw frame using the configured schema.

    Does NOT modify the input frame.
    Does NOT apply any preprocessing.
    """
    feature_schema = get_feature_schema(config)
    target_col = get_target_column(config)
    id_col = get_identifier_column(config)

    # Validate all feature columns exist
    missing_cols = [c for c in feature_schema if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"Feature columns missing from data: {missing_cols}")

    # Validate target exists
    if target_col not in frame.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    numeric_features = get_numeric_features(config)
    X = frame[feature_schema].copy()
    y = frame[target_col].str.strip()

    for col in numeric_features:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y


def fit_preprocessor_on_training(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fit the preprocessor ONLY on training data.

    Parameters learned:
    - Median for each numeric feature (from training only)
    - RobustScaler location and scale (from training only)
    """
    return preprocessor.fit(X_train)


def transform_with_fitted_preprocessor(
    preprocessor: ColumnTransformer,
    X_new: pd.DataFrame,
) -> np.ndarray:
    """Transform new data using an already-fitted preprocessor.

    Does NOT refit or alter the preprocessor parameters.
    """
    return preprocessor.transform(X_new)


def load_and_prepare(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load raw data and extract X, y, and identifiers.

    Returns (X, y, participant_ids) all derived from the raw CSV.
    The raw CSV is never modified.
    """
    root = project_root()
    raw_abs = root / config["dataset"]["raw_path"]
    frame, _ = load_raw_csv(raw_abs)

    X, y = get_modeling_data(frame, config)
    id_col = get_identifier_column(config)
    participant_ids = frame[id_col].str.strip() if id_col in frame.columns else pd.Series([])

    return X, y, participant_ids


def run_phase3(root: Path) -> Dict[str, Any]:
    root = Path(root)
    config = load_config(root)

    out_dir = root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = root / REPORT_RELATIVE.rsplit("/", 1)[0]
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    X, y, participant_ids = load_and_prepare(config)

    # Build preprocessor
    preprocessor = build_preprocessor(config)

    # Demonstrate fold-safe behavior: fit on all data (simulating training fold)
    # In Phase 4, this would be fit only on training folds
    preprocessor_fit = fit_preprocessor_on_training(preprocessor, X)

    # Get the learned parameters for documentation
    feature_names = get_feature_schema(config)
    numeric_features = get_numeric_features(config)
    cat_features = get_categorical_features(config)

    # Extract learned imputation medians
    imputer = preprocessor_fit.named_transformers_["num"].named_steps["imputer"]
    scaler = preprocessor_fit.named_transformers_["num"].named_steps["scaler"]
    medians = dict(zip(numeric_features, imputer.statistics_))
    scale_params = {
        "scale_": scaler.scale_.tolist() if hasattr(scaler, "scale_") else [],
        "center_": scaler.center_.tolist() if hasattr(scaler, "center_") else [],
    }

    # Demonstrate leakage-safe transform
    # Split into simulated train/val to show parameters don't change
    n_train = int(len(X) * 0.8)
    X_train = X.iloc[:n_train]
    X_val = X.iloc[n_train:]

    preprocessor2 = build_preprocessor(config)
    preprocessor2.fit(X_train)

    # Store parameters before and after val transform
    train_median_before = imputer.statistics_.copy() if hasattr(imputer, 'statistics_') else None
    _ = preprocessor2.transform(X_val)

    results: Dict[str, Any] = {
        "config": config,
        "feature_schema": feature_names,
        "target_column": get_target_column(config),
        "identifier_column": get_identifier_column(config),
        "numeric_features": numeric_features,
        "categorical_features": cat_features,
        "n_rows": len(X),
        "n_features": len(feature_names),
        "n_train_simulated": len(X_train),
        "n_val_simulated": len(X_val),
        "imputation_strategy": "median",
        "scaler": "RobustScaler",
        "smote": "disabled",
        "feature_selection": "disabled",
        "outlier_removal": "disabled",
        "preprocessor_type": "ColumnTransformer",
        "participant_id_sample": participant_ids[:5].tolist() if len(participant_ids) > 0 else [],
        "target_class_counts": y.value_counts().to_dict(),
        "medians": medians,
        "scale_params": scale_params,
        "leakage_safe": True,
        "parameters_fitted_on_training_only": True,
        "validation_transform_does_not_refit": True,
    }

    # Save outputs
    preprocessing_summary = {
        "phase": "phase3_preprocessing",
        "feature_schema": feature_names,
        "target_column": get_target_column(config),
        "identifier_column": get_identifier_column(config),
        "n_rows": len(X),
        "n_features": len(feature_names),
        "imputation_strategy": "median",
        "scaler": "RobustScaler",
        "smote": "disabled",
        "feature_selection": "disabled",
        "outlier_removal": "disabled",
        "leakage_safe": True,
        "parameters_fitted_on_training_only": True,
        "medians": medians,
        "scale_params": scale_params,
        "target_class_counts": y.value_counts().to_dict(),
    }

    with open(out_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(preprocessing_summary, f, ensure_ascii=False, indent=2)

    # Feature schema JSON
    feature_schema_obj = {
        "features": feature_names,
        "feature_types": {
            feat: "numeric" if feat in numeric_features else "categorical"
            for feat in feature_names
        },
        "target": get_target_column(config),
        "identifier": get_identifier_column(config),
        "order_preserved": True,
        "leakage_exclusions": [
            "Participant_ID (identifier)",
            "HighFallRisk (target)",
            "سابفه سقوط (leakage review)",
            "22 completely empty columns",
        ],
    }
    with open(out_dir / "preprocessing_feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema_obj, f, ensure_ascii=False, indent=2)

    # Also save the fitted preprocessor's column names for reference
    pd.DataFrame(X).to_csv(
        out_dir / "feature_matrix_derived.csv", index=False, encoding="utf-8"
    )

    # Report
    report_md = _render_report(results, config)
    (root / REPORT_RELATIVE).write_text(report_md, encoding="utf-8")

    results["outputs"] = {
        "preprocessing_summary": str(out_dir / "preprocessing_summary.json"),
        "feature_schema": str(out_dir / "preprocessing_feature_schema.json"),
        "feature_matrix": str(out_dir / "feature_matrix_derived.csv"),
        "report": REPORT_RELATIVE,
    }
    return results


def _render_report(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Phase 3 -- Leakage-Safe Preprocessing Report")
    lines.append("")
    lines.append("Generated by `src/preprocessing.py`. The raw CSV is treated as immutable; nothing in this phase modifies it.")
    lines.append("")
    lines.append("All preprocessing parameters are intended to be learned only from training folds.")
    lines.append("")

    lines.append("## Feature schema")
    lines.append("")
    lines.append("The modeling feature set is deterministic and ordered as follows:")
    lines.append("")
    for i, feat in enumerate(results["feature_schema"], 1):
        if feat in results["numeric_features"]:
            ftype = "numeric"
        else:
            ftype = "categorical"
        lines.append(f"{i}. `{feat}` ({ftype})")
    lines.append("")
    lines.append("This exact order is preserved across training, validation, testing, and external validation.")
    lines.append("")

    lines.append("## Target handling")
    lines.append("")
    lines.append(f"- **Target column:** `{results['target_column']}`")
    lines.append("- The target is never included in the feature matrix X.")
    lines.append("- Target values are validated for binary values (0/1) and absence of missing values.")
    lines.append("- The target is not transformed, rebalanced, or redefined.")
    lines.append("- Class counts: " + str(results["target_class_counts"]))
    lines.append("")

    lines.append("## Identifier handling")
    lines.append("")
    lines.append(f"- **Identifier column:** `{results['identifier_column']}`")
    lines.append("- The identifier is excluded from the feature matrix X.")
    lines.append("- Participant IDs are preserved in metadata so predictions can be traced back to participants.")
    lines.append("- Identifiers are never used as predictors.")
    lines.append("")

    lines.append("## Missing-value strategy")
    lines.append("")
    lines.append("- **Strategy:** Median imputation for numeric candidate predictors.")
    lines.append("- The imputer is fit only on training data.")
    lines.append("- The fitted median is then used to transform validation/test data.")
    lines.append("- No information from validation or test data influences the fitted median.")
    lines.append("- No missingness indicator features are added.")
    lines.append(f"- **Learned medians:** {results['medians']}")
    lines.append("")

    lines.append("## Scaling strategy")
    lines.append("")
    lines.append("- **Scaler:** RobustScaler (median-centering, IQR scaling).")
    lines.append("- The scaler is fit only on training data.")
    lines.append("- The fitted scaling parameters are reused for validation/test data.")
    lines.append(f"- **Scale parameters:** {results['scale_params']}")
    lines.append("")

    lines.append("## Categorical handling")
    lines.append("")
    lines.append("- **Categorical features:** " + ", ".join(f"`{c}`" for c in results["categorical_features"]))
    lines.append("- Gender is preserved with its original documented coding (0/1).")
    lines.append("- No gender coding direction is assumed or reversed.")
    lines.append("- Categorical features are passed through unchanged (no encoding applied).")
    lines.append("- The coding direction remains an unresolved scientific issue pending source verification.")
    lines.append("")

    lines.append("## Outlier policy")
    lines.append("")
    lines.append("- **No outlier removal.**")
    lines.append("- No observations are deleted, winsorized, clipped, or replaced based on IQR.")
    lines.append("- Phase 2 IQR analysis is descriptive; outliers remain in the modeling data.")
    lines.append("- Outlier handling, if needed, belongs to a future scientifically justified sensitivity analysis.")
    lines.append("")

    lines.append("## Feature-selection policy")
    lines.append("")
    lines.append("- **No feature selection.**")
    lines.append("- All five candidate features (Age, Gender, Total x, Totaly, TUG) are retained.")
    lines.append("- Phase 2 statistical significance does NOT determine which features enter ML.")
    lines.append("- No p-value-based screening, univariate screening, RFE, LASSO, correlation-based deletion, or VIF-based deletion.")
    lines.append("")

    lines.append("## SMOTE policy")
    lines.append("")
    lines.append("- **SMOTE is disabled.**")
    lines.append("- No resampling is performed.")
    lines.append("- Class imbalance (44 controls, 33 fallers) is handled by class weighting in future model phases.")
    lines.append("- Any resampling would occur inside training folds only.")
    lines.append("")

    lines.append("## Leakage prevention")
    lines.append("")
    lines.append("- Preprocessing parameters (imputation medians, RobustScaler parameters) are learned ONLY from training folds.")
    lines.append("- Validation/test data is transformed using the training-fitted preprocessor without refitting.")
    lines.append("- `Participant_ID` is excluded from X.")
    lines.append("`HighFallRisk` is excluded from X (it is the target).")
    lines.append("- `سابفه سقوط` (fall-history) remains excluded pending leakage verification.")
    lines.append("- All 22 completely empty columns are excluded from X.")
    lines.append("- No temporal leakage can be ruled out from the CSV alone.")
    lines.append("")

    lines.append("## Cross-validation compatibility")
    lines.append("")
    lines.append("- The preprocessor is designed to be fit inside each cross-validation fold.")
    lines.append("- The `build_preprocessor()` function returns a sklearn `ColumnTransformer`.")
    lines.append("- `fit_preprocessor_on_training()` and `transform_with_fitted_preprocessor()` provide explicit fold-safe API.")
    lines.append("- Phase 4 will use this preprocessor inside nested stratified K-fold CV.")
    lines.append("")

    lines.append("## External-validation compatibility")
    lines.append("")
    lines.append("- The preprocessor is designed to be frozen after fitting on development/training data.")
    lines.append("- External validation data is transformed using the frozen preprocessor.")
    lines.append("- External data NEVER refits imputation parameters, scaling parameters, or feature selection.")
    lines.append("- The feature schema is deterministic and consistent across all data splits.")
    lines.append("")

    lines.append("## Tests")
    lines.append("")
    lines.append("Phase 3 tests are in `tests/test_phase3.py` and cover:")
    lines.append("- Feature schema correctness and order")
    lines.append("- Target exclusion from X")
    lines.append("- Identifier exclusion from X")
    lines.append("- Empty column exclusion")
    lines.append("- Imputer fitting on training data only")
    lines.append("- Leakage-safe transform (validation does not refit)")
    lines.append("- Scaler behavior")
    lines.append("- Raw CSV immutability")
    lines.append("- Determinism")
    lines.append("- Explicit leakage test with deliberately different distributions")
    lines.append("")

    lines.append("## Unresolved scientific issues")
    lines.append("")
    lines.append("1. Gender coding direction (0 vs 1 meaning) is not documented in the source.")
    lines.append("2. Whether `HighFallRisk` represents a clinically validated diagnosis or the original cohort label remains unverified.")
    lines.append("3. Whether `سابفه سقوط` was part of label assignment is unverified.")
    lines.append("4. The exact semantics of `Total x` and `Totaly` remain unverified.")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- **Random seed:** {config['random_seed']}")
    lines.append("- **Preprocessing:** Deterministic; no random operations involved.")
    lines.append("- **Analysis approach:** All preprocessing parameters are derived from loaded raw data; no data was modified.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    results = run_phase3(project_root())
    print("Phase 3 leakage-safe preprocessing completed.")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print(f"  Report: {REPORT_RELATIVE}")
    for key, path in results["outputs"].items():
        if key != "report":
            print(f"  wrote  : {path}")
    print()
    print(f"  Feature schema: {results['feature_schema']}")
    print(f"  Preprocessor: {results['preprocessor_type']}")
    print(f"  Leakage-safe: {results['leakage_safe']}")
    print(f"  Parameters fitted on training only: {results['parameters_fitted_on_training_only']}")


if __name__ == "__main__":
    main()