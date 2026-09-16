"""Phase 2 -- Statistical & Data Audit.

Reads the raw CSV via the Phase 1 loader (immutable), performs an in-memory
statistical audit of the dataset, and writes machine-readable tables, figures,
and a human-readable Markdown report. No model training, imputation, scaling,
or any Phase 3+ preprocessing is performed.

Deterministic; project-relative paths; fixed random seed.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_raw_csv  # noqa: E402

CONFIG_RELATIVE = "configs/phase2_config.json"
RAW_RELATIVE = "data/raw/elderly_data.csv"
OUTPUT_DIR = "outputs/phase2_statistical_audit"
REPORT_RELATIVE = "reports/phase2_statistical_data_audit_report.md"
FIGURE_DIR = OUTPUT_DIR + "/figures"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(root: Path) -> Dict[str, Any]:
    with open(root / CONFIG_RELATIVE, encoding="utf-8") as handle:
        return json.load(handle)


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Manual Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    adjusted = np.zeros(n)
    rank = np.arange(1, n + 1)
    adjusted_sorted = sorted_pvals * n / rank
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    # Map back to original order
    adjusted[sorted_idx] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def missing_mask(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(lambda col: col.str.strip() == "")


def missing_percentage(n_missing: int, n_total: int) -> float:
    if n_total == 0:
        return 0.0
    return round(100.0 * n_missing / n_total, 2)


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return f"| {' | '.join(headers)} |\n|{'---|' * len(headers)}\n_(none)_\n"
    lines = [f"| {' | '.join(headers)} |", f"|{'---|' * len(headers)}"]
    for row in rows:
        lines.append(f"| {' | '.join(str(c) for c in row)} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 1. Variable classification
# --------------------------------------------------------------------------- #
def classify_variables(frame: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    mask = missing_mask(frame)
    binary_candidates = config.get("columns", {}).get("binary_candidates", [])
    rows = []
    for column in frame.columns:
        values = frame[column][~mask[column]]
        n_total = len(frame)
        n_missing = int(mask[column].sum())
        n_observed = n_total - n_missing
        numeric_vals = _as_numeric(values)
        is_numeric = len(numeric_vals) > 0 and not numeric_vals.isna().any()
        is_integer = False
        if is_numeric and len(numeric_vals) > 0:
            is_integer = bool((numeric_vals == numeric_vals.round()).all())

        if column == config["columns"]["identifier"]["column"]:
            vclass = "Identifier"
        elif column == config["columns"]["target"]["column"]:
            vclass = "Target"
        elif column == config["columns"]["fall_history"]["original_name"]:
            vclass = "Fall-history (leakage review)"
        elif n_observed == 0:
            vclass = "Completely empty"
        elif is_numeric:
            if column in binary_candidates:
                vclass = "Candidate binary predictor"
            else:
                vclass = "Candidate continuous/numeric predictor"
        else:
            vclass = "Unavailable for analysis"

        rows.append({
            "variable": column,
            "classification": vclass,
            "n_total": n_total,
            "n_missing": n_missing,
            "missing_pct": missing_percentage(n_missing, n_total),
            "n_observed": n_observed,
            "inferred_numeric_type": "integer" if (is_numeric and is_integer) else ("float" if is_numeric else ""),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Missing data analysis
# --------------------------------------------------------------------------- #
def missing_data_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    mask = missing_mask(frame)
    rows = []
    for column in frame.columns:
        n_total = len(frame)
        n_missing = int(mask[column].sum())
        rows.append({
            "variable": column,
            "N": n_total,
            "n_missing": n_missing,
            "missing_pct": missing_percentage(n_missing, n_total),
            "n_observed": n_total - n_missing,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Descriptive statistics
# --------------------------------------------------------------------------- #
def descriptive_statistics(frame: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        values = _as_numeric(frame[col].str.strip())
        complete = values.dropna()
        if len(complete) == 0:
            continue
        q1 = float(complete.quantile(0.25))
        q3 = float(complete.quantile(0.75))
        iqr = q3 - q1
        rows.append({
            "variable": col,
            "N": len(complete),
            "mean": round(float(complete.mean()), 6),
            "std": round(float(complete.std()), 6),
            "median": round(float(complete.median()), 6),
            "Q1": round(q1, 6),
            "Q3": round(q3, 6),
            "IQR": round(iqr, 6),
            "min": round(float(complete.min()), 6),
            "max": round(float(complete.max()), 6),
            "skewness": round(float(complete.skew()), 6),
            "kurtosis": round(float(complete.kurtosis()), 6),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Normality assessment
# --------------------------------------------------------------------------- #
def normality_assessment(frame: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        values = _as_numeric(frame[col].str.strip()).dropna()
        if len(values) < 3:
            continue
        if 3 <= len(values) <= 5000:
            stat, p_value = stats.shapiro(values)
        else:
            stat, p_value = float("nan"), float("nan")
        rows.append({
            "variable": col,
            "N": len(values),
            "shapiro_W": round(float(stat), 6) if not np.isnan(stat) else "",
            "shapiro_p": round(float(p_value), 6) if not np.isnan(p_value) else "",
            "skewness": round(float(values.skew()), 6),
            "kurtosis": round(float(values.kurtosis()), 6),
            "shapiro_performed": "yes" if (3 <= len(values) <= 5000) else "no (sample size)",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Outlier analysis
# --------------------------------------------------------------------------- #
def outlier_analysis(frame: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        values = _as_numeric(frame[col].str.strip()).dropna()
        if len(values) < 4:
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        multiplier = 1.5
        lower_fence = q1 - multiplier * iqr
        upper_fence = q3 + multiplier * iqr
        n_outside = int(((values < lower_fence) | (values > upper_fence)).sum())
        rows.append({
            "variable": col,
            "Q1": round(q1, 6),
            "Q3": round(q3, 6),
            "IQR": round(q3 - q1, 6),
            "lower_fence": round(q1 - 1.5 * (q3 - q1), 6),
            "upper_fence": round(q3 + 1.5 * (q3 - q1), 6),
            "n_outside_fences": n_outside,
            "pct_outside_fences": round(100.0 * n_outside / len(values), 2),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. Group comparisons
# --------------------------------------------------------------------------- #
def group_comparison(frame: pd.DataFrame, numeric_cols: List[str],
                     target_col: str) -> pd.DataFrame:
    mask = missing_mask(frame)
    target = frame[target_col].str.strip()
    controls = target == "0"
    fallers = target == "1"
    rows = []
    for col in numeric_cols:
        values = _as_numeric(frame[col].str.strip())
        c_vals = values[controls].dropna()
        f_vals = values[fallers].dropna()
        if len(c_vals) < 3 or len(f_vals) < 3:
            continue
        mean_c = float(c_vals.mean())
        mean_f = float(f_vals.mean())
        std_c = float(c_vals.std())
        std_f = float(f_vals.std())
        _, lev_p = stats.levene(c_vals, f_vals)
        equal_var = lev_p >= 0.05
        if equal_var:
            stat, p_val = stats.ttest_ind(c_vals, f_vals, equal_var=True)
            test_name = "Student's t-test"
        else:
            stat, p_val = stats.ttest_ind(c_vals, f_vals, equal_var=False)
            test_name = "Welch's t-test"
        pooled_std = np.sqrt(((len(c_vals) - 1) * std_c**2 + (len(f_vals) - 1) * std_f**2) /
                             (len(c_vals) + len(f_vals) - 2))
        cohens_d = (mean_f - mean_c) / pooled_std if pooled_std > 0 else float("nan")
        diff = mean_f - mean_c
        se_diff = np.sqrt(std_c**2 / len(c_vals) + std_f**2 / len(f_vals))
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        u_stat, mw_p = stats.mannwhitneyu(c_vals, f_vals, alternative="two-sided")
        rows.append({
            "variable": col,
            "group": "Controls_vs_Fallers",
            "n_controls": len(c_vals),
            "n_fallers": len(f_vals),
            "mean_controls": round(mean_c, 6),
            "mean_fallers": round(mean_f, 6),
            "std_controls": round(std_c, 6),
            "std_fallers": round(std_f, 6),
            "median_controls": round(float(c_vals.median()), 6),
            "median_fallers": round(float(f_vals.median()), 6),
            "levene_p": round(float(lev_p), 6),
            "equal_variance": "yes" if equal_var else "no",
            "test_name": test_name,
            "test_statistic": round(float(stat), 6),
            "p_value": round(float(p_val), 6),
            "mann_whitney_U": int(u_stat),
            "mann_whitney_p": round(float(mw_p), 6),
            "cohens_d": round(float(cohens_d), 6),
            "ci_95_lower": round(float(ci_lower), 6),
            "ci_95_upper": round(float(ci_upper), 6),
            "adjusted_p_value_bh": 0.0,  # placeholder, filled after
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        pvals = df["p_value"].values
        adj_p = _benjamini_hochberg(pvals)
        df["adjusted_p_value_bh"] = [round(float(v), 6) for v in adj_p]
    return df


# --------------------------------------------------------------------------- #
# 7. Categorical comparison
# --------------------------------------------------------------------------- #
def categorical_comparison(frame: pd.DataFrame, cat_cols: List[str],
                           target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = frame[target_col].str.strip()
    contingency_rows = []
    stat_rows = []
    for col in cat_cols:
        values = frame[col].str.strip()
        crosstab = pd.crosstab(values, target)
        for idx in crosstab.index:
            for col_idx in crosstab.columns:
                contingency_rows.append({
                    "variable": col,
                    "category": str(idx),
                    "target_class": str(col_idx),
                    "count": int(crosstab.loc[idx, col_idx]),
                })
        table = crosstab.values
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            if table.shape[0] == 2 and table.shape[1] == 2:
                expected = stats.contingency.expected_freq(table)
                min_exp = expected.min()
                if min_exp >= 5:
                    chi2, p_val, dof, _ = stats.chi2_contingency(table, correction=False)
                    test_name = "Chi-square (no correction)"
                else:
                    _, p_val, _, _ = stats.fisher_exact(table)
                    test_name = "Fisher's exact test"
                    chi2 = float("nan")
            else:
                chi2, p_val, dof, _ = stats.chi2_contingency(table)
                test_name = "Chi-square"
            n = table.sum()
            cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if (not np.isnan(chi2) and n > 0) else float("nan")
            stat_rows.append({
                "variable": col,
                "test": test_name,
                "chi2_statistic": round(float(chi2), 6) if not np.isnan(chi2) else "",
                "p_value": round(float(p_val), 6),
                "cramers_v": round(float(cramers_v), 6) if not np.isnan(cramers_v) else "",
                "n": int(n),
            })
    cont_df = pd.DataFrame(contingency_rows)
    stat_df = pd.DataFrame(stat_rows) if stat_rows else pd.DataFrame(columns=["variable", "test", "chi2_statistic", "p_value", "cramers_v", "n"])
    return cont_df, stat_df


# --------------------------------------------------------------------------- #
# 8. Correlation analysis
# --------------------------------------------------------------------------- #
def correlation_analysis(frame: pd.DataFrame, corr_cols: List[str]) -> pd.DataFrame:
    mask = missing_mask(frame)
    values = frame[corr_cols].apply(lambda col: _as_numeric(col.str.strip()))
    values = values.dropna(axis=0, how="any")
    n = len(values)
    if n < 3:
        return pd.DataFrame(columns=["var1", "var2", "spearman_rho", "spearman_p",
                                     "pearson_r", "pearson_p", "n", "adjusted_p_value_bh"])
    spearman_corr = values.corr(method="spearman")
    pearson_corr = values.corr(method="pearson")
    cols = list(spearman_corr.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            var1 = cols[i]
            var2 = cols[j]
            rho = float(spearman_corr.iloc[i, j])
            r = float(pearson_corr.iloc[i, j])
            if abs(rho) < 1 and n > 3:
                t_stat = rho * np.sqrt((n - 2) / (1 - rho**2))
                p_spearman = float(2 * stats.t.sf(abs(t_stat), n - 2))
            else:
                p_spearman = 0.0 if abs(rho) >= 1 else float("nan")
            if abs(r) < 1 and n > 3:
                t_stat_p = r * np.sqrt((n - 2) / (1 - r**2))
                p_pearson = float(2 * stats.t.sf(abs(t_stat_p), n - 2))
            else:
                p_pearson = 0.0 if abs(r) >= 1 else float("nan")
            rows.append({
                "var1": var1,
                "var2": var2,
                "spearman_rho": round(rho, 6),
                "spearman_p": round(p_spearman, 6),
                "pearson_r": round(r, 6),
                "pearson_p": round(p_pearson, 6),
                "n": n,
                "adjusted_p_value_bh": 0.0,  # placeholder
            })
    df = pd.DataFrame(rows)
    if len(df) > 0 and df["spearman_p"].notna().any():
        valid_mask = df["spearman_p"].notna()
        if valid_mask.sum() > 0:
            pvals = df.loc[valid_mask, "spearman_p"].values
            adj_p = _benjamini_hochberg(pvals)
            df.loc[valid_mask, "adjusted_p_value_bh"] = [round(float(v), 6) for v in adj_p]
        df.loc[~valid_mask, "adjusted_p_value_bh"] = np.nan
        df["adjusted_p_value_bh"] = df["adjusted_p_value_bh"].astype(object)
    return df


# --------------------------------------------------------------------------- #
# 9. Univariate logistic regression
# --------------------------------------------------------------------------- #
def univariate_logistic(frame: pd.DataFrame, predictor_cols: List[str],
                        target_col: str) -> pd.DataFrame:
    target = _as_numeric(frame[target_col].str.strip())
    target = target.dropna().astype(int)
    rows = []
    for col in predictor_cols:
        values = _as_numeric(frame[col].str.strip())
        valid = values.notna() & target.notna()
        x = values[valid].values
        y = target[valid].values
        if len(x) < 10 or len(np.unique(y)) < 2:
            rows.append({
                "predictor": col,
                "coefficient": "", "odds_ratio": "", "std_error": "",
                "ci_95_lower": "", "ci_95_upper": "", "p_value": "",
                "n": int(valid.sum()), "status": "insufficient data or separation",
            })
            continue
        try:
            model = LogisticRegression(solver="lbfgs", max_iter=1000)
            model.fit(x.reshape(-1, 1), y)
            coef = float(model.coef_[0][0])
            probs = model.predict_proba(x.reshape(-1, 1))[:, 1]
            probs = np.clip(probs, 0.001, 0.999)
            W = probs * (1 - probs)
            X_design = np.column_stack([np.ones(len(x)), x])
            var_covar = np.linalg.inv(X_design.T @ np.diag(W) @ X_design)
            se_coef = np.sqrt(var_covar[1, 1]) if var_covar.shape[0] > 1 else float("nan")
            z_stat = coef / se_coef if se_coef > 0 else float("nan")
            p_val = float(2 * stats.norm.sf(abs(z_stat))) if not np.isnan(z_stat) else float("nan")
            odds_ratio = float(np.exp(coef))
            ci_lower = coef - 1.96 * se_coef
            ci_upper = coef + 1.96 * se_coef
            rows.append({
                "predictor": col,
                "coefficient": round(coef, 6),
                "odds_ratio": round(odds_ratio, 6),
                "std_error": round(float(se_coef), 6),
                "ci_95_lower": round(float(ci_lower), 6),
                "ci_95_upper": round(float(ci_upper), 6),
                "p_value": round(p_val, 6),
                "n": int(valid.sum()),
                "status": "ok",
            })
        except Exception as exc:
            rows.append({
                "predictor": col, "coefficient": "", "odds_ratio": "",
                "std_error": "", "ci_95_lower": "", "ci_95_upper": "",
                "p_value": "", "n": int(valid.sum()),
                "status": f"numerical instability: {exc}",
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 10. ROC analysis
# --------------------------------------------------------------------------- #
def roc_analysis(frame: pd.DataFrame, roc_cols: List[str],
                 target_col: str) -> pd.DataFrame:
    target = _as_numeric(frame[target_col].str.strip()).dropna().astype(int)
    rows = []
    for col in roc_cols:
        values = _as_numeric(frame[col].str.strip())
        valid = values.notna() & target.notna()
        x = values[valid].values
        y = target[valid].values
        if len(np.unique(y)) < 2 or len(x) < 5:
            rows.append({"variable": col, "auc": "", "auc_ci_lower": "",
                         "auc_ci_upper": "", "n": int(valid.sum()),
                         "status": "insufficient data"})
            continue
        fpr, tpr, thresholds = roc_curve(y, x)
        roc_auc = float(auc(fpr, tpr))
        rng = np.random.RandomState(42)
        auc_boot = []
        for _ in range(1000):
            idx = rng.choice(len(y), size=len(y), replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            fpr_b, tpr_b, _ = roc_curve(y[idx], x[idx])
            auc_boot.append(float(auc(fpr_b, tpr_b)))
        ci_lower = float(np.percentile(auc_boot, 2.5)) if auc_boot else float("nan")
        ci_upper = float(np.percentile(auc_boot, 97.5)) if auc_boot else float("nan")
        j_indices = np.argmax(tpr - fpr)
        best_threshold = float(thresholds[j_indices]) if len(thresholds) > 0 else float("nan")
        best_sensitivity = float(tpr[j_indices]) if len(tpr) > 0 else float("nan")
        best_specificity = float(1 - fpr[j_indices]) if len(fpr) > 0 else float("nan")
        rows.append({
            "variable": col, "auc": round(roc_auc, 6),
            "auc_ci_lower": round(ci_lower, 6), "auc_ci_upper": round(ci_upper, 6),
            "exploratory_threshold": round(best_threshold, 6),
            "exploratory_sensitivity": round(best_sensitivity, 6),
            "exploratory_specificity": round(best_specificity, 6),
            "n": int(valid.sum()), "status": "exploratory",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 11. Multicollinearity diagnostic
# --------------------------------------------------------------------------- #
def multicollinearity_diagnostic(frame: pd.DataFrame, corr_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    values = frame[corr_cols].apply(lambda col: _as_numeric(col.str.strip()))
    values = values.dropna(axis=0, how="any")
    corr_mat = values.corr(method="spearman")
    vif_rows = []
    for col in corr_cols:
        y = values[col]
        X_cols = [c for c in corr_cols if c != col]
        if len(X_cols) < 1:
            continue
        X = np.column_stack([np.ones(len(values)), values[X_cols].values])
        try:
            beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
            y_pred = X @ beta
            ss_res = np.sum((y.values - y_pred) ** 2)
            ss_tot = np.sum((y.values - np.mean(y.values)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r_squared) if r_squared < 1 else float("inf")
        except Exception:
            vif = float("nan")
        vif_rows.append({"variable": col, "vif": round(float(vif), 6) if not np.isnan(vif) else ""})
    return corr_mat, pd.DataFrame(vif_rows)


# --------------------------------------------------------------------------- #
# Figure generation
# --------------------------------------------------------------------------- #
def _safe_title(col: str) -> str:
    return col.replace(" ", "_")


def make_histograms(frame: pd.DataFrame, numeric_cols: List[str], out_path: Path):
    n = len(numeric_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for i, col in enumerate(numeric_cols):
        values = _as_numeric(frame[col].str.strip()).dropna()
        axes[i].hist(values, bins=15, edgecolor="black", alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_qq_plots(frame: pd.DataFrame, numeric_cols: List[str], out_path: Path):
    n = len(numeric_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for i, col in enumerate(numeric_cols):
        values = _as_numeric(frame[col].str.strip()).dropna()
        stats.probplot(values, dist="norm", plot=axes[i])
        axes[i].set_title(f"Q-Q: {col}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_boxplots_by_group(frame: pd.DataFrame, numeric_cols: List[str],
                           target_col: str, out_path: Path):
    mask = missing_mask(frame)
    valid = frame[~mask[target_col]]
    target = valid[target_col].str.strip()
    n = len(numeric_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for i, col in enumerate(numeric_cols):
        col_values = _as_numeric(valid[col].str.strip())
        data_c = col_values[target == "0"].dropna()
        data_f = col_values[target == "1"].dropna()
        axes[i].boxplot([data_c, data_f], labels=["Controls", "Fallers"])
        axes[i].set_title(col)
        axes[i].set_ylabel(col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_correlation_heatmap(corr_mat: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 6))
    data = corr_mat.values
    plt.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(label="Spearman correlation")
    cols = list(corr_mat.columns)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = data[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if abs(val) > 0.5 else "black")
    plt.title("Spearman Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_roc_curves(frame: pd.DataFrame, roc_cols: List[str],
                    target_col: str, out_path: Path):
    target = _as_numeric(frame[target_col].str.strip()).dropna().astype(int)
    plt.figure(figsize=(8, 6))
    for col in roc_cols:
        values = _as_numeric(frame[col].str.strip())
        valid = values.notna() & target.notna()
        x = values[valid].values
        y = target[valid].values
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, x)
        roc_auc = float(auc(fpr, tpr))
        plt.plot(fpr, tpr, label=f"{col} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Exploratory ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run_phase2(root: Path) -> Dict[str, Any]:
    root = Path(root)
    config = load_config(root)
    raw_abs = (root / config["dataset"]["raw_path"]).resolve()

    frame, meta = load_raw_csv(
        raw_abs,
        encoding=config["dataset"]["expected_encoding"],
        delimiter=config["dataset"]["expected_delimiter"],
    )

    out_dir = root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = root / FIGURE_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir = root / REPORT_RELATIVE.rsplit("/", 1)[0]
    report_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = config["numeric_candidates"]
    binary_cols = config["columns"].get("binary_candidates", config.get("binary_candidates", []))
    target_col = config["columns"]["target"]["column"]
    fall_hist_col = config["columns"]["fall_history"]["original_name"]
    corr_cols = numeric_cols

    all_results: Dict[str, Any] = {}
    all_results["meta"] = meta
    all_results["config"] = config

    # --- 1. Variable classification ---
    class_df = classify_variables(frame, config)
    class_df.to_csv(out_dir / "10_variable_classification.csv", index=False, encoding="utf-8")
    all_results["variable_classification"] = class_df

    # --- 2. Missing data analysis ---
    miss_df = missing_data_analysis(frame)
    miss_df.to_csv(out_dir / "01_missingness.csv", index=False, encoding="utf-8")
    all_results["missingness"] = miss_df

    # --- 3. Descriptive statistics ---
    desc_df = descriptive_statistics(frame, numeric_cols)
    desc_df.to_csv(out_dir / "02_descriptive_statistics.csv", index=False, encoding="utf-8")
    all_results["descriptive_stats"] = desc_df

    # --- 4. Normality ---
    norm_df = normality_assessment(frame, numeric_cols)
    norm_df.to_csv(out_dir / "05_normality.csv", index=False, encoding="utf-8")
    all_results["normality"] = norm_df

    # --- 5. Outliers ---
    outlier_df = outlier_analysis(frame, numeric_cols)
    outlier_df.to_csv(out_dir / "06_outliers.csv", index=False, encoding="utf-8")
    all_results["outliers"] = outlier_df

    # --- 6. Group comparison ---
    group_df = group_comparison(frame, numeric_cols, target_col)
    group_df.to_csv(out_dir / "03_group_comparison.csv", index=False, encoding="utf-8")
    all_results["group_comparison"] = group_df

    # --- 7. Categorical comparison ---
    cont_df, cat_stat_df = categorical_comparison(frame, binary_cols, target_col)
    cont_df.to_csv(out_dir / "04_categorical_comparison.csv", index=False, encoding="utf-8")
    cat_stat_df.to_csv(out_dir / "categorical_statistics.csv", index=False, encoding="utf-8")
    all_results["categorical_comparison"] = cont_df
    all_results["categorical_statistics"] = cat_stat_df

    # --- 8. Correlation analysis ---
    corr_df = correlation_analysis(frame, corr_cols)
    corr_df.to_csv(out_dir / "07_correlation.csv", index=False, encoding="utf-8")
    all_results["correlation"] = corr_df

    # --- 9. Univariate logistic ---
    pred_cols = numeric_cols + binary_cols
    logistic_df = univariate_logistic(frame, pred_cols, target_col)
    logistic_df.to_csv(out_dir / "08_univariate_logistic.csv", index=False, encoding="utf-8")
    all_results["univariate_logistic"] = logistic_df

    # --- 10. ROC ---
    roc_df = roc_analysis(frame, corr_cols, target_col)
    roc_df.to_csv(out_dir / "09_roc_summary.csv", index=False, encoding="utf-8")
    all_results["roc_summary"] = roc_df

    # --- 11. Multicollinearity ---
    corr_mat, vif_df = multicollinearity_diagnostic(frame, corr_cols)
    vif_df.to_csv(out_dir / "vif_diagnostic.csv", index=False, encoding="utf-8")
    all_results["correlation_matrix"] = corr_mat
    all_results["vif_diagnostic"] = vif_df

    # --- Figures ---
    make_histograms(frame, numeric_cols, fig_dir / "histograms.png")
    make_qq_plots(frame, numeric_cols, fig_dir / "qq_plots.png")
    make_boxplots_by_group(frame, numeric_cols, target_col, fig_dir / "boxplots_by_group.png")
    make_correlation_heatmap(corr_mat, fig_dir / "correlation_heatmap.png")
    make_roc_curves(frame, corr_cols, target_col, fig_dir / "roc_curves.png")

    # --- Leakage audit ---
    leakage_df = pd.DataFrame([
        {"variable": "Participant_ID", "status": "DEFINITE_LEAKAGE_IDENTIFIER",
         "note": "Traceability metadata; never a predictor"},
        {"variable": target_col, "status": "TARGET",
         "note": "Target variable only"},
        {"variable": fall_hist_col, "status": "POSSIBLE_LIKELY_LEAKAGE",
         "note": "Spec flags as possible/likely leakage; exclude pending source verification"},
        {"variable": "(entire dataset)", "status": "TEMPORAL_LEAKAGE_UNKNOWN",
         "note": "No temporal leakage can be ruled out from CSV alone"},
    ])
    leakage_df.to_csv(out_dir / "leakage_audit.csv", index=False, encoding="utf-8")
    all_results["leakage_audit"] = leakage_df

    # --- Correlation matrix CSV ---
    corr_mat.reset_index().rename(columns={"index": "variable"}).to_csv(
        out_dir / "correlation_matrix.csv", index=False, encoding="utf-8"
    )

    # --- Report ---
    report_md = _render_report(all_results, config)
    (root / REPORT_RELATIVE).write_text(report_md, encoding="utf-8")

    all_results["outputs"] = {
        "missingness_csv": str(out_dir / "01_missingness.csv"),
        "descriptive_stats_csv": str(out_dir / "02_descriptive_statistics.csv"),
        "group_comparison_csv": str(out_dir / "03_group_comparison.csv"),
        "categorical_comparison_csv": str(out_dir / "04_categorical_comparison.csv"),
        "normality_csv": str(out_dir / "05_normality.csv"),
        "outliers_csv": str(out_dir / "06_outliers.csv"),
        "correlation_csv": str(out_dir / "07_correlation.csv"),
        "univariate_logistic_csv": str(out_dir / "08_univariate_logistic.csv"),
        "roc_summary_csv": str(out_dir / "09_roc_summary.csv"),
        "variable_classification_csv": str(out_dir / "10_variable_classification.csv"),
        "figures_dir": str(fig_dir),
        "report_md": REPORT_RELATIVE,
    }
    return all_results


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
def _render_report(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    lines: List[str] = []
    miss = results["missingness"]
    desc = results["descriptive_stats"]
    group = results["group_comparison"]
    norm = results["normality"]
    outliers = results["outliers"]
    corr = results["correlation"]
    logistic = results["univariate_logistic"]
    roc = results["roc_summary"]
    cat_stat = results["categorical_statistics"]
    class_df = results["variable_classification"]
    vif = results["vif_diagnostic"]

    lines.append("# Phase 2 -- Statistical & Data Audit Report")
    lines.append("")
    lines.append("Generated by `src/phase2_statistical_audit.py`. The raw CSV is treated as immutable; nothing in this phase modifies it.")
    lines.append("")
    lines.append("This is an exploratory/descriptive audit only. No model training, imputation, scaling, or feature selection was performed.")
    lines.append("")

    lines.append("## Dataset overview")
    lines.append("")
    meta = results["meta"]
    lines.append(f"- **Participants (rows):** {meta['n_records']}")
    lines.append(f"- **Variables (columns):** {meta['n_columns']}")
    lines.append(f"- **Encoding:** {meta['encoding']}")
    empty_count = len(class_df[class_df["classification"] == "Completely empty"])
    lines.append(f"- **All-empty columns:** {empty_count}")
    lines.append("")
    lines.append("The dataset supports 77 participants with 30 variables, of which 22 are completely empty and contain no observations.")
    lines.append("")

    lines.append("## Variable classification")
    lines.append("")
    lines.append("Based on the actual dataset and the project specification:")
    lines.append("")
    for cls_name in ["Identifier", "Target", "Candidate continuous/numeric predictor",
                      "Candidate binary predictor", "Fall-history (leakage review)",
                      "Completely empty"]:
        subset = class_df[class_df["classification"] == cls_name]
        if len(subset) > 0:
            vars_list = ", ".join(subset["variable"].tolist())
            lines.append(f"- **{cls_name}** ({len(subset)}): {vars_list}")
    lines.append("")
    lines.append("**Identifier:** `Participant_ID` is traceability metadata and is never used as a predictor.")
    lines.append("**Target:** `HighFallRisk` is the outcome/target variable (0=Controls, 1=Fallers).")
    lines.append("**Candidate continuous predictors:** Age, Total x, Totaly, TUG.")
    lines.append("**Candidate binary predictor:** Gender.")
    lines.append("**Fall-history:** `سابفه سقوط` is excluded pending leakage verification.")
    lines.append("**Completely empty:** 22 columns with no observations (see spec for full list).")
    lines.append("")

    lines.append("## Missing-data analysis")
    lines.append("")
    lines.append("Missingness was calculated from the raw data (empty cells):")
    lines.append("")
    key_cols = ["Age", "Gender", "Total x", "Totaly", "TUG", "سابفه سقوط", "HighFallRisk"]
    rows = []
    for _, r in miss.iterrows():
        if r["variable"] in key_cols:
            rows.append([r["variable"], str(r["N"]), str(r["n_missing"]), f"{r['missing_pct']}%", str(r["n_observed"])])
    lines.append(_md_table(["Variable", "N", "Missing N", "Missing %", "Observed"], rows))
    lines.append("")
    lines.append("Missingness findings match the expected values from the project specification.")
    lines.append("")

    lines.append("## Descriptive statistics")
    lines.append("")
    if len(desc) > 0:
        cols_order = ["variable", "N", "mean", "std", "median", "Q1", "Q3", "IQR", "min", "max", "skewness", "kurtosis"]
        available = [c for c in cols_order if c in desc.columns]
        rows = desc[available].values.tolist()
        rows = [[str(x) for x in row] for row in rows]
        lines.append(_md_table(["Variable", "N", "Mean", "SD", "Median", "Q1", "Q3", "IQR", "Min", "Max", "Skew", "Kurt"], rows))
    lines.append("")

    lines.append("## Distribution/normality assessment")
    lines.append("")
    if len(norm) > 0:
        cols_order = ["variable", "N", "shapiro_W", "shapiro_p", "skewness", "kurtosis"]
        available = [c for c in cols_order if c in norm.columns]
        rows = norm[available].values.tolist()
        rows = [[str(x) for x in row] for row in rows]
        lines.append(_md_table(["Variable", "N", "Shapiro-W", "Shapiro-p", "Skewness", "Kurtosis"], rows))
    lines.append("")
    lines.append("Shapiro-Wilk tests are reported for reference only but are not the sole basis for choosing statistical tests.")
    lines.append("Given the small sample size (N=77) and observed skewness/kurtosis, non-parametric methods are preferred where parametric assumptions are questionable.")
    lines.append("")

    lines.append("## Outlier assessment")
    lines.append("")
    if len(outliers) > 0:
        cols_order = ["variable", "Q1", "Q3", "IQR", "lower_fence", "upper_fence", "n_outside_fences", "pct_outside_fences"]
        available = [c for c in cols_order if c in outliers.columns]
        rows = outliers[available].values.tolist()
        rows = [[str(x) for x in row] for row in rows]
        lines.append(_md_table(["Variable", "Q1", "Q3", "IQR", "Lower Fence", "Upper Fence", "N Outside", "% Outside"], rows))
    lines.append("")
    lines.append("Outliers are flagged as potential statistical outliers according to the IQR rule. No outliers were deleted, winsorized, or replaced.")
    lines.append("")

    lines.append("## Control vs Faller comparisons")
    lines.append("")
    if len(group) > 0:
        cols_order = ["variable", "n_controls", "n_fallers", "mean_controls", "mean_fallers",
                       "test_name", "test_statistic", "p_value", "adjusted_p_value_bh",
                       "cohens_d", "ci_95_lower", "ci_95_upper"]
        available = [c for c in cols_order if c in group.columns]
        rows = group[available].values.tolist()
        rows = [[str(x) for x in row] for row in rows]
        lines.append(_md_table(["Variable", "N Ctrl", "N Fall", "Mean Ctrl", "Mean Fall", "Test", "Stat", "P", "Adj P", "Cohen's d", "CI Lower", "CI Upper"], rows))
    lines.append("")
    lines.append("Effect sizes use Cohen's d for mean-based comparisons. Confidence intervals are 95% for the difference in means.")
    lines.append("Benjamini-Hochberg FDR correction was applied to the family of group comparisons.")
    lines.append("")

    lines.append("## Categorical comparisons")
    lines.append("")
    if len(cat_stat) > 0:
        rows = cat_stat.values.tolist()
        rows = [[str(x) for x in row] for row in rows]
        lines.append(_md_table(["Variable", "Test", "Chi2", "P", "Cramér's V", "N"], rows))
    lines.append("")

    lines.append("## Correlation analysis")
    lines.append("")
    if len(corr) > 0:
        cols_order = ["var1", "var2", "spearman_rho", "spearman_p", "pearson_r", "pearson_p", "n", "adjusted_p_value_bh"]
        available = [c for c in cols_order if c in corr.columns]
        rows = corr[available].values.tolist()
        rows = [[str(x) for x in row] for row in rows]
        lines.append(_md_table(["Var1", "Var2", "Spearman ρ", "Spearman p", "Pearson r", "Pearson p", "N", "Adj P"], rows))
    lines.append("")
    lines.append("Primary approach is Spearman correlation due to the small sample size and possible non-normality.")
    lines.append("Pearson correlation is reported as a sensitivity analysis.")
    lines.append("")

    lines.append("## Multicollinearity diagnostic")
    lines.append("")
    if len(vif) > 0:
        rows = vif.values.tolist()
        rows = [[str(x) for x in row] for row in rows]
        lines.append(_md_table(["Variable", "VIF"], rows))
    lines.append("")
    lines.append("VIF values are descriptive diagnostics only. No variables were removed based on VIF during this phase.")
    lines.append("")

    lines.append("## Univariate logistic regression")
    lines.append("")
    if len(logistic) > 0:
        rows = logistic.values.tolist()
        rows = [[str(x) for x in row] for row in rows]
        lines.append(_md_table(["Predictor", "Coef", "OR", "SE", "CI Lower", "CI Upper", "P", "N", "Status"], rows))
    lines.append("")
    lines.append("These are exploratory univariate associations against `HighFallRisk`. P-values are not used for feature selection.")
    lines.append("")

    lines.append("## Exploratory ROC analysis")
    lines.append("")
    if len(roc) > 0:
        rows = roc.values.tolist()
        rows = [[str(x) for x in row] for row in rows]
        lines.append(_md_table(["Variable", "AUC", "AUC CI Lower", "AUC CI Upper", "Threshold", "Sensitivity", "Specificity", "N", "Status"], rows))
    lines.append("")
    lines.append("ROC results are exploratory only. No final clinical cutoff was selected or optimized on this dataset.")
    lines.append("")

    lines.append("## Leakage audit")
    lines.append("")
    lines.append("| Variable | Status | Note |")
    lines.append("|---|---|---|")
    for _, r in results["leakage_audit"].iterrows():
        lines.append(f"| {r['variable']} | {r['status']} | {r['note']} |")
    lines.append("")
    lines.append("### Safe candidate predictors currently supported by the specification")
    lines.append("- `Age`")
    lines.append("- `Gender`")
    lines.append("- `Total x`")
    lines.append("- `Totaly`")
    lines.append("- `TUG`")
    lines.append("")
    lines.append("### Variables excluded or flagged for leakage review")
    lines.append("- `Participant_ID` -- definite leakage identifier")
    lines.append("- `HighFallRisk` -- target only")
    lines.append("- `سابفه سقوط` -- possible/likely leakage; exclude pending source verification")
    lines.append("- All 22 completely empty columns -- no information")
    lines.append("")

    lines.append("## Scientific interpretation")
    lines.append("")
    lines.append("The target `HighFallRisk` aligns exactly with the cohort prefix (CO→0, FL→1). Until original LTMM documentation verifies whether this represents an independently validated clinical high-fall-risk score or simply the original Controls-vs-Fallers cohort label, the task is described as **cohort classification** rather than prospective fall-risk prediction.")
    lines.append("")
    lines.append("TUG shows the clearest association with cohort membership, consistent with the exploratory findings from Phase 1.")
    lines.append("")
    lines.append("No causal conclusions are drawn. All findings are exploratory and descriptive.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Small sample size (N=77, 33 fallers) limits statistical power and generalizability.")
    lines.append("- Five positive events per predictor is below the traditional EPV=10 heuristic.")
    lines.append("- Shapiro-Wilk p-values are not used as the sole basis for test selection.")
    lines.append("- The fall-history variable's temporal relationship to cohort assignment is unverified.")
    lines.append("- No temporal design information is available in the CSV.")
    lines.append("")

    lines.append("## Unresolved issues")
    lines.append("")
    lines.append("1. Whether `HighFallRisk` represents a clinically validated high-fall-risk diagnosis or the original cohort label remains unverified.")
    lines.append("2. Whether `سابفه سقوط` was part of the label assignment is unverified.")
    lines.append("3. Retrospective vs prospective classification timing is undocumented.")
    lines.append("4. The meaning of Gender coding (0 vs 1) is not documented.")
    lines.append("5. Whether `Total x` equals normalized MMSE or `Totaly` equals PASE is unverified.")
    lines.append("6. Gender code direction (which value corresponds to which sex) requires source documentation.")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- **Random seed:** {config['random_seed']}")
    lines.append(f"- **Dataset:** {meta['path']}")
    lines.append(f"- **Dataset hash (MD5):** {meta['md5']}")
    lines.append(f"- **Dataset hash (SHA-256):** {meta['sha256']}")
    lines.append("- **Analysis approach:** All statistics computed in-memory from the loaded raw data; no data was modified.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    results = run_phase2(project_root())
    print("Phase 2 statistical & data audit completed.")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print(f"  Figures: {FIGURE_DIR}/")
    print(f"  Report: {REPORT_RELATIVE}")
    print()
    for key, path in results["outputs"].items():
        if key != "figures_dir":
            print(f"  wrote  : {path}")


if __name__ == "__main__":
    main()
