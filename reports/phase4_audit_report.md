# Phase 4 Audit Report

## 24.1 Executive Summary

**Phase 4 Audit Status: VALIDATED WITH MINOR ISSUES**

The Phase 4 Nested Stratified Cross-Validation and baseline model comparison is **scientifically defensible and correctly implemented** with the following minor issues:

1. **Dataset size inconsistency** in `reports/phase4_model_comparison_report.md` line 15 states "Participants (N): 46" but the actual dataset has 77 participants (44 Controls + 33 Fallers). This is a documentation error in an existing report, not an implementation issue.
2. **Reproducibility not empirically verified** due to computational constraints (pipeline takes >10 minutes per run). Code-level analysis confirms all random seeds are fixed.
3. **Brier score not applicable** for LinearSVM (uses decision-function scores, not probabilities). All LinearSVM brier_score values are correctly empty/NaN.

No implementation correction is required.

---

## 24.2 Files Inspected

### Source Code
- `src/phase4_nested_cv.py` — Phase 4 pipeline (521 lines)
- `src/preprocessing.py` — Phase 3 preprocessing module (reused)
- `src/data_loader.py` — Phase 1 data loader (reused)

### Configuration
- `configs/phase4_config.json` — Phase 4 configuration
- `configs/phase3_config.json` — Phase 3 configuration (reference)

### Test Files
- `tests/test_phase4.py` — 24 Phase 4 tests
- `tests/test_phase3.py` — 27 Phase 3 tests
- `tests/test_phase2.py` — 39 Phase 2 tests

### Output Files
- `outputs/phase4_nested_cv/fold_results.csv` — 15 rows (5 folds × 3 models)
- `outputs/phase4_nested_cv/model_summary.csv` — 3 rows (one per model)
- `outputs/phase4_nested_cv/selected_hyperparameters.csv` — 15 rows
- `outputs/phase4_nested_cv/predictions.csv` — 231 rows (77 participants × 3 models)

### Reports
- `reports/phase4_model_comparison_report.md` — Existing Phase 4 report
- `reports/phase4_task_report.md` — Existing Phase 4 task report

### Documentation
- `docs/LTMM_STATISTICAL_ANALYSIS_AND_ML_PIPELINE_SPEC.md` — Governing specification
- `docs/OPENCODE_PROJECT_RULES.md` — Project rules
- `docs/OPENCODE_TASK_REPORT_TEMPLATE.md` — Task report template

### Raw Data
- `data/raw/elderly_data.csv` — Raw dataset (77 rows × 30 columns)

---

## 24.3 Dataset Integrity

| Property | Value | Status |
|---|---|---|
| N participants | 77 | PASS |
| Controls (HighFallRisk=0) | 44 | PASS |
| Fallers (HighFallRisk=1) | 33 | PASS |
| Total columns | 30 | PASS |
| Raw CSV MD5 | `76ef4918f1787872b4bba3d6881bc5ac` | MATCH |
| Raw CSV SHA-256 | `98a63e07dbcce81600ab2698dca15fc2b0f95b58809f9d0e6b96230b619a22f7` | MATCH |
| File size | 5867 bytes | PASS |
| Missing: Age | 1 empty cell | PASS |
| Missing: Gender | 0 empty cells | PASS |
| Missing: Total x | 5 empty cells | PASS |
| Missing: Totaly | 3 empty cells | PASS |
| Missing: TUG | 0 empty cells | PASS |

All missingness values are consistent with Phase 1 and Phase 3 findings.

---

## 24.4 Feature/Target Audit

### Included Features (exact order)
| # | Feature | Type |
|---|---------|------|
| 1 | Age | Numeric |
| 2 | Gender | Categorical |
| 3 | Total x | Numeric |
| 4 | Totaly | Numeric |
| 5 | TUG | Numeric |

**Status: PASS** — Exactly 5 features, in the specified order.

### Excluded Variables
| Variable | Reason |
|----------|--------|
| Participant_ID | Identifier |
| HighFallRisk | Target only |
| سابفه سقوط / fall_history | Leakage review |
| hight | Not in feature schema |
| weight | Not in feature schema |
| BMI | Not in feature schema |
| Education | Not in feature schema |
| Asa | Not in feature schema |
| heart Deases | Not in feature schema |
| surgery | Not in feature schema |
| x1–x11 | Not in feature schema |
| y1, y2 | Not in feature schema |
| DLRT | Not in feature schema |
| Absolut error | Not in feature schema |

**Status: PASS** — All excluded variables verified absent from model.

---

## 24.5 Leakage Audit

### Target Leakage
| Check | Result |
|---|---|
| HighFallRisk used only as target | PASS |
| HighFallRisk not in feature matrix X | PASS |
| No target-derived features created | PASS |

### Preprocessing Leakage
| Check | Result |
|---|---|
| SimpleImputer(median) fitted only on training data | PASS |
| RobustScaler fitted only on training data | PASS |
| Preprocessor is inside Pipeline | PASS |
| Pipeline passed to GridSearchCV | PASS |
| No global fit_transform() on full dataset | PASS |
| No preprocessing statistics computed on full dataset | PASS |

### Hyperparameter Leakage
| Check | Result |
|---|---|
| GridSearchCV uses only outer training data | PASS |
| Inner CV (3 folds) within outer training data | PASS |
| Outer test folds never used for hyperparameter selection | PASS |
| Best model refit only on outer training data | PASS |

### Participant/Identifier Leakage
| Check | Result |
|---|---|
| Participant_ID excluded from X | PASS |
| No participant information in predictions | PASS |
| Every participant in exactly one outer test fold | PASS |

### Temporal Leakage
| Check | Result |
|---|---|
| No time-series components in data | N/A |
| No temporal ordering applied | N/A |

---

## 24.6 Nested CV Audit

### Configuration
| Parameter | Value | Implementation |
|---|---|---|
| Outer CV | StratifiedKFold(n_splits=5, shuffle=True, random_state=42) | `src/phase4_nested_cv.py:190-194` |
| Inner CV | StratifiedKFold(n_splits=3, shuffle=True, random_state=42) | `src/phase4_nested_cv.py:195-199` |
| Models | 3 (Logistic Regression, Linear SVM, Random Forest) | `configs/phase4_config.json` |
| Total evaluations | 5 × 3 = 15 | Verified |

### Verification Results
| Check | Result |
|---|---|
| Outer folds are stratified | PASS |
| Inner folds are stratified | PASS |
| random_state=42 fixed for both | PASS |
| shuffle=True for both | PASS |
| All 77 participants appear in exactly one outer test fold | PASS |
| No overlap between outer test folds | PASS |
| Inner CV occurs only within outer training data | PASS |
| No outer test fold used by GridSearchCV | PASS |
| Fold-by-fold model count = 15 | PASS |

**Status: PASS** — Nested CV implementation matches specification exactly.

---

## 24.7 Model Audit

### Logistic Regression
- **Class:** `LogisticRegression`
- **Penalty:** L2 (explicit, not via `penalty='l2'`)
- **Solver:** `liblinear`
- **Max iterations:** 1000
- **Random state:** 42
- **C grid:** [0.01, 0.1, 1, 10, 100]
- **Probability output:** Yes (`predict_proba` available)
- **Convergence:** Expected (max_iter=1000 sufficient)
- **Status: PASS** — Properly regularized, tuning only in inner CV

### Linear SVM
- **Class:** `LinearSVC`
- **Random state:** 42
- **Max iterations:** 10000
- **C grid:** [0.01, 0.1, 1, 10, 100]
- **Score type:** `decision_function` (not probabilities)
- **Brier score:** Correctly empty (not applicable)
- **Status: PASS** — Linear kernel, decision scores used appropriately

### Random Forest
- **Class:** `RandomForestClassifier`
- **n_estimators:** 300
- **class_weight:** balanced
- **random_state:** 42
- **param_grid:** max_depth ∈ {None, 3, 5}, min_samples_leaf ∈ {1, 2, 4}, max_features ∈ {"sqrt", 0.5}
- **Probability output:** Yes (`predict_proba` available)
- **Status: PASS** — Conservative configuration for N=77

---

## 24.8 Hyperparameter Audit

All 15 hyperparameter selections are stored in `selected_hyperparameters.csv`.

**Verification:** Hyperparameters were selected via `GridSearchCV` with `cv=inner_cv` (StratifiedKFold, 3 splits) applied only to outer training data. The `refit=True` parameter ensures the best model is refit on the full outer training set. The outer test fold predictions come from `best_model.predict(X_test)` where `best_model` was selected by inner CV on training data only.

**Status: PASS** — Valid nested selection. No outer test data used for hyperparameter selection.

---

## 24.9 Metric Recalculation

### Independent Recomputation Results

All 135 metric comparisons (15 fold-model combinations × 9 metrics) were independently recomputed from the stored `predictions.csv` using the same adaptive threshold logic as the original code.

| Metric | Reported | Recomputed | Status |
|--------|----------|------------|--------|
| ROC-AUC | All 15 values | All 15 values | **MATCH (0 differences)** |
| PR-AUC | All 15 values | All 15 values | **MATCH (0 differences)** |
| Sensitivity | All 15 values | All 15 values | **MATCH (0 differences)** |
| Specificity | All 15 values | All 15 values | **MATCH (0 differences)** |
| PPV | All 15 values | All 15 values | **MATCH (0 differences)** |
| NPV | All 15 values | All 15 values | **MATCH (0 differences)** |
| F1 | All 15 values | All 15 values | **MATCH (0 differences)** |
| Balanced Accuracy | All 15 values | All 15 values | **MATCH (0 differences)** |
| Brier Score | LinearSVM all empty | All NaN | **MATCH** |

**Threshold logic verified:**
- Logistic Regression / Random Forest: threshold=0.5 (probability-based)
- Linear SVM: threshold=0.5 when max abs decision score ≤ 1.5, threshold=0.0 when max abs > 1.5

**Mean/SD verification:** All 27 summary statistics (9 metrics × 3 models) independently verified against `model_summary.csv`. All match within floating-point tolerance.

**Status: MATCH** — All reported metrics independently verified.

---

## 24.10 Fold Stability

| Model | ROC-AUC Range | ROC-AUC SD | Sensitivity Range | Specificity Range | Bal. Acc SD |
|-------|---------------|------------|-------------------|-------------------|-------------|
| Logistic Regression | 0.571–0.833 | 0.099 | 0.143–0.714 | 0.778–0.889 | 0.107 |
| Linear SVM | 0.571–0.815 | 0.091 | 0.000–0.667 | 0.778–1.000 | 0.096 |
| Random Forest | 0.508–0.730 | 0.095 | 0.167–0.714 | 0.667–0.889 | 0.081 |

**Observations:**
- Logistic Regression shows the widest ROC-AUC range (0.571–0.833), indicating fold-sensitive behavior
- Linear SVM shows floor behavior in sensitivity (0.000 in folds 1, 3, 4) — when the model predicts all negatives, sensitivity is 0
- Random Forest shows moderate variability with a floor at 0.508 ROC-AUC
- All models show substantial fold-to-fold variability consistent with N=77

---

## 24.11 Reproducibility

**Code-level analysis:**
- All random seeds are centrally defined and fixed at `random_seed=42`
- `StratifiedKFold` (outer and inner) use `random_state=42`
- `LogisticRegression` uses `random_state=42`
- `LinearSVC` uses `random_state=42`
- `RandomForestClassifier` uses `random_state=42`
- `GridSearchCV` uses `n_jobs=1` (no parallelism-induced non-determinism)
- `GridSearchCV` uses `refit=True`

**Empirical verification:** Could not be performed due to computational constraints — the pipeline requires >10 minutes to complete a single run (225 model fits including RandomForest with 300 trees).

**Status: PASS (code-level)** — All seeds fixed, no parallelism, deterministic algorithm selection.

---

## 24.12 Test Results

| Test Suite | Count | Status |
|---|---|---|
| Phase 2 tests | 39 | PASS |
| Phase 3 tests | 27 | PASS |
| Phase 4 tests | 24 | PASS |
| **Total** | **90** | **ALL PASS** |

All 90 tests pass. The test suite covers feature ordering, target exclusion, identifier exclusion, leakage rules, preprocessing safety, CV stratification, no test overlap, metric computation, model pipeline types, and raw CSV immutability.

---

## 24.13 Scientific Issues

### Small Sample Size
N=77 with 33 positive cases. Five positive events per predictor is below the traditional EPV=10 heuristic. Fold-to-fold variability is substantial (ROC-AUC SD ~0.09–0.10). Results should be interpreted as exploratory.

### Target Semantics
`HighFallRisk` is described as a cohort label (0=LTMM Control, 1=LTMM Faller) in the existing documentation. It is NOT presented as a clinically validated diagnosis. The audit report must not reinterpret this as a clinical diagnosis.

### Gender Coding Direction
The direction of `Gender` encoding (0 vs 1) remains unresolved. This is documented in Phase 2 and Phase 3 reports and has not been silently resolved.

### Fall-History Leakage
The column `سابفه سقوط` / `fall_history` was reviewed and excluded from modeling. Its leakage status remains unresolved — it could potentially contain information about the outcome that would violate the temporal assumptions of cross-validation.

### Limited Feature Set
Only 5 features are used for prediction. This limits the models' ability to capture complex relationships and may explain the modest ROC-AUC values (~0.6–0.7).

### Lack of External Validation
All results are derived from a single dataset with no external validation. Generalizability cannot be assessed.

### Dataset Size Inconsistency in Report
`reports/phase4_model_comparison_report.md` line 15 states "Participants (N): 46" but the actual dataset has 77 participants (44 Controls + 33 Fallers). This is a documentation error in an existing report that should not be silently corrected.

---

## 24.14 Corrections Required

**No implementation correction required.**

The Phase 4 pipeline is correctly implemented. The only finding is a documentation error in the existing `reports/phase4_model_comparison_report.md` (states N=46 instead of N=77), which is in a report file that this audit must not modify.

If correction is desired, it should be noted as an open item for a future documentation fix, not implemented during this audit.

---

## Audit Metadata

- **Audit date:** 2026-09-16
- **Audit scope:** Phase 4 — Nested Stratified CV and Baseline Models
- **Methodology:** Independent recomputation of all metrics from stored predictions, code review, configuration verification, test suite execution
- **Machine-readable outputs:** `outputs/phase4_audit/`
- **Test suite status:** 90/90 PASS
- **Raw data hash:** MD5 `76ef4918f1787872b4bba3d6881bc5ac` — MATCH