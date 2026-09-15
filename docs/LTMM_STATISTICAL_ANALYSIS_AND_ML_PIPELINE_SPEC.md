# LTMM Statistical Analysis & ML Pipeline Specification

## Status
- Dataset: `LTMM_harmonized.csv`
- N = 77 participants
- 30 columns
- Exploratory audit only; no model training
- Raw CSV must remain unchanged

## 1. Data audit
Usable columns:
`Participant_ID`, `Age`, `Gender`, `Total x`, `Totaly`, `TUG`, `سابفه سقوط`, `HighFallRisk`.

**Correction to the previous audit:** there are **22**, not 18, all-missing columns. They are:
`hight`, `weight`, `BMI`, `Education`, `Asa`, `heart Deases`, `surgery`, `x1`–`x11`, `y1`, `y2`, `DLRT`, `Absolut error`.

They must not be fabricated, reconstructed, or imputed.

Current integrity findings:
- 77 unique IDs
- 0 exact duplicate rows
- 0 duplicate IDs
- 0 empty/malformed rows
- `HighFallRisk`: 0 missing

## 2. Target definition — critical
`HighFallRisk` is exactly aligned with the ID cohort prefix:
- CO: 44 → 0
- FL: 33 → 1

Therefore the CSV supports interpreting it as the **original LTMM Controls-vs-Fallers cohort label**, not yet as an independently validated clinical high-fall-risk score.

Before final thesis interpretation, verify from original LTMM documentation:
1. how Controls/Fallers were defined;
2. retrospective vs prospective classification;
3. observation period;
4. whether `سابفه سقوط` was part of label assignment;
5. timing of all measurements relative to outcome/classification.

Until verified, describe the task as cohort classification rather than prospective fall-risk prediction.

## 3. Feature policy
Default candidate predictors:
- `Age`
- `Gender`
- `Total x`
- `Totaly`
- `TUG`

Excluded:
- `Participant_ID`: definite leakage/identifier
- `HighFallRisk`: target
- `سابفه سقوط`: possible/likely leakage; exclude pending source verification
- 22 all-missing columns: no information

Do not claim `Total x` = normalized MMSE or `Totaly` = PASE unless source documentation proves it.

## 4. Missing data
Current missingness:
| Variable | Missing N | Missing % |
|---|---:|---:|
| Age | 1 | 1.3% |
| Gender | 0 | 0.0% |
| Total x | 5 | 6.5% |
| Totaly | 3 | 3.9% |
| TUG | 0 | 0.0% |
| سابفه سقوط | 2 | 2.6% |
| HighFallRisk | 0 | 0.0% |

Future analysis must report missingness overall and by target class. Consider missingness indicators and Little's MCAR test only if appropriate. Do not claim MCAR/MAR/MNAR with certainty.

ML default: median imputation, fitted **inside training folds only**.

## 5. Descriptive statistics
For numeric variables report:
N, missing N, mean, SD, median, Q1, Q3, IQR, min, max, skewness, kurtosis.

For categorical variables report frequency and percentage.

Report overall and stratified by target.

## 6. Distribution/normality
For relevant continuous variables generate:
- histogram
- Q-Q plot
- boxplot
- skewness/kurtosis
- Shapiro-Wilk where appropriate

Do not classify normality from a p-value alone.

Observed pattern:
- Age: approximately symmetric
- Total x: bounded, near-discrete, ceiling-prone/left-skewed
- Totaly: right-skewed
- TUG: right-skewed/heavy-tailed
- fall history: count-like and cohort-asymmetric

## 7. Outliers
Use IQR fences, standardized-score checks where useful, boxplots, and clinical plausibility.

Classify outliers as:
1. probable data error
2. suspicious/investigate
3. legitimate clinical extreme

Do not automatically delete statistical outliers. Retain legitimate extremes. Robust scaling is preferred. Any winsorization must be a CV-internal sensitivity analysis.

## 8. Group comparisons
Continuous:
- Welch t-test when appropriate
- independent t-test when assumptions are defensible
- Mann-Whitney U when parametric assumptions are unsuitable

Report statistic, p-value, effect size, and CI where feasible.

Categorical:
- chi-square
- Fisher exact when expected counts are small
- Cramér's V

Use Benjamini-Hochberg FDR for families of exploratory tests where appropriate. P-values are exploratory, not automatic feature-selection rules.

## 9. Correlation
Primary: Spearman.
Secondary: Pearson when assumptions are suitable.

Assess correlations among safe continuous predictors:
Age, Total x, Totaly, TUG.

Do not correlate the target as though it were continuous. `سابفه سقوط` may be shown descriptively but remains excluded from default ML until leakage is resolved.

Current audit: no material correlation/redundancy among the four clean continuous predictors (absolute Spearman rho approximately <= 0.32).

## 10. Multicollinearity
Calculate VIF for candidate predictors when appropriate. Current audit found approximately:
- Age 1.06
- Total x 1.05
- Totaly 1.11
- TUG 1.12

No meaningful collinearity problem.

## 11. Leakage audit
`Participant_ID` = DEFINITE leakage.
`HighFallRisk` = target.
`سابفه سقوط` = POSSIBLE/LIKELY leakage; exclude until source/temporal verification.
No temporal leakage can be ruled out conclusively from the CSV alone.

## 12. Univariate association
Exploratory univariate logistic regression:
- OR per 1 SD for continuous predictors
- 95% CI
- p-value
- FDR where appropriate

Do not use univariate significance as the sole feature-selection method.

Current exploratory result: TUG has the clearest signal (Cohen's d ~ -0.82; AUC ~0.71). This does not prove TUG alone is sufficient.

## 13. ROC
For safe continuous predictors:
- ROC-AUC
- 95% CI
- sensitivity
- specificity

Exploratory AUCs from the audit:
- Age ~0.544
- Total x ~0.556
- Totaly ~0.551
- TUG ~0.711

Final thresholds must not be optimized on the entire dataset. Threshold optimization must be performed within the development/CV process.

## 14. Sample-size feasibility
N=77 and positive class n=33.

Suitable:
- regularized logistic regression
- linear SVM
- conservative Random Forest
- optionally conservative gradient boosting

Do not use deep learning.

Five predictors give approximately 6.6 positive events per predictor, below the traditional EPV=10 heuristic. Therefore prefer regularization, conservative complexity, nested CV, and uncertainty reporting.

## 15. Preprocessing
Default numeric pipeline:
1. median imputation
2. RobustScaler

Optional log transformation of TUG/Totaly may be tested as a documented CV-internal sensitivity analysis.

Gender is already binary; verify coding direction from source documentation.

Never fit imputation, scaling, transformation, feature selection, dimensionality reduction, or resampling on the full dataset before CV.

## 16. Class imbalance
44 controls (57.1%) vs 33 fallers (42.9%); ratio ~1.33:1.

Default:
- no SMOTE
- prefer class weighting
- threshold tuning may be evaluated separately

Any resampling must occur inside training folds.

## 17. Feature selection
Only five candidate predictors exist. Primary approach: retain the full safe candidate set.

Do not drop Age, Total x, or Totaly merely because their univariate p-values are non-significant.

Optional L1/RFE analyses must be nested/CV-internal.

## 18. Models
Primary:
- regularized Logistic Regression (L2)

Secondary:
- Linear SVM
- conservative Random Forest

Optional:
- HistGradientBoosting

Do not create a large model zoo.

## 19. Validation
Preferred:
- nested stratified K-fold CV
- outer 5 folds
- inner 3–5 folds
- fixed random seed
- all preprocessing inside the pipeline
- hyperparameter tuning only in inner CV

Avoid a single random train/test split as the primary estimate for N=77.

## 20. Metrics
Primary:
- ROC-AUC
- PR-AUC
- sensitivity

Secondary:
- specificity
- precision/PPV
- NPV
- F1
- balanced accuracy
- confusion matrix
- Brier score
- calibration curve

Never report accuracy alone.

## 21. Interpretability
Use:
- logistic coefficients / odds ratios
- permutation importance on held-out data

SHAP is optional. Feature importance is predictive association, not causality.

## 22. External validation
External validation is allowed only if:
1. target definitions are comparable;
2. feature constructs are comparable;
3. the LTMM model is frozen;
4. no tuning/feature selection uses local data;
5. preprocessing is not refit on local data.

If LTMM is cohort classification but the local target is defined differently, external validation is not justified until target harmonization.

## 23. Required outputs
Tables:
- dataset overview
- data dictionary
- missingness
- duplicates/integrity
- descriptive statistics
- distribution/normality
- outliers
- group comparisons
- effect sizes
- correlation
- VIF
- leakage audit
- feature quality
- univariate logistic regression
- ROC/AUC
- predictor list
- exclusions + reasons
- preprocessing
- CV/model configuration
- final performance

Figures:
- class distribution
- missingness plot
- histograms
- Q-Q plots
- target-stratified boxplots
- correlation heatmap
- ROC
- calibration

## 24. Final non-negotiable scientific rules
1. No fabricated data.
2. Raw CSV immutable.
3. No Participant_ID as predictor.
4. No target as predictor.
5. Exclude `سابفه سقوط` pending leakage verification.
6. Exclude all 22 all-missing columns.
7. No preprocessing before fold splitting.
8. No feature selection before fold splitting.
9. No resampling before fold splitting.
10. No external-data tuning.
11. No automatic outlier deletion.
12. No p-value-only feature selection.
13. Use stratified CV.
14. Prefer nested CV.
15. Report uncertainty.
16. Preserve provenance.
17. Save derived data separately.
18. Record seeds/configuration/package versions.
19. Do not overstate cohort classification as clinical prediction.
20. If scientific ambiguity cannot be resolved, stop and report it instead of guessing.

## 25. Machine-readable configuration
```yaml
target_column: HighFallRisk
positive_class: 1
negative_class: 0
target_definition_status: REQUIRES_VERIFICATION
candidate_features: [Age, Gender, Total x, Totaly, TUG]
excluded_features: [Participant_ID, سابفه سقوط, hight, weight, BMI, Education, Asa, heart Deases, surgery, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, y1, y2, DLRT, Absolut error]
numeric_features: [Age, Total x, Totaly, TUG]
binary_features: [Gender]
imputation: median
scaling: RobustScaler
outlier_policy: retain_unless_proven_error
class_balance: class_weight_preferred
smote_default: false
feature_selection: domain_driven
nested_cv: true
outer_cv: StratifiedKFold_5
inner_cv: StratifiedKFold_3
primary_metrics: [ROC_AUC, PR_AUC, sensitivity]
external_validation: conditional_on_target_and_feature_harmonization
```
