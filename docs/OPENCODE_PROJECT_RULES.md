# OPENCODE_PROJECT_RULES.md

## Purpose
Permanent engineering and scientific rules for the thesis codebase. The detailed statistical specification is in `LTMM_STATISTICAL_ANALYSIS_AND_ML_PIPELINE_SPEC.md`.

## Non-negotiable rules

1. **Task isolation:** implement only the requested task. Do not implement future phases.
2. **Minimal change:** inspect existing code first and change only files necessary for the current task.
3. **No unrelated refactoring:** do not rename, reorganize, or rewrite unrelated modules.
4. **Preserve existing work:** never overwrite existing analyses/results/configuration without explicit instruction.
5. **Raw data immutable:** never modify raw CSV/source files in place.
6. **No fabricated data:** never invent, synthesize, or reconstruct unavailable clinical variables.
7. **Truthful provenance:** never present public LTMM data as personally collected data.
8. **Target protection:** `HighFallRisk` is target only.
9. **Identifier protection:** `Participant_ID` is traceability metadata, never a feature.
10. **Leakage protection:** exclude `سابفه سقوط` until source documentation resolves its relationship to cohort assignment.
11. **Fold-safe preprocessing:** imputation, scaling, transformations, feature selection, dimensionality reduction, and resampling must be fitted inside training folds.
12. **No test/validation optimization:** never tune on outer folds or external validation data.
13. **External validation isolation:** local validation data is never used for development decisions.
14. **No automatic outlier deletion.**
15. **No p-value-only feature selection.**
16. **No SMOTE by default:** current imbalance is mild; class weighting is preferred.
17. **Small-data model discipline:** prioritize regularized classical ML; no deep learning.
18. **Validation discipline:** use stratified nested CV rather than a single random split as the primary estimate.
19. **Reproducibility:** record random seeds, configuration, dataset version/hash when possible, and package versions.
20. **Traceability:** every result must map to an input, script, configuration, and output.
21. **No silent assumptions:** if scientific ambiguity affects validity, stop and report it.
22. **No silent changes:** if implementation reveals a conflict, report the conflict and alternatives rather than guessing.
23. **Separate data layers:** raw → audited → modeling → fold-transformed → external validation.
24. **Centralize configuration:** do not scatter scientific parameters throughout code.
25. **Use project-relative paths:** avoid machine-specific absolute paths.
26. **Testing:** run relevant tests/checks after each implementation task.
27. **Scope report:** after each task, report changed files, tests, results, warnings, and exactly one recommended next step.
28. **Do not start the next phase automatically.**

## Current approved feature policy
Default predictors:
- Age
- Gender
- Total x
- Totaly
- TUG

Excluded:
- Participant_ID
- HighFallRisk
- سابفه سقوط pending leakage verification
- all 22 all-missing columns

This policy changes only through an explicit scientific decision.

## Definition of done
A task is complete only when the requested functionality works, unrelated files remain untouched, relevant checks pass, outputs are reproducible, and the work is logged using `OPENCODE_TASK_REPORT_TEMPLATE.md`.
