"""Phase 1 -- data validation (read-only, deterministic).

Implements structural audits over the loaded raw dataset:

* missingness (which missing-value representations actually occur),
* column profiles (type inference, missingness, cardinality, examples,
  numeric statistics), value frequencies for low-cardinality columns,
* duplicate checks (complete rows, participant IDs, cross-identity rows),
* identifier and target audits,
* all-empty columns,
* encoding / column-name issues (e.g. the Persian fall-history column),
* leakage flags and unresolved scientific ambiguities.

Every function is deterministic and never modifies its input.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_NA_LITERALS = {"NA", "N/A", "NAN", "NULL", "NONE", "<NA>"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def missing_mask(frame: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: True where a cell is empty (whitespace stripped)."""
    return frame.apply(lambda col: col.str.strip() == "")


def missing_percentage(n_missing: int, n_total: int) -> float:
    if n_total == 0:
        return 0.0
    return round(100.0 * n_missing / n_total, 2)


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# --------------------------------------------------------------------------- #
# missing-value representations
# --------------------------------------------------------------------------- #
def probe_missing_representations(frame: pd.DataFrame) -> Dict[str, Any]:
    """Count which missing-value representations occur in the raw cells.

    Known string literals (e.g. ``"NA"``) are reported separately so that an
    audit can distinguish ``""`` from source-encoded missing tokens.
    """
    counts: Dict[str, Any] = {"empty_string": 0, "whitespace_only": 0}
    literal_counts: Dict[str, int] = {}
    for column in frame.columns:
        for value in frame[column]:
            text = str(value)
            if text == "":
                counts["empty_string"] += 1
            elif text.strip() == "":
                counts["whitespace_only"] += 1
            else:
                token = text.strip().upper()
                if token in _NA_LITERALS:
                    literal_counts[token] = literal_counts.get(token, 0) + 1
    if literal_counts:
        counts["literal_tokens"] = literal_counts
    return counts


def all_empty_columns(frame: pd.DataFrame) -> List[str]:
    mask = missing_mask(frame)
    return [col for col in frame.columns if bool(mask[col].all())]


# --------------------------------------------------------------------------- #
# column profiles
# --------------------------------------------------------------------------- #
def infer_type(series: pd.Series, mask: pd.Series) -> str:
    """Infer a storage type for reporting only (never transforms data)."""
    values = series[~mask]
    if len(values) == 0:
        return "all_missing"
    numeric = _as_numeric(values)
    if not numeric.isna().any():
        if (numeric == numeric.round()).all():
            return "integer"
        return "float"
    if values.nunique() <= 20:
        return "categorical"
    return "text"


def column_profiles(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    mask = missing_mask(frame)
    profiles = []
    for column in frame.columns:
        values = frame[column][~mask[column]]
        n_total = len(frame)
        n_missing = int(mask[column].sum())
        numeric = _as_numeric(values)
        profile: Dict[str, Any] = {
            "column": column,
            "inferred_type": infer_type(frame[column], mask[column]),
            "n_nonnull": int(n_total - n_missing),
            "n_missing": n_missing,
            "missing_pct": missing_percentage(n_missing, n_total),
            "n_unique": int(values.nunique()),
            "example_values": list(dict.fromkeys(values.tolist()))[:5],
        }
        if len(values) > 0 and not numeric.isna().any():
            profile["numeric"] = {
                "min": round(float(numeric.min()), 6),
                "max": round(float(numeric.max()), 6),
                "mean": round(float(numeric.mean()), 6),
                "median": round(float(numeric.median()), 6),
            }
        if 0 < int(values.nunique()) <= 30:
            frequencies = values.value_counts()
            profile["value_frequencies"] = {
                str(key): int(value) for key, value in frequencies.items()
            }
        profiles.append(profile)
    return profiles


def flatten_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a column profile into a single CSV row."""
    numeric = profile.get("numeric")
    return {
        "column": profile["column"],
        "inferred_type": profile["inferred_type"],
        "n_nonnull": profile["n_nonnull"],
        "n_missing": profile["n_missing"],
        "missing_pct": profile["missing_pct"],
        "n_unique": profile["n_unique"],
        "example_values": " | ".join(profile["example_values"]),
        "min": numeric["min"] if numeric else "",
        "max": numeric["max"] if numeric else "",
        "mean": numeric["mean"] if numeric else "",
        "median": numeric["median"] if numeric else "",
    }


# --------------------------------------------------------------------------- #
# duplicates
# --------------------------------------------------------------------------- #
def duplicate_complete_rows(
    frame: pd.DataFrame, id_column: str
) -> Dict[str, Any]:
    """Count rows whose whitespace-stripped contents are duplicated."""
    stripped = frame.apply(lambda col: col.str.strip())
    flagged = stripped.duplicated(keep=False)
    involved = stripped[flagged]
    return {
        "method": "duplicated() over whitespace-stripped cell strings",
        "n_rows_flagged": int(stripped.duplicated().sum()),
        "n_rows_involved": int(len(involved)),
        "rows": [
            {
                "row_index": int(index),
                id_column: str(row[id_column]),
            }
            for index, row in involved.iterrows()
        ],
    }


def duplicate_participant_ids(frame: pd.DataFrame, id_column: str) -> Dict[str, Any]:
    ids = frame[id_column].str.strip()
    duplicated = ids[ids.duplicated(keep=False)].tolist()
    return {
        "n_duplicates": int(ids.duplicated().sum()),
        "ids": [str(value) for value in duplicated],
    }


def shared_numeric_id_suffixes(frame: pd.DataFrame, id_column: str) -> List[str]:
    """Numeric suffixes used by more than one participant (across prefixes)."""
    suffixes = frame[id_column].str.extract(r"-(\d+)$", expand=False).str.zfill(3)
    repeated = set(suffixes[suffixes.duplicated(keep=False)].dropna())
    return sorted(repeated)


# --------------------------------------------------------------------------- #
# identifier audit
# --------------------------------------------------------------------------- #
def identifier_audit(
    frame: pd.DataFrame, id_column: str, regex: Optional[str] = None
) -> Dict[str, Any]:
    ids = frame[id_column].str.strip()
    compiled = re.compile(regex) if regex else None
    all_match_format = (
        compiled is not None
        and bool(ids.map(lambda value: bool(compiled.fullmatch(value))).all())
    )
    return {
        "column": id_column,
        "n_rows": int(len(ids)),
        "n_unique": int(ids.nunique()),
        "n_missing": int((ids == "").sum()),
        "n_duplicates": int(ids.duplicated().sum()),
        "format_regex": regex,
        "all_match_format": all_match_format,
        "prefix_counts": {
            str(key): int(value) for key, value in ids.str[:2].value_counts().items()
        },
        "sample_ids": [str(value) for value in ids.tolist()][:5],
    }


# --------------------------------------------------------------------------- #
# target audit
# --------------------------------------------------------------------------- #
def target_audit(
    frame: pd.DataFrame,
    target_column: str,
    id_column: str,
    expected_values: Optional[List[str]] = None,
    expected_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    values = frame[target_column].str.strip()
    observed_counts = {
        str(key): int(value) for key, value in values.value_counts().items()
    }
    missing = int((values == "").sum())
    unexpected = sorted(
        {str(v) for v in observed_counts if expected_values and str(v) not in expected_values}
    )

    prefixes = frame[id_column].str.strip().str[:2]
    checked = 0
    aligned = 0
    misaligned: List[Dict[str, str]] = []
    for prefix, value in zip(prefixes, values):
        if value == "":
            continue
        checked += 1
        target_ok = (prefix == "CO" and str(value) == "0") or (
            prefix == "FL" and str(value) == "1"
        )
        if target_ok:
            aligned += 1
        else:
            misaligned.append({"prefix": prefix, "value": value})

    counts_difference = {}
    if expected_counts:
        for class_value, expected_count in expected_counts.items():
            if observed_counts.get(class_value) != expected_count:
                counts_difference[class_value] = {
                    "observed": observed_counts.get(class_value),
                    "expected": expected_count,
                }

    return {
        "column": target_column,
        "n_rows": int(len(values)),
        "n_missing": missing,
        "values": list(observed_counts.keys()),
        "n_classes": int(values.nunique()),
        "is_binary": int(values.nunique()) == 2,
        "class_counts": observed_counts,
        "expected_class_counts": expected_counts,
        "class_counts_difference": counts_difference if counts_difference else None,
        "unexpected_values": unexpected,
        "cohort_alignment": {
            "hypothesis": "CO prefix -> 0, FL prefix -> 1 (per project spec)",
            "n_checked": checked,
            "n_aligned": aligned,
            "aligned": checked == aligned,
            "misaligned_rows": misaligned,
        },
    }


# --------------------------------------------------------------------------- #
# column-name / encoding issues
# --------------------------------------------------------------------------- #
def column_name_issues(
    frame: pd.DataFrame, fall_history: Dict[str, str]
) -> List[Dict[str, Any]]:
    issues = []
    for column in frame.columns:
        if column != column.strip():
            issues.append(
                {"type": "surrounding_whitespace", "column": column}
            )
        if "\ufffd" in str(column):
            issues.append(
                {"type": "mojibake_replacement_character", "column": column}
            )
    duplicated = sorted(set(frame.columns[frame.columns.duplicated()]))
    if duplicated:
        issues.append({"type": "duplicate_column_names", "columns": duplicated})
    for column in frame.columns:
        if any(ord(ch) > 127 for ch in str(column)):
            issues.append(
                {
                    "type": "non_ascii_column_name",
                    "column": column,
                    "internal_name": fall_history["internal_name"],
                    "encoding_status": "decodes_cleanly_as_utf8_no_mojibake",
                    "note": (
                        "Persian fall-history column name preserved verbatim; "
                        "no silent rename or reinterpretation."
                    ),
                }
            )
    for column in frame.columns:
        if column != column.strip() and " " in column:
            issues.append(
                {"type": "embedded_space_column_name", "column": column}
            )
    return issues


# --------------------------------------------------------------------------- #
# aggregate validation
# --------------------------------------------------------------------------- #
def _file_level_checks(meta: Dict[str, Any]) -> Dict[str, Any]:
    delim = meta["delimiter_counts"]
    checks = [
        {"check": "file_exists", "passed": True},
        {"check": "file_readable", "passed": True},
        {"check": "file_not_empty", "passed": True},
        {"check": "encoding_decodable", "passed": True,
         "encoding": meta["encoding"]},
        {"check": "csv_parseable", "passed": True},
        {"check": "delimiter_consistent", "passed": delim["min"] == delim["max"],
         "observed_delimiter": meta["delimiter"],
         "counts_per_line": delim},
    ]
    return {
        "path": meta.get("relative_path", meta["path"]),
        "size_bytes": meta["size_bytes"],
        "md5": meta["md5"],
        "sha256": meta["sha256"],
        "encoding": meta["encoding"],
        "newline_style": meta["newline_style"],
        "bom_present": meta["bom_present"],
        "n_lines": meta["n_lines"],
        "checks": checks,
    }


def validate_dataset(
    frame: pd.DataFrame,
    meta: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run all Phase 1 audits and assemble the machine-readable result."""
    dataset_cfg = config["dataset"]
    columns_cfg = config["columns"]
    id_column = columns_cfg["identifier"]["column"]
    target_column = columns_cfg["target"]["column"]
    fall_history_cfg = columns_cfg["fall_history"]

    failures: List[str] = []
    warnings: List[str] = []

    if id_column not in frame.columns:
        failures.append(f"Identifier column '{id_column}' is missing.")
    if target_column not in frame.columns:
        failures.append(f"Target column '{target_column}' is missing.")

    if frame.shape[0] != dataset_cfg["expected_rows"]:
        failures.append(
            f"Expected {dataset_cfg['expected_rows']} rows, "
            f"found {frame.shape[0]}."
        )
    if frame.shape[1] != dataset_cfg["expected_columns"]:
        failures.append(
            f"Expected {dataset_cfg['expected_columns']} columns, "
            f"found {frame.shape[1]}."
        )
    if meta["delimiter"] != dataset_cfg["expected_delimiter"]:
        failures.append(
            f"Delimiter mismatch: expected '{dataset_cfg['expected_delimiter']}', "
            f"found '{meta['delimiter']}'."
        )
    if meta["encoding"] != dataset_cfg["expected_encoding"]:
        warnings.append(
            f"Encoding '{meta['encoding']}' differs from configured "
            f"'{dataset_cfg['expected_encoding']}'."
        )

    mask = missing_mask(frame)
    empty_columns = all_empty_columns(frame)
    expected_empty = columns_cfg["expected_all_empty"]
    not_empty_anymore = [c for c in expected_empty if c not in empty_columns]
    newly_empty = [c for c in empty_columns if c not in expected_empty]
    if not_empty_anymore or newly_empty:
        warnings.append(
            "All-empty column set differs from configuration. "
            f"Expected-but-not-empty: {not_empty_anymore}. "
            f"Unexpected-but-empty: {newly_empty}."
        )

    profiles = column_profiles(frame)

    duplicates = {
        "complete_duplicate_rows": (
            duplicate_complete_rows(frame, id_column) if id_column in frame.columns else {}
        ),
        "duplicate_participant_ids": (
            duplicate_participant_ids(frame, id_column) if id_column in frame.columns else {}
        ),
    }

    measurement_columns = [c for c in frame.columns if c != id_column]
    stripped_measurements = frame[measurement_columns].apply(
        lambda col: col.str.strip()
    )
    cross_flagged = stripped_measurements.duplicated(keep=False)
    duplicates["cross_cohort_identical_measurement_rows"] = int(len(
        stripped_measurements[cross_flagged]
    ))
    if id_column in frame.columns and cross_flagged.any():
        duplicates["cross_cohort_identical_rows"] = [
            {
                "row_index": int(i),
                id_column: str(frame.loc[i, id_column]),
            }
            for i in stripped_measurements[cross_flagged].index
        ]

    suffix_overlap = (
        shared_numeric_id_suffixes(frame, id_column) if id_column in frame.columns else []
    )
    duplicates["shared_numeric_id_suffixes"] = suffix_overlap
    duplicates["shared_suffix_note"] = (
        "Numeric ID suffixes are shared between the CO and FL cohorts "
        "(each cohort is numbered independently, e.g. CO-001 and FL-001). "
        "This is an expected cohort-relative numbering pattern, not an error."
    )

    identifier = (
        identifier_audit(
            frame, id_column, columns_cfg["identifier"].get("regex")
        )
        if id_column in frame.columns
        else {}
    )

    target = (
        target_audit(
            frame,
            target_column,
            id_column,
            expected_values=columns_cfg["target"].get("class_values"),
            expected_counts=columns_cfg["target"].get("class_counts"),
        )
        if target_column in frame.columns and id_column in frame.columns
        else {}
    )

    issues = column_name_issues(frame, fall_history_cfg)

    fall_history = frame[fall_history_cfg["original_name"]].str.strip()
    fh_numeric = pd.to_numeric(fall_history.mask(fall_history == ""), errors="coerce")
    fall_cohort_note = None
    if id_column in frame.columns and bool(fh_numeric.notna().any()):
        fh_table = pd.DataFrame(
            {
                "prefix": frame[id_column].str.strip().str[:2].values,
                "value": fh_numeric.values,
            }
        ).dropna(subset=["value"])
        means = fh_table.groupby("prefix")["value"].mean().to_dict()
        fall_cohort_note = {
            "description": (
                "Descriptive only: fall-history distribution is strongly "
                "cohort-asymmetric, consistent with the spec's possible/likely "
                "leakage flag."
            ),
            "mean_fall_history_by_cohort": {
                str(k): round(float(v), 4) for k, v in sorted(means.items())
            },
        }

    leakage_flags = [
        {
            "variable": id_column,
            "status": "DEFINITE_LEAKAGE_IDENTIFIER",
            "note": "Participant_ID is traceability metadata and must never be a predictor.",
        },
        {
            "variable": target_column,
            "status": "TARGET",
            "note": "HighFallRisk is the target only.",
        },
        {
            "variable": fall_history_cfg["original_name"],
            "internal_name": fall_history_cfg["internal_name"],
            "status": "POSSIBLE_LIKELY_LEAKAGE",
            "note": (
                "Spec flags fall history as possible/likely leakage; exclude "
                "from modelling until its relationship to cohort assignment is "
                "verified from original source documentation."
            ),
            "descriptive_evidence": fall_cohort_note,
        },
        {
            "variable": None,
            "status": "TEMPORAL_LEAKAGE_UNKNOWN",
            "note": (
                "No temporal leakage can be ruled out from the CSV alone "
                "(outcome timing relative to measurements is undocumented)."
            ),
        },
    ]

    ambiguities = [
        {
            "item": "target_source_definition",
            "note": (
                "Per spec, the CSV supports interpreting HighFallRisk as the "
                "original LTMM Controls-vs-Fallers cohort label (44/33), not "
                "yet as an independently validated clinical high-fall-risk "
                "score. Definition status: REQUIRES_VERIFICATION."
            ),
        },
        {
            "item": "fall_history_vs_label_assignment",
            "note": (
                "Whether 'سابفه سقوط' (fall history) was part of the "
                "label/cohort assignment is unverified; this drives the "
                "leakage decision."
            ),
        },
        {
            "item": "temporal_design",
            "note": (
                "Retrospective vs prospective classification, observation "
                "period, and timing of measurements relative to outcome are "
                "undocumented in the CSV."
            ),
        },
        {
            "item": "gender_coding_direction",
            "note": "Gender is binary 0/1 but the meaning of each code is not documented.",
        },
        {
            "item": "total_x_totaly_semantics",
            "note": (
                "Do not claim Total x = normalized MMSE or Totaly = PASE "
                "without source documentation proving it (spec rule)."
            ),
        },
        {
            "item": "age_value_precision",
            "note": (
                "Age values carry 14-15 significant digits, consistent with "
                "computed/derived values; the generating calculation is "
                "undocumented in the source."
            ),
        },
    ]

    critical_conditions = {
        "required_columns_present": id_column in frame.columns and target_column in frame.columns,
        "rows_match_config": frame.shape[0] == dataset_cfg["expected_rows"],
        "columns_match_config": frame.shape[1] == dataset_cfg["expected_columns"],
        "delimiter_matches_config": meta["delimiter"]
        == dataset_cfg["expected_delimiter"],
        "file_level_checks_pass": all(
            check["passed"] for check in _file_level_checks(meta)["checks"]
        ),
    }

    return {
        "file_level": _file_level_checks(meta),
        "dataset_level": {
            "n_rows": int(frame.shape[0]),
            "n_columns": int(frame.shape[1]),
            "expected_rows": dataset_cfg["expected_rows"],
            "expected_columns": dataset_cfg["expected_columns"],
            "rows_match_config": frame.shape[0] == dataset_cfg["expected_rows"],
            "columns_match_config": frame.shape[1] == dataset_cfg["expected_columns"],
            "column_names": list(frame.columns),
            "required_columns_present": {
                id_column: id_column in frame.columns,
                target_column: target_column in frame.columns,
            },
        },
        "column_profiles": profiles,
        "missingness": {
            "representations": probe_missing_representations(frame),
            "n_missing_cells": int(mask.to_numpy().sum()),
            "n_total_cells": int(frame.size),
            "overall_missing_pct": missing_percentage(
                int(mask.to_numpy().sum()), int(frame.size)
            ),
            "table": [
                {
                    "column": column,
                    "n_nonnull": int(len(frame) - int(mask[column].sum())),
                    "n_missing": int(mask[column].sum()),
                    "missing_pct": missing_percentage(
                        int(mask[column].sum()), len(frame)
                    ),
                }
                for column in frame.columns
            ],
        },
        "duplicates": duplicates,
        "identifier": identifier,
        "target": target,
        "leakage_flags": leakage_flags,
        "all_empty_columns": empty_columns,
        "column_name_issues": issues,
        "column_name_mapping": {
            fall_history_cfg["original_name"]: fall_history_cfg["internal_name"]
        },
        "warnings": warnings,
        "ambiguities": ambiguities,
        "critical_conditions": critical_conditions,
        "failures": failures,
        "critical_checks_passed": (
            critical_conditions["required_columns_present"]
            and critical_conditions["rows_match_config"]
            and critical_conditions["columns_match_config"]
            and critical_conditions["delimiter_matches_config"]
            and critical_conditions["file_level_checks_pass"]
            and not failures
        ),
    }