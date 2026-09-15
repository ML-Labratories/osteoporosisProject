"""Phase 1 -- raw CSV loader.

Loads the raw dataset CSV exactly as stored on disk and nothing more:

* every cell is kept as its original string (pandas ``dtype=str`` plus
  ``keep_default_na=False``), so empty cells stay ``""`` and values such as
  ``"1.0"`` or ``"NaN"`` are never remapped;
* no cleaning, type coercion, imputation, or row/column removal happens here;
* hard file-level assumptions (existence, readability, decodability,
  parseability, consistent delimiter) are enforced and raise
  :class:`DataLoaderError` with an explicit reason.

Dataset-level validation and auditing live in ``data_validator``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


class DataLoaderError(RuntimeError):
    """Raised when the raw CSV violates a hard file-level assumption."""


def _hash_bytes(data: bytes) -> Dict[str, str]:
    return {
        "md5": hashlib.md5(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def load_raw_csv(
    raw_path: Path,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> Tuple[pd.DataFrame, Dict]:
    """Load ``raw_path`` into an unmodified string-typed DataFrame.

    Returns ``(dataframe, meta)`` where ``meta`` records file provenance:
    hashes, size, encoding, BOM presence, delimiter consistency, line ending,
    and parsed dimensions.

    Raises ``DataLoaderError`` on any hard file-level violation.
    """
    path = Path(raw_path)
    if not path.exists():
        raise DataLoaderError(f"Raw data file does not exist: {path}")
    if not path.is_file():
        raise DataLoaderError(f"Raw data path is not a regular file: {path}")

    raw = path.read_bytes()
    if not raw.strip():
        raise DataLoaderError(f"Raw data file is empty: {path}")

    hashes = _hash_bytes(raw)
    bom_present = raw.startswith(b"\xef\xbb\xbf")

    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise DataLoaderError(
            f"File is not decodable with the declared encoding '{encoding}' "
            f"(byte position {exc.start}, reason: {exc.reason}). "
            "Aborting instead of guessing an encoding."
        ) from exc

    lines = text.splitlines()
    delim_counts = [line.count(delimiter) for line in lines]
    if not delim_counts or min(delim_counts) != max(delim_counts):
        raise DataLoaderError(
            "Inconsistent row widths: the number of delimiter characters "
            f"varies across lines (min={min(delim_counts)}, "
            f"max={max(delim_counts)}). The file is not a consistent CSV table."
        )

    pandas_encoding = "utf-8-sig" if bom_present else encoding
    try:
        frame = pd.read_csv(
            path,
            sep=delimiter,
            dtype=str,
            keep_default_na=False,
            encoding=pandas_encoding,
        )
    except Exception as exc:  # noqa: BLE001 - surface any parser failure
        raise DataLoaderError(
            f"CSV could not be parsed with delimiter {delimiter!r}: {exc}"
        ) from exc

    if frame.shape[1] == 0:
        raise DataLoaderError("CSV parsed to zero columns; expected a header row.")
    duplicated_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated_columns:
        raise DataLoaderError(
            f"Duplicate column names found: {sorted(set(duplicated_columns))}"
        )

    meta = {
        "path": path.as_posix(),
        "size_bytes": len(raw),
        "md5": hashes["md5"],
        "sha256": hashes["sha256"],
        "encoding": encoding,
        "bom_present": bom_present,
        "delimiter": delimiter,
        "delimiter_counts": {
            "min": min(delim_counts),
            "max": max(delim_counts),
        },
        "newline_style": "CRLF" if b"\r\n" in raw else "LF",
        "n_lines": len(lines),
        "n_records": int(len(frame)),
        "n_columns": int(frame.shape[1]),
    }
    return frame, meta