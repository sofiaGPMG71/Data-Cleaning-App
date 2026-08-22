"""
validation.py
--------------
Runs a final quality check over a cleaned DataFrame so the UI never claims
a dataset is fully clean when problems remain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class ValidationReport:
    is_fully_clean: bool
    remaining_missing_values: int
    columns_with_missing: List[str]
    remaining_duplicate_rows: int
    row_count: int
    column_count: int
    empty_columns: List[str]
    empty_rows: int
    dtype_summary: Dict[str, str]
    warnings: List[str] = field(default_factory=list)


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Inspect a cleaned DataFrame and report anything still unresolved."""
    warnings: List[str] = []

    missing_by_col = df.isna().sum()
    columns_with_missing = missing_by_col[missing_by_col > 0].index.tolist()
    remaining_missing = int(missing_by_col.sum())

    remaining_duplicates = int(df.duplicated().sum())

    empty_columns = [c for c in df.columns if df[c].isna().all()]
    empty_rows = int(df.isna().all(axis=1).sum())

    if columns_with_missing:
        warnings.append(
            "Missing values remain in: " + ", ".join(columns_with_missing)
        )
    if remaining_duplicates > 0:
        warnings.append(f"{remaining_duplicates} duplicate row(s) still present.")
    if empty_columns:
        warnings.append(
            "Completely empty column(s) detected: " + ", ".join(empty_columns)
        )
    if empty_rows > 0:
        warnings.append(f"{empty_rows} completely empty row(s) detected.")
    if df.shape[0] == 0:
        warnings.append("The cleaned dataset has no remaining rows.")

    is_fully_clean = (
        remaining_missing == 0
        and remaining_duplicates == 0
        and not empty_columns
        and empty_rows == 0
        and df.shape[0] > 0
    )

    dtype_summary = {col: str(df[col].dtype) for col in df.columns}

    return ValidationReport(
        is_fully_clean=is_fully_clean,
        remaining_missing_values=remaining_missing,
        columns_with_missing=columns_with_missing,
        remaining_duplicate_rows=remaining_duplicates,
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        empty_columns=empty_columns,
        empty_rows=empty_rows,
        dtype_summary=dtype_summary,
        warnings=warnings,
    )
