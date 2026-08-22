"""
analysis.py
------------
Read-only analysis helpers: dataset overview KPIs, column-level summaries,
a df.info()-style breakdown, and missing-value analysis. None of these
functions mutate the input DataFrame.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classify columns into numeric, categorical/string, and datetime buckets.

    A column is treated as datetime if it already has a datetime dtype, or if
    its name strongly suggests a date/time field AND its values can be parsed
    as dates without excessive failure (avoids false positives from column
    names alone, per the "no transformation based solely on column name"
    safety requirement).
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(series) or isinstance(
            series.dtype, pd.CategoricalDtype
        ):
            # Only classify as datetime if the column name hints at dates
            # AND a majority of non-null values actually parse as dates.
            name_hints_date = any(
                token in col.lower()
                for token in ("date", "time", "timestamp", "_at", "dob")
            )
            if name_hints_date and series.notna().sum() > 0:
                sample = series.dropna()
                if len(sample) > 200:
                    sample = sample.sample(200, random_state=42)
                parsed = pd.to_datetime(sample, errors="coerce")
                parse_rate = parsed.notna().mean() if len(sample) else 0
                if parse_rate >= 0.8:
                    datetime_cols.append(col)
                else:
                    categorical_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Compute the top-level KPI values shown on the Dataset Overview page."""
    column_types = detect_column_types(df)
    total_cells = df.shape[0] * df.shape[1] if df.shape[0] and df.shape[1] else 0
    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    return {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "total_missing_values": total_missing,
        "missing_percentage": round((total_missing / total_cells) * 100, 2)
        if total_cells
        else 0.0,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round((duplicate_rows / df.shape[0]) * 100, 2)
        if df.shape[0]
        else 0.0,
        "numeric_columns": len(column_types["numeric"]),
        "categorical_columns": len(column_types["categorical"]),
        "datetime_columns": len(column_types["datetime"]),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
    }


def get_column_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the column-by-column overview table (name, dtype, nulls, uniques)."""
    rows = []
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        rows.append(
            {
                "Column Name": col,
                "Data Type": str(series.dtype),
                "Non-Null Count": int(series.notna().sum()),
                "Missing Count": missing_count,
                "Missing %": round((missing_count / n_rows) * 100, 2)
                if n_rows
                else 0.0,
                "Unique Values": int(series.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


def get_dataset_info_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the essential content of df.info() as a DataFrame, since
    df.info() prints to stdout rather than returning a usable object.
    """
    rows = []
    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        null_count = int(series.isna().sum())
        rows.append(
            {
                "Column": col,
                "Dtype": str(series.dtype),
                "Non-Null Count": non_null,
                "Null Count": null_count,
                "Memory (KB)": round(series.memory_usage(deep=True) / 1024, 2),
            }
        )
    return pd.DataFrame(rows)


def get_dataset_info_summary_text(df: pd.DataFrame) -> str:
    """Human-readable one-liner summary, similar to the tail of df.info()."""
    column_types = detect_column_types(df)
    return (
        f"Dataset contains {df.shape[0]:,} rows and {df.shape[1]:,} columns. "
        f"{len(column_types['numeric'])} column(s) are numeric, "
        f"{len(column_types['categorical'])} are categorical/string, and "
        f"{len(column_types['datetime'])} are datetime."
    )


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, object]:
    """
    Produce a full missing-value report: a per-column table (only columns
    with at least one missing value) plus aggregate statistics.
    """
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols if n_rows and n_cols else 0

    per_column = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing > 0:
            per_column.append(
                {
                    "Column": col,
                    "Data Type": str(df[col].dtype),
                    "Missing Values": missing,
                    "Missing %": round((missing / n_rows) * 100, 2) if n_rows else 0.0,
                }
            )

    missing_table = pd.DataFrame(per_column).sort_values(
        "Missing Values", ascending=False, ignore_index=True
    ) if per_column else pd.DataFrame(
        columns=["Column", "Data Type", "Missing Values", "Missing %"]
    )

    total_missing_cells = int(df.isna().sum().sum())

    return {
        "table": missing_table,
        "total_missing_cells": total_missing_cells,
        "columns_with_missing": len(per_column),
        "overall_missing_percentage": round(
            (total_missing_cells / total_cells) * 100, 2
        )
        if total_cells
        else 0.0,
        "has_missing": total_missing_cells > 0,
    }


def find_problem_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify columns/rows that need special-case handling during cleaning."""
    all_null_columns = [c for c in df.columns if df[c].isna().all()]
    empty_rows_count = int(df.isna().all(axis=1).sum())
    mixed_type_columns = []

    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                type_set = {type(v) for v in non_null.head(500)}
                if len(type_set) > 1:
                    mixed_type_columns.append(col)

    return {
        "all_null_columns": all_null_columns,
        "mixed_type_columns": mixed_type_columns,
        "empty_row_count": empty_rows_count,
    }
