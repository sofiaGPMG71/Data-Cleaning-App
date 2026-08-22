"""
report.py
----------
Builds the before/after comparison table and produces downloadable
CSV / Excel bytes entirely in memory (no temp files written to disk).
"""

from __future__ import annotations

import io
from typing import Dict

import pandas as pd


def generate_cleaning_report(
    original_df: pd.DataFrame, cleaned_df: pd.DataFrame
) -> pd.DataFrame:
    """Build the Before / After / Change comparison table."""
    before_missing = int(original_df.isna().sum().sum())
    after_missing = int(cleaned_df.isna().sum().sum())
    before_duplicates = int(original_df.duplicated().sum())
    after_duplicates = int(cleaned_df.duplicated().sum())

    rows = [
        {
            "Metric": "Rows",
            "Before": original_df.shape[0],
            "After": cleaned_df.shape[0],
            "Change": cleaned_df.shape[0] - original_df.shape[0],
        },
        {
            "Metric": "Columns",
            "Before": original_df.shape[1],
            "After": cleaned_df.shape[1],
            "Change": cleaned_df.shape[1] - original_df.shape[1],
        },
        {
            "Metric": "Missing Values",
            "Before": before_missing,
            "After": after_missing,
            "Change": after_missing - before_missing,
        },
        {
            "Metric": "Duplicate Rows",
            "Before": before_duplicates,
            "After": after_duplicates,
            "Change": after_duplicates - before_duplicates,
        },
    ]
    return pd.DataFrame(rows)


def summarize_cleaning_impact(
    original_df: pd.DataFrame, cleaned_df: pd.DataFrame
) -> Dict[str, int]:
    """Small numeric summary used for the headline success message."""
    return {
        "rows_removed": int(original_df.shape[0] - cleaned_df.shape[0]),
        "rows_retained": int(cleaned_df.shape[0]),
        "missing_values_resolved": int(
            original_df.isna().sum().sum() - cleaned_df.isna().sum().sum()
        ),
        "duplicates_removed": int(
            original_df.duplicated().sum() - cleaned_df.duplicated().sum()
        ),
    }


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes (UTF-8), fully in memory."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Cleaned Data") -> bytes:
    """Serialize a DataFrame to an in-memory .xlsx file using BytesIO."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    buffer.seek(0)
    return buffer.getvalue()


def build_cleaned_filename(original_filename: str, extension: str) -> str:
    """
    Derive the downloadable filename from the original upload.
    e.g. 'customer_data.csv' -> 'customer_data_cleaned.csv'
    """
    base = original_filename.rsplit(".", 1)[0] if original_filename else "dataset"
    return f"{base}_cleaned.{extension}"
