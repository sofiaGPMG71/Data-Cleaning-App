"""
cleaning.py
------------
All data-mutating logic lives here, kept separate from the Streamlit UI.
Every function returns a *new* DataFrame — the caller's original object is
never modified in place — plus a log describing what happened, so the UI can
show before/after context without silently discarding data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analysis import detect_column_types

NUMERIC_STRATEGIES = ["Mean", "Median", "Forward Fill", "Backward Fill", "Interpolate"]
CATEGORICAL_STRATEGIES = ["Mode", "Forward Fill", "Backward Fill", "Custom Value"]
DATETIME_STRATEGIES = ["Forward Fill", "Backward Fill", "Interpolate"]


@dataclass
class CleaningLogEntry:
    column: str
    strategy: str
    values_filled: int
    note: str = ""


@dataclass
class CleaningResult:
    dataframe: pd.DataFrame
    log: List[CleaningLogEntry] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)


def build_auto_cleaning_plan(df: pd.DataFrame) -> Dict[str, str]:
    """
    Decide a safe default strategy per column, following:
        numeric      -> Median
        categorical  -> Mode (falls back to "Unknown" if no mode exists)
        datetime     -> Forward Fill
    Columns with no missing values are omitted from the plan entirely.
    """
    column_types = detect_column_types(df)
    plan: Dict[str, str] = {}

    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue

        if col in column_types["numeric"]:
            plan[col] = "Median"
        elif col in column_types["datetime"]:
            plan[col] = "Forward Fill"
        else:
            # Categorical / string / anything else falls back to Mode.
            mode_series = df[col].dropna().mode()
            plan[col] = "Mode" if not mode_series.empty else "Custom Value"

    return plan


def _fill_numeric(series: pd.Series, strategy: str) -> pd.Series:
    if strategy == "Mean":
        if series.dropna().empty:
            return series
        return series.fillna(series.mean())
    if strategy == "Median":
        if series.dropna().empty:
            return series
        return series.fillna(series.median())
    if strategy == "Forward Fill":
        return series.ffill()
    if strategy == "Backward Fill":
        return series.bfill()
    if strategy == "Interpolate":
        return series.interpolate(limit_direction="both")
    raise ValueError(f"Unsupported numeric strategy: {strategy}")


def _fill_categorical(
    series: pd.Series, strategy: str, custom_value: Optional[str] = None
) -> pd.Series:
    if strategy == "Mode":
        mode_series = series.dropna().mode()
        if mode_series.empty:
            return series.fillna("Unknown")
        return series.fillna(mode_series.iloc[0])
    if strategy == "Forward Fill":
        return series.ffill()
    if strategy == "Backward Fill":
        return series.bfill()
    if strategy == "Custom Value":
        value = custom_value if custom_value not in (None, "") else "Unknown"
        return series.fillna(value)
    raise ValueError(f"Unsupported categorical strategy: {strategy}")


def _fill_datetime(series: pd.Series, strategy: str) -> pd.Series:
    if strategy == "Forward Fill":
        return series.ffill()
    if strategy == "Backward Fill":
        return series.bfill()
    if strategy == "Interpolate":
        # Interpolate on the integer nanosecond representation, then cast back.
        try:
            numeric_view = series.view("int64")
            numeric_view = numeric_view.mask(series.isna())
            interpolated = numeric_view.interpolate(limit_direction="both")
            return pd.to_datetime(interpolated)
        except (TypeError, ValueError):
            return series.ffill().bfill()
    raise ValueError(f"Unsupported datetime strategy: {strategy}")


def handle_missing_values(
    df: pd.DataFrame,
    plan: Dict[str, str],
    custom_values: Optional[Dict[str, str]] = None,
) -> CleaningResult:
    """
    Apply the given per-column strategy plan to fill missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataset (not modified).
    plan : dict
        Mapping of column name -> strategy label (e.g. "Median", "Mode").
    custom_values : dict, optional
        Mapping of column name -> custom fill value, used only when the
        strategy for that column is "Custom Value".
    """
    custom_values = custom_values or {}
    cleaned = df.copy(deep=True)
    column_types = detect_column_types(df)
    log: List[CleaningLogEntry] = []
    skipped: List[str] = []

    for col, strategy in plan.items():
        if col not in cleaned.columns:
            continue

        before_missing = int(cleaned[col].isna().sum())
        if before_missing == 0:
            continue

        try:
            if col in column_types["numeric"]:
                if strategy not in NUMERIC_STRATEGIES:
                    skipped.append(col)
                    continue
                cleaned[col] = _fill_numeric(cleaned[col], strategy)
            elif col in column_types["datetime"]:
                if strategy not in DATETIME_STRATEGIES:
                    skipped.append(col)
                    continue
                cleaned[col] = _fill_datetime(cleaned[col], strategy)
            else:
                if strategy not in CATEGORICAL_STRATEGIES:
                    skipped.append(col)
                    continue
                cleaned[col] = _fill_categorical(
                    cleaned[col], strategy, custom_values.get(col)
                )

            after_missing = int(cleaned[col].isna().sum())
            filled = before_missing - after_missing
            note = ""
            if after_missing > 0:
                note = (
                    f"{after_missing} value(s) could not be filled "
                    "(e.g. fill/interpolate at dataset edges)."
                )
            log.append(
                CleaningLogEntry(
                    column=col, strategy=strategy, values_filled=filled, note=note
                )
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append(col)
            log.append(
                CleaningLogEntry(
                    column=col,
                    strategy=strategy,
                    values_filled=0,
                    note=f"Skipped due to error: {exc}",
                )
            )

    return CleaningResult(dataframe=cleaned, log=log, skipped=skipped)


def drop_missing_rows(df: pd.DataFrame, how: str = "any") -> "tuple[pd.DataFrame, int]":
    """
    Remove rows containing missing values.

    Parameters
    ----------
    how : str
        "any" -> drop rows with at least one missing value.
        "all" -> drop rows only when every value in the row is missing.
    """
    if how not in ("any", "all"):
        raise ValueError("how must be 'any' or 'all'")

    before = len(df)
    cleaned = df.dropna(how=how).reset_index(drop=True)
    removed = before - len(cleaned)
    return cleaned, removed


def preview_drop_missing_rows(df: pd.DataFrame, how: str = "any") -> int:
    """Return how many rows *would* be removed, without mutating anything."""
    if how not in ("any", "all"):
        raise ValueError("how must be 'any' or 'all'")
    return int(len(df) - len(df.dropna(how=how)))


def remove_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> "tuple[pd.DataFrame, int]":
    """Remove duplicate rows, returning the cleaned DataFrame and count removed."""
    before = len(df)
    cleaned = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removed = before - len(cleaned)
    return cleaned, removed


def preview_duplicate_count(df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
    """Return how many duplicate rows exist, without mutating anything."""
    return int(df.duplicated(subset=subset).sum())


def remove_empty_rows_and_columns(
    df: pd.DataFrame,
) -> "tuple[pd.DataFrame, int, List[str]]":
    """Drop rows and columns that are entirely empty (all values missing)."""
    empty_columns = [c for c in df.columns if df[c].isna().all()]
    cleaned = df.drop(columns=empty_columns) if empty_columns else df.copy()

    before = len(cleaned)
    cleaned = cleaned.dropna(how="all").reset_index(drop=True)
    removed_rows = before - len(cleaned)

    return cleaned, removed_rows, empty_columns
