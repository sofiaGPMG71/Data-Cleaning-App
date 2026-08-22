"""
data_loader.py
---------------
Handles reading uploaded CSV / Excel files into Pandas DataFrames safely,
with graceful error handling for corrupted or invalid files.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class LoadResult:
    """Container for the outcome of a dataset load attempt."""
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    filename: Optional[str] = None
    file_extension: Optional[str] = None
    error_message: Optional[str] = None


def _read_csv_with_fallback(file_bytes: bytes) -> pd.DataFrame:
    """Try reading CSV bytes with a series of common encodings."""
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_error: Optional[Exception] = None

    for encoding in encodings_to_try:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise exc
        except pd.errors.ParserError as exc:
            # Try the python engine as a more forgiving fallback.
            try:
                return pd.read_csv(
                    io.BytesIO(file_bytes), encoding=encoding, engine="python"
                )
            except Exception:
                last_error = exc
                continue

    # If every encoding failed, raise the most recent error encountered.
    raise last_error if last_error else ValueError("Unable to parse CSV file.")


def load_dataset(uploaded_file) -> LoadResult:
    """
    Load an uploaded CSV or Excel file into a Pandas DataFrame.

    Parameters
    ----------
    uploaded_file : streamlit.UploadedFile
        The file object returned by st.file_uploader.

    Returns
    -------
    LoadResult
        Structured result containing either the loaded DataFrame or a
        human-readable error message.
    """
    if uploaded_file is None:
        return LoadResult(success=False, error_message="No file was provided.")

    filename = uploaded_file.name
    extension = filename.split(".")[-1].lower() if "." in filename else ""

    try:
        file_bytes = uploaded_file.getvalue()

        if len(file_bytes) == 0:
            return LoadResult(
                success=False,
                filename=filename,
                file_extension=extension,
                error_message="The uploaded file is empty (0 bytes).",
            )

        if extension == "csv":
            df = _read_csv_with_fallback(file_bytes)
        elif extension in ("xlsx", "xls"):
            engine = "openpyxl" if extension == "xlsx" else None
            df = pd.read_excel(io.BytesIO(file_bytes), engine=engine)
        else:
            return LoadResult(
                success=False,
                filename=filename,
                file_extension=extension,
                error_message=(
                    f"Unsupported file type '.{extension}'. "
                    "Please upload a .csv, .xlsx, or .xls file."
                ),
            )

        if df.shape[1] == 0:
            return LoadResult(
                success=False,
                filename=filename,
                file_extension=extension,
                error_message="The file could not be parsed into any columns.",
            )

        if df.shape[0] == 0:
            return LoadResult(
                success=True,
                dataframe=df,
                filename=filename,
                file_extension=extension,
                error_message=(
                    "Warning: the file contains headers but no data rows."
                ),
            )

        return LoadResult(
            success=True, dataframe=df, filename=filename, file_extension=extension
        )

    except pd.errors.EmptyDataError:
        return LoadResult(
            success=False,
            filename=filename,
            file_extension=extension,
            error_message="The file appears to be empty or has no readable columns.",
        )
    except pd.errors.ParserError as exc:
        return LoadResult(
            success=False,
            filename=filename,
            file_extension=extension,
            error_message=f"The CSV file could not be parsed: {exc}",
        )
    except ValueError as exc:
        return LoadResult(
            success=False,
            filename=filename,
            file_extension=extension,
            error_message=f"Could not read the file: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - surface any unexpected error safely
        return LoadResult(
            success=False,
            filename=filename,
            file_extension=extension,
            error_message=f"An unexpected error occurred while reading the file: {exc}",
        )
