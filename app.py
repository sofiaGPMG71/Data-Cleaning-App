from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Data Cleaning & Quality Platform",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem;}
    .subtitle {color: #64748b; margin-bottom: 1.2rem;}
    .section-title {font-size: 1.25rem; font-weight: 650; margin-top: 0.8rem;}
    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 10px;
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "original_df": None,
        "cleaned_df": None,
        "source_name": None,
        "source_ext": None,
        "cleaning_plan": None,
        "cleaning_report": None,
        "remove_duplicates": True,
        "missing_strategy": "Automatic Handling",
        "custom_strategies": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_app() -> None:
    for key in [
        "original_df",
        "cleaned_df",
        "source_name",
        "source_ext",
        "cleaning_plan",
        "cleaning_report",
        "custom_strategies",
    ]:
        st.session_state[key] = None if key != "custom_strategies" else {}
    st.rerun()


def safe_read_csv(uploaded_file: Any) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    last_error = None
    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False)
        except Exception as exc:
            last_error = exc

    # Final fallback: pandas' Python engine can tolerate some malformed CSVs.
    try:
        return pd.read_csv(
            io.BytesIO(raw),
            encoding="latin1",
            engine="python",
            on_bad_lines="warn",
        )
    except Exception as exc:
        raise ValueError(f"Unable to read the CSV file. Last error: {last_error or exc}") from exc


def load_dataset(uploaded_file: Any) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()

    try:
        if suffix == ".csv":
            df = safe_read_csv(uploaded_file)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Please upload CSV, XLSX, or XLS.")
    except Exception as exc:
        raise ValueError(f"Could not load '{uploaded_file.name}': {exc}") from exc

    if df is None:
        raise ValueError("The file did not produce a dataset.")

    # Normalize pandas' completely empty dataframe case without changing valid data.
    if df.shape[1] == 0:
        raise ValueError("The file contains no columns.")
    if df.shape[0] == 0:
        raise ValueError("The dataset contains headers but no data rows.")

    return df


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    detected = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            detected.append(col)
            continue

        if not (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
        ):
            continue

        non_null = s.dropna()
        if non_null.empty:
            continue

        # Avoid converting arbitrary numeric-looking strings.
        sample = non_null.astype(str).head(500)
        parsed = pd.to_datetime(sample, errors="coerce")
        ratio = parsed.notna().mean()

        name_hint = bool(
            re.search(
                r"(date|time|dob|birth|admission|discharge|joining|effective|expiry)",
                str(col),
                flags=re.I,
            )
        )
        if ratio >= 0.80 and (name_hint or ratio >= 0.95):
            detected.append(col)
    return detected


def column_type(df: pd.DataFrame, col: str, datetime_cols: list[str]) -> str:
    if col in datetime_cols:
        return "datetime"
    if pd.api.types.is_numeric_dtype(df[col]):
        return "numeric"
    return "categorical/string"


def get_dataset_summary(df: pd.DataFrame) -> dict[str, int]:
    numeric = df.select_dtypes(include=[np.number]).shape[1]
    datetime_cols = detect_datetime_columns(df)
    categorical = df.shape[1] - numeric - len(datetime_cols)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing": missing,
        "duplicates": duplicates,
        "numeric": int(numeric),
        "categorical": int(max(categorical, 0)),
        "datetime": int(len(datetime_cols)),
    }


def get_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    datetime_cols = detect_datetime_columns(df)
    rows = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        rows.append(
            {
                "Column Name": str(col),
                "Data Type": column_type(df, col, datetime_cols),
                "Pandas dtype": str(s.dtype),
                "Non-Null Count": int(s.notna().sum()),
                "Missing Count": missing,
                "Missing %": round(missing / len(df) * 100, 2) if len(df) else 0.0,
                "Unique Values": int(s.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def missing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    profile = get_column_profile(df)
    return profile.loc[
        profile["Missing Count"] > 0,
        ["Column Name", "Data Type", "Missing Count", "Missing %"],
    ].reset_index(drop=True)


def mode_or_unknown(series: pd.Series) -> Any:
    modes = series.dropna().mode()
    if len(modes):
        return modes.iloc[0]
    return "Unknown"


def build_automatic_plan(df: pd.DataFrame) -> pd.DataFrame:
    datetime_cols = detect_datetime_columns(df)
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing == 0:
            continue
        typ = column_type(df, col, datetime_cols)
        if typ == "numeric":
            method = "Median"
        elif typ == "categorical/string":
            method = "Mode (fallback: Unknown)"
        else:
            method = "Forward Fill → Backward Fill"
        rows.append(
            {
                "Column": col,
                "Type": typ,
                "Missing Before": missing,
                "Planned Strategy": method,
            }
        )
    return pd.DataFrame(rows)


def apply_missing_strategy(
    df: pd.DataFrame,
    strategy: str,
    custom_strategies: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    out = df.copy(deep=True)
    datetime_cols = detect_datetime_columns(out)
    applied = {}
    unresolved = []

    if strategy == "Remove Rows (Any Missing)":
        before = len(out)
        out = out.dropna(how="any")
        applied["__rows_removed__"] = str(before - len(out))
        return out, applied, unresolved

    if strategy == "Remove Rows (All Values Missing)":
        before = len(out)
        out = out.dropna(how="all")
        applied["__rows_removed__"] = str(before - len(out))
        return out, applied, unresolved

    for col in out.columns:
        if out[col].isna().sum() == 0:
            continue

        typ = column_type(out, col, datetime_cols)
        method = (
            custom_strategies.get(col)
            if strategy == "Custom Handling" and custom_strategies
            else None
        )

        if method is None:
            if strategy == "Automatic Handling":
                method = {
                    "numeric": "Median",
                    "categorical/string": "Mode",
                    "datetime": "Forward Fill",
                }[typ]
            else:
                method = "Median" if typ == "numeric" else (
                    "Forward Fill" if typ == "datetime" else "Mode"
                )

        try:
            if method == "Mean" and typ == "numeric":
                value = out[col].mean()
                out[col] = out[col].fillna(value)
            elif method == "Median" and typ == "numeric":
                value = out[col].median()
                out[col] = out[col].fillna(value)
            elif method == "Mode" and typ == "categorical/string":
                out[col] = out[col].fillna(mode_or_unknown(out[col]))
            elif method == "Unknown" and typ == "categorical/string":
                out[col] = out[col].fillna("Unknown")
            elif method == "Forward Fill":
                if typ == "datetime" and not pd.api.types.is_datetime64_any_dtype(out[col]):
                    out[col] = pd.to_datetime(out[col], errors="coerce")
                out[col] = out[col].ffill()
            elif method == "Backward Fill":
                if typ == "datetime" and not pd.api.types.is_datetime64_any_dtype(out[col]):
                    out[col] = pd.to_datetime(out[col], errors="coerce")
                out[col] = out[col].bfill()
            elif method == "Interpolation" and typ == "numeric":
                out[col] = out[col].interpolate()
            else:
                unresolved.append(f"{col}: strategy '{method}' is not suitable for {typ}.")
                continue

            applied[col] = method
        except Exception as exc:
            unresolved.append(f"{col}: {method} failed ({exc}).")

    return out, applied, unresolved


def clean_dataset(
    original_df: pd.DataFrame,
    missing_strategy: str,
    remove_dupes: bool,
    custom_strategies: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = original_df.copy(deep=True)

    before_rows = len(cleaned)
    before_missing = int(cleaned.isna().sum().sum())
    before_duplicates = int(cleaned.duplicated().sum())

    cleaned, applied, unresolved = apply_missing_strategy(
        cleaned, missing_strategy, custom_strategies
    )

    duplicate_removed = 0
    if remove_dupes:
        duplicate_removed = int(cleaned.duplicated().sum())
        cleaned = cleaned.drop_duplicates().copy()

    after_missing = int(cleaned.isna().sum().sum())
    report = {
        "before_rows": before_rows,
        "after_rows": len(cleaned),
        "before_columns": original_df.shape[1],
        "after_columns": cleaned.shape[1],
        "before_missing": before_missing,
        "after_missing": after_missing,
        "before_duplicates": before_duplicates,
        "after_duplicates": int(cleaned.duplicated().sum()),
        "rows_removed_total": before_rows - len(cleaned),
        "missing_resolved": max(before_missing - after_missing, 0),
        "duplicates_removed": duplicate_removed,
        "applied": applied,
        "unresolved": unresolved,
        "affected_columns": [
            c for c in original_df.columns
            if original_df[c].isna().sum() > 0 and cleaned[c].isna().sum() < original_df[c].isna().sum()
        ],
    }
    return cleaned, report


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    empty_cols = [str(c) for c in df.columns if df[c].isna().all()]
    empty_rows = int(df.isna().all(axis=1).sum()) if len(df.columns) else 0
    return {
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "rows": len(df),
        "columns": df.shape[1],
        "empty_columns": empty_cols,
        "empty_rows": empty_rows,
        "dtypes": pd.DataFrame(
            {"Column": df.columns.astype(str), "Data Type": df.dtypes.astype(str)}
        ),
    }


def excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
    output.seek(0)
    return output.getvalue()


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def filename_for_download(source_name: str | None, extension: str) -> str:
    stem = Path(source_name or "dataset").stem
    return f"{stem}_cleaned.{extension}"


init_state()

with st.sidebar:
    st.markdown("## 🧹 Data Cleaning")
    st.caption("Professional data-quality workflow")
    if st.session_state.original_df is not None:
        st.divider()
        if st.button("↻ Start Over", use_container_width=True):
            reset_app()
    st.divider()
    st.info(
        "Your original dataset is kept separately in memory. "
        "Cleaning operations are only applied to a copy."
    )

st.markdown('<div class="main-title">DATA CLEANING & QUALITY PLATFORM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload, profile, safely clean, validate and download your dataset.</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload Your Dataset",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, XLSX and XLS.",
)

if uploaded is not None:
    is_new_file = (
        st.session_state.source_name != uploaded.name
        or st.session_state.original_df is None
    )
    if is_new_file:
        try:
            df = load_dataset(uploaded)
            st.session_state.original_df = df.copy(deep=True)
            st.session_state.cleaned_df = None
            st.session_state.source_name = uploaded.name
            st.session_state.source_ext = Path(uploaded.name).suffix.lower()
            st.session_state.cleaning_plan = None
            st.session_state.cleaning_report = None
            st.session_state.custom_strategies = {}
            st.success(f"Successfully loaded **{uploaded.name}** — {len(df):,} rows × {df.shape[1]:,} columns.")
        except Exception as exc:
            st.error(str(exc))

if st.session_state.original_df is None:
    st.info("Upload a CSV or Excel dataset to begin.")
    st.stop()

original_df = st.session_state.original_df
summary = get_dataset_summary(original_df)

st.markdown("### Dataset Overview")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Rows", f"{summary['rows']:,}")
k2.metric("Total Columns", f"{summary['columns']:,}")
k3.metric("Missing Values", f"{summary['missing']:,}")
k4.metric("Duplicate Rows", f"{summary['duplicates']:,}")
k5.metric("Numeric Columns", f"{summary['numeric']:,}")
k6.metric("Categorical/String", f"{summary['categorical']:,}")

tab_info, tab_preview, tab_missing, tab_clean, tab_results = st.tabs(
    ["Dataset Information", "Data Preview", "Missing Analysis", "Cleaning Configuration", "Cleaning Results"]
)

with tab_info:
    st.markdown("### Column Names & Profile")
    st.dataframe(
        get_column_profile(original_df),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("### Dataset Information")
    mem_mb = original_df.memory_usage(deep=True).sum() / (1024**2)
    st.write(
        f"Dataset contains **{len(original_df):,} rows** and **{original_df.shape[1]:,} columns**. "
        f"Memory usage is approximately **{mem_mb:.2f} MB**."
    )
    st.caption(
        "The profile above reproduces the useful operational information normally inspected with pandas df.info()."
    )

with tab_preview:
    st.markdown("### Data Preview")
    n = st.selectbox("Rows to view", [5, 10, 25, 50, 100], index=1)
    preview_mode = st.radio(
        "Preview mode",
        ["First N", "Last N", "Random Sample"],
        horizontal=True,
    )
    if preview_mode == "First N":
        view = original_df.head(n)
    elif preview_mode == "Last N":
        view = original_df.tail(n)
    else:
        view = original_df.sample(min(n, len(original_df)), random_state=42)
    st.dataframe(view, use_container_width=True, hide_index=True)

with tab_missing:
    st.markdown("### Missing Value Analysis")
    miss = missing_analysis(original_df)
    if miss.empty:
        st.success("No missing values were detected in this dataset.")
    else:
        total_cells = original_df.shape[0] * original_df.shape[1]
        total_missing = int(original_df.isna().sum().sum())
        cols_with_missing = len(miss)
        pct_dataset = total_missing / total_cells * 100 if total_cells else 0
        a, b, c = st.columns(3)
        a.metric("Total Missing Cells", f"{total_missing:,}")
        b.metric("Columns With Missing", f"{cols_with_missing:,}")
        c.metric("Dataset Missing %", f"{pct_dataset:.2f}%")
        st.dataframe(miss, use_container_width=True, hide_index=True)

with tab_clean:
    st.markdown("### Cleaning Configuration")

    strategy = st.radio(
        "Missing Value Strategy",
        [
            "Automatic Handling",
            "Remove Rows (Any Missing)",
            "Remove Rows (All Values Missing)",
            "Custom Handling",
        ],
        index=0,
    )
    st.session_state.missing_strategy = strategy

    custom_strategies: dict[str, str] = {}
    if strategy == "Custom Handling":
        missing_cols = missing_analysis(original_df)
        if missing_cols.empty:
            st.info("There are no missing values requiring configuration.")
        else:
            st.caption("Only suitable strategies are offered for each detected data type.")
            for _, row in missing_cols.iterrows():
                col = row["Column Name"]
                typ = row["Data Type"]
                if typ == "numeric":
                    options = ["Median", "Mean", "Forward Fill", "Backward Fill", "Interpolation"]
                    default = "Median"
                elif typ == "datetime":
                    options = ["Forward Fill", "Backward Fill"]
                    default = "Forward Fill"
                else:
                    options = ["Mode", "Unknown", "Forward Fill", "Backward Fill"]
                    default = "Mode"
                custom_strategies[col] = st.selectbox(
                    f"{col} ({typ})",
                    options,
                    index=options.index(default),
                    key=f"strategy_{col}",
                )
        st.session_state.custom_strategies = custom_strategies

    st.markdown("#### Duplicate Handling")
    remove_dupes = st.checkbox(
        "Remove Duplicate Records",
        value=True,
        help="Duplicates are removed only from the cleaned copy.",
    )
    st.session_state.remove_duplicates = remove_dupes

    st.markdown("#### Planned Cleaning Operations")
    if strategy in {"Automatic Handling", "Custom Handling"}:
        if strategy == "Automatic Handling":
            plan = build_automatic_plan(original_df)
        else:
            plan_rows = []
            miss = missing_analysis(original_df)
            for _, row in miss.iterrows():
                plan_rows.append(
                    {
                        "Column": row["Column Name"],
                        "Type": row["Data Type"],
                        "Missing Before": row["Missing Count"],
                        "Planned Strategy": custom_strategies.get(row["Column Name"], "—"),
                    }
                )
            plan = pd.DataFrame(plan_rows)
    else:
        plan = pd.DataFrame(
            {
                "Operation": [strategy],
                "Rows potentially affected": [
                    int(original_df.isna().any(axis=1).sum())
                    if "Any" in strategy
                    else int(original_df.isna().all(axis=1).sum())
                ],
            }
        )

    if not plan.empty:
        st.dataframe(plan, use_container_width=True, hide_index=True)

    if remove_dupes:
        st.info(
            f"{summary['duplicates']:,} duplicate rows are currently detected and "
            "will be removed when cleaning is executed."
        )

    st.markdown("#### Confirmation")
    confirm = st.checkbox(
        "I have reviewed the planned operations and approve applying them to a copy of the dataset.",
        value=False,
    )

    if st.button("▶ Preview & Execute Cleaning", type="primary", use_container_width=True):
        if not confirm:
            st.warning("Please review the plan and tick the confirmation box before cleaning.")
        else:
            with st.spinner("Cleaning and validating dataset..."):
                cleaned, report = clean_dataset(
                    original_df=original_df,
                    missing_strategy=strategy,
                    remove_dupes=remove_dupes,
                    custom_strategies=custom_strategies,
                )
                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_report = report
                st.session_state.cleaning_plan = plan
            st.success("Cleaning completed. Review the Cleaning Results tab before downloading.")

with tab_results:
    if st.session_state.cleaned_df is None:
        st.info("No cleaning run yet. Configure and execute the cleaning process first.")
    else:
        cleaned_df = st.session_state.cleaned_df
        report = st.session_state.cleaning_report
        st.markdown("### Cleaning Results")

        comparison = pd.DataFrame(
            {
                "Metric": ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
                "Before": [
                    report["before_rows"],
                    report["before_columns"],
                    report["before_missing"],
                    report["before_duplicates"],
                ],
                "After": [
                    report["after_rows"],
                    report["after_columns"],
                    report["after_missing"],
                    report["after_duplicates"],
                ],
            }
        )
        comparison["Change"] = comparison["After"] - comparison["Before"]
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        a, b, c, d = st.columns(4)
        a.metric("Rows Removed", f"{report['rows_removed_total']:,}")
        b.metric("Missing Resolved", f"{report['missing_resolved']:,}")
        c.metric("Duplicates Removed", f"{report['duplicates_removed']:,}")
        d.metric("Rows Retained", f"{len(cleaned_df):,}")

        if report["affected_columns"]:
            st.write("**Columns affected by missing-value cleaning:**")
            st.write(", ".join(map(str, report["affected_columns"])))
        else:
            st.write("**Columns affected by missing-value cleaning:** None")

        validation = validate_dataset(cleaned_df)
        st.markdown("### Data Quality Status")
        if validation["missing"] == 0:
            st.success("✅ No missing values remain.")
        else:
            st.warning(
                f"⚠️ {validation['missing']:,} missing values remain. "
                "These require manual review."
            )

        if validation["duplicates"] == 0:
            st.success("✅ No duplicate records remain.")
        else:
            st.warning(f"⚠️ {validation['duplicates']:,} duplicate records remain.")

        if validation["empty_columns"]:
            st.warning(
                "⚠️ Completely empty columns remain: "
                + ", ".join(validation["empty_columns"])
            )
        else:
            st.success("✅ No completely empty columns.")

        if validation["empty_rows"] > 0:
            st.warning(f"⚠️ {validation['empty_rows']:,} completely empty rows remain.")
        else:
            st.success("✅ No completely empty rows.")

        if report["unresolved"]:
            with st.expander("Review unresolved cleaning items"):
                for item in report["unresolved"]:
                    st.warning(item)

        st.markdown("### Cleaned Dataset")
        preview_n = st.selectbox(
            "Rows to inspect",
            [5, 10, 25, 50, 100],
            index=1,
            key="cleaned_preview_n",
        )
        mode = st.radio(
            "Inspect",
            ["First N", "Last N", "Random Sample"],
            horizontal=True,
            key="cleaned_preview_mode",
        )
        if mode == "First N":
            cleaned_view = cleaned_df.head(preview_n)
        elif mode == "Last N":
            cleaned_view = cleaned_df.tail(preview_n)
        else:
            cleaned_view = cleaned_df.sample(
                min(preview_n, len(cleaned_df)), random_state=42
            )
        st.dataframe(cleaned_view, use_container_width=True, hide_index=True)

        st.markdown("### Download Cleaned Dataset")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇ Download CSV",
                data=csv_bytes(cleaned_df),
                file_name=filename_for_download(st.session_state.source_name, "csv"),
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇ Download Excel",
                data=excel_bytes(cleaned_df),
                file_name=filename_for_download(st.session_state.source_name, "xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

st.divider()
st.caption(
    "Data Cleaning & Quality Platform • Original data is never overwritten • "
    "Cleaning actions are explicitly confirmed before execution"
)
