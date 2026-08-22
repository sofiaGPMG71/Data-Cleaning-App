"""
app.py
-------
Data Cleaning & Quality Platform
A professional Streamlit dashboard for uploading, inspecting, cleaning, and
exporting tabular datasets (CSV / Excel).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from modules.data_loader import load_dataset
from modules.analysis import (
    detect_column_types,
    get_dataset_summary,
    get_column_overview_table,
    get_dataset_info_table,
    get_dataset_info_summary_text,
    analyze_missing_values,
    find_problem_columns,
)
from modules.cleaning import (
    build_auto_cleaning_plan,
    handle_missing_values,
    drop_missing_rows,
    preview_drop_missing_rows,
    remove_duplicates,
    preview_duplicate_count,
    NUMERIC_STRATEGIES,
    CATEGORICAL_STRATEGIES,
    DATETIME_STRATEGIES,
)
from modules.validation import validate_dataset
from modules.report import (
    generate_cleaning_report,
    summarize_cleaning_impact,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    build_cleaned_filename,
)

# --------------------------------------------------------------------------
# Page configuration & styling
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Data Cleaning & Quality Platform",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #6b7280;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.35rem;
        font-weight: 600;
        color: #111827;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.9rem 0.8rem 0.6rem 0.8rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "original_df": None,
        "working_df": None,
        "filename": None,
        "file_extension": None,
        "cleaning_log": [],
        "rows_removed_missing": 0,
        "duplicates_removed": 0,
        "missing_step_applied": False,
        "duplicate_step_applied": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_app() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


init_session_state()


# --------------------------------------------------------------------------
# Sidebar: upload + reset
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🧹 Data Cleaning & Quality Platform")
    st.caption("Upload a dataset to begin.")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Drag and drop your CSV or Excel file here, or browse your computer.",
    )

    if uploaded_file is not None:
        # Only (re)load if this is a new file, so we don't clobber cleaning
        # progress on every rerun.
        if st.session_state["filename"] != uploaded_file.name or st.session_state["original_df"] is None:
            result = load_dataset(uploaded_file)
            if result.success:
                st.session_state["original_df"] = result.dataframe
                st.session_state["working_df"] = result.dataframe.copy(deep=True)
                st.session_state["filename"] = result.filename
                st.session_state["file_extension"] = result.file_extension
                st.session_state["cleaning_log"] = []
                st.session_state["rows_removed_missing"] = 0
                st.session_state["duplicates_removed"] = 0
                st.session_state["missing_step_applied"] = False
                st.session_state["duplicate_step_applied"] = False
                st.success(f"Loaded '{result.filename}' successfully.")
                if result.error_message:
                    st.warning(result.error_message)
            else:
                st.error(result.error_message)

    st.divider()
    if st.session_state["original_df"] is not None:
        if st.button("🔄 Start Over / Reset Dataset", width="stretch"):
            reset_app()
            st.rerun()


# --------------------------------------------------------------------------
# Main header
# --------------------------------------------------------------------------

st.markdown('<p class="main-header">Data Cleaning & Quality Platform</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload, inspect, clean, and export your dataset with confidence.</p>',
    unsafe_allow_html=True,
)

original_df: pd.DataFrame | None = st.session_state["original_df"]

if original_df is None:
    st.info("👈 Upload a CSV or Excel file from the sidebar to get started.")
    st.markdown(
        """
        **What this app does:**
        - Analyzes data quality (missing values, duplicates, types)
        - Lets you choose or auto-select cleaning strategies
        - Shows exactly what will change before applying anything
        - Produces a downloadable cleaned CSV / Excel file
        """
    )
    st.stop()

working_df: pd.DataFrame = st.session_state["working_df"]
column_types = detect_column_types(original_df)
problem_cols = find_problem_columns(original_df)

if working_df.shape[0] == 0 or working_df.shape[1] == 0:
    st.warning(
        "This file only contains headers (no data rows) or has no columns. "
        "Upload a dataset with data to continue."
    )
    st.stop()

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tab_overview, tab_preview, tab_missing, tab_duplicates, tab_clean, tab_download = st.tabs(
    [
        "📊 Overview",
        "🔍 Data Preview",
        "❓ Missing Values",
        "📑 Duplicates",
        "✅ Clean & Validate",
        "⬇️ Download",
    ]
)

# ---------------------------- Overview tab --------------------------------
with tab_overview:
    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)
    summary = get_dataset_summary(original_df)

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Total Rows", f"{summary['total_rows']:,}")
    kpi_cols[1].metric("Total Columns", f"{summary['total_columns']:,}")
    kpi_cols[2].metric("Missing Values", f"{summary['total_missing_values']:,}")
    kpi_cols[3].metric("Duplicate Rows", f"{summary['duplicate_rows']:,}")
    kpi_cols[4].metric("Numeric Columns", summary["numeric_columns"])
    kpi_cols[5].metric("Categorical Columns", summary["categorical_columns"])

    st.markdown('<p class="section-header">Column Names</p>', unsafe_allow_html=True)
    st.dataframe(get_column_overview_table(original_df), width="stretch", hide_index=True)

    st.markdown('<p class="section-header">Dataset Information</p>', unsafe_allow_html=True)
    st.dataframe(get_dataset_info_table(original_df), width="stretch", hide_index=True)
    st.info(get_dataset_info_summary_text(original_df))
    st.caption(f"Approximate memory usage: {summary['memory_usage_mb']} MB")

    if problem_cols["all_null_columns"] or problem_cols["mixed_type_columns"]:
        with st.expander("⚠️ Data Quality Warnings", expanded=True):
            if problem_cols["all_null_columns"]:
                st.warning(
                    "Completely empty column(s): " + ", ".join(problem_cols["all_null_columns"])
                )
            if problem_cols["mixed_type_columns"]:
                st.warning(
                    "Column(s) with mixed data types: " + ", ".join(problem_cols["mixed_type_columns"])
                )

# ---------------------------- Data Preview tab -----------------------------
with tab_preview:
    st.markdown('<p class="section-header">Data Preview</p>', unsafe_allow_html=True)
    row_options = [5, 10, 25, 50, 100]
    n_rows = st.selectbox("Rows to display", row_options, index=1)
    st.dataframe(original_df.head(n_rows), width="stretch")
    st.caption(f"Showing {min(n_rows, len(original_df))} of {len(original_df):,} rows.")

# ---------------------------- Missing Values tab ----------------------------
with tab_missing:
    st.markdown('<p class="section-header">Missing Value Analysis</p>', unsafe_allow_html=True)
    missing_report = analyze_missing_values(working_df)

    if not missing_report["has_missing"]:
        st.success("✅ No missing values were detected in this dataset.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Missing Cells", f"{missing_report['total_missing_cells']:,}")
        m2.metric("Columns Affected", missing_report["columns_with_missing"])
        m3.metric("% of Dataset Missing", f"{missing_report['overall_missing_percentage']}%")
        st.dataframe(missing_report["table"], width="stretch", hide_index=True)

        st.markdown('<p class="section-header">Handle Missing Values</p>', unsafe_allow_html=True)

        if st.session_state["missing_step_applied"]:
            st.success("Missing-value handling has already been applied for this session.")
            st.caption("Use 'Start Over' in the sidebar to redo this step from scratch.")
        else:
            strategy_mode = st.radio(
                "Choose a missing-value handling approach:",
                ["Remove Rows", "Automatic Fill", "Custom Per-Column Fill"],
                horizontal=True,
            )

            # --- Remove rows ---
            if strategy_mode == "Remove Rows":
                how = st.radio(
                    "Removal rule",
                    ["Remove rows with ANY missing value", "Remove rows where ALL values are missing"],
                )
                how_arg = "any" if "ANY" in how else "all"
                preview_count = preview_drop_missing_rows(working_df, how=how_arg)
                st.info(f"This operation will remove **{preview_count:,}** row(s).")

                if st.button("✅ Confirm: Remove Rows", type="primary"):
                    cleaned, removed = drop_missing_rows(working_df, how=how_arg)
                    st.session_state["working_df"] = cleaned
                    st.session_state["rows_removed_missing"] = removed
                    st.session_state["missing_step_applied"] = True
                    st.success(f"Removed {removed:,} row(s).")
                    st.rerun()

            # --- Automatic fill ---
            elif strategy_mode == "Automatic Fill":
                plan = build_auto_cleaning_plan(working_df)
                st.write("Planned operations:")
                plan_df = pd.DataFrame(
                    [{"Column": c, "Strategy": s} for c, s in plan.items()]
                )
                st.dataframe(plan_df, width="stretch", hide_index=True)
                st.caption(
                    "Numeric columns use Median, categorical columns use Mode "
                    "(falling back to 'Unknown' if no mode exists), and datetime "
                    "columns use Forward Fill."
                )

                if st.button("✅ Confirm: Apply Automatic Fill", type="primary"):
                    result = handle_missing_values(working_df, plan)
                    st.session_state["working_df"] = result.dataframe
                    st.session_state["cleaning_log"] = result.log
                    st.session_state["missing_step_applied"] = True
                    st.success("Automatic missing-value handling applied.")
                    st.rerun()

            # --- Custom per-column fill ---
            else:
                st.write("Choose a strategy for each column that has missing values:")
                custom_plan = {}
                custom_values = {}

                for col in missing_report["table"]["Column"].tolist():
                    if col in column_types["numeric"]:
                        options = NUMERIC_STRATEGIES
                    elif col in column_types["datetime"]:
                        options = DATETIME_STRATEGIES
                    else:
                        options = CATEGORICAL_STRATEGIES

                    with st.expander(f"Column: {col}", expanded=False):
                        choice = st.selectbox(
                            "Strategy", options, key=f"strategy_{col}"
                        )
                        custom_plan[col] = choice
                        if choice == "Custom Value":
                            custom_values[col] = st.text_input(
                                "Fill value", value="Unknown", key=f"customval_{col}"
                            )

                if st.button("✅ Confirm: Apply Custom Fill", type="primary"):
                    result = handle_missing_values(working_df, custom_plan, custom_values)
                    st.session_state["working_df"] = result.dataframe
                    st.session_state["cleaning_log"] = result.log
                    st.session_state["missing_step_applied"] = True
                    st.success("Custom missing-value handling applied.")
                    st.rerun()

# ---------------------------- Duplicates tab --------------------------------
with tab_duplicates:
    st.markdown('<p class="section-header">Duplicate Records</p>', unsafe_allow_html=True)
    dup_count = preview_duplicate_count(working_df)
    dup_pct = round((dup_count / len(working_df)) * 100, 2) if len(working_df) else 0.0

    d1, d2 = st.columns(2)
    d1.metric("Duplicate Rows", f"{dup_count:,}")
    d2.metric("Duplicate %", f"{dup_pct}%")

    if dup_count == 0:
        st.success("✅ No duplicate records were detected.")
    elif st.session_state["duplicate_step_applied"]:
        st.success("Duplicate removal has already been applied for this session.")
    else:
        st.info(f"**{dup_count:,}** duplicate record(s) were detected and will be removed.")
        if st.button("✅ Confirm: Remove Duplicate Records", type="primary"):
            cleaned, removed = remove_duplicates(working_df)
            st.session_state["working_df"] = cleaned
            st.session_state["duplicates_removed"] = removed
            st.session_state["duplicate_step_applied"] = True
            st.success(f"Removed {removed:,} duplicate row(s).")
            st.rerun()

# ---------------------------- Clean & Validate tab ---------------------------
with tab_clean:
    st.markdown('<p class="section-header">Cleaning Results</p>', unsafe_allow_html=True)

    if not st.session_state["missing_step_applied"] and not st.session_state["duplicate_step_applied"]:
        st.info(
            "No cleaning operations have been applied yet. "
            "Use the 'Missing Values' and 'Duplicates' tabs to configure and confirm cleaning steps."
        )
    else:
        working_df = st.session_state["working_df"]
        report_table = generate_cleaning_report(original_df, working_df)
        st.dataframe(report_table, width="stretch", hide_index=True)

        impact = summarize_cleaning_impact(original_df, working_df)
        st.success(
            f"Cleaning completed successfully.\n\n"
            f"- {impact['missing_values_resolved']:,} missing value(s) resolved\n"
            f"- {impact['duplicates_removed']:,} duplicate row(s) removed\n"
            f"- {impact['rows_retained']:,} row(s) retained"
        )

        if st.session_state["cleaning_log"]:
            with st.expander("Missing-value handling details"):
                log_rows = [
                    {
                        "Column": entry.column,
                        "Strategy": entry.strategy,
                        "Values Filled": entry.values_filled,
                        "Note": entry.note,
                    }
                    for entry in st.session_state["cleaning_log"]
                ]
                st.dataframe(pd.DataFrame(log_rows), width="stretch", hide_index=True)

        st.markdown('<p class="section-header">Cleaned Dataset Preview</p>', unsafe_allow_html=True)
        view_mode = st.radio("View", ["First rows", "Last rows", "Random sample"], horizontal=True)
        n = st.slider("Number of rows", 5, min(100, max(5, len(working_df))), min(10, max(5, len(working_df))))

        if view_mode == "First rows":
            st.dataframe(working_df.head(n), width="stretch")
        elif view_mode == "Last rows":
            st.dataframe(working_df.tail(n), width="stretch")
        else:
            sample_n = min(n, len(working_df))
            st.dataframe(working_df.sample(sample_n) if sample_n > 0 else working_df, width="stretch")

        st.markdown('<p class="section-header">Data Quality Status</p>', unsafe_allow_html=True)
        validation = validate_dataset(working_df)

        if validation.is_fully_clean:
            st.success("✅ No missing values\n\n✅ No duplicate records\n\n✅ Dataset successfully cleaned")
        else:
            st.warning("⚠️ Some issues remain and may require manual review:")
            for w in validation.warnings:
                st.write(f"- {w}")

# ---------------------------- Download tab -----------------------------------
with tab_download:
    st.markdown('<p class="section-header">Download Cleaned Dataset</p>', unsafe_allow_html=True)

    final_df = st.session_state["working_df"]
    if not st.session_state["missing_step_applied"] and not st.session_state["duplicate_step_applied"]:
        st.warning(
            "No cleaning has been applied yet — this would download the dataset unchanged. "
            "Consider applying cleaning steps first, or proceed if no cleaning is needed."
        )

    original_filename = st.session_state["filename"] or "dataset.csv"
    csv_bytes = dataframe_to_csv_bytes(final_df)
    csv_name = build_cleaned_filename(original_filename, "csv")

    excel_bytes = dataframe_to_excel_bytes(final_df)
    excel_name = build_cleaned_filename(original_filename, "xlsx")

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Download as CSV",
            data=csv_bytes,
            file_name=csv_name,
            mime="text/csv",
            width="stretch",
            type="primary",
        )
    with dl2:
        st.download_button(
            "⬇️ Download as Excel",
            data=excel_bytes,
            file_name=excel_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

    st.caption(f"Final dataset: {final_df.shape[0]:,} rows × {final_df.shape[1]:,} columns.")
