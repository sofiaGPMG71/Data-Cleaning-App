# Data Cleaning & Quality Platform

A professional Streamlit dashboard for uploading, inspecting, cleaning, and
exporting tabular datasets (CSV / Excel).

## Features

- Upload CSV, XLSX, or XLS files with robust error handling (empty files,
  corrupted Excel files, unsupported types, encoding issues, ragged CSVs).
- **Dataset Overview**: KPI cards, per-column summary table, and a
  `df.info()`-style breakdown.
- **Data Preview**: adjustable row count (5 / 10 / 25 / 50 / 100).
- **Missing Value Analysis**: per-column missing counts/percentages plus
  three handling strategies:
  - Remove rows (any missing value, or only fully-empty rows)
  - Automatic fill (Median for numeric, Mode for categorical — falling back
    to `"Unknown"` if no mode exists — Forward Fill for datetime)
  - Custom per-column strategy (Mean/Median/Forward Fill/Backward
    Fill/Interpolate for numeric; Mode/Forward Fill/Backward Fill/Custom
    Value for categorical; Forward/Backward Fill/Interpolate for datetime)
- **Duplicate Detection**: count, percentage, and one-click safe removal
  with a preview of how many rows will be affected before you confirm.
- **Cleaning Results**: before/after comparison table, a plain-language
  summary, and a detailed log of exactly which columns were filled with
  which strategy.
- **Data Quality Status / Validation**: after cleaning, the app re-checks
  for remaining missing values, duplicates, empty rows/columns, and reports
  honestly if the dataset is not fully clean.
- **Download**: cleaned dataset as CSV or Excel (generated in memory via
  `BytesIO`, no temp files), named `<original>_cleaned.<ext>`.
- **Reset**: start over at any point without restarting the app.

The original uploaded dataset is never mutated — all cleaning operations
work on a separate in-memory copy (`working_df`), so you can always compare
before/after or reset back to the original.

## Project Structure

```
app.py                     Streamlit UI (page layout, tabs, session state)
modules/
  data_loader.py           File upload parsing + error handling
  analysis.py               Read-only dataset analysis (overview, info, missing values)
  cleaning.py                Data-mutating logic (fill strategies, dedup, auto plan)
  validation.py              Post-cleaning quality checks
  report.py                   Before/after report + CSV/Excel export helpers
requirements.txt
```

## Installation

1. (Recommended) create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in
your browser.

## Notes

- Large files: performance depends on available memory since the app works
  with in-memory Pandas DataFrames; very large files (hundreds of MB+) may
  be slow in a browser-based interface like Streamlit.
- Datetime columns are only auto-detected when a column's name hints at a
  date/time field (e.g. contains "date", "time", "timestamp") **and** its
  values actually parse as dates — this avoids accidentally reinterpreting
  unrelated columns based on name alone.
