# Data Cleaning & Quality Platform

A professional Streamlit application for profiling, cleaning, validating and exporting CSV/Excel datasets.

## Included

- `app.py` — complete Streamlit application
- `requirements.txt` — Python dependencies

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The application opens in your browser.

## Supported input

- CSV
- XLSX
- XLS

## Cleaning safeguards

- The original uploaded DataFrame is stored separately.
- Cleaning is performed on a deep copy.
- Planned operations are shown before execution.
- The user must explicitly confirm the cleaning operation.
- Cleaning statistics and validation checks are displayed before download.
- CSV and Excel downloads are generated from the cleaned DataFrame.

## Automatic missing-value rules

- Numeric → median
- Categorical/string → mode, with `Unknown` fallback
- Datetime → forward fill

Custom strategies are also available where appropriate.

## Notes

The application intentionally does not silently delete records. Every removal is reported in the cleaning results.
