# Datathon-25 — Streamlit Forecast & Anomaly Explorer

A small Streamlit app for exploring Arizona time-series forecasting and anomaly reports used in the Datathon-25 project.

**Quick Summary:**
- **What:** Interactive Streamlit dashboard to view forecasts, weekly summaries, and anomaly reports from included CSV datasets.
- **Where:** Launch locally with `streamlit run app.py`.

## Files
- `app.py`: Streamlit application entrypoint.
- `requirements.txt`: Python dependencies for the app.
- `AZ_Master_Cleaned_Data.csv`: Master cleaned dataset used by the app.
- `AZ_Weekly_Summary.csv`: Weekly summary metrics.
- `AZ_Anomaly_Report.csv`: Detected anomalies and related info.
- `AZ_6Month_Forecast.csv`: Six-month forecast output used for visualization.

## Setup (local)
1. Create and activate a virtual environment (recommended):

	python -m venv .venv
	source .venv/bin/activate

2. Install dependencies:

	pip install -r requirements.txt

3. Run the Streamlit app:

	streamlit run app.py / python3 -m streamlit run app.py


The app will open in your browser at the address printed by Streamlit (usually http://localhost:8501).

## Usage
- Use the sidebar controls in the app to select datasets, date ranges, and visualization options.
- View forecasts, compare historical vs predicted values, and inspect anomaly reports.
- Export or download CSV snapshots if the app exposes those actions (check the app UI).

## Development notes
- The app expects the CSV files to be present in the repository root (same folder as `app.py`).
- If you add larger datasets, update `app.py` as needed to handle loading and caching.

## Data
The repository includes sample CSV exports used by the Datathon. Confirm provenance and licensing before using for any external purpose.

## Troubleshooting
- If Streamlit fails to start, ensure you have a supported Python version (3.8+ recommended) and that dependencies installed successfully.
- If CSVs fail to load, check that filenames match those listed above and are in the repository root.

## License & Contact
This repository is provided as-is for the Datathon. For questions, open an issue or contact the maintainer.
