# MMA3001 Occupancy Analysis

## Project Overview

This project analyses occupancy sensor measurements collected across five spaces between May and December 2024 created for MMA3001.

The computational workflow reconstructs hourly occupancy, evaluates methods for forecasting future occupancy, identifies unusual behaviour, and analyses space underutilisation.

The project uses Python and combines numerical data processing, machine learning, anomaly detection, and validation.

## Project Structure

- `main.py` — Runs the complete analysis.
- `mma3001_stage1_occupancy_audit.py` — Audits the raw dataset and reconstructs hourly occupancy.
- `mma3001_stage2_occupancy_forecasting.py` — Develops and evaluates future occupancy forecasts.
- `mma3001_stage3_anomaly_detection.py` — Identifies unusual occupancy patterns residuals from forecasting.
- `mma3001_stage4_underutilisation.py` — Evaluates occupancy underutilisation across the monitored spaces.
- `mma3001_stage5_final_validation.py` — Performs additional validation checks on the analysis.
- `requirements.txt` — Lists the Python packages required.
- `data/` — Directory for the original dataset, which is not included.
- `docs/` — HTML source-code documentation.

## Dataset

The project uses the dataset provided by the unit:

`5occupancySensor_MayToDec2024_9MRows.csv`

This dataset contains approximately 9.09 million records covering May to December 2024.

**Dataset source:** [MMA3001 unit-provided Google Drive folder](https://drive.google.com/drive/folders/1sB4UyZc4c6h67iKfmWcCm0Kv3DSjnCb2)

The original dataset is not provided within this GitHub repository. To reproduce the analysis, place the CSV from the unit-provided source in the project's `data/` directory.
Expected location:

`data/5occupancySensor_MayToDec2024_9MRows.csv`

## Requirements

- Python 3.10 or later
- NumPy
- pandas
- scikit-learn
- Matplotlib

Install the required packages by opening a terminal in the project directory and running:

python -m pip install -r requirements.txt

## Running the Project

1. Download the original occupancy CSV from the Google Drive folder provided by the unit.
2. Place the CSV in the `data/` directory with its original filename.
3. Install the Python dependencies using `requirements.txt`.
4. Open a terminal in the main project directory.
5. Run:

python main.py

The workflow executes the five analysis stages one after the other and generates the corresponding results and validation outputs.

## Analysis Methods

### Stage 1 — Occupancy Reconstruction

Audits the raw occupancy records and reconstructs hourly mean occupancy using time weighted calculations.

### Stage 2 — Occupancy Forecasting

Evaluates methods of predicting mean occupancy for the next hour using historical occupancy and calendar features, including comparisons against baseline methods.

### Stage 3 — Anomaly Detection

Use residuals from occupancy forecast to identify and summarise unusual deviations from expected occupancy.

### Stage 4 — Underutilisation Analysis

Quantifies time weighted occupancy and empty space fractions, including analysis for working hours and sensitivity testing.

### Stage 5 — Final Validation

Performs numerical checks to assess consistency, correctness, and reliability of the computational workflow.

## Validation

The final workflow was successfully executed in a clean testing folder using the original dataset.
All 16 numerical validation checks passed during the final verification run. These checks support the internal numerical consistency of the implemented methods but do not independently establish the accuracy of the original sensor measurements.

## Documentation

The Python source files contain function docstrings and inline comments.

Generated HTML documentation available in the `docs/` directory.

## Dataset Reference

Monash University n.d., Occupancy sensor dataset: May–December 2024 [CSV dataset], MMA3001 Numerical Methods and Machine Learning, provided by the unit, viewed 13 October 2026, https://drive.google.com/drive/folders/1sB4UyZc4c6h67iKfmWcCm0Kv3DSjnCb2.