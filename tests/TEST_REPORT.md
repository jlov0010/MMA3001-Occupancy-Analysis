# Validation Test Report

The project uses four validation functions in `mma3001_stage5_final_validation.py`:

- `validate_integration()` — Checks occupancy and utilisation calculations.
- `validate_forecasts()` — Verifies MAE, RMSE and R².
- `validate_anomalies()` — Verifies anomaly and event counts.
- `validate_rankings()` — Checks ranking stability across teaching-hour windows.

**Results: 16 checks passed, 0 failed.**

Detailed results are provided in `validation_checks.csv`, with supporting CSV tables and figures in this folder.

To reproduce the tests, run `python main.py` with the required dataset installed.

These checks verify numerical consistency, not the accuracy of the original sensor measurements.