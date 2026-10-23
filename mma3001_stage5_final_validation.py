
"""
MMA3001 Stage 5: Final validation and cross-stage synthesis.

Independently checks integrated occupancy, forecasting metrics,
anomaly counts and teaching-window ranking stability.

Recalculates key results from earlier-stage outputs and compares them
against reported values to verify numerical consistency.

Combines utilisation, forecasting and anomaly results into a final
floorspace comparison with supporting figures.

Inputs
------
Stage 1 hourly occupancy, Stage 2 forecast results,
Stage 3 anomaly results and Stage 4 utilisation results.

Outputs
-------
Validation tables, consolidated floorspace comparisons and figures.

Run
---
python mma3001_stage5_final_validation.py
"""

from pathlib import Path
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Supports scripts stored in either the project root or src/.
PROJECT_ROOT = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name == "src"
    else SCRIPT_DIR
)

# Directories containing the outputs of each stage.
S1 = PROJECT_ROOT / "mma3001_stage1_outputs"
S2 = PROJECT_ROOT / "mma3001_stage2_outputs"
S3 = PROJECT_ROOT / "mma3001_stage3_outputs"
S4 = PROJECT_ROOT / "mma3001_stage4_outputs"

# Final validation and synthesis outputs.
OUT = PROJECT_ROOT / "mma3001_stage5_outputs"

# Use a common tolerance for numerical consistency checks.
TOLERANCE = 1e-6
TIMEZONE = "Australia/Melbourne"


def read_required(path):
    """
    Read a required CSV and report missing inputs.

    Parameters
    ----------
    path : pathlib.Path
        Path to the required CSV file.

    Returns
    -------
    pandas.DataFrame
        Contents of the input CSV.

    Raises
    ------
    FileNotFoundError
        If the required input does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required input missing:\n{path}")
    return pd.read_csv(path)


def find_column(df, candidates, description):
    """
    Locate the first supported name for a required column.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing the required column.
    candidates : list of str
        Accepted column names in priority order.
    description : str
        Column description used in error messages.

    Returns
    -------
    str
        First matching column name.

    Raises
    ------
    ValueError
        If none of the supported column names exist.
    """
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        f"Cannot find {description}. Available: {list(df.columns)}"
    )


def close_enough(a, b):
    """
    Compare numerical values within the validation tolerance.

    Parameters
    ----------
    a : array-like or float
        Recalculated value or values.
    b : array-like or float
        Reported value or values.

    Returns
    -------
    bool
        Whether the values agree within the tolerance.

    Notes
    -----
    Relative and absolute tolerances are both applied.
    Matching NaN values are treated as equal.
    """
    return bool(np.allclose(
        a, b, rtol=TOLERANCE, atol=TOLERANCE, equal_nan=True
    ))


def normalize_model(value):
    """
    Standardise model names for cross-stage comparisons.

    Parameters
    ----------
    value : object
        Model name to normalise.

    Returns
    -------
    str
        Lowercase model name with spaces replaced by underscores.
    """
    return str(value).strip().lower().replace(" ", "_")


def check(checks, name, passed, detail):
    """
    Record and print the result of a validation check.

    Parameters
    ----------
    checks : list
        Mutable collection of validation records.
    name : str
        Identifier for the validation check.
    passed : bool
        Whether the check passed.
    detail : str
        Supporting information for the result.
    """
    status = "PASS" if passed else "FAIL"
    checks.append({
        "check": name, "status": status, "detail": str(detail)
    })
    print(f"  {status}: {name}")


def save_table(rows, filename):
    """
    Save validation records as a CSV table.

    Parameters
    ----------
    rows : list
        Records to convert into a DataFrame.
    filename : str
        Output CSV filename.

    Returns
    -------
    pandas.DataFrame
        Saved validation table.
    """
    result = pd.DataFrame(rows)
    result.to_csv(OUT / filename, index=False)
    return result


def save_figure(fig, filename):
    """
    Save and close a final-stage figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to export.
    filename : str
        Output image filename.
    """
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def load_inputs():
    """
    Load the required outputs from Stages 1–4.

    Returns
    -------
    dict
        Named DataFrames containing hourly occupancy, forecast
        metrics, anomaly results and utilisation summaries.

    Raises
    ------
    FileNotFoundError
        If an earlier-stage output is missing.
    """
    # Load the specific outputs needed for independent validation.
    files = {
        "hourly": S1 / "hourly_occupancy.csv",
        "forecast": S2 / "model_comparison.csv",
        "anomalies": S3 / "anomaly_summary.csv",
        "events": S3 / "anomaly_events.csv",
        "utilisation": S4 / "space_utilisation_summary.csv",
        "ranking": S4 / "teaching_hours_space_ranking.csv",
        "sensitivity": S4 / "sensitivity_analysis.csv",
    }
    return {
        name: read_required(path) for name, path in files.items()
    }


def validate_integration(data, checks):
    """
    Independently recalculate working-hour utilisation measures.

    Parameters
    ----------
    data : dict
        Outputs loaded from earlier stages.
    checks : list
        Collection of validation results to update.

    Returns
    -------
    pandas.DataFrame
        Recalculated and reported utilisation measures by space.

    Notes
    -----
    Uses observed-duration weighting and weekday 9 AM–5 PM
    Melbourne time to reproduce Stage 4's working-hour results.
    """
    hourly = data["hourly"].copy()
    empty_col = find_column(hourly, [
        "empty_fraction", "mean_empty_fraction",
        "hourly_empty_fraction", "fraction_empty",
        "empty_time_fraction", "zero_headcount_fraction",
    ], "integrated hourly empty fraction")

    # Independently identify the weekday working-hour observations.
    local = pd.to_datetime(
        hourly["hour_start_utc"], utc=True
    ).dt.tz_convert(TIMEZONE)

    teaching = hourly.loc[
        (local.dt.dayofweek < 5)
        & local.dt.hour.between(9, 16)
        & (hourly["coverage_fraction"] >= 0.999)
    ]
    reported = data["utilisation"].loc[
        lambda df: df["period"] == "weekday_9_to_17"
    ].set_index("floorspaceid")

    rows = []
    for space, group in teaching.groupby("floorspaceid"):
        # Recombine hourly values using their observed durations.
        weights = group["coverage_fraction"].to_numpy(float)
        empty = np.average(
            group[empty_col].to_numpy(float), weights=weights
        )
        mean = np.average(
            group["mean_headcount"].to_numpy(float), weights=weights
        )
        observed = weights.sum()

        if space not in reported.index:
            check(
                checks, f"Integration {space[:8]}",
                False, "Missing Stage 4 summary"
            )
            continue

        # Verify empty fraction, mean headcount and observed hours.
        expected = reported.loc[space]
        passed = all([
            close_enough(empty, expected["empty_fraction"]),
            close_enough(mean, expected["mean_headcount"]),
            close_enough(observed, expected["observed_hours"]),
        ])
        check(
            checks, f"Integration {space[:8]}", passed,
            f"Recalculated empty={empty:.6f}, "
            f"Stage 4 empty={expected['empty_fraction']:.6f}"
        )
        rows.append({
            "floorspaceid": space,
            "recalculated_empty_fraction": empty,
            "reported_empty_fraction": expected["empty_fraction"],
            "absolute_difference": abs(
                empty - expected["empty_fraction"]
            ),
            "recalculated_mean_headcount": mean,
            "reported_mean_headcount": expected["mean_headcount"],
            "recalculated_observed_hours": observed,
            "reported_observed_hours": expected["observed_hours"],
        })

    return save_table(rows, "integration_validation.csv")


def validate_forecasts(data, checks):
    """
    Independently recalculate linear-regression holdout metrics.

    Parameters
    ----------
    data : dict
        Outputs loaded from earlier stages.
    checks : list
        Collection of validation results to update.

    Returns
    -------
    pandas.DataFrame
        Recalculated and reported MAE, RMSE and R-squared by space.

    Notes
    -----
    Metrics are calculated directly from Stage 2 holdout
    predictions rather than copied from its summary table.
    """
    comparison = data["forecast"].copy()
    model_col = find_column(
        comparison, ["model", "model_name"], "model"
    )
    space_col = find_column(
        comparison, ["floorspaceid", "space"], "space"
    )
    metric_cols = {
        name: find_column(comparison, options, name)
        for name, options in {
            "mae": ["mae", "MAE"],
            "rmse": ["rmse", "RMSE"],
            "r2": ["r2", "R2", "r_squared"],
        }.items()
    }
    comparison["model_key"] = comparison[model_col].map(
        normalize_model
    )

    rows = []
    for space in sorted(data["hourly"]["floorspaceid"].unique()):
        predictions = read_required(
            S2 / f"test_predictions_{space}.csv"
        )
        actual_col = find_column(
            predictions, ["actual_headcount", "actual", "y_true"],
            "actual test headcount"
        )
        predicted_col = find_column(
            predictions,
            ["linear_regression", "linear_regression_prediction"],
            "linear regression predictions"
        )

        # Recalculate errors directly from the saved test predictions.
        actual = predictions[actual_col].to_numpy(float)
        predicted = predictions[predicted_col].to_numpy(float)
        errors = actual - predicted
        denominator = np.sum((actual - actual.mean()) ** 2)

        # Use the standard definitions of MAE, RMSE and R-squared.
        metrics = {
            "mae": np.mean(np.abs(errors)),
            "rmse": np.sqrt(np.mean(errors ** 2)),
            "r2": (
                1 - np.sum(errors ** 2) / denominator
                if denominator > 0 else np.nan
            ),
        }

        # Compare against the unique linear-regression summary row.
        matching = comparison.loc[
            (comparison[space_col] == space)
            & (comparison["model_key"] == "linear_regression")
        ]
        if len(matching) != 1:
            check(
                checks, f"Forecast {space[:8]}", False,
                "Expected one linear regression comparison row"
            )
            continue

        reported = matching.iloc[0]
        passed = all(
            close_enough(value, reported[metric_cols[name]])
            for name, value in metrics.items()
        )
        check(
            checks, f"Forecast {space[:8]}", passed,
            f"MAE={metrics['mae']:.6f}, "
            f"RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.6f}"
        )

        # Preserve the original CSV column ordering.
        rows.append({
            "floorspaceid": space,
            "test_hours": len(actual),
            "recalculated_mae": metrics["mae"],
            "reported_mae": reported[metric_cols["mae"]],
            "recalculated_rmse": metrics["rmse"],
            "reported_rmse": reported[metric_cols["rmse"]],
            "recalculated_r2": metrics["r2"],
            "reported_r2": reported[metric_cols["r2"]],
        })

    return save_table(rows, "forecast_validation.csv")


def validate_anomalies(data, checks):
    """
    Verify anomaly-hour classifications and grouped event counts.

    Parameters
    ----------
    data : dict
        Outputs loaded from earlier stages.
    checks : list
        Collection of validation results to update.

    Returns
    -------
    pandas.DataFrame
        Recalculated and reported anomaly counts for each space.

    Notes
    -----
    Hourly classifications are counted independently and
    compared with the Stage 3 anomaly and event summaries.
    """
    summary = data["anomalies"].set_index("floorspaceid")
    events = data["events"]
    rows = []

    for space in summary.index:
        hourly = read_required(
            S3 / f"hourly_anomalies_{space}.csv"
        )
        types = hourly["anomaly_type"]

        # Recount positive, negative and total anomalous hours.
        high = int((types == "unexpectedly_high").sum())
        low = int((types == "unexpectedly_low").sum())
        total = int((types != "normal").sum())

        # Verify grouped event totals against the event table.
        event_count = int(
            (events["floorspaceid"] == space).sum()
        )
        reported = summary.loc[space]

        passed = (
            high == int(reported["unexpectedly_high_hours"])
            and low == int(reported["unexpectedly_low_hours"])
            and total == int(reported["total_anomalous_hours"])
            and event_count == int(reported["number_of_events"])
        )
        check(
            checks, f"Anomalies {space[:8]}", passed,
            f"{total} anomalous hours, {event_count} events"
        )
        rows.append({
            "floorspaceid": space,
            "recalculated_anomalous_hours": total,
            "reported_anomalous_hours":
                int(reported["total_anomalous_hours"]),
            "recalculated_events": event_count,
            "reported_events": int(reported["number_of_events"]),
            "high_hours": high,
            "low_hours": low,
        })

    return save_table(rows, "anomaly_validation.csv")


def validate_rankings(data, checks):
    """
    Check working-hour ranking stability across weekday windows.

    Parameters
    ----------
    data : dict
        Outputs loaded from earlier stages.
    checks : list
        Collection of validation results to update.

    Returns
    -------
    pandas.DataFrame
        Floorspace rankings for each weekday analysis window.

    Notes
    -----
    Identical rankings across 8 AM–6 PM, 9 AM–5 PM and
    10 AM–4 PM indicate stability to the selected time window.
    """
    sensitivity = data["sensitivity"]
    windows = [
        "weekday_8_to_18",
        "weekday_9_to_17",
        "weekday_10_to_16",
    ]
    rows = []

    for window in windows:
        # Re-rank spaces independently for each analysis window.
        subset = sensitivity.loc[
            (sensitivity["analysis_window"] == window)
            & (sensitivity["low_mean_headcount_threshold"] == 1.0)
        ].sort_values("empty_fraction", ascending=False)

        for rank, (_, item) in enumerate(
            subset.iterrows(), start=1
        ):
            rows.append({
                "analysis_window": window,
                "floorspaceid": item["floorspaceid"],
                "empty_fraction": item["empty_fraction"],
                "rank": rank,
            })

    # Confirm each space holds the same rank in all three windows.
    result = pd.DataFrame(rows)
    pivot = result.pivot(
        index="floorspaceid",
        columns="analysis_window",
        values="rank"
    )
    stable = bool(
        (pivot.nunique(axis=1) == 1).all()
        and pivot.notna().all().all()
    )
    check(
        checks, "Weekday window ranking stability", stable,
        "Rankings identical across all three windows"
        if stable else "Rankings change or data are missing"
    )
    result.to_csv(OUT / "ranking_stability.csv", index=False)
    return result


def build_consolidated_table(data):
    """
    Combine utilisation, forecasting and anomaly results by space.

    Parameters
    ----------
    data : dict
        Outputs loaded from earlier stages.

    Returns
    -------
    pandas.DataFrame
        Consolidated engineering comparison ordered by
        working-hour underutilisation rank.
    """
    forecast = data["forecast"].copy()
    model_col = find_column(
        forecast, ["model", "model_name"], "model"
    )

    # Select the linear-regression results used in final comparisons.
    linear = forecast.loc[
        forecast[model_col].map(normalize_model)
        == "linear_regression",
        ["floorspaceid", "mae", "rmse", "r2"]
    ].rename(columns={
        "mae": "linear_regression_mae",
        "rmse": "linear_regression_rmse",
        "r2": "linear_regression_r2",
    })

    # Combine the key results from Stages 2, 3 and 4.
    combined = (
        data["ranking"][[
            "floorspaceid", "underutilisation_rank",
            "empty_fraction", "occupied_fraction",
            "mean_headcount", "observed_hours",
        ]]
        .merge(linear, on="floorspaceid", how="left")
        .merge(data["anomalies"][[
            "floorspaceid", "total_anomalous_hours",
            "anomaly_percentage", "number_of_events",
        ]], on="floorspaceid", how="left")
        .sort_values("underutilisation_rank")
    )
    combined.to_csv(
        OUT / "final_space_comparison.csv", index=False
    )
    return combined


def plot_bar(combined, sort_col, value_col, filename,
             title, ylabel, descending=False,
             percent=False, annotate=False):
    """
    Draw a final comparison bar chart.

    Parameters
    ----------
    combined : pandas.DataFrame
        Consolidated floorspace results.
    sort_col : str
        Column used to order spaces.
    value_col : str
        Column containing plotted values.
    filename : str
        Output image filename.
    title : str
        Chart title.
    ylabel : str
        Vertical axis label.
    descending : bool, optional
        Whether to sort values in descending order.
    percent : bool, optional
        Whether to multiply plotted fractions by 100.
    annotate : bool, optional
        Whether to label bars with percentage values.
    """
    ordered = combined.sort_values(
        sort_col, ascending=not descending
    )
    values = ordered[value_col] * (100 if percent else 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        ordered["floorspaceid"].str[:8], values
    )

    # Add percentage labels when requested for the ranking chart.
    if annotate:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{value:.1f}%",
                ha="center", fontsize=9
            )
        ax.set_ylim(0, max(values) * 1.2)

    ax.set(ylabel=ylabel, title=title)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, filename)


def plot_cross_stage(combined):
    """
    Plot underutilisation against forecast error and anomaly count.

    Parameters
    ----------
    combined : pandas.DataFrame
        Consolidated floorspace results.

    Notes
    -----
    Horizontal position represents working-hour empty fraction,
    vertical position represents linear-regression RMSE, and
    marker size represents the number of anomalous test hours.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    x = combined["empty_fraction"] * 100
    y = combined["linear_regression_rmse"]

    # Scale marker area by the number of anomalous hours.
    sizes = 50 + 40 * combined["total_anomalous_hours"]
    ax.scatter(x, y, s=sizes, alpha=0.65)

    # Label each point using its floorspace identifier.
    for _, row in combined.iterrows():
        ax.annotate(
            row["floorspaceid"][:8],
            (
                row["empty_fraction"] * 100,
                row["linear_regression_rmse"]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.set(
        xlabel="Working hours empty time proportion (%)",
        ylabel="Linear regression RMSE",
        title=(
            "Underutilisation vs forecast error\n"
            "Marker size represents anomalous-hour count"
        )
    )
    ax.grid(alpha=0.2)
    save_figure(fig, "final_cross_stage_comparison.png")


def plot_results(combined):
    """
    Generate the four final-stage engineering comparison figures.

    Parameters
    ----------
    combined : pandas.DataFrame
        Consolidated floorspace results.
    """
    # Compare spaces by working-hour empty-time fraction.
    plot_bar(
        combined,
        "underutilisation_rank", "empty_fraction",
        "final_underutilisation_ranking.png",
        "Observed underutilisation: Weekdays 9 AM–5 PM",
        "Empty time fraction (%)",
        percent=True, annotate=True
    )

    # Compare holdout forecast error across floorspaces.
    plot_bar(
        combined,
        "linear_regression_rmse", "linear_regression_rmse",
        "final_forecast_accuracy.png",
        "Linear regression holdout forecast accuracy",
        "RMSE (average headcount)"
    )

    # Compare the share of anomalous test hours by floorspace.
    plot_bar(
        combined,
        "anomaly_percentage", "anomaly_percentage",
        "final_anomaly_rates.png",
        "Forecast residual anomaly rates by space",
        "Anomalous test hours (%)",
        descending=True
    )

    # Show utilisation, forecast error and anomalies together.
    plot_cross_stage(combined)


def main():
    """
    Execute Stage 5 validation and final engineering synthesis.

    Notes
    -----
    Runs integration, forecast, anomaly and ranking checks,
    then saves validation tables, consolidated results and plots.
    Failed checks are reported without automatically stopping
    final output generation.
    """
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("\nMMA3001 Stage 5: Final validation and synthesis")
    print("=" * 55)

    data = load_inputs()
    checks = []

    # Run independent numerical and cross-stage consistency checks.
    for heading, function in [
        ("Numerical integration validation", validate_integration),
        ("Forecast metric validation", validate_forecasts),
        ("Anomaly count validation", validate_anomalies),
        ("Ranking sensitivity validation", validate_rankings),
    ]:
        print(f"\n{heading}")
        function(data, checks)

    print("\nConsolidating results")
    combined = build_consolidated_table(data)
    plot_results(combined)

    # Save the complete pass/fail record for reproducibility.
    validation = save_table(checks, "validation_checks.csv")
    passed = int((validation["status"] == "PASS").sum())
    failed = int((validation["status"] == "FAIL").sum())

    print("\nFinal space comparison:")
    print(combined.to_string(index=False))
    print(f"\nValidation summary:\nPassed: {passed}\nFailed: {failed}")

    # Flag failed checks rather than silently accepting the results.
    if failed:
        print(
            "\nWARNING: Some validation checks failed. "
            "Review validation_checks.csv before using the results."
        )
    else:
        print("\nAll numerical consistency checks passed.")

    print(
        f"\nCompleted in "
        f"{time.perf_counter() - started:.1f} seconds"
    )
    print(f"Outputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
