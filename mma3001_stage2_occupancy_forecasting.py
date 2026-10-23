
"""
MMA3001 Stage 2 — Occupancy Forecasting.

Forecasts hourly mean occupancy one hour ahead using reconstructed
Stage 1 data. Compares persistence, historical averages, linear
regression and decision-tree regression.

Evaluation includes a chronological 80/20 holdout, expanding-window
validation and sensitivity to long gaps between sensor events.

Inputs
------
- Stage 1 hourly_occupancy.csv
- Stage 1 cached event_timestamps_<space>.npy files

Outputs
-------
- Forecast accuracy and model comparison CSVs
- Holdout predictions and diagnostic plots
- Expanding-window and gap-sensitivity results

Run
---
python mma3001_stage2_occupancy_forecasting.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent

# Supports scripts stored in either the project root or src/.
PROJECT_ROOT = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name == "src"
    else SCRIPT_DIR
)

# Same dataset directory convention as Stage 1.
DATA_DIR = Path(
    os.environ.get("MMA3001_DATA_DIR", str(PROJECT_ROOT / "data"))
).expanduser().resolve()

# Read reconstructed hourly occupancy from Stage 1.
HOURLY_FILE = (
    PROJECT_ROOT / "mma3001_stage1_outputs" / "hourly_occupancy.csv"
)

# Retained for consistency, although Stage 2 now reads
# cached sensor timestamps rather than the raw CSV.
RAW_FILE = DATA_DIR / "5occupancySensor_MayToDec2024_9MRows.csv"

# Save forecasting results in the Stage 2 output directory.
OUT = PROJECT_ROOT / "mma3001_stage2_outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Lag offsets are relative to the forecast origin, not the target.
# Lag 0 is current occupancy; lag 23 is the previous day's target hour.
LAGS = (0, 1, 2, 23, 24, 167)

# Test sensitivity to raw-event gaps exceeding these durations.
THRESHOLDS = (1, 6, 24)

# Expanding training windows with successive 10% evaluation periods.
FOLDS = ((0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00))

# Combine recent occupancy history with calendar-based features.
FEATURES = [f"lag_{n}" for n in LAGS] + [
    "hour_sin", "hour_cos", "day_sin", "day_cos", "weekend"
]


def load_hourly():
    """
    Load and validate reconstructed hourly occupancy from Stage 1.

    Returns
    -------
    pandas.DataFrame
        Hourly occupancy records with parsed UTC timestamps and
        numeric headcount and coverage fields.

    Raises
    ------
    ValueError
        If required columns are missing or a floorspace contains
        duplicate hourly timestamps.
    """
    df = pd.read_csv(HOURLY_FILE)
    required = {"floorspaceid", "hour_start_utc", "mean_headcount", "coverage_fraction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing Stage 1 columns: {missing}")

    # Enforce consistent types before aligning hourly observations.
    df["hour_start_utc"] = pd.to_datetime(df["hour_start_utc"], utc=True)
    df["mean_headcount"] = pd.to_numeric(df["mean_headcount"], errors="raise")
    df["coverage_fraction"] = pd.to_numeric(df["coverage_fraction"], errors="raise")

    # Each floorspace must have at most one observation per UTC hour.
    if df.duplicated(["floorspaceid", "hour_start_utc"]).any():
        raise ValueError("Duplicate hourly timestamps within a space")
    return df


def read_raw_timestamps(spaces):
    """
    Load Stage 1's sorted, unique raw-event timestamps.

    Preserve the original Stage 2 behaviour:
    - all events in [study_start, study_end)
    - the nearest preceding event, if available
    - the nearest following event, if available

    Parameters
    ----------
    spaces : iterable of str
        Floorspace identifiers requiring event timestamps.

    Returns
    -------
    dict
        Floorspace identifiers mapped to UTC event timestamps.

    Raises
    ------
    FileNotFoundError
        If a required Stage 1 timestamp cache is missing.
    ValueError
        If cached timestamps are empty, invalid or not strictly
        increasing.
    """
    # Use the same Melbourne-local study window as Stage 1.
    start = pd.Timestamp(
        "2024-05-01", tz="Australia/Melbourne"
    ).tz_convert("UTC").value

    end = pd.Timestamp(
        "2025-01-01", tz="Australia/Melbourne"
    ).tz_convert("UTC").value

    cache_dir = HOURLY_FILE.parent
    result = {}

    for space in spaces:
        path = cache_dir / f"event_timestamps_{space}.npy"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing timestamp cache: {path}\n"
                "Run Stage 1 before Stage 2."
            )

        # Load nanosecond timestamps without rereading the raw CSV.
        events = np.load(path, allow_pickle=False)

        if events.ndim != 1 or not np.issubdtype(events.dtype, np.integer):
            raise ValueError(f"Invalid timestamp cache: {path}")

        if len(events) == 0:
            raise ValueError(f"No raw timestamps for {space}")

        if np.any(events[1:] <= events[:-1]):
            raise ValueError(f"Timestamp cache is not sorted/unique: {path}")

        # Locate events inside the study period.
        left = np.searchsorted(events, start, side="left")
        right = np.searchsorted(events, end, side="left")

        # Match the old before/inside/after event selection exactly.
        selected = events[
            max(0, left - 1):min(len(events), right + 1)
        ]

        if len(selected) == 0:
            raise ValueError(f"No raw timestamps for {space}")

        result[space] = pd.to_datetime(selected, utc=True)
        print(
            f"Loaded {len(selected):,} cached timestamps "
            f"for {space[:8]}"
        )

    return result


def gap_flags(hours, events, threshold_hours):
    """
    Flag hourly intervals affected by long raw-event gaps.

    Parameters
    ----------
    hours : array-like
        UTC starting times of hourly intervals.
    events : array-like
        Sorted UTC sensor-event timestamps.
    threshold_hours : float
        Minimum gap duration used for flagging.

    Returns
    -------
    numpy.ndarray
        Boolean flags identifying affected hours.

    Notes
    -----
    Flags are diagnostic and do not establish sensor failure.
    A difference array marks overlapping hours efficiently.
    """
    h = pd.DatetimeIndex(hours).asi8
    e = pd.DatetimeIndex(events).asi8
    hour_ns = pd.Timedelta(hours=1).value
    limit_ns = pd.Timedelta(hours=threshold_hours).value
    changes = np.zeros(len(h) + 1, dtype=np.int32)

    # Mark hour ranges overlapping gaps longer than the threshold.
    for j in np.flatnonzero(np.diff(e) > limit_ns):
        left = max(0, np.searchsorted(h, e[j] - hour_ns, side="right"))
        right = min(len(h), np.searchsorted(h, e[j + 1], side="left"))
        if left < right:
            changes[left] += 1
            changes[right] -= 1

    # Recover overlap flags using cumulative range updates.
    flagged = np.cumsum(changes[:-1]) > 0

    # Flag hours outside sufficiently supported event boundaries.
    flagged |= h + hour_ns <= e[0]
    flagged |= h + hour_ns > e[-1] + limit_ns
    return flagged


def build_examples(space_df, events):
    """
    Construct one-hour-ahead forecasting examples for a floorspace.

    Parameters
    ----------
    space_df : pandas.DataFrame
        Stage 1 hourly occupancy records for one floorspace.
    events : pandas.DatetimeIndex
        UTC timestamps of its raw sensor events.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Complete forecasting examples and data-availability diagnostics.

    Notes
    -----
    The target is occupancy at t+1, with features available at t.
    Lag 23 represents the target hour one day earlier, while lag 24
    represents the origin hour one day earlier.
    Only hours with at least 99.9% coverage are used.
    """
    # Align observations to a complete UTC hourly calendar.
    space_df = space_df.sort_values("hour_start_utc")
    indexed = space_df.set_index("hour_start_utc")
    hours = pd.date_range(indexed.index.min(), indexed.index.max(), freq="h", tz="UTC")
    aligned = indexed.reindex(hours)

    # Exclude inadequately covered hours from targets and lag features.
    y = aligned["mean_headcount"].where(aligned["coverage_fraction"] >= 0.999)
    data = pd.DataFrame(index=hours)
    data.index.name = "forecast_origin_utc"

    # Predict the next hour using information at the forecast origin.
    data["target"] = y.shift(-1)
    data["target_time"] = pd.Series(hours, index=hours).shift(-1)

    # Construct historical features without using future headcounts.
    for lag in LAGS:
        data[f"lag_{lag}"] = y.shift(lag)

    # Encode target-hour calendar cycles in Melbourne local time.
    local = data["target_time"].dt.tz_convert("Australia/Melbourne")
    hour, weekday = local.dt.hour, local.dt.dayofweek
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["day_sin"] = np.sin(2 * np.pi * weekday / 7)
    data["day_cos"] = np.cos(2 * np.pi * weekday / 7)
    data["weekend"] = (weekday >= 5).astype(int)
    data["hour_of_week"] = weekday * 24 + hour

    # Flag examples if the target or any lag overlaps a long gap.
    for threshold in THRESHOLDS:
        flags = pd.Series(gap_flags(hours, events, threshold), index=hours)
        affected = flags.shift(-1, fill_value=True)
        for lag in LAGS:
            affected = affected | flags.shift(lag, fill_value=True)
        data[f"affected_{threshold}h"] = affected

    # Remove incomplete examples before chronological model evaluation.
    examples = data.dropna(subset=FEATURES + ["target", "target_time"]).reset_index()
    diagnostics = {
        "raw_hourly_rows": len(space_df),
        "calendar_hours_in_span": len(hours),
        "missing_hourly_rows": len(hours) - len(space_df),
        "usable_forecast_examples": len(examples),
        "excluded_forecast_origins": len(hours) - 1 - len(examples),
    }
    return examples, diagnostics


def evaluate(train, test, space, evaluation, label, save_predictions=False):
    """
    Fit and compare four one-hour-ahead forecasting methods.

    Parameters
    ----------
    train : pandas.DataFrame
        Chronologically earlier training examples.
    test : pandas.DataFrame
        Later examples used for evaluation.
    space : str
        Floorspace identifier.
    evaluation : str
        Evaluation type recorded in the results.
    label : str
        Label identifying the holdout, fold or sensitivity case.
    save_predictions : bool, optional
        Whether to return individual predictions and residuals.

    Returns
    -------
    tuple[list, pandas.DataFrame or None]
        Model metrics and optional per-example predictions.

    Notes
    -----
    Persistence and historical hour-of-week averages provide
    benchmarks for linear regression and decision-tree regression.
    Models are assessed using MAE, RMSE and R-squared.
    """
    # Skip evaluations with insufficient training or test examples.
    if len(train) < 100 or len(test) < 24:
        return [], None

    X_train = train[FEATURES].to_numpy(float)
    X_test = test[FEATURES].to_numpy(float)
    y_train = train["target"].to_numpy(float)
    y_test = test["target"].to_numpy(float)

    outputs, runtimes = {}, {}

    # Persistence baseline: next-hour occupancy equals current occupancy.
    start = time.perf_counter()
    outputs["persistence"] = test["lag_0"].to_numpy(float)
    runtimes["persistence"] = (0.0, time.perf_counter() - start)

    # Historical baseline: average training occupancy by hour of week.
    start = time.perf_counter()
    means = train.groupby("hour_of_week")["target"].mean()
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    outputs["historical_hour_of_week"] = (
        test["hour_of_week"].map(means).fillna(y_train.mean()).to_numpy(float)
    )
    runtimes["historical_hour_of_week"] = (
        fit_time, time.perf_counter() - start
    )

    # Compare a linear model with a depth-limited nonlinear tree.
    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=24, random_state=42
        ),
    }

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        start = time.perf_counter()

        # Prevent model predictions from implying negative occupancy.
        outputs[name] = np.maximum(model.predict(X_test), 0)
        runtimes[name] = (fit_time, time.perf_counter() - start)

    results = []
    predictions = pd.DataFrame({
        "floorspaceid": space,
        "forecast_origin_utc": train.iloc[:0]["forecast_origin_utc"]
        if False else test["forecast_origin_utc"].astype(str).to_numpy(),
        "target_time_utc": test["target_time"].astype(str).to_numpy(),
        "actual_headcount": y_test,
    }) if save_predictions else None

    # Record accuracy, runtime and test-period boundaries for each method.
    for name, predicted in outputs.items():
        fit_time, predict_time = runtimes[name]
        results.append({
            "floorspaceid": space,
            "evaluation_type": evaluation,
            "evaluation_label": label,
            "model": name,
            "mae": mean_absolute_error(y_test, predicted),
            "rmse": np.sqrt(mean_squared_error(y_test, predicted)),
            "r2": r2_score(y_test, predicted),
            "fit_seconds": fit_time,
            "predict_seconds": predict_time,
            "train_examples": len(train),
            "test_examples": len(test),
            "first_test_target_utc": str(test["target_time"].iloc[0]),
            "last_test_target_utc": str(test["target_time"].iloc[-1]),
        })
        if predictions is not None:
            predictions[name] = predicted
            predictions[f"residual_{name}"] = y_test - predicted

    return results, predictions


def plot_forecast(predictions, space):
    """
    Plot the first 168 holdout predictions for a floorspace.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Actual headcounts and forecasts from all four methods.
    space : str
        Floorspace identifier used in the plot title and filename.
    """
    # Show the first 168 test examples for a readable comparison.
    week = predictions.head(168)
    timestamps = pd.to_datetime(week["target_time_utc"], utc=True)
    fig, ax = plt.subplots(figsize=(12, 4))

    for column in [
        "actual_headcount", "persistence", "linear_regression",
        "decision_tree", "historical_hour_of_week"
    ]:
        ax.plot(timestamps, week[column], label=column, linewidth=1)

    ax.set(
        title=f"One hour ahead forecasts: {space}",
        xlabel="Time (UTC)",
        ylabel="Average headcount"
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / f"forecast_first_week_{space}.png", dpi=130)
    plt.close(fig)


def plot_full_test_period(predictions, space):
    """
    Plot actual versus linear-regression predictions across
    the complete chronological holdout period.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Holdout targets and model predictions.
    space : str
        Floorspace identifier used in the plot title and filename.
    """
    # Display the full holdout period in Melbourne local time.
    timestamps = pd.to_datetime(
        predictions["target_time_utc"], utc=True
    ).dt.tz_convert("Australia/Melbourne")

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(
        timestamps,
        predictions["actual_headcount"],
        label="Actual hourly mean headcount",
        linewidth=0.9,
        alpha=0.85,
    )

    ax.plot(
        timestamps,
        predictions["linear_regression"],
        label="Linear regression forecast",
        linewidth=0.9,
        alpha=0.8,
    )

    ax.set(
        title=f"Full test period: Actual vs predicted occupancy\nSpace: {space}",
        xlabel="Date",
        ylabel="Hourly mean headcount",
    )

    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        OUT / f"forecast_full_test_period_{space}.png",
        dpi=150,
    )
    plt.close(fig)


def main():
    """
    Execute Stage 2 forecasting and validation for all floorspaces.

    Notes
    -----
    Each space uses a chronological 80/20 training/test split.
    Additional expanding-window folds and gap-exclusion experiments
    assess robustness. Results are saved as CSV files and plots.

    Raises
    ------
    ValueError
        If a floorspace has insufficient forecasting examples.
    """
    start = time.perf_counter()
    hourly = load_hourly()
    spaces = sorted(hourly["floorspaceid"].astype(str).unique())

    print(f"Loaded {len(hourly):,} hourly records across {len(spaces)} spaces.")
    print("Checking raw sensor gaps...")
    events = read_raw_timestamps(spaces)

    holdout, rolling, sensitivity = [], [], []
    split_diagnostics, gap_diagnostics = [], []

    for space, group in hourly.groupby("floorspaceid", sort=True):
        space = str(space)
        examples, diagnostics = build_examples(group, events[space])

        # Reserve the final 20% of examples for chronological testing.
        cut = int(len(examples) * 0.8)
        train, test = examples.iloc[:cut], examples.iloc[cut:]

        if len(train) < 100 or len(test) < 24:
            raise ValueError(f"Insufficient forecast examples for {space}")

        diagnostics.update({
            "floorspaceid": space,
            "status": "success",
            "train_examples": len(train),
            "test_examples": len(test),
            "first_test_target_utc": str(test["target_time"].iloc[0]),
            "last_test_target_utc": str(test["target_time"].iloc[-1]),
        })
        split_diagnostics.append(diagnostics)

        # Evaluate all four methods on the same held-out examples.
        scores, predictions = evaluate(
            train, test, space, "original_holdout", "chronological_80_20", True
        )
        holdout.extend(scores)
        predictions.to_csv(OUT / f"test_predictions_{space}.csv", index=False)
        plot_forecast(predictions, space)
        plot_full_test_period(predictions, space)

        # Expand the training window while testing on later observations.
        n = len(examples)
        for fold, (left, right) in enumerate(FOLDS, 1):
            a, b = int(n * left), int(n * right)
            scores, _ = evaluate(
                examples.iloc[:a], examples.iloc[a:b],
                space, "expanding_window", f"fold_{fold}"
            )
            rolling.extend(scores)

        # Re-evaluate after excluding examples affected by long event gaps.
        for threshold in THRESHOLDS:
            flag = f"affected_{threshold}h"
            clean_train = train.loc[~train[flag]]
            clean_test = test.loc[~test[flag]]

            gap_diagnostics.append({
                "floorspaceid": space,
                "threshold_hours": threshold,
                "original_train_examples": len(train),
                "retained_train_examples": len(clean_train),
                "excluded_train_examples": len(train) - len(clean_train),
                "original_test_examples": len(test),
                "retained_test_examples": len(clean_test),
                "excluded_test_examples": len(test) - len(clean_test),
            })

            scores, _ = evaluate(
                clean_train, clean_test, space,
                "gap_sensitivity", f"exclude_gaps_over_{threshold}h"
            )
            sensitivity.extend(scores)

        print(f"Completed {space}: {len(examples):,} forecast examples")

    holdout_df = pd.DataFrame(holdout)
    rolling_df = pd.DataFrame(rolling)

    # Save holdout metrics and rankings by mean absolute error.
    holdout_df.to_csv(OUT / "model_comparison.csv", index=False)
    holdout_df.sort_values(
        ["floorspaceid", "mae"]
    ).to_csv(OUT / "models_ranked_by_mae.csv", index=False)

    pd.DataFrame(split_diagnostics).to_csv(
        OUT / "data_split_diagnostics.csv", index=False
    )
    rolling_df.to_csv(OUT / "expanding_window_results.csv", index=False)

    # Summarise model performance across expanding-window folds.
    rolling_df.groupby(["floorspaceid", "model"], as_index=False).agg(
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_r2=("r2", "mean"),
        folds=("mae", "count"),
    ).to_csv(OUT / "expanding_window_summary.csv", index=False)

    # Save gap-exclusion metrics and retained sample counts.
    pd.DataFrame(sensitivity).to_csv(
        OUT / "gap_sensitivity_results.csv", index=False
    )
    pd.DataFrame(gap_diagnostics).to_csv(
        OUT / "gap_sensitivity_diagnostics.csv", index=False
    )

    print(f"\nStage 2 complete in {time.perf_counter() - start:.1f} seconds.")
    print(f"Results saved to: {OUT}")


if __name__ == "__main__":
    main()
