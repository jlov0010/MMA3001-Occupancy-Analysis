
"""
MMA3001 Stage 3 — Occupancy Anomaly Detection.

Detects unexpectedly high or low occupancy using residuals from the
Stage 2 linear-regression forecasts.

Residual thresholds are calibrated using an earlier portion of the
training data, without using holdout test targets. Consecutive
anomalous hours of the same type are grouped into events.

Sensor freshness is recorded separately to distinguish unusual
occupancy from potentially stale sensor observations.

Inputs
------
- Stage 1 hourly_occupancy.csv
- Stage 1 cached event timestamps
- Stage 2 holdout predictions

Outputs
-------
- Hourly anomaly classifications
- Anomaly summaries and grouped events
- Top 30 anomaly events
- Occupancy and residual diagnostic plots

Run
---
python mma3001_stage3_anomaly_detection.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent

# Supports scripts stored in either the project root or src/.
PROJECT_ROOT = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name == "src"
    else SCRIPT_DIR
)

# Consistent dataset directory across all stages.
DATA_DIR = Path(
    os.environ.get("MMA3001_DATA_DIR", str(PROJECT_ROOT / "data"))
).expanduser().resolve()

# Stage 1 reconstructed occupancy and timestamp caches.
HOURLY_FILE = (
    PROJECT_ROOT / "mma3001_stage1_outputs" / "hourly_occupancy.csv"
)

# Stage 2 forecasting predictions.
STAGE2_DIR = PROJECT_ROOT / "mma3001_stage2_outputs"

# Original dataset path (retained for consistency).
RAW_FILE = DATA_DIR / "5occupancySensor_MayToDec2024_9MRows.csv"

# Stage 3 anomaly detection outputs.
OUT = PROJECT_ROOT / "mma3001_stage3_outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Match the forecasting features used in Stage 2.
LAGS = (0, 1, 2, 23, 24, 167)
FEATURES = [f"lag_{n}" for n in LAGS] + [
    "hour_sin", "hour_cos", "day_sin", "day_cos", "weekend"
]

# Calibrate residual thresholds without using holdout test targets.
TRAIN_FRACTION = 0.80
CALIBRATION_FRACTION = 0.20
ANOMALY_QUANTILE = 0.99

# Flag stale observations separately from occupancy anomalies.
FRESHNESS_LIMIT_HOURS = 6
MIN_CALIBRATION_EXAMPLES = 100


def build_examples(group):
    """
    Recreate Stage 2 forecasting features for anomaly detection.

    Parameters
    ----------
    group : pandas.DataFrame
        Stage 1 hourly occupancy records for one floorspace.

    Returns
    -------
    pandas.DataFrame
        Complete one-hour-ahead examples with occupancy lags,
        calendar features, targets and target timestamps.

    Notes
    -----
    Features use information available at the forecast origin.
    Only hourly observations with at least 99.9% coverage are used.
    """
    # Align observations to a complete UTC hourly calendar.
    group = group.sort_values("hour_start_utc")
    indexed = group.set_index("hour_start_utc")
    hours = pd.date_range(
        indexed.index.min(), indexed.index.max(), freq="h", tz="UTC"
    )
    aligned = indexed.reindex(hours)

    # Exclude inadequately covered observations from forecasting.
    y = aligned["mean_headcount"].where(
        aligned["coverage_fraction"] >= 0.999
    )

    # Predict occupancy at t+1 using features available at time t.
    data = pd.DataFrame(index=hours)
    data.index.name = "forecast_origin_utc"
    data["target"] = y.shift(-1)
    data["target_time"] = pd.Series(hours, index=hours).shift(-1)

    # Match the historical occupancy lags used in Stage 2.
    for lag in LAGS:
        data[f"lag_{lag}"] = y.shift(lag)

    # Encode target-hour calendar cycles in Melbourne local time.
    local = data["target_time"].dt.tz_convert("Australia/Melbourne")
    hour = local.dt.hour
    weekday = local.dt.dayofweek

    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["day_sin"] = np.sin(2 * np.pi * weekday / 7)
    data["day_cos"] = np.cos(2 * np.pi * weekday / 7)
    data["weekend"] = (weekday >= 5).astype(int)

    # Remove incomplete examples to preserve Stage 2 alignment.
    return data.dropna(
        subset=FEATURES + ["target", "target_time"]
    ).reset_index()


def read_raw_timestamps(spaces):
    """
    Load the complete per-space event history exported by Stage 1.

    Stage 3 uses all valid timestamps, not only study-window events,
    when calculating the age of the most recent sensor observation.

    Parameters
    ----------
    spaces : iterable of str
        Floorspace identifiers requiring timestamp histories.

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
    cache_dir = HOURLY_FILE.parent
    result = {}

    for space in spaces:
        path = cache_dir / f"event_timestamps_{space}.npy"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing timestamp cache: {path}\n"
                "Run Stage 1 before Stage 3."
            )

        # Reuse cached events rather than scanning the large raw CSV.
        events = np.load(path, allow_pickle=False)

        if events.ndim != 1 or not np.issubdtype(events.dtype, np.integer):
            raise ValueError(f"Invalid timestamp cache: {path}")

        if len(events) == 0:
            raise ValueError(f"No raw timestamps for {space}")

        if np.any(events[1:] <= events[:-1]):
            raise ValueError(f"Timestamp cache is not sorted/unique: {path}")

        result[space] = pd.to_datetime(events, utc=True)

        print(
            f"Loaded {len(events):,} cached timestamps "
            f"for {space[:8]}"
        )

    return result


def observation_age_hours(hour_starts, raw_timestamps):
    """
    Calculate the age of the latest sensor event at each hour's end.

    Parameters
    ----------
    hour_starts : array-like
        UTC starting times of hourly intervals.
    raw_timestamps : array-like
        Sorted UTC sensor-event timestamps.

    Returns
    -------
    numpy.ndarray
        Hours since the latest recorded event, or NaN if none exists.

    Notes
    -----
    A zero-order-held value may be much older than its hourly timestamp.
    """
    # Evaluate freshness at the end of each hourly interval.
    hour_end = pd.DatetimeIndex(hour_starts).asi8 + pd.Timedelta(hours=1).value
    events = pd.DatetimeIndex(raw_timestamps).asi8

    # Locate the most recent event at or before each hour's end.
    indices = np.searchsorted(events, hour_end, side="right") - 1
    age = np.full(len(hour_end), np.nan)

    valid = indices >= 0
    age[valid] = (
        hour_end[valid] - events[indices[valid]]
    ) / pd.Timedelta(hours=1).value

    return age


def calibrate_thresholds(examples):
    """
    Calibrate separate positive and negative residual thresholds.

    Parameters
    ----------
    examples : pandas.DataFrame
        Chronologically ordered forecasting examples.

    Returns
    -------
    tuple[float, float, int]
        Positive threshold, negative threshold and number of
        calibration examples.

    Raises
    ------
    ValueError
        If fewer than the required calibration examples are available.

    Notes
    -----
    The earliest 80% of examples form the training period.
    Its final 20% is reserved for threshold calibration.
    Holdout test targets are excluded from threshold selection.
    """
    # Keep the chronological holdout separate from calibration.
    train_end = int(len(examples) * TRAIN_FRACTION)
    training = examples.iloc[:train_end]

    # Fit on earlier training data and calibrate on later data.
    fit_end = int(len(training) * (1 - CALIBRATION_FRACTION))
    fit = training.iloc[:fit_end]
    calibration = training.iloc[fit_end:]

    if len(calibration) < MIN_CALIBRATION_EXAMPLES:
        raise ValueError("Insufficient calibration observations")

    model = LinearRegression()
    model.fit(fit[FEATURES], fit["target"])

    # Restrict predicted headcounts to non-negative values.
    predicted = np.maximum(
        model.predict(calibration[FEATURES]), 0
    )
    residual = calibration["target"].to_numpy() - predicted

    # Separate thresholds allow different positive/negative behaviour.
    positive_threshold = max(
        0.0, float(np.quantile(residual, ANOMALY_QUANTILE))
    )
    negative_threshold = min(
        0.0, float(np.quantile(residual, 1 - ANOMALY_QUANTILE))
    )

    return positive_threshold, negative_threshold, len(calibration)


def detect_anomalies(space, examples, predictions, raw_timestamps):
    """
    Classify holdout residuals and record sensor freshness.

    Parameters
    ----------
    space : str
        Floorspace identifier.
    examples : pandas.DataFrame
        Complete chronological forecasting examples.
    predictions : pandas.DataFrame
        Stage 2 holdout predictions for the floorspace.
    raw_timestamps : pandas.DatetimeIndex
        Complete UTC sensor-event history.

    Returns
    -------
    tuple[pandas.DataFrame, int]
        Hourly anomaly classifications and calibration sample count.

    Raises
    ------
    ValueError
        If Stage 2 test timestamps or actual headcounts do not
        match the reconstructed Stage 3 examples.

    Notes
    -----
    Positive residuals indicate occupancy above the forecast.
    Negative residuals indicate occupancy below the forecast.
    Sensor freshness flags do not automatically exclude anomalies.
    """
    # Calibrate thresholds using only pre-holdout observations.
    positive, negative, calibration_n = calibrate_thresholds(examples)

    test = examples.iloc[int(len(examples) * TRAIN_FRACTION):].copy()
    predictions = predictions.copy()

    test_times = pd.to_datetime(test["target_time"], utc=True)
    predicted_times = pd.to_datetime(
        predictions["target_time_utc"], utc=True
    )

    # Verify exact chronological alignment with Stage 2 predictions.
    if len(test) != len(predictions) or not np.array_equal(
        test_times.to_numpy(), predicted_times.to_numpy()
    ):
        raise ValueError(f"Stage 2 test timestamps do not match for {space}")

    actual = test["target"].to_numpy(float)
    predicted = predictions["linear_regression"].to_numpy(float)

    # Check that both stages use the same observed target headcounts.
    if not np.allclose(
        actual, predictions["actual_headcount"].to_numpy(float),
        rtol=1e-8, atol=1e-8
    ):
        raise ValueError(f"Stage 2 actual headcounts do not match for {space}")

    # Residual sign distinguishes unexpectedly high and low occupancy.
    residual = actual - predicted

    results = pd.DataFrame({
        "floorspaceid": space,
        "target_time_utc": test_times.to_numpy(),
        "actual_headcount": actual,
        "predicted_headcount": predicted,
        "residual": residual,
        "absolute_residual": np.abs(residual),
    })

    # Compare holdout residuals against calibrated directional limits.
    results["anomaly_type"] = np.select(
        [residual > positive, residual < negative],
        ["unexpectedly_high", "unexpectedly_low"],
        default="normal"
    )

    # Sensor freshness is a diagnostic, not an automatic exclusion.
    results["sensor_age_hours"] = observation_age_hours(
        test_times, raw_timestamps
    )
    results["sensor_gap_flag"] = (
        results["sensor_age_hours"].isna()
        | (results["sensor_age_hours"] > FRESHNESS_LIMIT_HOURS)
    )

    # Preserve local timestamps for engineering interpretation.
    local = pd.to_datetime(
        results["target_time_utc"], utc=True
    ).dt.tz_convert("Australia/Melbourne")
    results["target_time_melbourne"] = local.astype(str)

    results["positive_threshold"] = positive
    results["negative_threshold"] = negative

    return results, calibration_n


def group_events(results):
    """
    Combine consecutive anomalous hours of the same type.

    Parameters
    ----------
    results : pandas.DataFrame
        Hourly anomaly classifications for one floorspace.

    Returns
    -------
    pandas.DataFrame
        Event timing, duration, peak residual and sensor-gap
        diagnostics, or an empty DataFrame if no anomalies exist.

    Notes
    -----
    An event ends when anomalous hours are not consecutive or
    the anomaly type changes.
    """
    anomalies = results[
        results["anomaly_type"] != "normal"
    ].sort_values("target_time_utc").copy()

    if anomalies.empty:
        return pd.DataFrame()

    # Start a new event after a time gap or change in anomaly type.
    time_diff = anomalies["target_time_utc"].diff()
    new_event = (
        time_diff.ne(pd.Timedelta(hours=1))
        | anomalies["anomaly_type"].ne(
            anomalies["anomaly_type"].shift()
        )
    )

    anomalies["event_number"] = new_event.cumsum()
    events = []

    for event_number, group in anomalies.groupby("event_number"):
        # Identify the most extreme residual within each event.
        peak = group.loc[group["absolute_residual"].idxmax()]

        events.append({
            "floorspaceid": group["floorspaceid"].iloc[0],
            "event_number": int(event_number),
            "anomaly_type": group["anomaly_type"].iloc[0],
            "start_utc": group["target_time_utc"].iloc[0],
            "end_utc": group["target_time_utc"].iloc[-1],
            "start_melbourne": group["target_time_melbourne"].iloc[0],
            "end_melbourne": group["target_time_melbourne"].iloc[-1],
            "duration_hours": len(group),
            "peak_absolute_residual": peak["absolute_residual"],
            "peak_residual": peak["residual"],
            "peak_actual_headcount": peak["actual_headcount"],
            "peak_predicted_headcount": peak["predicted_headcount"],
            "peak_time_melbourne": peak["target_time_melbourne"],
            "gap_flagged_hours": int(group["sensor_gap_flag"].sum()),
            "any_sensor_gap_flag": bool(group["sensor_gap_flag"].any()),
        })

    return pd.DataFrame(events)


def plot_anomalies(results, space):
    """
    Plot test-period occupancy, residuals and anomaly thresholds.

    Parameters
    ----------
    results : pandas.DataFrame
        Hourly predictions, residuals and anomaly classifications.
    space : str
        Floorspace identifier used in the title and filename.
    """
    local = pd.to_datetime(
        results["target_time_utc"], utc=True
    ).dt.tz_convert("Australia/Melbourne")

    # Compare observed occupancy with the Stage 2 forecast.
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(
        local, results["actual_headcount"],
        label="Actual", linewidth=0.9
    )
    axes[0].plot(
        local, results["predicted_headcount"],
        label="Linear regression", linewidth=0.9, alpha=0.8
    )

    # Mark positive and negative anomalies with different symbols.
    for kind, marker in [
        ("unexpectedly_high", "^"),
        ("unexpectedly_low", "v")
    ]:
        mask = results["anomaly_type"].eq(kind)
        axes[0].scatter(
            local[mask],
            results.loc[mask, "actual_headcount"],
            marker=marker, s=28, label=kind, zorder=3
        )

    axes[0].set_ylabel("Average headcount")
    axes[0].set_title(f"Stage 3 anomaly detection — {space}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    # Show forecast residuals against the calibrated thresholds.
    axes[1].plot(
        local, results["residual"],
        linewidth=0.8, label="Forecast residual"
    )
    axes[1].axhline(
        results["positive_threshold"].iloc[0],
        linestyle="--", label="Positive threshold"
    )
    axes[1].axhline(
        results["negative_threshold"].iloc[0],
        linestyle="--", label="Negative threshold"
    )
    axes[1].axhline(0, linewidth=0.7)
    axes[1].set_ylabel("Actual minus predicted")
    axes[1].set_xlabel("Time")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / f"anomalies_{space[:8]}.png", dpi=140)
    plt.close(fig)


def main():
    """
    Execute anomaly detection for all floorspaces.

    Notes
    -----
    Reconstructs the Stage 2 examples, checks prediction alignment,
    calibrates residual thresholds and groups anomalous hours.
    Saves per-space results and consolidated event rankings.

    Raises
    ------
    FileNotFoundError
        If a required Stage 2 prediction file is missing.
    ValueError
        If prediction alignment or calibration validation fails.
    """
    started = time.perf_counter()

    hourly = pd.read_csv(HOURLY_FILE)
    hourly["hour_start_utc"] = pd.to_datetime(
        hourly["hour_start_utc"], utc=True
    )

    spaces = sorted(hourly["floorspaceid"].astype(str).unique())
    print(f"Stage 3: analysing {len(spaces)} spaces")
    print("Reading raw timestamps for sensor freshness...")
    raw_timestamps = read_raw_timestamps(spaces)

    all_events = []
    summaries = []

    for space in spaces:
        group = hourly[hourly["floorspaceid"] == space]

        # Recreate forecasting examples to match the Stage 2 holdout.
        examples = build_examples(group)

        prediction_file = STAGE2_DIR / f"test_predictions_{space}.csv"
        if not prediction_file.exists():
            raise FileNotFoundError(prediction_file)

        predictions = pd.read_csv(prediction_file)

        # Detect residual anomalies and attach sensor freshness flags.
        results, calibration_n = detect_anomalies(
            space, examples, predictions, raw_timestamps[space]
        )

        # Combine adjacent anomalous hours into interpretable events.
        events = group_events(results)
        if not events.empty:
            all_events.append(events)

        anomalies = results["anomaly_type"] != "normal"

        summaries.append({
            "floorspaceid": space,
            "test_hours": len(results),
            "calibration_hours": calibration_n,
            "positive_threshold": results["positive_threshold"].iloc[0],
            "negative_threshold": results["negative_threshold"].iloc[0],
            "unexpectedly_high_hours": int(
                (results["anomaly_type"] == "unexpectedly_high").sum()
            ),
            "unexpectedly_low_hours": int(
                (results["anomaly_type"] == "unexpectedly_low").sum()
            ),
            "total_anomalous_hours": int(anomalies.sum()),
            "anomaly_percentage": 100 * anomalies.mean(),
            "gap_flagged_test_hours": int(results["sensor_gap_flag"].sum()),
            "gap_flagged_anomalous_hours": int(
                (anomalies & results["sensor_gap_flag"]).sum()
            ),
            "number_of_events": len(events),
        })

        results.to_csv(
            OUT / f"hourly_anomalies_{space}.csv", index=False
        )
        plot_anomalies(results, space)

        print(
            f"{space[:8]}: {int(anomalies.sum())} anomalous hours, "
            f"{len(events)} events"
        )

    # Save per-space anomaly rates and calibration diagnostics.
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "anomaly_summary.csv", index=False)

    # Rank events by their largest absolute forecast residual.
    if all_events:
        event_table = pd.concat(all_events, ignore_index=True)
        event_table = event_table.sort_values(
            "peak_absolute_residual", ascending=False
        )
    else:
        event_table = pd.DataFrame(columns=[
            "floorspaceid", "event_number", "anomaly_type",
            "start_utc", "end_utc", "duration_hours",
            "peak_absolute_residual", "any_sensor_gap_flag"
        ])

    event_table.to_csv(OUT / "anomaly_events.csv", index=False)
    event_table.head(30).to_csv(
        OUT / "top_30_anomaly_events.csv", index=False
    )

    print("\nStage 3 summary:")
    print(summary.to_string(index=False))
    print(f"\nCompleted in {time.perf_counter() - started:.1f} seconds")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
