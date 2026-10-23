
"""
MMA3001 Stage 4: Time-weighted space underutilisation.

Uses Stage 1's integrated hourly empty fraction, not mean_headcount == 0.
Weekday 09:00–17:00 Melbourne time is a proxy for working hours.

Calculates time-weighted occupancy and empty-time measures across
floorspaces, calendar periods and hourly usage patterns.

Compares alternative working-hour windows and low-headcount thresholds
to assess the sensitivity of space underutilisation rankings.

Inputs
------
Stage 1 hourly_occupancy.csv.

Outputs
-------
Space utilisation summaries, working-hour rankings, daily and hourly
usage patterns, sensitivity results and diagnostic plots.

Run
---
python mma3001_stage4_underutilisation.py
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

# Consistent raw dataset directory across the project.
DATA_DIR = Path(
    os.environ.get("MMA3001_DATA_DIR", str(PROJECT_ROOT / "data"))
).expanduser().resolve()

# Read the reconstructed hourly occupancy from Stage 1.
INPUT_FILE = (
    PROJECT_ROOT / "mma3001_stage1_outputs" / "hourly_occupancy.csv"
)

# Save Stage 4 outputs alongside the other stage outputs.
OUT = PROJECT_ROOT / "mma3001_stage4_outputs"

# Use Melbourne local time and retain nearly complete hourly observations.
TIMEZONE = "Australia/Melbourne"
MIN_COVERAGE = 0.999

# Define low use and a weekday working-hours proxy.
LOW_USE_THRESHOLD = 1.0
TEACHING_START, TEACHING_END = 9, 17

# Compare alternative weekday analysis windows.
WINDOWS = {
    "all_hours": None,
    "weekday_9_to_17": (9, 17),
    "weekday_8_to_18": (8, 18),
    "weekday_10_to_16": (10, 16),
}

# Test whether low-use classifications depend on the threshold.
LOW_USE_THRESHOLDS = (0.5, 1.0, 2.0)

# Keep output measures consistent across summary tables.
METRICS = [
    "calendar_hours", "analysed_hours", "observed_hours",
    "empty_hours", "occupied_hours", "empty_fraction",
    "occupied_fraction", "mean_headcount", "person_hours",
    "low_mean_headcount_hour_fraction",
]


def load_data():
    """
    Validate Stage 1 occupancy data and derive local calendar features.

    Returns
    -------
    pandas.DataFrame
        Valid hourly observations with Melbourne-local time fields,
        coverage flags and occupied/empty fractions.

    Raises
    ------
    FileNotFoundError
        If the Stage 1 hourly occupancy file is missing.
    ValueError
        If required columns are missing, timestamps are duplicated
        or occupancy and coverage values are invalid.

    Notes
    -----
    Empty-time fraction must come from Stage 1 integration.
    Mean headcount equal to zero is not an equivalent measure.
    """
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Stage 1 file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    required = {
        "floorspaceid", "hour_start_utc",
        "mean_headcount", "coverage_fraction",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Accept supported names for the integrated empty-time fraction.
    candidates = [
        "empty_fraction", "mean_empty_fraction",
        "hourly_empty_fraction", "fraction_empty",
        "empty_time_fraction", "zero_headcount_fraction",
    ]
    empty_col = next((c for c in candidates if c in df.columns), None)
    if empty_col is None:
        raise ValueError(
            "Integrated empty-time fraction column not found. "
            "Do not replace it with mean_headcount == 0."
        )

    # Standardise column names and numerical types.
    df = df.rename(columns={empty_col: "empty_fraction"})
    df["hour_start_utc"] = pd.to_datetime(
        df["hour_start_utc"], utc=True, errors="coerce"
    )
    numeric = ["mean_headcount", "coverage_fraction", "empty_fraction"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove incomplete rows and reject duplicate space-hour records.
    df = df.dropna(
        subset=["floorspaceid", "hour_start_utc"] + numeric
    ).copy()
    if df.duplicated(["floorspaceid", "hour_start_utc"]).any():
        raise ValueError("Duplicate space-hour rows found")

    # Enforce physical bounds for occupancy and time fractions.
    valid = (
        df["coverage_fraction"].between(0, 1)
        & df["empty_fraction"].between(0, 1)
        & df["mean_headcount"].ge(0)
    )
    if not valid.all():
        raise ValueError(f"{(~valid).sum()} invalid numeric rows")

    # Use Melbourne-local dates and hours for operational comparisons.
    local = df["hour_start_utc"].dt.tz_convert(TIMEZONE)
    df["sufficient_coverage"] = df["coverage_fraction"] >= MIN_COVERAGE
    df["local_time"] = local
    df["local_hour"] = local.dt.hour
    df["local_date"] = local.dt.date
    df["weekday"] = local.dt.dayofweek
    df["is_weekend"] = df["weekday"] >= 5

    # Treat weekday 9 AM–5 PM as a working-hours proxy.
    df["is_teaching_hours"] = (
        (df["weekday"] < 5)
        & df["local_hour"].between(TEACHING_START, TEACHING_END - 1)
    )

    df["occupied_fraction"] = 1 - df["empty_fraction"]
    df["observed_hours"] = df["coverage_fraction"]
    return df


def summarise(group):
    """
    Recombine Stage 1 integrals using observed-duration weights.

    Returns observed hours, person-hours, time-empty fraction and
    low-use-hour fraction. No room capacity is assumed.

    Parameters
    ----------
    group : pandas.DataFrame
        Hourly observations for a space and analysis period.

    Returns
    -------
    dict
        Time-weighted utilisation measures and observation counts.

    Notes
    -----
    Only hours meeting the minimum coverage threshold contribute
    to the weighted measures. Empty time is based on integrated
    empty fractions rather than zero mean headcount.
    """
    # Exclude hours with insufficient observed coverage.
    valid = group[group["sufficient_coverage"]]
    result = dict.fromkeys(METRICS, np.nan)
    result.update(
        calendar_hours=len(group),
        analysed_hours=len(valid),
        observed_hours=0,
    )
    if valid.empty:
        return result

    # Weight each hourly measure by its observed duration.
    duration = valid["observed_hours"].to_numpy(float)
    total = duration.sum()
    empty = np.sum(valid["empty_fraction"].to_numpy(float) * duration)
    occupied = np.sum(valid["occupied_fraction"].to_numpy(float) * duration)
    people = np.sum(valid["mean_headcount"].to_numpy(float) * duration)

    # Recombine time integrals before calculating overall fractions.
    result.update(
        observed_hours=total,
        empty_hours=empty,
        occupied_hours=occupied,
        empty_fraction=empty / total,
        occupied_fraction=occupied / total,
        mean_headcount=people / total,
        person_hours=people,
        low_mean_headcount_hour_fraction=np.average(
            valid["mean_headcount"].to_numpy(float) < LOW_USE_THRESHOLD,
            weights=duration,
        ),
    )
    return result


def grouped_summary(df, keys):
    """
    Apply the same time-weighted summary to selected groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Hourly occupancy observations.
    keys : list of str
        Columns defining each group.

    Returns
    -------
    pandas.DataFrame
        Group identifiers and their utilisation measures.
    """
    rows = []
    for values, group in df.groupby(keys):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values))
        row.update(summarise(group))
        rows.append(row)
    return pd.DataFrame(rows)


def main_summaries(df):
    """
    Export overall utilisation summaries and working-hour rankings.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated hourly occupancy observations.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Period-specific utilisation summaries and working-hour ranking.

    Notes
    -----
    Spaces are ranked by descending empty-time fraction during
    weekday 9 AM–5 PM hours.
    """
    # Compare overall, weekday, weekend and working-hour use.
    subsets = {
        "all_hours": pd.Series(True, index=df.index),
        "weekdays_all_hours": ~df["is_weekend"],
        "weekends_all_hours": df["is_weekend"],
        "weekday_9_to_17": df["is_teaching_hours"],
        "weekday_outside_9_to_17": (
            ~df["is_weekend"] & ~df["is_teaching_hours"]
        ),
    }
    frames = []
    for name, mask in subsets.items():
        part = grouped_summary(df.loc[mask], ["floorspaceid"])
        part.insert(1, "period", name)
        frames.append(part)

    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(OUT / "space_utilisation_summary.csv", index=False)

    # Rank working-hour underutilisation by empty-time fraction.
    ranking = summary.loc[
        summary["period"] == "weekday_9_to_17"
    ].sort_values("empty_fraction", ascending=False).copy()
    ranking["underutilisation_rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(
        OUT / "teaching_hours_space_ranking.csv", index=False
    )
    return summary, ranking


def usage_patterns(df):
    """
    Export hourly and daily time-weighted occupancy patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated hourly occupancy observations.

    Returns
    -------
    pandas.DataFrame
        Weekday and weekend usage patterns by local hour.

    Notes
    -----
    Daily summaries are also saved, grouped by floorspace
    and Melbourne-local calendar date.
    """
    # Separate weekday and weekend profiles by local hour.
    hourly = grouped_summary(
        df.loc[df["sufficient_coverage"]],
        ["floorspaceid", "is_weekend", "local_hour"],
    )
    hourly["day_type"] = np.where(
        hourly["is_weekend"], "weekend", "weekday"
    )
    hourly = hourly[
        ["floorspaceid", "day_type", "local_hour"] + METRICS
    ]
    hourly.to_csv(OUT / "hourly_usage_patterns.csv", index=False)

    # Summarise daily occupancy without averaging unweighted fractions.
    daily = grouped_summary(df, ["floorspaceid", "local_date"])
    weekend = df.groupby(
        ["floorspaceid", "local_date"]
    )["is_weekend"].first().rename("is_weekend")
    daily = daily.merge(
        weekend.reset_index(),
        on=["floorspaceid", "local_date"],
        how="left",
    )
    daily = daily[
        ["floorspaceid", "local_date", "is_weekend"] + METRICS
    ]
    daily.to_csv(OUT / "daily_usage_patterns.csv", index=False)
    return hourly


def sensitivity_analysis(df):
    """
    Compare alternative time windows and low-use thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated hourly occupancy observations.

    Returns
    -------
    pandas.DataFrame
        Utilisation metrics and rankings for each analysis setting.

    Notes
    -----
    Varying time windows tests the working-hours assumption.
    Varying headcount thresholds tests low-use classifications.
    """
    rows = []
    for name, window in WINDOWS.items():
        if window is None:
            subset = df
        else:
            start, end = window
            subset = df.loc[
                ~df["is_weekend"]
                & df["local_hour"].between(start, end - 1)
            ]

        for space, group in subset.groupby("floorspaceid"):
            valid = group[group["sufficient_coverage"]]
            if valid.empty:
                continue

            base = summarise(group)
            duration = valid["observed_hours"].to_numpy(float)
            counts = valid["mean_headcount"].to_numpy(float)

            # Recalculate low-use fractions for each threshold.
            for threshold in LOW_USE_THRESHOLDS:
                rows.append({
                    "floorspaceid": space,
                    "analysis_window": name,
                    "low_mean_headcount_threshold": threshold,
                    "empty_fraction": base["empty_fraction"],
                    "occupied_fraction": base["occupied_fraction"],
                    "mean_headcount": base["mean_headcount"],
                    "observed_hours": base["observed_hours"],
                    "fraction_of_hours_with_low_mean_headcount":
                        np.average(counts < threshold, weights=duration),
                })

    result = pd.DataFrame(rows)
    keys = ["analysis_window", "low_mean_headcount_threshold"]

    # Compare space rankings under each sensitivity setting.
    for rank, metric in [
        ("empty_fraction_rank", "empty_fraction"),
        (
            "low_mean_headcount_rank",
            "fraction_of_hours_with_low_mean_headcount",
        ),
    ]:
        result[rank] = result.groupby(keys)[metric].rank(
            ascending=False, method="min"
        ).astype(int)

    result.to_csv(OUT / "sensitivity_analysis.csv", index=False)
    return result


def save_plot(fig, filename):
    """
    Save and close a diagnostic figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filename : str
        Requested output filename.

    Notes
    -----
    Selected filenames are shortened before saving.
    """
    fig.tight_layout()

    # Use shorter filenames for the two comparison figures.
    short_names = {
        "space_empty_fraction_comparison.png": "space_comparison.png",
        "time_window_sensitivity.png": "sensitivity.png",
    }

    filename = short_names.get(filename, filename)

    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def plot_space_comparison(summary):
    """
    Compare overall and working-hour empty-time fractions by space.

    Parameters
    ----------
    summary : pandas.DataFrame
        Period-specific utilisation summaries.
    """
    # Rank the displayed spaces by working-hour empty fraction.
    pivot = summary.loc[
        summary["period"].isin(["all_hours", "weekday_9_to_17"])
    ].pivot(
        index="floorspaceid",
        columns="period",
        values="empty_fraction",
    ).sort_values("weekday_9_to_17", ascending=False)

    x, width = np.arange(len(pivot)), 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x - width / 2, pivot["all_hours"] * 100,
        width, label="All hours",
    )
    ax.bar(
        x + width / 2, pivot["weekday_9_to_17"] * 100,
        width, label="Weekdays 9 AM–5 PM",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([s[:8] for s in pivot.index])
    ax.set(
        ylabel="Empty time fraction (%)",
        title="Time-weighted empty fraction by space",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    save_plot(fig, "space_empty_fraction_comparison.png")


def plot_hourly_patterns(patterns):
    """
    Plot weekday and weekend occupancy profiles for each space.

    Parameters
    ----------
    patterns : pandas.DataFrame
        Hourly headcount and empty-time summaries by day type.
    """
    for space, part in patterns.groupby("floorspaceid"):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for day_type in ["weekday", "weekend"]:
            subset = part.loc[part["day_type"] == day_type]
            axes[0].plot(
                subset["local_hour"], subset["mean_headcount"],
                marker="o", markersize=3, label=day_type,
            )
            axes[1].plot(
                subset["local_hour"], subset["empty_fraction"] * 100,
                marker="o", markersize=3, label=day_type,
            )

        axes[0].set_ylabel("Mean headcount")
        axes[0].set_title(f"Daily usage patterns — {space[:8]}")
        axes[1].set_ylabel("Empty time proportion (%)")
        axes[1].set_xlabel("Hour of day")
        axes[1].set_xticks(range(0, 24, 2))
        axes[1].set_ylim(0, 100)

        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.2)
        save_plot(fig, f"hourly_patterns_{space[:8]}.png")


def plot_weekday_heatmaps(df):
    """
    Plot mean headcount by local weekday and hour for each space.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated hourly occupancy observations.
    """
    # Use only sufficiently covered hours for the heatmaps.
    valid = df.loc[df["sufficient_coverage"]]
    for space, group in valid.groupby("floorspaceid"):
        table = grouped_summary(
            group, ["weekday", "local_hour"]
        ).pivot(
            index="weekday",
            columns="local_hour",
            values="mean_headcount",
        ).reindex(index=range(7), columns=range(24))

        fig, ax = plt.subplots(figsize=(12, 4.5))
        image = ax.imshow(
            table.to_numpy(float),
            aspect="auto", interpolation="nearest",
        )
        ax.set_yticks(range(7))
        ax.set_yticklabels(
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        )
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        ax.set_xlabel("Hour of day (Melbourne)")
        ax.set_title(
            f"Average headcount by weekday and hour — {space[:8]}"
        )
        fig.colorbar(image, ax=ax, label="Mean headcount")
        save_plot(fig, f"usage_heatmap_{space[:8]}.png")


def plot_sensitivity(sensitivity):
    """
    Plot empty-time fractions across alternative analysis windows.

    Parameters
    ----------
    sensitivity : pandas.DataFrame
        Space utilisation results for each sensitivity setting.
    """
    # Use a common low-use threshold for the window comparison.
    selected = sensitivity.loc[
        sensitivity["low_mean_headcount_threshold"] == 1.0
    ]
    pivot = selected.pivot(
        index="floorspaceid",
        columns="analysis_window",
        values="empty_fraction",
    )
    x, columns = np.arange(len(pivot)), list(pivot.columns)
    width = 0.8 / len(columns)
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, column in enumerate(columns):
        ax.bar(
            x + (i - (len(columns) - 1) / 2) * width,
            pivot[column] * 100,
            width,
            label=column.replace("_", " "),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([s[:8] for s in pivot.index])
    ax.set_ylabel("Empty time fraction (%)")
    ax.set_title("Sensitivity to the analysis time window")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    save_plot(fig, "time_window_sensitivity.png")


def main():
    """
    Run Stage 4 underutilisation analysis and export results.

    Notes
    -----
    Produces period summaries, working-hour rankings, usage
    profiles, sensitivity results and diagnostic figures.
    """
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Stage 4: Space underutilisation analysis")
    df = load_data()
    print(
        f"Loaded {len(df):,} hourly records "
        f"across {df['floorspaceid'].nunique()} spaces"
    )
    print(
        f"Rows meeting coverage threshold: "
        f"{df['sufficient_coverage'].sum():,}"
    )

    # Generate utilisation summaries and sensitivity results.
    summary, ranking = main_summaries(df)
    patterns = usage_patterns(df)
    sensitivity = sensitivity_analysis(df)

    # Export comparison charts, daily profiles and heatmaps.
    plot_space_comparison(summary)
    plot_hourly_patterns(patterns)
    plot_weekday_heatmaps(df)
    plot_sensitivity(sensitivity)

    print("\nWeekday 9 AM–5 PM underutilisation ranking:")
    print(ranking[[
        "underutilisation_rank", "floorspaceid",
        "empty_fraction", "occupied_fraction",
        "mean_headcount", "observed_hours",
    ]].to_string(index=False))

    print("\nSensitivity analysis:")
    print(sensitivity.loc[
        sensitivity["low_mean_headcount_threshold"] == 1.0,
        [
            "floorspaceid", "analysis_window", "empty_fraction",
            "fraction_of_hours_with_low_mean_headcount",
        ],
    ].to_string(index=False))

    print(f"\nCompleted in {time.perf_counter() - started:.1f} seconds")
    print(f"Outputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
