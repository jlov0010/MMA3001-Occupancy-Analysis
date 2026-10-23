
"""
MMA3001 Stage 1 — Occupancy Audit and Hourly Reconstruction.

Reconstructs hourly occupancy from irregular sensor events using exact
zero-order-hold integration. Produces the CSV outputs required by Stages 2–5.

Install: pip install pandas numpy matplotlib
Run:     python mma3001_stage1_occupancy_audit.py

Assumptions:
- Each valid headcount persists until the next event.
- Invalid headcounts contribute no observed coverage.
- Long event gaps are retained, not labelled as sensor failures.
- Only complete hours between recorded events are reconstructed.
- All analysis boundaries use Melbourne local time.
"""

from collections import Counter
from pathlib import Path
import json
import logging
import os
import re
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# Support scripts stored in the project root or src/.
PROJECT_ROOT = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name == "src"
    else SCRIPT_DIR
)

# Allow the dataset location to be set through an environment variable.
DATA_DIR = Path(
    os.environ.get("MMA3001_DATA_DIR", str(PROJECT_ROOT / "data"))
).expanduser().resolve()

OCCUPANCY_FILE = (
    DATA_DIR / "5occupancySensor_MayToDec2024_9MRows.csv"
)

# Store reconstructed data and audit outputs together.
OUTPUT_DIR = PROJECT_ROOT / "mma3001_stage1_outputs"

# Define the common analysis window in Melbourne local time.
TIMEZONE = "Australia/Melbourne"
START = pd.Timestamp("2024-05-01", tz=TIMEZONE).tz_convert("UTC")
END = pd.Timestamp("2025-01-01", tz=TIMEZONE).tz_convert("UTC")

# Limit CSV memory usage and define diagnostic gap thresholds.
CHUNK_SIZE = 150_000
GAP_THRESHOLDS = (6, 24, 72, 168)
COLUMNS = [
    "id", "deviceid", "floorspaceid",
    "occupancystatus", "headcount", "collecteddate"
]


def log(message):
    """
    Write a progress message to the console and run log.

    Parameters
    ----------
    message : str
        Message to record.
    """
    logging.info(message)


def safe_name(value):
    """
    Convert an identifier into a safe output filename component.

    Parameters
    ----------
    value : object
        Floorspace identifier to convert.

    Returns
    -------
    str
        Identifier containing only permitted filename characters.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))[:100]


def save_counts(counter, identifier, filename):
    """
    Save identifier frequencies to a CSV file.

    Parameters
    ----------
    counter : collections.Counter
        Number of records for each identifier.
    identifier : str
        Name of the identifier column.
    filename : str
        Output CSV filename.
    """
    pd.DataFrame(
        [{identifier: key, "records": count} for key, count in counter.items()],
        columns=[identifier, "records"]
    ).to_csv(OUTPUT_DIR / filename, index=False)


# ---------------------------------------------------------------------
# Large CSV partitioning and metadata
# ---------------------------------------------------------------------

def partition_dataset(temp_dir):
    """
    Read the occupancy CSV in chunks and partition events by floorspace.

    Parameters
    ----------
    temp_dir : pathlib.Path
        Temporary directory for per-space CSV files.

    Returns
    -------
    tuple[dict, dict]
        Space-to-file mapping and dataset metadata.

    Raises
    ------
    ValueError
        If required CSV columns are missing.
    """
    # Validate the expected input columns before processing the dataset.
    available = pd.read_csv(
        OCCUPANCY_FILE, nrows=0, encoding="utf-8-sig"
    ).columns
    missing = sorted(set(COLUMNS) - set(available))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    spaces, devices, statuses = Counter(), Counter(), Counter()
    paths = {}
    total = missing_spaces = invalid_headcounts = 0

    # Read the large dataset in chunks to limit memory usage.
    reader = pd.read_csv(
        OCCUPANCY_FILE, usecols=COLUMNS, dtype="string",
        chunksize=CHUNK_SIZE, encoding="utf-8-sig", low_memory=False
    )

    for number, chunk in enumerate(reader, 1):
        total += len(chunk)

        # Collect dataset-wide counts and data-quality statistics.
        spaces.update(chunk["floorspaceid"].dropna().value_counts().to_dict())
        devices.update(chunk["deviceid"].dropna().value_counts().to_dict())
        statuses.update(
            chunk["occupancystatus"].fillna("<missing>").value_counts().to_dict()
        )
        missing_spaces += int(chunk["floorspaceid"].isna().sum())

        counts = pd.to_numeric(chunk["headcount"], errors="coerce")
        invalid_headcounts += int((counts.isna() | counts.lt(0)).sum())

        # Partition events by floorspace for independent reconstruction.
        for space_id, group in chunk.dropna(
            subset=["floorspaceid"]
        ).groupby("floorspaceid", sort=False):
            space_id = str(space_id)
            if space_id not in paths:
                paths[space_id] = temp_dir / f"space_{len(paths):04d}.csv"
            path = paths[space_id]
            group.to_csv(
                path, mode="a", index=False, header=not path.exists()
            )

        if number % 10 == 0:
            log(f"Processed {total:,} raw records")

    save_counts(spaces, "floorspaceid", "space_record_counts.csv")
    save_counts(devices, "deviceid", "device_record_counts.csv")

    # Preserve dataset inventory and analysis-period information.
    metadata = {
        "source_file": str(OCCUPANCY_FILE),
        "total_records": total,
        "distinct_spaces": len(spaces),
        "distinct_devices": len(devices),
        "missing_space_records": missing_spaces,
        "invalid_headcount_records": invalid_headcounts,
        "status_counts": dict(statuses),
        "records_per_space": dict(spaces),
        "records_per_device": dict(devices),
        "analysis_start_local": "2024-05-01",
        "analysis_end_local_inclusive": "2024-12-31",
    }
    return paths, metadata


# ---------------------------------------------------------------------
# Exact zero-order-hold numerical integration
# ---------------------------------------------------------------------

def reconstruct_hourly_occupancy(events):
    """
    Reconstruct complete hourly intervals using zero-order hold.

    Parameters
    ----------
    events : pandas.DataFrame
        Time-sorted events with UTC `timestamp` and numeric
        `headcount_numeric` columns. Timestamps must be unique.

    Returns
    -------
    pandas.DataFrame
        Hourly mean headcount, occupied/empty fractions, coverage,
        UTC and Melbourne-local time fields.

    Notes
    -----
    Each headcount applies from its timestamp until the next event.
    Mean headcount is the time integral divided by valid covered time.
    Invalid or negative headcounts are excluded from coverage.
    """
    if len(events) < 2:
        return pd.DataFrame()

    timestamps = events["timestamp"]
    counts = events["headcount_numeric"].to_numpy(dtype=float)

    # Restrict reconstruction to complete hours within the analysis window.
    first = max(timestamps.iloc[0].ceil("h"), START)
    last = min(timestamps.iloc[-1].floor("h"), END)
    if last <= first:
        return pd.DataFrame()

    boundaries = pd.date_range(first, last, freq="h")
    event_ns = timestamps.astype("int64").to_numpy()
    boundary_ns = boundaries.astype("int64").to_numpy()

    # Calculate event durations and exclude invalid observations.
    duration = np.diff(event_ns) / 1e9
    valid = np.isfinite(counts[:-1]) & (counts[:-1] >= 0) & (duration > 0)

    # Three exact piecewise-constant integrals:
    # headcount*time, valid observed time, occupied time.
    integrands = (
        np.where(valid, counts[:-1] * duration, 0.0),
        np.where(valid, duration, 0.0),
        np.where(valid & (counts[:-1] > 0), duration, 0.0),
    )

    # Locate the active sensor reading at each hourly boundary.
    index = np.searchsorted(event_ns, boundary_ns, side="right") - 1
    index = np.clip(index, 0, len(event_ns) - 2)
    partial = np.clip(
        (boundary_ns - event_ns[index]) / 1e9, 0, duration[index]
    )

    def integrate_at_boundaries(segment_values, partial_values):
        """
        Evaluate cumulative integrals at hourly boundaries.

        Parameters
        ----------
        segment_values : numpy.ndarray
            Complete event-interval contributions.
        partial_values : numpy.ndarray
            Partial contributions at hourly boundaries.

        Returns
        -------
        numpy.ndarray
            Cumulative integral at each hourly boundary.
        """
        cumulative = np.r_[0.0, np.cumsum(segment_values)]
        return cumulative[index] + partial_values

    # Subtract consecutive boundary integrals to obtain hourly totals.
    integral, coverage, occupied = [
        np.diff(integrate_at_boundaries(
            segment_values,
            np.where(valid[index], rate * partial, 0.0)
        ))
        for segment_values, rate in zip(
            integrands,
            (counts[index], np.ones(len(index)),
             (counts[index] > 0).astype(float))
        )
    ]

    # Weight occupancy intervals by duration for fair hourly contribution.
    mean = np.divide(
        integral, coverage,
        out=np.full(len(coverage), np.nan),
        where=coverage > 0
    )

    # Calculate the occupied share of valid observed time.
    occupied_fraction = np.divide(
        occupied, coverage,
        out=np.full(len(coverage), np.nan),
        where=coverage > 0
    )
    occupied_fraction = np.clip(occupied_fraction, 0, 1)

    hour_seconds = np.diff(boundary_ns) / 1e9
    local = boundaries[:-1].tz_convert(TIMEZONE)

    # Preserve UTC timestamps and Melbourne-local calendar information.
    return pd.DataFrame({
        "hour_start_utc": boundaries[:-1].astype(str),
        "hour_start_local": local.astype(str),
        "local_date": local.strftime("%Y-%m-%d"),
        "local_hour": local.hour,
        "local_weekday": local.dayofweek,
        "is_weekend": local.dayofweek >= 5,
        "mean_headcount": mean,
        "occupied_fraction": occupied_fraction,
        "empty_fraction": 1 - occupied_fraction,
        "covered_minutes": coverage / 60,
        "hour_duration_minutes": hour_seconds / 60,
        "coverage_fraction": coverage / hour_seconds,
    })


def test_numerical_integration():
    """
    Verify reconstruction against a known time-weighted example.

    Notes
    -----
    Headcounts of 1 for 20 minutes and 4 for 40 minutes give
    a mean of (1*20 + 4*40)/60 = 3 occupants.
    The hour must also be fully occupied.
    """
    events = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-06-01T00:00:00Z",
            "2024-06-01T00:20:00Z",
            "2024-06-01T01:00:00Z"
        ], utc=True),
        "headcount_numeric": [1.0, 4.0, 0.0],
    })
    result = reconstruct_hourly_occupancy(events)
    assert len(result) == 1
    assert np.isclose(result.iloc[0]["mean_headcount"], 3.0, atol=1e-8)
    assert np.isclose(result.iloc[0]["occupied_fraction"], 1.0, atol=1e-8)
    log("Numerical integration self-test passed")


# ---------------------------------------------------------------------
# Daily summaries and diagnostic plots
# ---------------------------------------------------------------------

def save_daily_outputs(space_id, hourly):
    """
    Save daily occupancy summaries and a headcount plot.

    Parameters
    ----------
    space_id : str
        Floorspace identifier used in output filenames.
    hourly : pandas.DataFrame
        Reconstructed hourly occupancy observations.
    """
    if hourly.empty:
        return

    name = safe_name(space_id)

    # Aggregate hourly observations by Melbourne-local date.
    daily = hourly.groupby("local_date", as_index=False).agg(
        mean_hourly_headcount=("mean_headcount", "mean"),
        mean_hourly_empty_fraction=("empty_fraction", "mean"),
        hours=("mean_headcount", "size"),
        mean_coverage_fraction=("coverage_fraction", "mean"),
    )
    daily.to_csv(OUTPUT_DIR / f"daily_summary_{name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(
        pd.to_datetime(daily["local_date"]),
        daily["mean_hourly_headcount"], linewidth=0.9
    )
    ax.set(
        title=f"Daily mean reconstructed headcount\nSpace: {space_id}",
        xlabel="Date", ylabel="Mean headcount"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"plot_daily_{name}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Individual floorspace audit
# ---------------------------------------------------------------------

def analyse_space(space_id, file_path):
    """
    Audit sensor events and reconstruct occupancy for one floorspace.

    Parameters
    ----------
    space_id : str
        Identifier of the floorspace being analysed.
    file_path : pathlib.Path
        Path to its partitioned event CSV.

    Returns
    -------
    tuple[dict, pandas.DataFrame]
        Per-space audit statistics and reconstructed hourly observations.

    Notes
    -----
    Duplicate timestamps retain the last event after stable sorting.
    Events outside the analysis window define integration boundaries.
    """
    log(f"Analysing space {space_id}")
    df = pd.read_csv(file_path, dtype="string", low_memory=False)
    raw_records = len(df)

    # Convert raw timestamps to UTC and headcounts to numeric values.
    df["timestamp"] = pd.to_datetime(
        df["collecteddate"], errors="coerce", utc=True
    )
    df["headcount_numeric"] = pd.to_numeric(
        df["headcount"], errors="coerce"
    )

    # Remove invalid timestamps and retain the last duplicate event.
    invalid_timestamps = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"]).sort_values(
        ["timestamp", "id"], kind="stable"
    )
    duplicate_timestamps = int(df["timestamp"].duplicated().sum())
    df = df.drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    # Reuse the exact sorted, unique event timestamps in Stages 2 and 3.
    # int64 stores UTC nanoseconds without string conversion or precision loss.
    cache_path = OUTPUT_DIR / f"event_timestamps_{safe_name(space_id)}.npy"
    np.save(cache_path, df["timestamp"].astype("int64").to_numpy())
    log(f"Saved shared timestamp cache: {cache_path.name}")

    if df.empty:
        return {
            "floorspaceid": space_id,
            "raw_records": raw_records,
            "invalid_timestamps": invalid_timestamps,
        }, pd.DataFrame()

    timestamps = df["timestamp"]
    counts = df["headcount_numeric"]
    status = df["occupancystatus"].fillna("<missing>")

    # Measure event gaps and changes in reported occupancy.
    gaps = timestamps.diff().dt.total_seconds() / 3600
    status_changes = status.ne(status.shift()).fillna(False)
    count_changes = counts.ne(counts.shift()).fillna(False)

    # Keep boundary events so carry-forward integration is well-defined.
    before = df.loc[timestamps < START].tail(1)
    within = df.loc[(timestamps >= START) & (timestamps <= END)]
    after = df.loc[timestamps > END].head(1)
    events = pd.concat([before, within, after]).sort_values(
        "timestamp"
    ).reset_index(drop=True)

    hourly = reconstruct_hourly_occupancy(events)
    if not hourly.empty:
        hourly.insert(0, "floorspaceid", space_id)

    # Summarise event quality, sensor consistency and hourly occupancy.
    summary = {
        "floorspaceid": space_id,
        "raw_records": raw_records,
        "usable_events": len(df),
        "distinct_devices": int(df["deviceid"].nunique()),
        "first_event_local": str(timestamps.iloc[0].tz_convert(TIMEZONE)),
        "last_event_local": str(timestamps.iloc[-1].tz_convert(TIMEZONE)),
        "invalid_timestamps": invalid_timestamps,
        "duplicate_timestamps": duplicate_timestamps,
        "invalid_headcounts": int((counts.isna() | counts.lt(0)).sum()),
        "status_changes_between_rows": int(status_changes.iloc[1:].sum()),
        "headcount_changes_between_rows": int(count_changes.iloc[1:].sum()),
        "repeated_status_and_headcount": int(
            (~status_changes & ~count_changes).sum()
        ),
        "notoccupied_with_positive_headcount": int(
            (status.eq("NotOccupied") & counts.gt(0)).sum()
        ),
        "currentlyoccupied_with_zero_headcount": int(
            (status.eq("CurrentlyOccupied") & counts.eq(0)).sum()
        ),
        "median_gap_minutes": float(gaps.median() * 60),
        "p95_gap_hours": float(gaps.quantile(0.95)),
        "max_gap_hours": float(gaps.max()),
        "events_within_common_window": int(
            ((timestamps >= START) & (timestamps < END)).sum()
        ),
        "hourly_rows_common_window": len(hourly),
        "mean_hourly_headcount_common_window": (
            float(hourly["mean_headcount"].mean())
            if not hourly.empty else np.nan
        ),
        "mean_hourly_empty_fraction_common_window": (
            float(hourly["empty_fraction"].mean())
            if not hourly.empty else np.nan
        ),
    }
    summary.update({
        f"gaps_over_{threshold}h": int((gaps > threshold).sum())
        for threshold in GAP_THRESHOLDS
    })

    # Event gaps are diagnostic; they do not prove sensor malfunction.
    indices = gaps.nlargest(30).index
    longest = df.loc[
        indices, ["timestamp", "occupancystatus", "headcount_numeric"]
    ].copy()
    longest.insert(0, "floorspaceid", space_id)
    longest["gap_hours"] = gaps.loc[indices]
    longest["preceding_headcount"] = counts.shift().loc[indices]
    longest.sort_values("gap_hours", ascending=False).to_csv(
        OUTPUT_DIR / f"longest_gaps_{safe_name(space_id)}.csv",
        index=False
    )

    save_daily_outputs(space_id, hourly)
    return summary, hourly


# ---------------------------------------------------------------------
# Output documentation and main workflow
# ---------------------------------------------------------------------

def write_output_notes():
    """
    Save descriptions of Stage 1 outputs and modelling assumptions.
    """
    notes = """MMA3001 STAGE 1 OUTPUTS

hourly_occupancy.csv:
  Hourly time-weighted headcount, occupied/empty fractions and coverage.
space_audit_summary.csv:
  Per-space event counts, consistency checks and gap statistics.
run_metadata.json:
  Dataset inventory, record counts and analysis period.
space_record_counts.csv / device_record_counts.csv:
  Raw event counts grouped by identifier.
longest_gaps_<space>.csv:
  Largest intervals between recorded events.
daily_summary_<space>.csv:
  Daily aggregates of reconstructed hourly occupancy.
plot_daily_<space>.png:
  Daily mean headcount time-series plot.
run_log.txt:
  Execution messages and timing.

ASSUMPTIONS
- Headcount persists until the next recorded event (zero-order hold).
- Invalid headcounts are excluded from valid observed coverage.
- Headcount > 0 means occupied; headcount = 0 means empty.
- Long event gaps are retained and are not automatically outages.
- occupancystatus is used for basic consistency checks only.
- previousoccupancystatus is not analysed.
- No extrapolation occurs before the first or after the last event.
- The common analysis period is May–December 2024 in Melbourne time.
- No room capacities or energy savings are assumed.
- Reconstructed occupancy is not independently verified ground truth.
"""
    (OUTPUT_DIR / "README_outputs.txt").write_text(
        notes, encoding="utf-8"
    )


def main():
    """
    Execute the Stage 1 audit and hourly reconstruction.

    Raises
    ------
    FileNotFoundError
        If the expected occupancy CSV is missing.
    ValueError
        If the dataset lacks required columns.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                OUTPUT_DIR / "run_log.txt", mode="w", encoding="utf-8"
            ),
        ],
        force=True,
    )
    start_time = time.perf_counter()
    log("MMA3001 Stage 1 — Occupancy Audit and Hourly Reconstruction")

    # Check the input dataset before beginning reconstruction.
    if not OCCUPANCY_FILE.is_file():
        candidates = sorted(DATA_DIR.glob("*occupancy*.csv"))
        details = "\n".join(f"  {p.name}" for p in candidates)
        raise FileNotFoundError(
            f"Occupancy CSV not found:\n{OCCUPANCY_FILE}\n"
            f"Other occupancy CSVs in data folder:\n"
            f"{details or '  None found'}\n"
            "Check DATA_DIR and OCCUPANCY_FILE."
        )

    write_output_notes()
    test_numerical_integration()

    # Replace the previous hourly CSV to prevent duplicate results.
    hourly_path = OUTPUT_DIR / "hourly_occupancy.csv"
    if hourly_path.exists():
        hourly_path.unlink()

    summaries = []

    # Use temporary partitions to avoid holding all events in memory.
    with tempfile.TemporaryDirectory(prefix="mma3001_occupancy_") as folder:
        partitions, metadata = partition_dataset(Path(folder))
        log(
            f"Raw records: {metadata['total_records']:,}; "
            f"spaces: {metadata['distinct_spaces']}"
        )

        for number, (space_id, path) in enumerate(partitions.items(), 1):
            summary, hourly = analyse_space(space_id, path)
            summaries.append(summary)

            # Append each space's results to the common hourly dataset.
            if not hourly.empty:
                hourly.to_csv(
                    hourly_path, mode="a", index=False,
                    header=not hourly_path.exists()
                )
            log(
                f"Completed space {number}/{len(partitions)}: "
                f"{len(hourly):,} hourly records"
            )

    # Save consolidated audit statistics and dataset metadata.
    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "space_audit_summary.csv", index=False
    )
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    log(f"Stage 1 complete in {time.perf_counter() - start_time:.1f} seconds")
    log(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
