
"""
MMA3001 - Occupancy Analysis Pipeline

Run the five project stages in sequence or execute selected stages.

Stage scripts are located beside main.py or in the src/ folder.
Each stage runs as a separate Python process. Execution stops if a
required script is missing or a stage fails.

Usage
-----
python main.py
python main.py --stage 3
python main.py --from-stage 4
python main.py --list
python main.py --dry-run
"""

from pathlib import Path
import argparse
import subprocess
import sys
import time


# Define stage order, descriptions and corresponding Python scripts.
STAGES = {
    1: (
        "Occupancy reconstruction",
        "mma3001_stage1_occupancy_audit.py",
    ),
    2: (
        "Occupancy forecasting",
        "mma3001_stage2_occupancy_forecasting.py",
    ),
    3: (
        "Anomaly detection",
        "mma3001_stage3_anomaly_detection.py",
    ),
    4: (
        "Space underutilisation",
        "mma3001_stage4_underutilisation.py",
    ),
    5: (
        "Final validation and synthesis",
        "mma3001_stage5_final_validation.py",
    ),
}


def find_script(filename, root):
    """
    Locate a stage script in the project directory.

    Parameters
    ----------
    filename : str
        Name of the Python stage script.
    root : pathlib.Path
        Project directory containing main.py.

    Returns
    -------
    pathlib.Path or None
        Resolved script path, or None if not found.
    """
    # Prefer src/ when scripts exist in both locations.
    for folder in (root / "src", root):
        path = folder / filename
        if path.is_file():
            return path.resolve()
    return None


def run_stage(number, script, root):
    """
    Execute one pipeline stage as a separate Python process.

    Parameters
    ----------
    number : int
        Stage number from 1 to 5.
    script : pathlib.Path
        Path to the stage script.
    root : pathlib.Path
        Project directory used as the working directory.

    Returns
    -------
    int
        Zero on success; non-zero if execution fails.
    """
    name = STAGES[number][0]

    print("\n" + "=" * 60, flush=True)
    print(f"STAGE {number}/5: {name}", flush=True)
    print("=" * 60, flush=True)

    start = time.perf_counter()

    try:
        # Use the current Python interpreter for consistent execution.
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=root,
            check=False,
        )
    except OSError as error:
        print(f"Could not start Stage {number}: {error}")
        return 1

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        # Stop the pipeline to avoid using incomplete stage outputs.
        print(
            f"\nStage {number} FAILED "
            f"(exit code {result.returncode})."
        )
        print("Remaining stages will not run.")
        return result.returncode or 1

    print(
        f"\nStage {number} completed in {elapsed:.1f} seconds.",
        flush=True,
    )
    return 0


def main():
    """
    Select, validate and execute occupancy analysis stages.

    Command-line options allow individual stages, stage ranges,
    script listing and dry-run validation.

    Returns
    -------
    int
        Zero on success, 2 for missing scripts, or the exit
        status of a failed stage.

    Notes
    -----
    Invalid command-line arguments are handled by argparse.
    """
    parser = argparse.ArgumentParser(
        description="MMA3001 occupancy analysis pipeline"
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=range(1, 6),
        help="Run one stage only.",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        choices=range(1, 6),
        default=1,
        help="First stage to run (default: 1).",
    )
    parser.add_argument(
        "--to-stage",
        type=int,
        choices=range(1, 6),
        default=5,
        help="Last stage to run (default: 5).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List stages and locate scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check scripts without executing them.",
    )

    args = parser.parse_args()

    # Resolve single-stage or inclusive stage-range selection.
    if args.stage is not None:
        if args.from_stage != 1 or args.to_stage != 5:
            parser.error(
                "--stage cannot be combined with "
                "--from-stage or --to-stage."
            )
        selected = [args.stage]
    else:
        if args.from_stage > args.to_stage:
            parser.error(
                "--from-stage must be <= --to-stage."
            )
        selected = list(
            range(args.from_stage, args.to_stage + 1)
        )

    root = Path(__file__).resolve().parent

    print("\nMMA3001 OCCUPANCY ANALYSIS")
    print(f"Project folder: {root}")

    if args.list:
        selected = list(STAGES)

    # Check that every selected stage script is available.
    scripts = {}
    missing = []

    for number in selected:
        name, filename = STAGES[number]
        script = find_script(filename, root)
        scripts[number] = script

        status = str(script) if script else "NOT FOUND"
        print(f"\nStage {number}: {name}")
        print(f"  {status}")

        if script is None:
            missing.append(filename)

    if args.list:
        return 0

    if missing:
        print("\nERROR: Missing required scripts:")
        for filename in missing:
            print(f"  - {filename}")
        print(
            "\nPlace them beside main.py "
            "or inside the src folder."
        )
        return 2

    if args.dry_run:
        print("\nAll selected scripts found.")
        print("Dry run complete; no analysis executed.")
        return 0

    overall_start = time.perf_counter()

    # Execute stages sequentially and stop on the first failure.
    for number in selected:
        status = run_stage(
            number,
            scripts[number],
            root,
        )
        if status != 0:
            return status

    elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Stages executed: {selected}")
    print(f"Total runtime: {elapsed:.1f} seconds")
    print(
        "Results saved in the existing stage output folders."
    )

    return 0


if __name__ == "__main__":
    # Return the pipeline status to the operating system.
    sys.exit(main())
