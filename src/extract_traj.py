"""
extract_traj.py - Trajectory Post-Processing
The CSV of raw (id, frame, x, y) is now written directly by track.py.
This script adds the speed_mps column and produces a clean output file.
"""

import argparse
from pathlib import Path

import pandas as pd


def compute_speeds(df: pd.DataFrame, fps: float = 30.0, px_per_m: float = 50.0) -> pd.DataFrame:
    """
    Add a speed_mps column (metres per second) to the trajectory DataFrame.

    Args:
        df:        Trajectory DataFrame with columns [id, frame, x, y].
        fps:       Video frame rate (default: 30 for Samsung A56 FHD).
        px_per_m:  Pixels per metre conversion factor (calibrate per video).

    Returns:
        DataFrame with additional speed_mps column.
    """
    df = df.copy().sort_values(["id", "frame"])
    df["dx"] = df.groupby("id")["x"].diff()
    df["dy"] = df.groupby("id")["y"].diff()
    df["dt"] = df.groupby("id")["frame"].diff() / fps
    df["speed_mps"] = (df["dx"] ** 2 + df["dy"] ** 2) ** 0.5 / df["dt"] / px_per_m
    df.drop(columns=["dx", "dy", "dt"], inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Post-process raw trajectory CSV: add speed column"
    )
    parser.add_argument("--input",     default="data/trajectories_yaounde.csv",
                        help="Raw CSV written by track.py (default: data/trajectories_yaounde.csv)")
    parser.add_argument("--output",    default="data/trajectories_yaounde.csv",
                        help="Output CSV path (can be the same file, default: overwrites input)")
    parser.add_argument("--fps",       type=float, default=30.0,
                        help="Video frame rate (default: 30 for Samsung A56 FHD)")
    parser.add_argument("--px-per-m",  type=float, default=50.0,
                        help="Pixels per metre for speed conversion (calibrate per video)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Trajectory CSV not found at {in_path}.\n"
            "Run src/track.py first to generate it."
        )

    print(f"[extract] Loading {in_path} ...")
    df = pd.read_csv(in_path)

    n_ped    = df["id"].nunique()
    n_frames = df["frame"].nunique()
    print(f"[extract] {n_ped} unique pedestrians across {n_frames} frames.")

    df = compute_speeds(df, fps=args.fps, px_per_m=args.px_per_m)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[extract] Saved with speed column -> {out_path}")


if __name__ == "__main__":
    main()
