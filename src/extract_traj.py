"""
extract_traj.py - Trajectory Extraction
Reads raw YOLO tracking results (or re-runs tracking) and exports
per-pedestrian trajectories to a CSV (id, frame, x, y).
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd


def extract_trajectories(results: list) -> pd.DataFrame:
    """
    Extract (id, frame, x, y) from a list of Ultralytics Results.

    x, y are the bounding-box centre coordinates in pixels.

    Args:
        results: List of Ultralytics Results objects (one per frame).

    Returns:
        DataFrame with columns [id, frame, x, y].
    """
    records = []
    for frame_idx, result in enumerate(results):
        if result.boxes is None:
            continue
        for box in result.boxes:
            if box.id is None:
                continue
            person_id = int(box.id)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            records.append({"id": person_id, "frame": frame_idx, "x": cx, "y": cy})

    df = pd.DataFrame(records, columns=["id", "frame", "x", "y"])
    df.sort_values(["id", "frame"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_speeds(df: pd.DataFrame, fps: float = 30.0, px_per_m: float = 50.0) -> pd.DataFrame:
    """
    Add a 'speed_mps' column (metres per second) to the trajectory DataFrame.

    Args:
        df:        Trajectory DataFrame (id, frame, x, y).
        fps:       Video frame rate.
        px_per_m:  Pixel-to-metre conversion factor (calibrate per video).

    Returns:
        DataFrame with additional 'speed_mps' column.
    """
    df = df.copy().sort_values(["id", "frame"])
    df["dx"] = df.groupby("id")["x"].diff()
    df["dy"] = df.groupby("id")["y"].diff()
    df["dt"] = df.groupby("id")["frame"].diff() / fps          # seconds
    df["speed_mps"] = (df["dx"]**2 + df["dy"]**2)**0.5 / df["dt"] / px_per_m
    df.drop(columns=["dx", "dy", "dt"], inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract pedestrian trajectories from YOLO tracking results")
    parser.add_argument(
        "--raw-results", default="results/raw_results.pkl",
        help="Path to pickled raw results from track.py"
    )
    parser.add_argument(
        "--output", default="data/trajectories_yaounde.csv",
        help="Output CSV path"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate (default: 30 for Samsung A56 FHD)")
    parser.add_argument("--px-per-m", type=float, default=50.0, help="Pixels per metre (for speed calc)")
    args = parser.parse_args()

    raw_path = Path(args.raw_results)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw results not found at {raw_path}. "
            "Run src/track.py first, or point --raw-results to the correct path."
        )

    print(f"[extract] Loading raw results from {raw_path} ...")
    with open(raw_path, "rb") as f:
        results = pickle.load(f)

    df = extract_trajectories(results)
    df = compute_speeds(df, fps=args.fps, px_per_m=args.px_per_m)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_pedestrians = df["id"].nunique()
    n_frames      = df["frame"].nunique()
    print(f"[extract] {n_pedestrians} unique pedestrians tracked across {n_frames} frames.")
    print(f"[extract] Trajectories saved to {out_path}")


if __name__ == "__main__":
    main()
