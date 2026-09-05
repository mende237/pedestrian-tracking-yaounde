"""
compare_eth_ucy.py - Comparative Analysis: Yaounde vs ETH/UCY
Loads the standard ETH/UCY annotation format (space-separated .txt files)
and compares speed distributions, trajectory lengths, and path linearity
against the Yaounde dataset.

ETH/UCY format (per line): frame_id  person_id  x  y
(positions are in metres, already world-coordinate)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ETH/UCY loader
# ---------------------------------------------------------------------------

def load_eth_ucy(data_dir: str, fps: float = 2.5) -> pd.DataFrame:
    """
    Load all .txt annotation files in data_dir following ETH/UCY format.

    Each file has space-separated columns: frame_id  person_id  x  y

    Returns a DataFrame with columns [id, frame, x, y, speed_mps, dataset].
    """
    records = []
    data_path = Path(data_dir)
    txt_files = list(data_path.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    for txt_file in txt_files:
        subset_name = txt_file.stem
        raw = pd.read_csv(txt_file, sep=r"\s+", header=None,
                          names=["frame", "id", "x", "y"])
        raw["dataset"] = subset_name
        records.append(raw)

    df = pd.concat(records, ignore_index=True)
    df.sort_values(["dataset", "id", "frame"], inplace=True)

    # Compute speed in m/s (positions already in metres)
    df["dx"] = df.groupby(["dataset", "id"])["x"].diff()
    df["dy"] = df.groupby(["dataset", "id"])["y"].diff()
    df["dt"] = df.groupby(["dataset", "id"])["frame"].diff() / fps
    df["speed_mps"] = (df["dx"]**2 + df["dy"]**2)**0.5 / df["dt"]
    df.drop(columns=["dx", "dy", "dt"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def path_linearity(traj: pd.DataFrame) -> float:
    """
    Linearity = straight-line distance / total path length.
    1.0 = perfectly straight, 0.0 = no net displacement.
    """
    if len(traj) < 2:
        return np.nan
    pts = traj[["x", "y"]].values
    total_len = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    straight  = np.linalg.norm(pts[-1] - pts[0])
    return straight / total_len if total_len > 0 else np.nan


def per_pedestrian_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute per-pedestrian stats: mean speed, path length, linearity."""
    rows = []
    group_cols = ["dataset", "id"] if "dataset" in df.columns else ["id"]
    for keys, grp in df.groupby(group_cols):
        grp = grp.sort_values("frame")
        pts = grp[["x", "y"]].values
        path_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else np.nan
        rows.append({
            "source":      label,
            "mean_speed":  grp["speed_mps"].mean(),
            "path_length": path_len,
            "linearity":   path_linearity(grp),
            "n_frames":    len(grp),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison(yaounde_stats: pd.DataFrame, eth_stats: pd.DataFrame,
                    output_dir: str = "results") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Speed comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Yaounde vs ETH/UCY — Pedestrian Behaviour Comparison", fontsize=14)

    metrics = [
        ("mean_speed",  "Mean Speed (m/s)",        "Speed Distribution"),
        ("path_length", "Path Length (m or px)",   "Path Length Distribution"),
        ("linearity",   "Linearity (0=erratic, 1=straight)", "Trajectory Linearity"),
    ]

    for ax, (col, xlabel, title) in zip(axes, metrics):
        y_vals = yaounde_stats[col].dropna()
        e_vals = eth_stats[col].dropna()

        ax.hist(y_vals, bins=25, alpha=0.7, color="#e07b39", label="Yaounde", density=True)
        ax.hist(e_vals, bins=25, alpha=0.7, color="#3b82f6", label="ETH/UCY", density=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()

    fig.tight_layout()
    out_path = f"{output_dir}/speed_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] Comparison plot saved to {out_path}")

    # --- Summary table ---
    summary = pd.DataFrame({
        "Metric":  ["Mean speed (m/s)", "Median path length", "Median linearity"],
        "Yaounde": [
            f"{yaounde_stats['mean_speed'].mean():.3f}",
            f"{yaounde_stats['path_length'].median():.1f}",
            f"{yaounde_stats['linearity'].median():.3f}",
        ],
        "ETH/UCY": [
            f"{eth_stats['mean_speed'].mean():.3f}",
            f"{eth_stats['path_length'].median():.1f}",
            f"{eth_stats['linearity'].median():.3f}",
        ],
    })
    summary_path = f"{output_dir}/comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[compare] Summary table saved to {summary_path}")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare Yaounde vs ETH/UCY pedestrian trajectories")
    parser.add_argument("--yaounde-csv",  default="data/trajectories_yaounde.csv",
                        help="Yaounde trajectory CSV from extract_traj.py")
    parser.add_argument("--eth-dir",      default="data/eth_ucy",
                        help="Directory containing ETH/UCY .txt annotation files")
    parser.add_argument("--eth-fps",      type=float, default=2.5,
                        help="Frame rate used in ETH/UCY annotations (default: 2.5 for ETH)")
    parser.add_argument("--yaounde-fps",  type=float, default=30.0,
                        help="Frame rate of Yaounde video (default: 30.0 for Samsung A56 FHD)")
    parser.add_argument("--yaounde-px-per-m", type=float, default=50.0,
                        help="Pixels per metre for Yaounde video (for speed conversion)")
    parser.add_argument("--output-dir",   default="results",
                        help="Directory to save comparison plots")
    args = parser.parse_args()

    # --- Load Yaounde ---
    print(f"[compare] Loading Yaounde trajectories from {args.yaounde_csv} ...")
    yaounde_df = pd.read_csv(args.yaounde_csv)

    # Filter only pedestrians if the obj_class column exists
    if "obj_class" in yaounde_df.columns:
        yaounde_df = yaounde_df[yaounde_df["obj_class"] == 0]

    # Convert pixel positions to metres if speed_mps already computed
    if "speed_mps" not in yaounde_df.columns:
        # Recompute speed
        yaounde_df.sort_values(["id", "frame"], inplace=True)
        yaounde_df["dx"] = yaounde_df.groupby("id")["x"].diff()
        yaounde_df["dy"] = yaounde_df.groupby("id")["y"].diff()
        yaounde_df["dt"] = yaounde_df.groupby("id")["frame"].diff() / args.yaounde_fps
        yaounde_df["speed_mps"] = (
            (yaounde_df["dx"]**2 + yaounde_df["dy"]**2)**0.5
            / yaounde_df["dt"]
            / args.yaounde_px_per_m
        )
        yaounde_df.drop(columns=["dx", "dy", "dt"], inplace=True)

    # Convert x,y to metres for linearity/path-length comparability
    yaounde_df["x"] = yaounde_df["x"] / args.yaounde_px_per_m
    yaounde_df["y"] = yaounde_df["y"] / args.yaounde_px_per_m

    yaounde_stats = per_pedestrian_stats(yaounde_df, label="Yaounde")

    # --- Load ETH/UCY ---
    print(f"[compare] Loading ETH/UCY data from {args.eth_dir} ...")
    eth_df    = load_eth_ucy(args.eth_dir, fps=args.eth_fps)
    eth_stats = per_pedestrian_stats(eth_df, label="ETH/UCY")

    # --- Plot ---
    plot_comparison(yaounde_stats, eth_stats, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
