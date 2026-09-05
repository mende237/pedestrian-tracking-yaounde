"""
visualize.py - Trajectory Visualization
Generates:
  - Heatmap of pedestrian density
  - Speed distribution histogram (Yaounde)
  - Individual trajectory overlays on a blank canvas
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame, frame_w: int = 1920, frame_h: int = 1080,
                 output: str = "results/heatmap.png") -> None:
    """Plot a 2-D kernel density heatmap of pedestrian positions."""
    fig, ax = plt.subplots(figsize=(12, 7))

    h = ax.hist2d(
        df["x"], df["y"],
        bins=[frame_w // 10, frame_h // 10],
        range=[[0, frame_w], [0, frame_h]],
        cmap="hot",
    )
    fig.colorbar(h[3], ax=ax, label="Density (frame counts)")
    ax.set_title("Pedestrian Density Heatmap — Yaoundé", fontsize=14)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.invert_yaxis()   # video coords: y=0 at top

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Heatmap saved to {output}")


# ---------------------------------------------------------------------------
# Trajectory overlays
# ---------------------------------------------------------------------------

def plot_trajectories(df: pd.DataFrame, frame_w: int = 1920, frame_h: int = 1080,
                      max_ids: int = 50, output: str = "results/trajectories.png") -> None:
    """Plot individual pedestrian trajectories on a blank canvas."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, frame_w)
    ax.set_ylim(frame_h, 0)   # y-axis flipped to match image coords
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    cmap = plt.get_cmap("tab20")
    unique_ids = df["id"].unique()[:max_ids]

    for i, pid in enumerate(unique_ids):
        traj = df[df["id"] == pid].sort_values("frame")
        color = cmap(i % 20)
        ax.plot(traj["x"], traj["y"], linewidth=0.8, alpha=0.7, color=color)
        # Mark start
        ax.scatter(traj["x"].iloc[0], traj["y"].iloc[0], s=10, color=color, zorder=5)

    ax.set_title("Pedestrian Trajectories — Yaoundé", fontsize=14, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualize] Trajectory plot saved to {output}")


# ---------------------------------------------------------------------------
# Speed distribution
# ---------------------------------------------------------------------------

def plot_speed_distribution(df: pd.DataFrame, output: str = "results/speed_distribution.png") -> None:
    """Histogram of individual pedestrian mean speeds."""
    mean_speeds = df.groupby("id")["speed_mps"].mean().dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mean_speeds, bins=30, color="#e07b39", edgecolor="white", linewidth=0.4)
    ax.set_title("Mean Pedestrian Speed Distribution — Yaoundé", fontsize=13)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Number of pedestrians")
    ax.axvline(mean_speeds.median(), color="red", linestyle="--", label=f"Median = {mean_speeds.median():.2f} m/s")
    ax.legend()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Speed distribution saved to {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize pedestrian trajectories and density")
    parser.add_argument("--csv", default="data/trajectories_yaounde.csv", help="Trajectories CSV")
    parser.add_argument("--frame-w", type=int, default=1920, help="Video frame width in pixels")
    parser.add_argument("--frame-h", type=int, default=1080, help="Video frame height in pixels")
    parser.add_argument("--output-dir", default="results", help="Output directory for plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    
    # Filter only pedestrians if the obj_class column exists
    if "obj_class" in df.columns:
        df = df[df["obj_class"] == 0]

    out = args.output_dir

    plot_heatmap(df, args.frame_w, args.frame_h, output=f"{out}/heatmap.png")
    plot_trajectories(df, args.frame_w, args.frame_h, output=f"{out}/trajectories.png")

    if "speed_mps" in df.columns:
        plot_speed_distribution(df, output=f"{out}/speed_distribution.png")
    else:
        print("[visualize] No speed_mps column found — skipping speed distribution plot.")


if __name__ == "__main__":
    main()
