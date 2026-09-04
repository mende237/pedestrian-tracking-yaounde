"""
run_all.py - Full pipeline orchestrator for all recordings.

Runs tracking + speed extraction on every video, then merges trajectories
by acquisition group:

    Group        | Files                        | Vantage point
    -------------|------------------------------|------------------
    ground       | Record_1, Record_2           | Street level
    balcony      | Record 3-6                   | 3rd-floor balcony

Outputs
-------
    data/trajectories/record_<N>.csv      per-recording trajectories
    data/trajectories/all_ground.csv      merged ground recordings
    data/trajectories/all_balcony.csv     merged balcony recordings

Usage
-----
    # Full pipeline (track + extract + merge)
    python src/run_all.py

    # Skip tracking if CSVs already exist (re-merge / re-extract only)
    python src/run_all.py --skip-tracking

    # Only process balcony videos (recommended for trajectory analysis)
    python src/run_all.py --groups balcony

    # Custom px-per-metre conversion (calibrate per camera setup)
    python src/run_all.py --px-per-m-ground 45 --px-per-m-balcony 30
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Recording registry
# ---------------------------------------------------------------------------

RECORDINGS = [
    # name         video file            group
    ("record_1",  "data/Record_1.mp4",  "ground"),
    ("record_2",  "data/Record_2.mp4",  "ground"),
    ("record_3",  "data/Record 3.mp4",  "balcony"),
    ("record_4",  "data/Record 4.mp4",  "balcony"),
    ("record_5",  "data/Record 5.mp4",  "balcony"),
    ("record_6",  "data/Record 6.mp4",  "balcony"),
]

TRAJ_DIR = Path("data/trajectories")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_track(name, video, csv_out, model, conf, iou, imgsz):
    """Run YOLOv8 + ByteTrack on a single video. Returns True on success."""
    from src.track import run_tracking

    video_path = Path(video)
    if not video_path.exists():
        print(f"[run_all] WARNING  Video not found, skipping: {video_path}")
        return False

    print(f"\n{'='*60}")
    print(f"[run_all] Tracking: {video_path.name}  ->  {csv_out}")
    print(f"{'='*60}")

    run_tracking(
        source=str(video_path),
        output_dir=f"results/tracking/{name}",
        csv_path=str(csv_out),
        model_size=model,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
    )
    return True


def step_extract(csv_path, fps, px_per_m):
    """Add speed column to a trajectory CSV (in-place)."""
    from src.extract_traj import compute_speeds

    if not csv_path.exists():
        print(f"[run_all] WARNING  CSV not found, skipping extract: {csv_path}")
        return

    print(f"[run_all] Computing speeds: {csv_path.name}  "
          f"(fps={fps}, px_per_m={px_per_m})")

    df = pd.read_csv(csv_path)
    df = compute_speeds(df, fps=fps, px_per_m=px_per_m)
    df.to_csv(csv_path, index=False)
    print(f"[run_all] Saved with speed column -> {csv_path}")


def step_merge(group, names):
    """Concatenate per-recording CSVs for a group into a single file."""
    frames = []
    for name in names:
        csv_path = TRAJ_DIR / f"{name}.csv"
        if not csv_path.exists():
            print(f"[run_all] WARNING  Missing CSV for merge: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["recording"] = name           # keep track of source
        frames.append(df)

    if not frames:
        print(f"[run_all] WARNING  No data to merge for group '{group}'.")
        return None

    merged = pd.concat(frames, ignore_index=True)
    out = TRAJ_DIR / f"all_{group}.csv"
    merged.to_csv(out, index=False)
    n_ped = merged["id"].nunique()
    print(f"[run_all] Merged {len(frames)} recording(s) -> {out}  "
          f"({len(merged)} rows, {n_ped} unique IDs)")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full tracking pipeline over all recordings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--groups", nargs="+", choices=["ground", "balcony"], default=["ground", "balcony"],
        help="Which groups to process.",
    )
    parser.add_argument("--skip-tracking", action="store_true",
                        help="Skip tracking step (use existing CSVs).")
    parser.add_argument("--skip-extract",  action="store_true",
                        help="Skip speed extraction step.")
    parser.add_argument("--skip-merge",    action="store_true",
                        help="Skip merging step.")

    # YOLO parameters
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--conf",   type=float, default=0.3)
    parser.add_argument("--iou",    type=float, default=0.5)
    parser.add_argument("--imgsz",  type=int,   default=640)

    # Speed calibration (adjust after ground-truth measurement)
    parser.add_argument("--fps",                type=float, default=30.0,
                        help="Frame rate for all recordings.")
    parser.add_argument("--px-per-m-ground",    type=float, default=50.0,
                        help="Pixels per metre for ground-level recordings.")
    parser.add_argument("--px-per-m-balcony",   type=float, default=30.0,
                        help="Pixels per metre for balcony recordings (wider FOV).")

    args = parser.parse_args()

    TRAJ_DIR.mkdir(parents=True, exist_ok=True)

    px_per_m = {"ground": args.px_per_m_ground, "balcony": args.px_per_m_balcony}

    # Filter recordings to requested groups
    recordings = [(n, v, g) for n, v, g in RECORDINGS if g in args.groups]

    # ------------------------------------------------------------------
    # Step 1 - Tracking
    # ------------------------------------------------------------------
    if not args.skip_tracking:
        for name, video, group in recordings:
            csv_out = TRAJ_DIR / f"{name}.csv"
            step_track(name, video, csv_out,
                       model=args.model, conf=args.conf,
                       iou=args.iou, imgsz=args.imgsz)
    else:
        print("[run_all] Tracking skipped.")

    # ------------------------------------------------------------------
    # Step 2 - Speed extraction
    # ------------------------------------------------------------------
    if not args.skip_extract:
        for name, _, group in recordings:
            csv_path = TRAJ_DIR / f"{name}.csv"
            step_extract(csv_path, fps=args.fps, px_per_m=px_per_m[group])
    else:
        print("[run_all] Speed extraction skipped.")

    # ------------------------------------------------------------------
    # Step 3 - Merge by group
    # ------------------------------------------------------------------
    if not args.skip_merge:
        for group in args.groups:
            names = [n for n, _, g in recordings if g == group]
            step_merge(group, names)
    else:
        print("[run_all] Merge skipped.")

    print("\n[run_all] Pipeline complete.")
    print(f"[run_all] Merged CSVs available in: {TRAJ_DIR.resolve()}")
    print("[run_all] Next step -> python src/compare_eth_ucy.py "
          "--yaounde-csv data/trajectories/all_balcony.csv "
          "--eth-dir data/eth_ucy")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
