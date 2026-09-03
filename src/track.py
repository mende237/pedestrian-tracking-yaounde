"""
track.py - Pedestrian Detection & Tracking
Uses YOLOv8n with ByteTrack to detect and track pedestrians in a video.
Saves an annotated output video and raw tracking results for trajectory extraction.
"""

import argparse
import pickle
from pathlib import Path

from ultralytics import YOLO


def run_tracking(
    source: str,
    output_dir: str = "results",
    model_size: str = "yolov8n.pt",
    conf: float = 0.3,
    iou: float = 0.5,
    save_raw: bool = True,
) -> list:
    """
    Run YOLOv8 + ByteTrack on a video source and return raw results.

    Args:
        source:     Path to video file (e.g. 'data/yaounde_video.mp4').
        output_dir: Directory where annotated video is saved.
        model_size: YOLO checkpoint (yolov8n/s/m/l/x).
        conf:       Detection confidence threshold.
        iou:        NMS IoU threshold.
        save_raw:   Whether to pickle raw results for offline reuse.

    Returns:
        List of Ultralytics Results objects, one per frame.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = YOLO(model_size)  # downloads automatically on first run

    results = model.track(
        source=source,
        classes=[0],                # class 0 = person
        tracker="bytetrack.yaml",
        conf=conf,
        iou=iou,
        save=True,
        project=output_dir,
        name="tracking_run",
        exist_ok=True,
        show=False,
        stream=True,               # memory-efficient for long videos
    )

    # Materialise the generator so we can reuse the results list
    results_list = list(results)

    if save_raw:
        raw_path = Path(output_dir) / "raw_results.pkl"
        with open(raw_path, "wb") as f:
            pickle.dump(results_list, f)
        print(f"[track] Raw results saved to {raw_path}")

    print(f"[track] Processed {len(results_list)} frames.")
    return results_list


def main():
    parser = argparse.ArgumentParser(description="Pedestrian tracking with YOLOv8 + ByteTrack")
    parser.add_argument("source", help="Path to input video file")
    parser.add_argument("--output-dir", default="results", help="Output directory (default: results/)")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO checkpoint (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--iou",  type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    run_tracking(
        source=args.source,
        output_dir=args.output_dir,
        model_size=args.model,
        conf=args.conf,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
