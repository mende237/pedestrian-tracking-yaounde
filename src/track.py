import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def run_tracking(
    source: str,
    output_dir: str = "results",
    csv_path: str = "data/trajectories_yaounde.csv",
    model_size: str = "yolov8n.pt",
    conf: float = 0.3,
    iou: float = 0.5,
    imgsz: int = 640,
) -> Path:
    """
    Run YOLOv8 + ByteTrack on a video source.

    Processes frames one-by-one in streaming mode to avoid OOM errors on
    long FHD videos. Writes (id, frame, x, y) rows to a CSV in real time.

    Args:
        source:     Path to video file (e.g. 'data/Record_1.mp4').
        output_dir: Directory where the annotated video is saved.
        csv_path:   Destination CSV for trajectory data.
        model_size: YOLO checkpoint  (yolov8n / s / m / l / x).
        conf:       Detection confidence threshold.
        iou:        NMS IoU threshold.
        imgsz:      Inference image size (pixels). 640 is the YOLO default;
                    lower (e.g. 480) reduces VRAM/RAM usage further.

    Returns:
        Path to the written CSV file.
    """
    # Resolve to an absolute path so YOLO does NOT prepend its own
    # global runs_dir (which would give runs/results/traking instead of results/traking).
    abs_project = str(Path(output_dir).resolve())
    Path(abs_project).mkdir(parents=True, exist_ok=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_size)  # downloads automatically on first run

    # stream=True is essential: yields one Result object at a time
    # and discards it immediately after processing -> O(1) memory per frame.
    results_gen = model.track(
        source=source,
        classes=[0],             # 0 = person
        tracker="bytetrack.yaml",
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=True,               # saves annotated video to abs_project/traking
        project=abs_project,
        name="traking",
        exist_ok=True,
        show=False,
        stream=True,             # KEY: do NOT call list() on this
    )

    frame_idx = 0
    total_detections = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "frame", "x", "y"])  # header

        for result in results_gen:
            if result.boxes is not None:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    person_id = int(box.id)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    writer.writerow([person_id, frame_idx, round(cx, 2), round(cy, 2)])
                    total_detections += 1

            frame_idx += 1
            if frame_idx % 300 == 0:   # progress every ~10 s at 30 fps
                print(f"[track] Frame {frame_idx} | detections so far: {total_detections}")

    print(f"[track] Done. {frame_idx} frames processed, {total_detections} detections.")
    print(f"[track] Trajectories saved -> {csv_path}")
    return Path(csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Pedestrian tracking with YOLOv8 + ByteTrack (memory-efficient streaming)"
    )
    parser.add_argument("source",          help="Path to input video file")
    parser.add_argument("--output-dir",    default="results",
                        help="Directory for annotated video (default: results/)")
    parser.add_argument("--csv",           default="data/trajectories_yaounde.csv",
                        help="Output CSV path (default: data/trajectories_yaounde.csv)")
    parser.add_argument("--model",         default="yolov8n.pt",
                        help="YOLO checkpoint (default: yolov8n.pt)")
    parser.add_argument("--conf",          type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--iou",           type=float, default=0.5,
                        help="NMS IoU threshold (default: 0.5)")
    parser.add_argument("--imgsz",         type=int,   default=640,
                        help="Inference image size in pixels (default: 640). "
                             "Use 480 or 320 to reduce RAM usage further.")
    args = parser.parse_args()

    run_tracking(
        source=args.source,
        output_dir=args.output_dir,
        csv_path=args.csv,
        model_size=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
