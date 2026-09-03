"""
make_gif.py - Create animated GIF from tracked video frames
Extracts sampled frames from the annotated tracking video and stitches
them into a demo GIF for the GitHub README.
"""

import argparse
from pathlib import Path

import cv2


def video_to_gif(video_path: str, output: str = "results/tracking_demo.gif",
                 max_frames: int = 60, fps: int = 8, scale: float = 0.5) -> None:
    """
    Convert a video file to an animated GIF by sampling frames uniformly.

    Args:
        video_path: Path to annotated tracking video (.mp4 / .avi).
        output:     Destination GIF file.
        max_frames: Maximum number of frames to include in the GIF.
        fps:        Frame rate of the output GIF.
        scale:      Scale factor applied to each frame (0 < scale <= 1).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // max_frames)

    frames = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        # cv2 uses BGR; convert to RGB for saving
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        if len(frames) >= max_frames:
            break

    cap.release()

    if not frames:
        print("[make_gif] No frames extracted.")
        return

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            output,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=int(1000 / fps),
            optimize=True,
        )
        print(f"[make_gif] GIF saved to {output}  ({len(frames)} frames @ {fps} fps)")
    except ImportError:
        print("[make_gif] Pillow not installed. Run: pip install Pillow")


def main():
    parser = argparse.ArgumentParser(description="Convert tracking video to animated GIF")
    parser.add_argument("video", help="Path to annotated tracking video")
    parser.add_argument("--output",     default="results/tracking_demo.gif")
    parser.add_argument("--max-frames", type=int,   default=60)
    parser.add_argument("--fps",        type=int,   default=8)
    parser.add_argument("--scale",      type=float, default=0.5)
    args = parser.parse_args()

    video_to_gif(args.video, args.output, args.max_frames, args.fps, args.scale)


if __name__ == "__main__":
    main()
