"""
agent.py —> Traffic Sign & Traffic Light Inference Pipeline
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import cv2


############################# START #############################

OUTPUT_DIR  = Path("/data/output")
OUTPUT_FILE = OUTPUT_DIR / "output.csv"
CACHE_DIR   = Path("/cache")


VIDEO_ROOT  = Path(os.environ.get("VIDEO_ROOT", "/data/input"))

FRAME_STEP: int = 1

# YOLOv8 confidence threshold (lower = more detections, more FP)
CONF_THRESHOLD: float = 0.25

# YOLOv8 model variant: "yolov8n.pt" (fastest) … "yolov8x.pt" (most accurate)
MODEL_FILENAME: str = "yolov8s.pt"

# COCO class IDs we care about
COCO_TRAFFIC_LIGHT: int = 9
COCO_STOP_SIGN: int     = 11   # only road-sign class in COCO → traffic_sign

# Map COCO class id → BDD100K category name
CATEGORY_MAP: dict[int, str] = {
    COCO_TRAFFIC_LIGHT: "traffic_light",
    COCO_STOP_SIGN:     "traffic_sign",
}


def _model_path() -> Path:
    return CACHE_DIR / MODEL_FILENAME


def _get_video_paths() -> list[Path]:
    """Return sorted list of video.mp4 paths under VIDEO_ROOT."""
    paths = sorted(VIDEO_ROOT.glob("*/video.mp4"))
    if not paths:
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            paths = sorted(VIDEO_ROOT.glob(f"*/{ext}"))
            if paths:
                break
    return paths


def _video_id_from_path(video_path: Path) -> str:
    """Extract video_id from parent directory name, e.g. video_0003."""
    return video_path.parent.name


def _run_inference_on_video(model, video_path: Path, video_id: str) -> list[dict]:
    """
    Run YOLOv8 inference on a single video.
    """
    from collections import defaultdict

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}", file=sys.stderr)
        return []

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP == 0:
            results = model(frame, verbose=False, conf=CONF_THRESHOLD)

            boxes_by_cat: dict[str, list[list[float]]] = defaultdict(list)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id not in CATEGORY_MAP:
                        continue
                    category = CATEGORY_MAP[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes_by_cat[category].append([
                        round(x1, 6), round(y1, 6),
                        round(x2, 6), round(y2, 6),
                    ])

            for category, boxes in boxes_by_cat.items():
                rows.append({
                    "video_id":  video_id,
                    "frame_idx": frame_idx,
                    "category":  category,
                    "n_items":   len(boxes),
                    "box_2d":    json.dumps(boxes),
                })

        frame_idx += 1

    cap.release()
    return rows


############################# END #############################


def setup() -> None:
    """
    Download YOLOv8s weights to /cache/.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(CACHE_DIR)

    dest = _model_path()
    if dest.exists():
        print(f"[setup] Model already cached at {dest}. Skipping download.")
        return

    print(f"[setup] Downloading {MODEL_FILENAME} to {dest} …")
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        result = attempt_download_asset(str(dest))
        print(f"[setup] Download complete: {result}")

        if not dest.exists():
            raise FileNotFoundError(f"Expected {dest} after download but file not found.")

    except Exception as exc:
        print(f"[setup] ERROR: {exc}", file=sys.stderr)
        raise

    print("[setup] Done.")


def infer() -> None:
    """
    Load model from /cache/, run inference on all videos,
    write output.csv to /data/output/.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["YOLO_CONFIG_DIR"] = str(CACHE_DIR)

    model_path = _model_path()
    if not model_path.exists():
        print(
            f"[infer] ERROR: model not found at {model_path}. "
            "Run --setup first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[infer] Loading model from {model_path} …")
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    # Force CPU inference
    model.to("cpu")

    video_paths = _get_video_paths()
    if not video_paths:
        print(f"[infer] No videos found under {VIDEO_ROOT}", file=sys.stderr)
        sys.exit(1)

    print(f"[infer] Found {len(video_paths)} video(s). FRAME_STEP={FRAME_STEP}")

    all_rows: list[dict] = []

    for i, vp in enumerate(video_paths, 1):
        vid_id = _video_id_from_path(vp)
        print(f"[infer] [{i}/{len(video_paths)}] Processing {vid_id} …")
        rows = _run_inference_on_video(model, vp, vid_id)
        all_rows.extend(rows)
        print(f"         → {len(rows)} detection rows")

    # Write CSV
    fieldnames = ["video_id", "frame_idx", "category", "n_items", "box_2d"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[infer] Wrote {len(all_rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    if "--setup" in sys.argv:
        setup()
    elif "--infer" in sys.argv:
        infer()
    else:
        print("Usage: python agent.py --setup | --infer")
        sys.exit(1)