"""
agent.py —> Traffic Sign & Traffic Light Inference Pipeline
-------------------------------------
Minimal agent for testing the validator pipeline end-to-end.

Setup phase: downloads a tiny file from HuggingFace to verify
             network access and /cache/ write access work.

Infer phase: writes hardcoded predictions to output.csv —
             no model loading, no GPU needed.

Keep all your agent logic between START and END.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2

# Constants / configuration

OUTPUT_DIR = Path("/data/output")
OUTPUT_FILE = OUTPUT_DIR / "output.csv"
CACHE_DIR = Path("/cache")

VIDEO_ROOT = Path(os.environ.get("VIDEO_ROOT", "/data/input"))

# Process every Nth frame (1 = every frame, 2 = every other frame, …)
FRAME_STEP: int = 1

# YOLOv8 confidence threshold
CONF_THRESHOLD: float = 0.25

# YOLOv8 model variant
MODEL_FILENAME: str = "yolov8s.pt"


# COCO class mapping

COCO_VEHICLE_IDS: frozenset[int] = frozenset(
    {
        2,  # car
        5,  # bus
        7,  # truck
    }
)

COCO_TRAFFIC_LIGHT: int = 9
COCO_STOP_SIGN: int = 11  # only road-sign class in COCO

# Map every relevant COCO id → output category string
CATEGORY_MAP: dict[int, str] = {
    COCO_TRAFFIC_LIGHT: "traffic_light",
    COCO_STOP_SIGN: "traffic_sign",
    **{cid: "vehicle" for cid in COCO_VEHICLE_IDS},
}

CSV_FIELDNAMES = ["video_id", "frame_idx", "category", "n_items", "box_2d"]


# LuminarAgent
class LuminarAgent:
    """
    End-to-end inference agent for traffic object detection.

    Usage
    -----
    agent = LuminarAgent()
    agent.setup()   # download weights once
    agent.infer()   # run inference, write output.csv
    """

    def __init__(
        self,
        model_filename: str = MODEL_FILENAME,
        conf_threshold: float = CONF_THRESHOLD,
        frame_step: int = FRAME_STEP,
        video_root: Path = VIDEO_ROOT,
        output_file: Path = OUTPUT_FILE,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self.model_filename = model_filename
        self.conf_threshold = conf_threshold
        self.frame_step = frame_step
        self.video_root = video_root
        self.output_file = output_file
        self.cache_dir = cache_dir
        self._model = None  # loaded lazily in infer()

    # Internal helpers

    def _model_path(self) -> Path:
        return self.cache_dir / self.model_filename

    def _get_video_paths(self) -> list[Path]:
        """Return sorted list of video.mp4 paths under video_root."""
        paths = sorted(self.video_root.glob("*/video.mp4"))
        if not paths:
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                paths = sorted(self.video_root.glob(f"*/{ext}"))
                if paths:
                    break
        return paths

    @staticmethod
    def _video_id_from_path(video_path: Path) -> str:
        """Extract video_id from parent directory name, e.g. video_0003."""
        return video_path.parent.name

    def _run_inference_on_video(
        self,
        video_path: Path,
        video_id: str,
    ) -> list[dict]:
        """
        Run YOLOv8 on a single video file.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [WARN] Cannot open {video_path}", file=sys.stderr)
            return []

        rows: list[dict] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_step == 0:
                results = self._model(
                    frame,
                    verbose=False,
                    conf=self.conf_threshold,
                )

                # Collect boxes grouped by output category
                boxes_by_cat: dict[str, list[list[float]]] = defaultdict(list)

                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        category = CATEGORY_MAP.get(cls_id)
                        if category is None:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        boxes_by_cat[category].append(
                            [
                                round(x1, 6),
                                round(y1, 6),
                                round(x2, 6),
                                round(y2, 6),
                            ]
                        )

                for category, boxes in boxes_by_cat.items():
                    rows.append(
                        {
                            "video_id": video_id,
                            "frame_idx": frame_idx,
                            "category": category,
                            "n_items": len(boxes),
                            "box_2d": json.dumps(boxes),
                        }
                    )

            frame_idx += 1

        cap.release()
        return rows

    # Public API

    def setup(self) -> None:
        """
        Download YOLOv8 weights to cache_dir.
        Safe to call multiple times; skips download if already present.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(self.cache_dir)

        dest = self._model_path()
        if dest.exists():
            print(f"[setup] Model already cached at {dest}. Skipping download.")
            return

        print(f"[setup] Downloading {self.model_filename} to {dest} …")
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

    def infer(self) -> None:
        """
        Load model, run inference on all videos, write output.csv.
        """
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(self.cache_dir)

        model_path = self._model_path()
        if not model_path.exists():
            print(
                f"[infer] ERROR: model not found at {model_path}. Run --setup first.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[infer] Loading model from {model_path} …")
        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self._model.to("cpu")

        video_paths = self._get_video_paths()
        if not video_paths:
            print(
                f"[infer] No videos found under {self.video_root}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"[infer] Found {len(video_paths)} video(s). "
            f"FRAME_STEP={self.frame_step}  CONF={self.conf_threshold}"
        )

        all_rows: list[dict] = []

        for i, vp in enumerate(video_paths, 1):
            vid_id = self._video_id_from_path(vp)
            print(f"[infer] [{i}/{len(video_paths)}] Processing {vid_id} …")
            rows = self._run_inference_on_video(vp, vid_id)
            all_rows.extend(rows)
            print(f"         → {len(rows)} detection rows")

        # Write CSV
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"[infer] Wrote {len(all_rows)} rows → {self.output_file}")
        print(
            f"[infer] To access this file on your host, ensure you mounted "
            f"the output directory:\n"
            f"        docker run … -v /host/output/dir:{self.output_file.parent} …"
        )


# CLI entry-point
def main() -> None:
    agent = LuminarAgent()

    if "--setup" in sys.argv:
        agent.setup()
    elif "--infer" in sys.argv:
        agent.infer()
    else:
        print("Usage: python agent.py --setup | --infer")
        sys.exit(1)


if __name__ == "__main__":
    main()
