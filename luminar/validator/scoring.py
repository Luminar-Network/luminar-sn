"""
Computes robust object-detection score for the new per-frame CSV format.
GT and output both contain: video_id,frame_idx,category,n_items,box_2d
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import pandas as pd

from luminar.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class ScoreResult:
    final_score: float  # micro F1 (0.0-1.0)
    f1: float  # same as final_score (kept for backward compatibility)
    precision: float
    recall: float
    n_total: int  # total GT boxes across all groups
    n_predicted: int  # total predicted boxes (includes extras)
    n_matched: int  # true-positive boxes (IoU >= 0.5)


def _parse_boxes(box_str: str) -> list[tuple[float, float, float, float]]:
    """Safely parse '[[x1,y1,x2,y2], ...]' string."""
    if pd.isna(box_str) or not str(box_str).strip() or str(box_str).strip() in ("[]", ""):
        return []
    try:
        boxes = json.loads(str(box_str))
        return [tuple(float(x) for x in box) for box in boxes]
    except (json.JSONDecodeError, TypeError, ValueError):
        log.warning("Failed to parse box_2d: %s", box_str[:200])
        return []


def _iou(b1: tuple[float, ...], b2: tuple[float, ...]) -> float:
    """Standard axis-aligned IoU."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _match_boxes(
    gt_boxes: list[tuple[float, ...]],
    pred_boxes: list[tuple[float, ...]],
    iou_thresh: float = 0.5,
) -> int:
    """Greedy best-IoU matching (handles wrong order, partial matches)."""
    if not gt_boxes or not pred_boxes:
        return 0

    matches = []
    for i, gb in enumerate(gt_boxes):
        for j, pb in enumerate(pred_boxes):
            iou_val = _iou(gb, pb)
            if iou_val >= iou_thresh:
                matches.append((iou_val, i, j))

    matches.sort(reverse=True, key=lambda x: x[0])  # best IoU first

    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)
    tp = 0
    for _, gi, pj in matches:
        if not gt_matched[gi] and not pred_matched[pj]:
            gt_matched[gi] = True
            pred_matched[pj] = True
            tp += 1
    return tp


def score_output(
    output_csv_bytes: bytes,
    ground_truth_bytes: bytes,
) -> ScoreResult:
    try:
        gt_df = pd.read_csv(io.BytesIO(ground_truth_bytes))
        out_df = pd.read_csv(io.BytesIO(output_csv_bytes))
    except Exception as exc:
        log.warning("CSV parse error: %s", exc)
        return ScoreResult(0.0, 0.0, 0.0, 0.0, 0, 0, 0)

    # Normalise columns
    gt_df.columns = gt_df.columns.str.strip().str.lower()
    out_df.columns = out_df.columns.str.strip().str.lower()

    required = ["video_id", "frame_idx", "category", "box_2d"]
    if not all(c in gt_df.columns for c in required):
        log.warning("GT missing required columns: %s", list(gt_df.columns))
        return ScoreResult(0.0, 0.0, 0.0, 0.0, 0, 0, 0)
    if not all(c in out_df.columns for c in required):
        log.warning("Output missing required columns: %s", list(out_df.columns))
        return ScoreResult(0.0, 0.0, 0.0, 0.0, len(gt_df), 0, 0)

    # Parse boxes
    gt_df["boxes"] = gt_df["box_2d"].apply(_parse_boxes)
    out_df["boxes"] = out_df["box_2d"].apply(_parse_boxes)

    # Group by (video_id, frame_idx, category)
    gt_groups = {}
    for (vid, fidx, cat), g in gt_df.groupby(["video_id", "frame_idx", "category"]):
        key = (str(vid), int(fidx), str(cat))
        gt_groups[key] = g["boxes"].iloc[0]

    out_groups = {}
    for (vid, fidx, cat), g in out_df.groupby(["video_id", "frame_idx", "category"]):
        key = (str(vid), int(fidx), str(cat))
        out_groups[key] = g["boxes"].iloc[0]

    # Compute micro F1 with extra-group penalty
    gt_keys = set(gt_groups.keys())
    extra_keys = set(out_groups.keys()) - gt_keys

    total_gt_boxes = sum(len(b) for b in gt_groups.values())
    total_tp = 0
    total_pred_boxes = 0

    for key in gt_keys:
        gt_b = gt_groups[key]
        pred_b = out_groups.get(key, [])
        tp = _match_boxes(gt_b, pred_b)
        total_tp += tp
        total_pred_boxes += len(pred_b)

    for key in extra_keys:
        total_pred_boxes += len(out_groups[key])

    if total_gt_boxes == 0:
        final = 1.0 if total_pred_boxes == 0 else 0.0
        prec = 1.0
        rec = 1.0
    else:
        rec = total_tp / total_gt_boxes
        prec = total_tp / total_pred_boxes if total_pred_boxes > 0 else 0.0
        final = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    result = ScoreResult(
        final_score=round(final, 6),
        f1=round(final, 6),
        precision=round(prec, 6),
        recall=round(rec, 6),
        n_total=total_gt_boxes,
        n_predicted=total_pred_boxes,
        n_matched=total_tp,
    )

    log.info(
        "Scoring complete — final=%.4f  f1=%.4f  prec=%.4f  rec=%.4f  "
        "matched=%d/%d  extra_boxes=%d",
        result.final_score,
        result.f1,
        result.precision,
        result.recall,
        result.n_matched,
        result.n_total,
        result.n_predicted - result.n_matched,
    )
    return result
