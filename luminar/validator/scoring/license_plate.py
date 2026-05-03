"""
Scores license-plate benchmark submissions.
Image-based detection with OCR: image_id, objects (containing bbox and plate text)
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import pandas as pd

from luminar.common.logging import get_logger
from luminar.validator.scoring import ScoreResult

log = get_logger(__name__)


@dataclass
class _GroupRecord:
    boxes: list[tuple[float, float, float, float]]
    n_items: int
    plates: list[str] | None = None


def _parse_license_plate_objects(obj_str: str) -> tuple[list[tuple[float, float, float, float]], list[str]]:
    """Parse license-plate objects field which contains {'bbox': [...], 'plate': [...]}"""
    if pd.isna(obj_str) or not str(obj_str).strip() or str(obj_str).strip() in ("[]", ""):
        return [], []
    try:
        # Handle both JSON format and Python dict format
        obj_str = str(obj_str).strip()
        if obj_str.startswith("'") or obj_str.startswith("{"):
            # Try to evaluate as Python dict
            obj = eval(obj_str)  # noqa: S307
        else:
            obj = json.loads(obj_str)
        
        if not isinstance(obj, dict):
            log.warning("Objects field is not a dict: %s", obj_str[:200])
            return [], []
        
        bboxes = obj.get("bbox", [])
        plates = obj.get("plate", [])
        
        # Convert bbox [x, y, w, h] to [x1, y1, x2, y2]
        converted_boxes = []
        for bbox in bboxes:
            if len(bbox) >= 4:
                x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                x1, y1, x2, y2 = x, y, x + w, y + h
                converted_boxes.append((x1, y1, x2, y2))
        
        # Ensure plates are strings
        plates = [str(p) for p in plates] if plates else []
        
        return converted_boxes, plates
    except Exception as exc:
        log.warning("Failed to parse objects field: %s", obj_str[:200])
        return [], []


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


def _match_license_plates(
    gt_plates: list[str],
    pred_plates: list[str],
) -> int:
    """Count exact matches between ground truth and predicted plates."""
    if not gt_plates or not pred_plates:
        return 0
    
    matches = 0
    gt_matched = [False] * len(gt_plates)
    pred_matched = [False] * len(pred_plates)
    
    # Greedy matching: prioritize exact matches
    for i, gp in enumerate(gt_plates):
        for j, pp in enumerate(pred_plates):
            if not gt_matched[i] and not pred_matched[j] and gp == pp:
                gt_matched[i] = True
                pred_matched[j] = True
                matches += 1
                break
    
    return matches


def score_license_plate_output(
    output_csv_bytes: bytes,
    ground_truth_bytes: bytes,
) -> ScoreResult:
    """Score license plate detection and OCR (plate text recognition)."""
    try:
        gt_df = pd.read_csv(io.BytesIO(ground_truth_bytes))
        out_df = pd.read_csv(io.BytesIO(output_csv_bytes))
    except Exception as exc:
        log.warning("CSV parse error: %s", exc)
        return ScoreResult(0.0, 0.0, 0.0, 0.0, 0, 0, 0)

    # Normalise columns
    gt_df.columns = gt_df.columns.str.strip().str.lower()
    out_df.columns = out_df.columns.str.strip().str.lower()

    required = ["image_id", "objects"]
    if not all(c in gt_df.columns for c in required):
        log.warning("GT missing required columns: %s", list(gt_df.columns))
        return ScoreResult(0.0, 0.0, 0.0, 0.0, 0, 0, 0)
    if not all(c in out_df.columns for c in required):
        log.warning("Output missing required columns: %s", list(out_df.columns))
        return ScoreResult(0.0, 0.0, 0.0, 0.0, len(gt_df), 0, 0)

    # Parse objects (bbox + plate)
    gt_df["boxes"], gt_df["plates"] = zip(
        *gt_df["objects"].apply(_parse_license_plate_objects)
    )
    out_df["boxes"], out_df["plates"] = zip(
        *out_df["objects"].apply(_parse_license_plate_objects)
    )

    # Build per-image groups
    gt_groups = {}
    out_groups = {}
    
    for row in gt_df.itertuples(index=False):
        img_id = str(getattr(row, "image_id"))
        boxes = list(getattr(row, "boxes"))
        plates = list(getattr(row, "plates"))
        gt_groups[img_id] = _GroupRecord(boxes=boxes, n_items=len(boxes), plates=plates)
    
    for row in out_df.itertuples(index=False):
        img_id = str(getattr(row, "image_id"))
        boxes = list(getattr(row, "boxes"))
        plates = list(getattr(row, "plates"))
        out_groups[img_id] = _GroupRecord(boxes=boxes, n_items=len(boxes), plates=plates)

    # Compute metrics
    gt_keys = set(gt_groups.keys())
    out_keys = set(out_groups.keys())
    overlap_keys = gt_keys & out_keys
    missing_keys = gt_keys - out_keys
    extra_keys = out_keys - gt_keys

    total_gt_boxes = sum(record.n_items for record in gt_groups.values())
    total_tp = 0
    total_plate_matches = 0
    total_pred_boxes = 0

    # Match boxes and plates
    for img_id in gt_keys:
        gt_b = gt_groups[img_id].boxes
        gt_p = gt_groups[img_id].plates or []
        
        pred_b = out_groups[img_id].boxes if img_id in out_groups else []
        pred_p = out_groups[img_id].plates or [] if img_id in out_groups else []
        
        # Box IoU matching
        tp = _match_boxes(gt_b, pred_b)
        total_tp += tp
        total_pred_boxes += len(pred_b)
        
        # Plate text matching
        plate_matches = _match_license_plates(gt_p, pred_p)
        total_plate_matches += plate_matches

    for img_id in extra_keys:
        total_pred_boxes += len(out_groups[img_id].boxes)

    # Compute F1 for boxes
    if total_gt_boxes == 0:
        box_f1 = 1.0 if total_pred_boxes == 0 else 0.0
        box_prec = 1.0 if total_pred_boxes == 0 else 0.0
        box_rec = 1.0
    else:
        box_rec = total_tp / total_gt_boxes
        box_prec = total_tp / total_pred_boxes if total_pred_boxes > 0 else 0.0
        box_f1 = 2 * box_prec * box_rec / (box_prec + box_rec) if (box_prec + box_rec) > 0 else 0.0

    # Compute F1 for plate text
    total_gt_plates = sum(
        len(record.plates) if record.plates else 0
        for record in gt_groups.values()
    )
    
    if total_gt_plates == 0:
        plate_f1 = 1.0 if total_plate_matches == 0 else 0.0
        plate_prec = 1.0 if total_plate_matches == 0 else 0.0
        plate_rec = 1.0
    else:
        plate_rec = total_plate_matches / total_gt_plates
        plate_prec = (
            total_plate_matches / sum(
                len(record.plates) if record.plates else 0
                for record in out_groups.values()
            )
            if sum(len(record.plates) if record.plates else 0 for record in out_groups.values()) > 0
            else 0.0
        )
        plate_f1 = 2 * plate_prec * plate_rec / (plate_prec + plate_rec) if (plate_prec + plate_rec) > 0 else 0.0

    # Compute row metrics
    expected_rows = len(gt_keys)
    output_rows_unique = len(out_keys)
    overlap_rows = len(overlap_keys)

    row_rec = overlap_rows / expected_rows if expected_rows > 0 else 1.0
    row_prec = overlap_rows / output_rows_unique if output_rows_unique > 0 else (1.0 if expected_rows == 0 else 0.0)
    row_f1 = 2 * row_prec * row_rec / (row_prec + row_rec) if (row_prec + row_rec) > 0 else 0.0

    # Combined score: average box detection and plate recognition
    final = 0.5 * box_f1 + 0.5 * plate_f1

    unexpected_rows = len(extra_keys)
    missing_rows = len(missing_keys)

    result = ScoreResult(
        final_score=round(final, 6),
        f1=round(final, 6),
        precision=round(box_prec, 6),
        recall=round(box_rec, 6),
        n_total=total_gt_boxes,
        n_predicted=total_pred_boxes,
        n_matched=total_tp,
        box_f1=round(box_f1, 6),
        row_f1=round(row_f1, 6),
        item_f1=round(plate_f1, 6),
        exact_count_ratio=round(plate_f1, 6),
        n_expected_rows=expected_rows,
        n_output_rows=output_rows_unique,
        n_missing_rows=missing_rows,
        n_unexpected_rows=unexpected_rows,
        n_duplicate_rows=0,
        n_extra_rows=unexpected_rows,
        n_count_mismatch_rows=0,
        n_declared_mismatch_rows=0,
    )

    log.info(
        "License-plate scoring complete -- final=%.4f box_f1=%.4f plate_f1=%.4f row_f1=%.4f "
        "box_matched=%d/%d plate_matched=%d/%d missing_images=%d unexpected_images=%d",
        result.final_score,
        result.box_f1,
        result.item_f1,
        result.row_f1,
        result.n_matched,
        result.n_total,
        total_plate_matches,
        total_gt_plates,
        result.n_missing_rows,
        result.n_unexpected_rows,
    )
    return result
