"""
Scores traffic benchmark submissions.
Video-based detection with categories: video_id, frame_idx, category, n_items, box_2d
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


def _to_non_negative_int(value: object, default: int = 0) -> int:
    """Convert value to non-negative int; fallback to default on invalid values."""
    try:
        val = int(float(value))
        return val if val >= 0 else default
    except (TypeError, ValueError):
        return default


def _build_groups(
    df: pd.DataFrame,
) -> tuple[
    dict[tuple[str, int, str], _GroupRecord],
    int,
    int,
]:
    """
    Build per-(video_id, frame_idx, category) groups.

    Duplicate rows are merged into a single key for box matching, but the
    duplicate count is tracked and penalized separately.
    """
    groups: dict[tuple[str, int, str], _GroupRecord] = {}
    duplicate_rows = 0
    declared_mismatch_rows = 0

    for row in df.itertuples(index=False):
        vid = str(getattr(row, "video_id"))
        fidx = _to_non_negative_int(getattr(row, "frame_idx"), default=0)
        cat = str(getattr(row, "category"))
        boxes = list(getattr(row, "boxes"))
        n_items = _to_non_negative_int(getattr(row, "n_items"), default=len(boxes))

        if n_items != len(boxes):
            declared_mismatch_rows += 1

        key = (vid, fidx, cat)
        if key in groups:
            duplicate_rows += 1
            groups[key].boxes.extend(boxes)
            groups[key].n_items += n_items
        else:
            groups[key] = _GroupRecord(boxes=boxes, n_items=n_items)

    return groups, duplicate_rows, declared_mismatch_rows


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

    required = ["video_id", "frame_idx", "category", "n_items", "box_2d"]
    if not all(c in gt_df.columns for c in required):
        log.warning("GT missing required columns: %s", list(gt_df.columns))
        return ScoreResult(0.0, 0.0, 0.0, 0.0, 0, 0, 0)
    if not all(c in out_df.columns for c in required):
        log.warning("Output missing required columns: %s", list(out_df.columns))
        return ScoreResult(0.0, 0.0, 0.0, 0.0, len(gt_df), 0, 0)

    # Parse boxes
    gt_df["boxes"] = gt_df["box_2d"].apply(_parse_boxes)
    out_df["boxes"] = out_df["box_2d"].apply(_parse_boxes)
    gt_df["n_items"] = gt_df["n_items"].apply(_to_non_negative_int)
    out_df["n_items"] = out_df["n_items"].apply(_to_non_negative_int)

    # Build strict per-key groups and track malformed/duplicate rows.
    gt_groups, gt_duplicate_rows, gt_declared_mismatch_rows = _build_groups(gt_df)
    out_groups, out_duplicate_rows, out_declared_mismatch_rows = _build_groups(out_df)

    if gt_duplicate_rows > 0:
        log.warning("GT contains %d duplicate rows for the same key.", gt_duplicate_rows)
    if gt_declared_mismatch_rows > 0:
        log.warning(
            "GT has %d rows where n_items does not match parsed box count.",
            gt_declared_mismatch_rows,
        )

    # Compute strict diagnostics and box-level metrics.
    gt_keys = set(gt_groups.keys())
    out_keys = set(out_groups.keys())
    overlap_keys = gt_keys & out_keys
    missing_keys = gt_keys - out_keys
    extra_keys = set(out_groups.keys()) - gt_keys

    total_gt_boxes = sum(len(record.boxes) for record in gt_groups.values())
    total_tp = 0
    total_pred_boxes = 0

    for key in gt_keys:
        gt_b = gt_groups[key].boxes
        pred_b = out_groups[key].boxes if key in out_groups else []
        tp = _match_boxes(gt_b, pred_b)
        total_tp += tp
        total_pred_boxes += len(pred_b)

    for key in extra_keys:
        total_pred_boxes += len(out_groups[key].boxes)

    if total_gt_boxes == 0:
        box_f1 = 1.0 if total_pred_boxes == 0 else 0.0
        prec = 1.0 if total_pred_boxes == 0 else 0.0
        rec = 1.0
    else:
        rec = total_tp / total_gt_boxes
        prec = total_tp / total_pred_boxes if total_pred_boxes > 0 else 0.0
        box_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    expected_rows = len(gt_keys)
    output_rows_unique = len(out_keys)
    output_rows_total = output_rows_unique + out_duplicate_rows
    overlap_rows = len(overlap_keys)

    row_rec = overlap_rows / expected_rows if expected_rows > 0 else 1.0
    if output_rows_total == 0:
        row_prec = 1.0 if expected_rows == 0 else 0.0
    else:
        row_prec = overlap_rows / output_rows_total
    row_f1 = 2 * row_prec * row_rec / (row_prec + row_rec) if (row_prec + row_rec) > 0 else 0.0

    gt_item_total = sum(record.n_items for record in gt_groups.values())
    pred_item_total = sum(record.n_items for record in out_groups.values())
    matched_item_total = sum(
        min(gt_groups[key].n_items, out_groups[key].n_items) for key in overlap_keys
    )

    if gt_item_total == 0:
        item_f1 = 1.0 if pred_item_total == 0 else 0.0
    else:
        item_rec = matched_item_total / gt_item_total
        item_prec = matched_item_total / pred_item_total if pred_item_total > 0 else 0.0
        item_f1 = (
            2 * item_prec * item_rec / (item_prec + item_rec)
            if (item_prec + item_rec) > 0
            else 0.0
        )

    exact_count_rows = 0
    for key in gt_keys:
        if key in out_groups and out_groups[key].n_items == gt_groups[key].n_items:
            exact_count_rows += 1
    count_mismatch_rows = expected_rows - exact_count_rows
    exact_count_ratio = exact_count_rows / expected_rows if expected_rows > 0 else 1.0

    declared_mismatch_ratio = (
        out_declared_mismatch_rows / output_rows_total if output_rows_total > 0 else 0.0
    )
    declared_consistency = 1.0 - declared_mismatch_ratio

    # Strict score combines localization quality with row and count correctness.
    count_component = 0.5 * (item_f1 + exact_count_ratio)
    final = box_f1 * row_f1 * count_component * declared_consistency

    unexpected_rows = len(extra_keys)
    missing_rows = len(missing_keys)
    extra_rows = unexpected_rows + out_duplicate_rows

    result = ScoreResult(
        final_score=round(final, 6),
        f1=round(final, 6),
        precision=round(prec, 6),
        recall=round(rec, 6),
        n_total=total_gt_boxes,
        n_predicted=total_pred_boxes,
        n_matched=total_tp,
        box_f1=round(box_f1, 6),
        row_f1=round(row_f1, 6),
        item_f1=round(item_f1, 6),
        exact_count_ratio=round(exact_count_ratio, 6),
        n_expected_rows=expected_rows,
        n_output_rows=output_rows_total,
        n_missing_rows=missing_rows,
        n_unexpected_rows=unexpected_rows,
        n_duplicate_rows=out_duplicate_rows,
        n_extra_rows=extra_rows,
        n_count_mismatch_rows=count_mismatch_rows,
        n_declared_mismatch_rows=out_declared_mismatch_rows,
    )

    log.info(
        "Scoring complete -- strict=%.4f box_f1=%.4f row_f1=%.4f item_f1=%.4f "
        "exact_count=%.4f matched=%d/%d extra_boxes=%d missing_rows=%d "
        "unexpected_rows=%d duplicate_rows=%d count_mismatch_rows=%d "
        "declared_mismatch_rows=%d",
        result.final_score,
        result.box_f1,
        result.row_f1,
        result.item_f1,
        result.exact_count_ratio,
        result.n_matched,
        result.n_total,
        result.n_predicted - result.n_matched,
        result.n_missing_rows,
        result.n_unexpected_rows,
        result.n_duplicate_rows,
        result.n_count_mismatch_rows,
        result.n_declared_mismatch_rows,
    )
    return result
