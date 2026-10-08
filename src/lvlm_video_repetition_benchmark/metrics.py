from __future__ import annotations

from math import sqrt
import math
from typing import Any, Iterable


def temporal_iou(
    gt_start: float,
    gt_end: float,
    pred_start: float,
    pred_end: float,
) -> float:
    """Compute temporal IoU for two half-open intervals in seconds."""
    values = (gt_start, gt_end, pred_start, pred_end)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Interval boundaries must be finite")
    if gt_start < 0 or pred_start < 0 or gt_end <= gt_start or pred_end <= pred_start:
        raise ValueError("Intervals must be non-negative and have positive duration")

    intersection = max(0.0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = (gt_end - gt_start) + (pred_end - pred_start) - intersection
    return intersection / union


def compute_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    """Compute count metrics for rows with valid ground truth and predictions."""
    valid_rows = [
        row
        for row in rows
        if row.get("gt_count") is not None and row.get("pred_count") is not None
    ]
    empty_metrics: dict[str, float | int | None] = {
            "n": 0,
            "mae_n": 0,
            "mae_percent": None,
            "nmae": None,
            "legacy_mae_count": None,
            "rmse_count": None,
            "obz_percent": None,
            "obo_percent": None,
            "n_localization_predictions": 0,
            "mean_temporal_iou": None,
            "tiou_at_0_3_percent": None,
            "tiou_at_0_5_percent": None,
            "tiou_at_0_75_percent": None,
            "start_mae_sec": None,
            "end_mae_sec": None,
            "boundary_mae_sec": None,
    }
    if not valid_rows:
        return empty_metrics

    errors = [float(row["pred_count"]) - float(row["gt_count"]) for row in valid_rows]
    absolute_errors = [abs(error) for error in errors]
    relative_errors = [
        abs(float(row["pred_count"]) - float(row["gt_count"])) / float(row["gt_count"])
        for row in valid_rows
        if float(row["gt_count"]) > 0
    ]
    count = len(valid_rows)
    mean_gt_count = sum(float(row["gt_count"]) for row in valid_rows) / count
    normalized_mae = (
        sum(absolute_errors) / count / mean_gt_count
        if mean_gt_count > 0
        else None
    )

    metrics: dict[str, float | int | None] = {
        "n": count,
        "mae_n": len(relative_errors),
        "mae_percent": 100.0 * sum(relative_errors) / len(relative_errors)
        if relative_errors
        else None,
        "nmae": normalized_mae,
        "legacy_mae_count": sum(absolute_errors) / count,
        "rmse_count": sqrt(sum(error * error for error in errors) / count),
        "obz_percent": 100.0 * sum(error == 0 for error in absolute_errors) / count,
        "obo_percent": 100.0 * sum(error <= 1 for error in absolute_errors) / count,
    }

    localization_rows = [
        row
        for row in valid_rows
        if all(
            row.get(key) is not None
            for key in (
                "gt_sequence_start_sec",
                "gt_sequence_end_sec",
                "pred_sequence_start_sec",
                "pred_sequence_end_sec",
            )
        )
    ]
    if not localization_rows:
        return {**empty_metrics, **metrics}

    ious = [
        temporal_iou(
            float(row["gt_sequence_start_sec"]),
            float(row["gt_sequence_end_sec"]),
            float(row["pred_sequence_start_sec"]),
            float(row["pred_sequence_end_sec"]),
        )
        for row in localization_rows
    ]
    start_errors = [
        abs(float(row["pred_sequence_start_sec"]) - float(row["gt_sequence_start_sec"]))
        for row in localization_rows
    ]
    end_errors = [
        abs(float(row["pred_sequence_end_sec"]) - float(row["gt_sequence_end_sec"]))
        for row in localization_rows
    ]
    loc_count = len(localization_rows)
    metrics.update(
        {
            "n_localization_predictions": loc_count,
            "mean_temporal_iou": sum(ious) / loc_count,
            "tiou_at_0_3_percent": 100.0 * sum(iou >= 0.3 for iou in ious) / loc_count,
            "tiou_at_0_5_percent": 100.0 * sum(iou >= 0.5 for iou in ious) / loc_count,
            "tiou_at_0_75_percent": 100.0 * sum(iou >= 0.75 for iou in ious) / loc_count,
            "start_mae_sec": sum(start_errors) / loc_count,
            "end_mae_sec": sum(end_errors) / loc_count,
            "boundary_mae_sec": sum(start_errors + end_errors) / (2 * loc_count),
        }
    )
    return {**empty_metrics, **metrics}