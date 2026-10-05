from __future__ import annotations

from math import sqrt
from typing import Any, Iterable


def compute_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    """Compute count metrics for rows with valid ground truth and predictions."""
    valid_rows = [
        row
        for row in rows
        if row.get("gt_count") is not None and row.get("pred_count") is not None
    ]
    if not valid_rows:
        return {
            "n": 0,
            "mae_n": 0,
            "mae_percent": None,
            "legacy_mae_count": None,
            "rmse_count": None,
            "obz_percent": None,
            "obo_percent": None,
        }

    errors = [float(row["pred_count"]) - float(row["gt_count"]) for row in valid_rows]
    absolute_errors = [abs(error) for error in errors]
    relative_errors = [
        abs(float(row["pred_count"]) - float(row["gt_count"])) / float(row["gt_count"])
        for row in valid_rows
        if float(row["gt_count"]) > 0
    ]
    count = len(valid_rows)

    return {
        "n": count,
        "mae_n": len(relative_errors),
        "mae_percent": 100.0 * sum(relative_errors) / len(relative_errors)
        if relative_errors
        else None,
        "legacy_mae_count": sum(absolute_errors) / count,
        "rmse_count": sqrt(sum(error * error for error in errors) / count),
        "obz_percent": 100.0 * sum(error == 0 for error in absolute_errors) / count,
        "obo_percent": 100.0 * sum(error <= 1 for error in absolute_errors) / count,
    }