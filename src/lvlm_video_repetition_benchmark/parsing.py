from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ParsedResponse:
    count: int
    action_description: str
    reasoning: str
    window_counts: tuple[int, ...] | None = None
    window_count_sum: int | None = None
    window_count_consistent: bool | None = None
    sequence_start_fraction: float | None = None
    sequence_end_fraction: float | None = None
    localization_error: str | None = None


def parse_response(
    response: str,
    require_localization: bool = False,
    expected_fields: Iterable[str] | None = None,
    expected_window_count: int | None = None,
) -> ParsedResponse | None:
    try:
        payload = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return None

    required_fields = {"count", "action_description", "reasoning"}
    localization_fields = {"sequence_start_fraction", "sequence_end_fraction"}
    supported_fields = required_fields | localization_fields
    if expected_window_count is not None:
        if expected_window_count < 1:
            return None
        supported_fields.add("windows")
    if not isinstance(payload, dict):
        return None
    fields = set(payload)
    if (
        "count" not in fields
        or fields - supported_fields
        or (expected_window_count is not None and "windows" not in fields)
    ):
        return None
    if expected_fields is not None:
        selected_fields = tuple(expected_fields)
        if (
            len(set(selected_fields)) != len(selected_fields)
            or "count" not in selected_fields
            or set(selected_fields) - (required_fields | ({"windows"} if expected_window_count is not None else set()))
            or (expected_window_count is not None and "windows" not in selected_fields)
        ):
            return None
    has_localization = localization_fields.issubset(fields)
    has_partial_localization = bool(fields.intersection(localization_fields)) and not has_localization

    count = payload["count"]
    action_description = payload.get("action_description", "")
    reasoning = payload.get("reasoning", "")
    if type(count) is not int or count < 0:
        return None
    if not isinstance(action_description, str):
        action_description = ""
    if not isinstance(reasoning, str):
        reasoning = ""

    window_counts = None
    window_count_sum = None
    window_count_consistent = None
    if expected_window_count is not None:
        raw_window_counts = payload["windows"]
        if (
            not isinstance(raw_window_counts, list)
            or len(raw_window_counts) != expected_window_count
            or any(type(window_count) is not int or window_count < 0 for window_count in raw_window_counts)
        ):
            return None
        window_counts = tuple(raw_window_counts)
        window_count_sum = sum(window_counts)
        window_count_consistent = window_count_sum == count

    sequence_start_fraction = None
    sequence_end_fraction = None
    localization_error = None
    if require_localization and not has_localization:
        localization_error = "Missing normalized sequence interval"
    elif has_partial_localization:
        localization_error = "Incomplete normalized sequence interval"
    elif has_localization:
        sequence_start = payload["sequence_start_fraction"]
        sequence_end = payload["sequence_end_fraction"]
        if (
            type(sequence_start) not in (int, float)
            or type(sequence_end) not in (int, float)
        ):
            localization_error = "Normalized sequence interval must contain numbers"
        else:
            sequence_start_fraction = float(sequence_start)
            sequence_end_fraction = float(sequence_end)
            if (
                not math.isfinite(sequence_start_fraction)
                or not math.isfinite(sequence_end_fraction)
                or sequence_start_fraction < 0
                or sequence_end_fraction > 1
                or sequence_start_fraction >= sequence_end_fraction
            ):
                sequence_start_fraction = None
                sequence_end_fraction = None
                localization_error = "Invalid normalized sequence interval"

    return ParsedResponse(
        count=count,
        action_description=action_description.strip(),
        reasoning=reasoning.strip(),
        window_counts=window_counts,
        window_count_sum=window_count_sum,
        window_count_consistent=window_count_consistent,
        sequence_start_fraction=sequence_start_fraction,
        sequence_end_fraction=sequence_end_fraction,
        localization_error=localization_error,
    )


def parse_count(response: str) -> int | None:
    parsed = parse_response(response)
    return parsed.count if parsed is not None else None