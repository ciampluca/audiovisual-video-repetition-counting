from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedResponse:
    count: int
    action_description: str
    reasoning: str
    sequence_start_fraction: float | None = None
    sequence_end_fraction: float | None = None
    localization_error: str | None = None


def parse_response(
    response: str,
    require_localization: bool = False,
) -> ParsedResponse | None:
    try:
        payload = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return None

    required_fields = {"count", "action_description", "reasoning"}
    localization_fields = {"sequence_start_fraction", "sequence_end_fraction"}
    if not isinstance(payload, dict):
        return None
    fields = set(payload)
    if not required_fields.issubset(fields) or fields - required_fields - localization_fields:
        return None
    has_localization = localization_fields.issubset(fields)
    has_partial_localization = bool(fields.intersection(localization_fields)) and not has_localization

    count = payload["count"]
    action_description = payload["action_description"]
    reasoning = payload["reasoning"]
    if type(count) is not int or count < 0:
        return None
    if not isinstance(action_description, str) or not action_description.strip():
        return None
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None

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
        sequence_start_fraction=sequence_start_fraction,
        sequence_end_fraction=sequence_end_fraction,
        localization_error=localization_error,
    )


def parse_count(response: str) -> int | None:
    parsed = parse_response(response)
    return parsed.count if parsed is not None else None