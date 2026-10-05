from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedResponse:
    count: int
    action_description: str
    reasoning: str


def parse_response(response: str) -> ParsedResponse | None:
    try:
        payload = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return None

    required_fields = {"count", "action_description", "reasoning"}
    if not isinstance(payload, dict) or set(payload) != required_fields:
        return None

    count = payload["count"]
    action_description = payload["action_description"]
    reasoning = payload["reasoning"]
    if type(count) is not int or count < 0:
        return None
    if not isinstance(action_description, str) or not action_description.strip():
        return None
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None

    return ParsedResponse(
        count=count,
        action_description=action_description.strip(),
        reasoning=reasoning.strip(),
    )


def parse_count(response: str) -> int | None:
    parsed = parse_response(response)
    return parsed.count if parsed is not None else None