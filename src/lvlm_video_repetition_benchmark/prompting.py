from __future__ import annotations

import math
from string import Formatter
from typing import Iterable, Sequence


class PromptTemplateError(ValueError):
    pass


class PromptContextError(ValueError):
    pass


def compose_prompt(
    task_prompt: str,
    localization_instruction: str | None = None,
    response_fields: Iterable[str] | None = None,
    window_intervals: Sequence[tuple[float, float]] | None = None,
    reasoning_mode: str = "none",
) -> str:
    if reasoning_mode not in {"none", "step_by_step"}:
        raise ValueError("reasoning_mode must be 'none' or 'step_by_step'")
    sections = [task_prompt.strip()]
    field_descriptions = {
        "count": '"count" (a non-negative integer)',
        "action_description": '"action_description" (a short description of the repeated action)',
        "reasoning": '"reasoning" (one concise sentence grounded in visible motion)',
    }
    if reasoning_mode == "step_by_step":
        field_descriptions["reasoning"] = (
            '"reasoning" (a concise, structured summary of the target action, '
            "complete-cycle criterion, and count verification)"
        )
    if window_intervals is not None:
        field_descriptions["windows"] = (
            '"windows" (an array of non-negative integer counts, one per listed window '
            "in the same order)"
        )
    selected_fields = tuple(response_fields) if response_fields else tuple(field_descriptions)
    if reasoning_mode == "step_by_step" and "reasoning" not in selected_fields:
        selected_fields += ("reasoning",)
    if window_intervals is not None and "windows" not in selected_fields:
        selected_fields += ("windows",)
    if len(set(selected_fields)) != len(selected_fields):
        raise ValueError("Prompt response_fields must not contain duplicates")
    if "count" not in selected_fields or set(selected_fields) - field_descriptions.keys():
        raise ValueError("Prompt response_fields must include count and use supported fields")
    if ("windows" in selected_fields) != (window_intervals is not None):
        raise ValueError("Prompt windows field requires temporal window intervals")
    fields = [field_descriptions[field] for field in selected_fields]
    if localization_instruction:
        sections.append(localization_instruction.strip())
        fields.extend(
            [
                '"sequence_start_fraction" (a number from 0.0 to 1.0)',
                '"sequence_end_fraction" (a number from 0.0 to 1.0, greater than the start fraction)',
            ]
        )
    if reasoning_mode == "step_by_step":
        sections.append(
            "Before answering, count the video using these steps:\n"
            "1. Identify the target repeated action from the video and the prompt.\n"
            "2. Define one complete repetition using observable motion.\n"
            "3. Review the sampled video chronologically, counting each complete cycle "
            "once; exclude partial cycles and do not extrapolate unseen motion.\n"
            "4. Recount the sequence as a consistency check.\n"
            'In "reasoning", give a concise, evidence-based summary of the action, '
            "cycle criterion, and count check."
        )
    if window_intervals is not None:
        if not window_intervals:
            raise ValueError("Temporal windows must not be empty")
        window_lines = []
        previous_end = 0.0
        for index, (start, end) in enumerate(window_intervals, start=1):
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < previous_end
                or end <= start
            ):
                raise ValueError("Temporal windows must be finite, ordered, and non-empty")
            window_lines.append(f"{index}. [{start:g}, {end:g}) seconds")
            previous_end = end
        sections.append(
            "Also count complete repetitions separately in these non-overlapping windows:\n"
            + "\n".join(window_lines)
            + "\nAssign each repetition to the window in which it completes. Estimate the "
            "whole-video total and the window counts, then check that they agree. If they "
            "do not, review the video and revise the counts before answering."
        )
    sections.append(
        "Return exactly one JSON object with exactly these fields: "
        + ", ".join(fields)
        + ". Do not include markdown or any text outside the JSON object."
    )
    return "\n\n".join(sections)


def build_temporal_windows(
    duration_seconds: float,
    window_seconds: float,
) -> tuple[tuple[float, float], ...]:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Video duration must be finite and greater than zero")
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Window size must be finite and greater than zero")
    window_count = math.ceil(duration_seconds / window_seconds)
    return tuple(
        (
            index * window_seconds,
            min((index + 1) * window_seconds, duration_seconds),
        )
        for index in range(window_count)
    )


def render_prompt(template: str, annotation: object, context_fields: Iterable[str] = ()) -> str:
    configured_fields = tuple(context_fields)
    if len(set(configured_fields)) != len(configured_fields):
        raise ValueError("Prompt context_fields must not contain duplicates")

    template_fields = tuple(
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    )
    if set(template_fields) != set(configured_fields):
        raise PromptTemplateError(
            "Prompt placeholders must exactly match configured context_fields; "
            f"placeholders={sorted(set(template_fields))}, "
            f"context_fields={sorted(configured_fields)}"
        )

    values: dict[str, str] = {}
    for field_name in configured_fields:
        value = getattr(annotation, field_name, None)
        if not isinstance(value, str) or not value.strip() or value.strip().casefold() == "unknown":
            raise PromptContextError(
                f"Required prompt context field {field_name!r} is missing or invalid"
            )
        values[field_name] = value.strip()

    return template.format_map(values)