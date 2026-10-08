from __future__ import annotations

from string import Formatter
from typing import Iterable


class PromptTemplateError(ValueError):
    pass


class PromptContextError(ValueError):
    pass


def compose_prompt(
    task_prompt: str,
    localization_instruction: str | None = None,
    response_fields: Iterable[str] | None = None,
) -> str:
    sections = [task_prompt.strip()]
    field_descriptions = {
        "count": '"count" (a non-negative integer)',
        "action_description": '"action_description" (a short description of the repeated action)',
        "reasoning": '"reasoning" (one concise sentence grounded in visible motion)',
    }
    selected_fields = tuple(response_fields or field_descriptions)
    if len(set(selected_fields)) != len(selected_fields):
        raise ValueError("Prompt response_fields must not contain duplicates")
    if "count" not in selected_fields or set(selected_fields) - field_descriptions.keys():
        raise ValueError("Prompt response_fields must include count and use supported fields")
    fields = [field_descriptions[field] for field in selected_fields]
    if localization_instruction:
        sections.append(localization_instruction.strip())
        fields.extend(
            [
                '"sequence_start_fraction" (a number from 0.0 to 1.0)',
                '"sequence_end_fraction" (a number from 0.0 to 1.0, greater than the start fraction)',
            ]
        )
    sections.append(
        "Return exactly one JSON object with exactly these fields: "
        + ", ".join(fields)
        + ". Do not include markdown or any text outside the JSON object."
    )
    return "\n\n".join(sections)


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