from __future__ import annotations

from string import Formatter
from typing import Iterable


class PromptTemplateError(ValueError):
    pass


class PromptContextError(ValueError):
    pass


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