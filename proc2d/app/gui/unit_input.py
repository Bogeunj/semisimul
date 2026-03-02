"""Helpers for parsing GUI numeric inputs with optional units."""

from __future__ import annotations

import re


_NUMERIC_WITH_OPTIONAL_UNIT = re.compile(
    r"^\s*"
    r"([+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*"
    r"(?:\[\s*([^\]]+)\s*\]|([^\s\[\]]+))?"
    r"\s*$"
)


def parse_number_with_optional_unit(
    raw: str, *, expected_unit: str, field_name: str
) -> float:
    """Parse numeric text, optionally allowing the expected unit suffix."""
    text = str(raw).strip()
    match = _NUMERIC_WITH_OPTIONAL_UNIT.match(text)
    if match is None:
        raise ValueError(f"Invalid numeric value for {field_name}: {raw!r}")

    number_text = match.group(1).replace(",", "")
    try:
        value = float(number_text)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for {field_name}: {raw!r}") from exc

    parsed_unit = (match.group(2) or match.group(3) or "").strip()
    if parsed_unit and parsed_unit != expected_unit:
        raise ValueError(
            f"{field_name} must use unit [{expected_unit}] when unit is provided."
        )

    return value
