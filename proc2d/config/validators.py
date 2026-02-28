"""Shared config validation helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def as_mapping(value: Any, context: str) -> dict[str, Any]:
    """Require mapping value."""
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping.")
    return value


def opt_mapping(value: Any, context: str) -> dict[str, Any]:
    """Return mapping or empty mapping for None."""
    if value is None:
        return {}
    return as_mapping(value, context)


def required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    """Require mapping key existence."""
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in {context}.")
    return mapping[key]


def to_float(value: Any, key: str, context: str) -> float:
    """Convert value to float with contextual error message."""
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{context}.{key} must be a number, got {value!r}.") from exc


def to_int(value: Any, key: str, context: str) -> int:
    """Convert value to int with contextual error message."""
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{context}.{key} must be an integer, got {value!r}.") from exc


def ensure_choice(name: str, value: str, allowed: Sequence[str]) -> str:
    """Validate str choice and return normalized value."""
    val = str(value)
    if val not in allowed:
        joined = ", ".join(allowed)
        raise ValueError(f"{name} must be one of: {joined}. Got '{val}'.")
    return val


def ensure_nonnegative(name: str, value: float, *, allow_zero: bool = True) -> float:
    """Validate scalar non-negativity for already-numeric values."""
    x = float(value)
    if allow_zero:
        if x < 0.0:
            raise ValueError(f"{name} must be >= 0.")
    elif x <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return float(value)
