"""Common parsing helpers for pipeline step modules."""

from __future__ import annotations

from typing import Any, Mapping

from ...config.validators import as_mapping as _as_mapping
from ...config.validators import opt_mapping as _opt_mapping
from ...config.validators import required as _required
from ...config.validators import to_float as _to_float
from ...errors import DeckError


def as_mapping(value: Any, context: str) -> dict[str, Any]:
    """Require mapping and raise DeckError on failure."""
    try:
        return _as_mapping(value, context)
    except ValueError as exc:
        raise DeckError(str(exc)) from exc


def opt_mapping(value: Any, context: str) -> dict[str, Any]:
    """Optional mapping parser that raises DeckError on failure."""
    try:
        return _opt_mapping(value, context)
    except ValueError as exc:
        raise DeckError(str(exc)) from exc


def required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    """Require key and raise DeckError on failure."""
    try:
        return _required(mapping, key, context)
    except ValueError as exc:
        raise DeckError(str(exc)) from exc


def to_float(value: Any, key: str, context: str) -> float:
    """Convert to float and raise DeckError on failure."""
    try:
        return _to_float(value, key, context)
    except ValueError as exc:
        raise DeckError(str(exc)) from exc
