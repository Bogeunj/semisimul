"""Unit tests for config validators."""

from __future__ import annotations

import pytest

from proc2d.config.validators import as_mapping, required, to_float, to_int


pytestmark = pytest.mark.unit


def test_as_mapping_and_required() -> None:
    payload = as_mapping({"a": 1}, "ctx")
    assert required(payload, "a", "ctx") == 1


def test_required_raises() -> None:
    with pytest.raises(ValueError, match="Missing required key"):
        required({}, "missing", "ctx")


def test_numeric_converters_raise_contextual_error() -> None:
    with pytest.raises(ValueError, match="ctx.x must be a number"):
        to_float("abc", "x", "ctx")
    with pytest.raises(ValueError, match="ctx.y must be an integer"):
        to_int("abc", "y", "ctx")
