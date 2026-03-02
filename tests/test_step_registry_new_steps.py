"""Registry coverage tests for newly added step handlers."""

from __future__ import annotations

import pytest

from proc2d.errors import DeckError
from proc2d.pipeline.engine import run
from proc2d.pipeline.registry import StepRegistry
from proc2d.pipeline.steps import build_default_step_handlers


pytestmark = pytest.mark.unit


def test_default_step_handlers_include_new_step_types() -> None:
    handlers = build_default_step_handlers()
    assert "deposition" in handlers
    assert "etch" in handlers
    assert "electrical" in handlers


def test_unknown_step_error_lists_new_step_types() -> None:
    with pytest.raises(DeckError) as excinfo:
        run([{"type": "unknown"}], object(), StepRegistry())

    msg = str(excinfo.value)
    assert "deposition" in msg
    assert "etch" in msg
    assert "electrical" in msg
