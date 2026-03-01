"""Unit tests for pipeline engine error handling."""

from __future__ import annotations

from typing import Any

import pytest

from proc2d.errors import DeckError
from proc2d.pipeline.engine import run
from proc2d.pipeline.registry import StepRegistry


pytestmark = pytest.mark.unit


def test_run_rejects_unknown_step_type() -> None:
    context = object()
    registry: StepRegistry[object] = StepRegistry()

    with pytest.raises(DeckError, match=r"steps\[0\]\.type 'unknown' is not supported"):
        run([{"type": "unknown"}], context, registry)


def test_run_wraps_non_deck_errors_with_step_context() -> None:
    context = object()

    def _boom_runner(_ctx: object, _step: dict[str, Any], _idx: int) -> None:
        raise RuntimeError("boom")

    registry = StepRegistry({"implant": _boom_runner})

    with pytest.raises(
        DeckError, match=r"Step 0 \('implant'\) failed: boom"
    ) as excinfo:
        run([{"type": "implant"}], context, registry)

    assert isinstance(excinfo.value.__cause__, RuntimeError)
