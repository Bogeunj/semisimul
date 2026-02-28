"""Step runner interfaces used by pipeline engine/registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .context import SimulationState

StepRunner = Callable[[SimulationState, dict[str, Any], int], None]

__all__ = ["StepRunner"]
