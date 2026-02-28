"""Simulation service built on top of a step registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from ..pipeline import StepRegistry, StepRunner, create_step_registry

StateT = TypeVar("StateT")


class UnsupportedStepTypeError(KeyError):
    """Raised when no handler is registered for the requested step type."""


@dataclass(slots=True)
class SimulationService:
    """Execute simulation steps through the configured registry."""

    registry: StepRegistry

    def run_step(self, *, state: StateT, step_type: str, step: dict[str, Any], idx: int) -> None:
        runner = self.registry.resolve(step_type)
        if runner is None:
            raise UnsupportedStepTypeError(step_type)
        runner(state, step, idx)


def build_simulation_service(handlers: dict[str, StepRunner[StateT]]) -> SimulationService:
    """Create a SimulationService from step handlers."""
    return SimulationService(registry=create_step_registry(handlers))
