"""Simulation service built on top of pipeline primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import parse_domain_config, parse_step_configs, parse_steps
from ..domain import build_initial_state
from ..domain.state import SimulationState
from ..errors import DeckError
from ..pipeline.engine import run as run_pipeline
from ..pipeline.registry import StepRegistry, create_step_registry
from ..pipeline.step_base import StepRunner
from ..pipeline.steps import build_default_step_handlers


class UnsupportedStepTypeError(KeyError):
    """Backward-compatible alias for unsupported step type errors."""


@dataclass(slots=True)
class SimulationService:
    """Execute simulation steps through the configured registry."""

    registry: StepRegistry

    def run_step(self, *, state: SimulationState, step_type: str, step: dict[str, Any], idx: int) -> None:
        runner = self.registry.resolve(step_type)
        if runner is None:
            raise UnsupportedStepTypeError(step_type)
        runner(state, step, idx)

    def run_steps(self, *, state: SimulationState, steps: list[dict[str, Any]]) -> SimulationState:
        """Run prepared step payload list against an initialized state."""
        return run_pipeline(steps, state, self.registry)

    def run_payload(
        self,
        deck: dict[str, Any],
        *,
        deck_path: str | Path | None = None,
        out_override: str | Path | None = None,
    ) -> SimulationState:
        """Run simulation from in-memory deck payload."""
        path = Path("__in_memory_deck__.yaml") if deck_path is None else Path(deck_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        if not isinstance(deck, dict):
            raise DeckError("deck must be a mapping.")

        try:
            parse_step_configs(deck)
            steps = parse_steps(deck)
            domain = parse_domain_config(deck)
        except (TypeError, ValueError) as exc:
            raise DeckError(str(exc)) from exc

        try:
            state = build_initial_state(domain=domain, deck_path=path, steps=steps, out_override=out_override)
        except (TypeError, ValueError) as exc:
            raise DeckError(str(exc)) from exc
        return self.run_steps(state=state, steps=steps)


def build_simulation_service(handlers: dict[str, StepRunner]) -> SimulationService:
    """Create SimulationService from explicit step handlers."""
    return SimulationService(registry=create_step_registry(handlers))


def build_default_simulation_service() -> SimulationService:
    """Create SimulationService using default built-in step handlers."""
    return build_simulation_service(build_default_step_handlers())
