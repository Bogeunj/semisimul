"""Step registry for pipeline execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

StateT = TypeVar("StateT")
StepRunner = Callable[[StateT, dict[str, Any], int], None]


class StepRegistry(Generic[StateT]):
    """Registry that maps normalized step types to handlers."""

    def __init__(self, handlers: dict[str, StepRunner[StateT]] | None = None) -> None:
        self._handlers: dict[str, StepRunner[StateT]] = {}
        if handlers:
            for step_type, runner in handlers.items():
                self.register(step_type, runner)

    @staticmethod
    def _normalize(step_type: str) -> str:
        return str(step_type).lower()

    def register(self, step_type: str, runner: StepRunner[StateT]) -> None:
        if not callable(runner):
            raise TypeError(f"Runner for '{step_type}' must be callable.")
        self._handlers[self._normalize(step_type)] = runner

    def resolve(self, step_type: str) -> StepRunner[StateT] | None:
        return self._handlers.get(self._normalize(step_type))

    def supported_types(self) -> tuple[str, ...]:
        return tuple(self._handlers)


def create_step_registry(handlers: dict[str, StepRunner[StateT]]) -> StepRegistry[StateT]:
    """Build a registry from a step-type -> handler mapping."""
    return StepRegistry(handlers=handlers)
