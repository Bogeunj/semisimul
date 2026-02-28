"""Pipeline execution engine."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..errors import DeckError
from .context import SimulationState
from .registry import StepRegistry


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise DeckError(f"Missing required key '{key}' in {context}.")
    return mapping[key]


def run(steps: Iterable[Mapping[str, Any]], context: SimulationState, registry: StepRegistry) -> SimulationState:
    """Run deck steps sequentially against the given context."""
    for idx, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            raise DeckError(f"steps[{idx}] must be a mapping.")

        step = raw_step
        stype = str(_required(step, "type", f"steps[{idx}]"))
        stype_l = stype.lower()

        runner = registry.resolve(stype_l)
        if runner is None:
            supported = ", ".join(registry.supported_types())
            raise DeckError(
                f"steps[{idx}].type '{stype_l}' is not supported. "
                f"Use one of: {supported}."
            )

        try:
            runner(context, step, idx)
        except DeckError:
            raise
        except Exception as exc:
            raise DeckError(f"Step {idx} ('{stype_l}') failed: {exc}") from exc

    return context
