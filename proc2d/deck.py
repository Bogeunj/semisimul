"""YAML deck parser and execution entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .domain.state import SimulationState
from .errors import DeckError
from .pipeline.steps.anneal_step import arrhenius_diffusivity
from .services import build_default_simulation_service


def _as_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeckError(f"{context} must be a mapping.")
    return value


def _arrhenius_D(D0_cm2_s: float, Ea_eV: float, T_C: float) -> float:
    """Backward-compatible Arrhenius helper export."""
    return arrhenius_diffusivity(D0_cm2_s=D0_cm2_s, Ea_eV=Ea_eV, T_C=T_C)


def load_deck(deck_path: str | Path) -> dict[str, Any]:
    """Load YAML deck from file."""
    path = Path(deck_path)
    if not path.exists():
        raise DeckError(f"Deck file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise DeckError(f"Failed to parse YAML deck: {path}") from exc

    if payload is None:
        raise DeckError(f"Deck is empty: {path}")
    return _as_mapping(payload, "deck")


def run_simulation_payload(
    deck: dict[str, Any],
    *,
    deck_path: str | Path | None = None,
    out_override: str | Path | None = None,
) -> SimulationState:
    """Run simulation from an in-memory deck payload."""
    service = build_default_simulation_service()
    return service.run_payload(deck, deck_path=deck_path, out_override=out_override)


def run_deck(deck_path: str | Path, out_override: str | Path | None = None) -> SimulationState:
    """Run all process steps from a YAML deck."""
    path = Path(deck_path).resolve()
    deck = load_deck(path)
    return run_simulation_payload(deck, deck_path=path, out_override=out_override)


__all__ = ["DeckError", "SimulationState", "_arrhenius_D", "load_deck", "run_deck", "run_simulation_payload"]
