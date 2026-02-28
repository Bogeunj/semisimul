"""Mask step runner."""

from __future__ import annotations

from typing import Any

from ...domain.state import SimulationState
from ...errors import DeckError
from ...physics.mask import build_mask_1d, openings_from_any, smooth_mask_1d, validate_mask
from .common import required, to_float


def run_mask_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    """Apply mask generation and smoothing step."""
    context = f"steps[{idx}] (mask)"
    openings_raw = required(step, "openings_um", context)
    if not isinstance(openings_raw, list):
        raise DeckError(f"{context}.openings_um must be a list of [start, end] pairs.")

    sigma_lat_um = to_float(step.get("sigma_lat_um", 0.0), "sigma_lat_um", context)
    if sigma_lat_um < 0.0:
        raise DeckError(f"{context}.sigma_lat_um must be >= 0.")

    openings = openings_from_any(openings_raw)
    mask_raw = build_mask_1d(state.grid.x_um, openings)
    state.mask_eff = smooth_mask_1d(mask_raw, sigma_lat_um=sigma_lat_um, dx_um=state.grid.dx_um)
    validate_mask(state.mask_eff, state.grid.Nx)
