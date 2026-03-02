"""Deposition step runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...domain.state import SimulationState
from ...errors import DeckError
from ...physics.deposition import apply_deposition
from ...physics.mask import full_open_mask
from ...units import ensure_positive
from .common import required, to_float


def run_deposition_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    """Apply deposition and persist updated physical fields."""
    context = f"steps[{idx}] (deposition)"
    state.ensure_oxide_fields()

    thickness_um = to_float(
        required(step, "thickness_um", context), "thickness_um", context
    )
    ensure_positive(f"{context}.thickness_um", thickness_um, allow_zero=True)

    apply_on = str(step.get("apply_on", "all")).lower()
    mask_weighting = str(step.get("mask_weighting", "binary")).lower()
    open_threshold = to_float(
        step.get("open_threshold", 0.5), "open_threshold", context
    )
    update_materials = bool(step.get("update_materials", True))

    mask_eff = (
        state.mask_eff if state.mask_eff is not None else full_open_mask(state.grid.Nx)
    )
    assert state.tox_um is not None
    try:
        Cn, tox_new, materials_new, _ = apply_deposition(
            state.C,
            state.grid,
            state.tox_um,
            mask_eff,
            thickness_um=thickness_um,
            apply_on=apply_on,
            mask_weighting=mask_weighting,
            open_threshold=open_threshold,
        )
    except ValueError as exc:
        raise DeckError(f"{context} failed: {exc}") from exc

    if np.any(tox_new > float(state.grid.Ly_um) + 1e-12):
        raise DeckError(
            f"{context} failed: oxide thickness exceeds domain depth Ly_um={state.grid.Ly_um}"
        )

    state.C = Cn
    state.tox_um = tox_new
    if update_materials:
        state.materials = materials_new
