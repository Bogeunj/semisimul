"""Oxidation step runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...domain.state import SimulationState
from ...errors import DeckError
from ...physics.mask import full_open_mask
from ...physics.oxidation import apply_oxidation
from ...units import ensure_positive
from .common import required, to_float


def run_oxidation_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    """Apply oxidation and optional material map update."""
    context = f"steps[{idx}] (oxidation)"
    state.ensure_oxide_fields()

    model = str(step.get("model", "deal_grove")).lower()
    if model != "deal_grove":
        raise DeckError(f"{context}.model must be 'deal_grove', got '{model}'.")

    time_s = to_float(required(step, "time_s", context), "time_s", context)
    A_um = to_float(required(step, "A_um", context), "A_um", context)
    B_um2_s = to_float(required(step, "B_um2_s", context), "B_um2_s", context)
    gamma = to_float(step.get("gamma", 2.27), "gamma", context)
    ensure_positive(f"{context}.time_s", time_s, allow_zero=True)
    ensure_positive(f"{context}.A_um", A_um, allow_zero=True)
    ensure_positive(f"{context}.B_um2_s", B_um2_s, allow_zero=True)
    ensure_positive(f"{context}.gamma", gamma)

    apply_on = str(step.get("apply_on", "all")).lower()
    consume_dopants = bool(step.get("consume_dopants", True))
    update_materials = bool(step.get("update_materials", True))

    if "tox_init_um" in step and np.allclose(state.tox_um, 0.0):
        tox_init = to_float(step["tox_init_um"], "tox_init_um", context)
        ensure_positive(f"{context}.tox_init_um", tox_init, allow_zero=True)
        state.tox_um = np.full(state.grid.Nx, tox_init, dtype=float)

    mask_eff = state.mask_eff if state.mask_eff is not None else full_open_mask(state.grid.Nx)
    Cn, tox_new, materials_new, _ = apply_oxidation(
        state.C,
        state.grid,
        state.tox_um,
        mask_eff,
        time_s=time_s,
        A_um=A_um,
        B_um2_s=B_um2_s,
        gamma=gamma,
        apply_on=apply_on,
        consume_dopants=consume_dopants,
    )

    if np.any(tox_new > float(state.grid.Ly_um) + 1e-12):
        raise DeckError(
            f"{context} failed: oxide thickness exceeds domain depth Ly_um={state.grid.Ly_um}"
        )

    state.C = Cn
    state.tox_um = tox_new
    if update_materials:
        state.materials = materials_new
