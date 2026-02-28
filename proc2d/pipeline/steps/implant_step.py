"""Implant step runner."""

from __future__ import annotations

from typing import Any

from ...domain.state import SimulationState
from ...physics.implant import implant_2d_gaussian
from ...physics.mask import full_open_mask
from ...units import ensure_positive
from .common import required, to_float


def run_implant_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    """Apply implant source profile to concentration field."""
    context = f"steps[{idx}] (implant)"
    dose_cm2 = to_float(required(step, "dose_cm2", context), "dose_cm2", context)
    Rp_um = to_float(required(step, "Rp_um", context), "Rp_um", context)
    dRp_um = to_float(required(step, "dRp_um", context), "dRp_um", context)

    ensure_positive(f"{context}.dose_cm2", dose_cm2)
    ensure_positive(f"{context}.dRp_um", dRp_um)
    state.ensure_oxide_fields()

    mask_eff = state.mask_eff if state.mask_eff is not None else full_open_mask(state.grid.Nx)
    dC = implant_2d_gaussian(
        grid=state.grid,
        dose_cm2=dose_cm2,
        Rp_um=Rp_um,
        dRp_um=dRp_um,
        mask_eff=mask_eff,
        tox_um=state.tox_um,
    )
    state.C = state.C + dC
