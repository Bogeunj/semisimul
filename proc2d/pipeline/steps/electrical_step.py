"""Electrical step runner."""

from __future__ import annotations

from typing import Any

from ...domain.state import SimulationState
from ...electrical import channel_doping_from_state, estimate_mosfet_metrics
from ...errors import DeckError
from .common import required, to_float


def run_electrical_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    """Compute electrical metrics and merge into state.metrics."""
    context = f"steps[{idx}] (electrical)"
    model = str(step.get("model", "mosfet_long_channel")).lower()
    if model != "mosfet_long_channel":
        raise DeckError(
            f"{context}.model must be 'mosfet_long_channel', got '{model}'."
        )

    state.ensure_oxide_fields()
    assert state.tox_um is not None

    mobility_cm2_Vs = to_float(
        step.get("mobility_cm2_Vs", 250.0), "mobility_cm2_Vs", context
    )
    Vgs_V = to_float(required(step, "Vgs_V", context), "Vgs_V", context)
    Vds_V = to_float(required(step, "Vds_V", context), "Vds_V", context)
    W_um = to_float(required(step, "W_um", context), "W_um", context)
    L_um = to_float(required(step, "L_um", context), "L_um", context)

    try:
        Nch_cm3 = channel_doping_from_state(state.C, state.tox_um, state.grid.dy_um)
        tox_eff_um = float(state.tox_um.mean())
        report = estimate_mosfet_metrics(
            Nch_cm3=Nch_cm3,
            tox_um=tox_eff_um,
            mobility_cm2_Vs=mobility_cm2_Vs,
            Vgs_V=Vgs_V,
            Vds_V=Vds_V,
            W_um=W_um,
            L_um=L_um,
        )
    except ValueError as exc:
        raise DeckError(f"{context} failed: {exc}") from exc

    metrics = dict(state.metrics) if isinstance(state.metrics, dict) else {}
    metrics["electrical"] = report
    state.metrics = metrics
