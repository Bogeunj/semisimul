"""Unit tests for electrical model and electrical step output."""

from __future__ import annotations

import pytest

from proc2d.deck import run_simulation_payload
from proc2d.electrical import estimate_mosfet_metrics


pytestmark = pytest.mark.unit


def test_estimate_mosfet_metrics_id_decreases_with_thicker_oxide() -> None:
    thin = estimate_mosfet_metrics(
        Nch_cm3=1.0e17,
        tox_um=0.005,
        mobility_cm2_Vs=250.0,
        Vgs_V=2.5,
        Vds_V=0.1,
        W_um=1.0,
        L_um=0.5,
    )
    thick = estimate_mosfet_metrics(
        Nch_cm3=1.0e17,
        tox_um=0.03,
        mobility_cm2_Vs=250.0,
        Vgs_V=2.5,
        Vds_V=0.1,
        W_um=1.0,
        L_um=0.5,
    )

    assert thin["Cox_F_cm2"] > thick["Cox_F_cm2"]
    assert thin["Vth_V"] < thick["Vth_V"]
    assert thin["Id_A"] > thick["Id_A"]


def test_estimate_mosfet_metrics_id_increases_with_mobility() -> None:
    low = estimate_mosfet_metrics(
        Nch_cm3=1.0e17,
        tox_um=0.01,
        mobility_cm2_Vs=120.0,
        Vgs_V=2.5,
        Vds_V=0.1,
        W_um=1.0,
        L_um=0.5,
    )
    high = estimate_mosfet_metrics(
        Nch_cm3=1.0e17,
        tox_um=0.01,
        mobility_cm2_Vs=320.0,
        Vgs_V=2.5,
        Vds_V=0.1,
        W_um=1.0,
        L_um=0.5,
    )

    assert high["Id_A"] > low["Id_A"]


def test_electrical_step_writes_metrics_to_state() -> None:
    deck = {
        "domain": {"Lx_um": 1.0, "Ly_um": 0.3, "Nx": 81, "Ny": 61},
        "background_doping_cm3": 1.0e15,
        "steps": [
            {"type": "implant", "dose_cm2": 1.0e13, "Rp_um": 0.04, "dRp_um": 0.015},
            {
                "type": "electrical",
                "model": "mosfet_long_channel",
                "Vgs_V": 2.5,
                "Vds_V": 0.1,
                "W_um": 1.0,
                "L_um": 0.5,
            },
        ],
    }

    state = run_simulation_payload(deck)

    assert state.metrics is not None
    assert "electrical" in state.metrics
    assert float(state.metrics["electrical"]["Id_A"]) >= 0.0
