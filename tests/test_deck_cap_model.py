from __future__ import annotations

import numpy as np
import pytest
import yaml

from proc2d.deck import run_deck


pytestmark = pytest.mark.integration


def _run_cap_model(tmp_path, cap_model: str):
    deck = {
        "domain": {"Lx_um": 1.0, "Ly_um": 0.3, "Nx": 61, "Ny": 51},
        "background_doping_cm3": 1.0e15,
        "steps": [
            {
                "type": "oxidation",
                "time_s": 0.0,
                "A_um": 0.1,
                "B_um2_s": 0.0,
                "apply_on": "all",
                "tox_init_um": 0.02,
                "consume_dopants": False,
            },
            {
                "type": "implant",
                "dose_cm2": 8.0e12,
                "Rp_um": 0.04,
                "dRp_um": 0.015,
            },
            {
                "type": "anneal",
                "D_cm2_s": 1.0e-14,
                "total_t_s": 2.0,
                "dt_s": 0.25,
                "cap_model": cap_model,
                "cap_eps_um": 0.001,
                "cap_len_um": 0.2,
                "top_bc": {
                    "open": {"type": "robin", "h_cm_s": 1.0e-3, "Ceq_cm3": 0.0},
                    "blocked": {"type": "neumann"},
                },
            },
            {
                "type": "export",
                "outdir": "outputs/cap_model",
                "formats": ["npy"],
            },
        ],
    }

    deck_path = tmp_path / f"deck_cap_{cap_model}.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False), encoding="utf-8")
    state = run_deck(deck_path, out_override=tmp_path / f"out_{cap_model}")
    return state


def test_deck_anneal_cap_model_exp_runs(tmp_path) -> None:
    state = _run_cap_model(tmp_path, "exp")
    assert state.C.shape == state.grid.shape


def test_cap_model_exp_changes_final_mass_vs_hard(tmp_path) -> None:
    state_hard = _run_cap_model(tmp_path, "hard")
    state_exp = _run_cap_model(tmp_path, "exp")

    assert state_hard.tox_um is not None
    assert state_exp.tox_um is not None
    assert float(np.max(state_hard.tox_um)) > 0.001

    mass_hard = float(
        np.sum(state_hard.C) * state_hard.grid.dx_cm * state_hard.grid.dy_cm
    )
    mass_exp = float(np.sum(state_exp.C) * state_exp.grid.dx_cm * state_exp.grid.dy_cm)
    assert mass_exp < mass_hard
