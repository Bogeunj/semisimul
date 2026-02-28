from __future__ import annotations

import yaml

from proc2d.deck import run_deck


def test_deck_anneal_cap_model_exp_runs(tmp_path) -> None:
    deck = {
        "domain": {"Lx_um": 1.0, "Ly_um": 0.3, "Nx": 61, "Ny": 51},
        "background_doping_cm3": 1.0e15,
        "steps": [
            {
                "type": "implant",
                "dose_cm2": 8.0e12,
                "Rp_um": 0.04,
                "dRp_um": 0.015,
            },
            {
                "type": "anneal",
                "D_cm2_s": 1.0e-14,
                "total_t_s": 0.5,
                "dt_s": 0.25,
                "cap_model": "exp",
                "cap_len_um": 0.02,
                "top_bc": {
                    "open": {"type": "robin", "h_cm_s": 1.0e-5, "Ceq_cm3": 0.0},
                    "blocked": {"type": "neumann"},
                },
            },
            {
                "type": "export",
                "outdir": "outputs/cap_exp",
                "formats": ["npy"],
            },
        ],
    }

    deck_path = tmp_path / "deck_cap_exp.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False), encoding="utf-8")
    state = run_deck(deck_path, out_override=tmp_path / "out")
    assert state.C.shape == state.grid.shape
