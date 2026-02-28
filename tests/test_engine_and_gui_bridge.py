from __future__ import annotations

from pathlib import Path

import numpy as np

from proc2d.deck import load_deck, run_deck_data
from proc2d.engine import run_deck_mapping
from proc2d.gui_bridge import build_deck_from_ui


def test_engine_matches_run_deck_data_for_same_mapping(tmp_path) -> None:
    deck_path = Path("examples/deck_analysis_history_vtk.yaml").resolve()
    deck = load_deck(deck_path)

    state_direct = run_deck_data(deck, deck_path=deck_path, out_override=tmp_path / "direct")
    result_engine = run_deck_mapping(deck, base_dir=deck_path.parent, out_override=tmp_path / "engine")
    state_engine = result_engine.state

    np.testing.assert_allclose(state_direct.C, state_engine.C, rtol=1e-12, atol=1e-6)
    assert bool(state_direct.metrics) == bool(state_engine.metrics)
    if state_direct.metrics and state_engine.metrics:
        m0 = float(state_direct.metrics["total_mass_cm1"])
        m1 = float(state_engine.metrics["total_mass_cm1"])
        assert abs(m0 - m1) <= max(1e-9, 1e-9 * abs(m0))


def test_build_deck_from_ui_supports_arrhenius_schedule(tmp_path) -> None:
    params = {
        "Lx_um": 1.0,
        "Ly_um": 0.3,
        "Nx": 81,
        "Ny": 61,
        "background_doping_cm3": 1.0e15,
        "openings_um": [[0.4, 0.6]],
        "sigma_lat_um": 0.01,
        "oxidation_enable": False,
        "dose_cm2": 5.0e12,
        "Rp_um": 0.04,
        "dRp_um": 0.015,
        "D_cm2_s": 1.0e-14,
        "total_t_s": 1.0,
        "dt_s": 0.25,
        "anneal_use_arrhenius": True,
        "arrhenius_D0_cm2_s": 1.0e-3,
        "arrhenius_Ea_eV": 3.5,
        "arrhenius_T_C": 1000.0,
        "anneal_schedule_text": "- {t_s: 0.6, T_C: 900}\n- {t_s: 0.4, T_C: 1000}",
        "oxide_D_scale": 0.0,
        "cap_eps_um": 0.001,
        "cap_model": "hard",
        "cap_len_um": 0.01,
        "top_open_type": "neumann",
        "h_cm_s": 0.0,
        "Ceq_cm3": 0.0,
        "dirichlet_value_cm3": 0.0,
        "formats": ["npy", "png"],
        "export_vtk": False,
        "linecut_x_um": 0.5,
        "linecut_y_um": 0.05,
        "plot_log10": True,
        "plot_vmin": 1.0e14,
        "plot_vmax": 1.0e20,
        "compute_metrics": True,
        "record_history": False,
        "history_every_s": 0.25,
        "history_save_csv": True,
        "history_save_png": True,
        "export_tox_csv": False,
        "export_tox_png": False,
    }

    deck = build_deck_from_ui(params, outdir=tmp_path / "ui")
    anneal = [s for s in deck["steps"] if s["type"] == "anneal"][0]
    diff = anneal["diffusivity"]
    assert diff["model"] == "arrhenius"
    assert "schedule" in diff
    assert len(diff["schedule"]) == 2

    state = run_deck_data(deck, deck_path=tmp_path / "ui.yaml", out_override=tmp_path / "run")
    assert state.metrics is not None
