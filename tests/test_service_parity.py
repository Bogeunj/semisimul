"""Regression tests for shared simulation service across deck and GUI adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("streamlit")

from proc2d.config import GuiRunConfig, build_deck_from_gui_config
from proc2d.deck import run_deck
from proc2d.gui_app import run_simulation

pytestmark = pytest.mark.integration


def _sample_gui_params(outdir: Path) -> dict[str, object]:
    return {
        "Lx_um": 1.0,
        "Ly_um": 0.3,
        "Nx": 41,
        "Ny": 31,
        "background_doping_cm3": 1.0e15,
        "openings_um": [[0.2, 0.8]],
        "sigma_lat_um": 0.01,
        "dose_cm2": 8.0e12,
        "Rp_um": 0.04,
        "dRp_um": 0.015,
        "oxidation_enable": True,
        "oxidation_time_s": 1.0,
        "oxidation_A_um": 0.08,
        "oxidation_B_um2_s": 0.005,
        "oxidation_gamma": 2.27,
        "oxidation_apply_on": "all",
        "oxidation_consume_dopants": True,
        "oxidation_update_materials": True,
        "top_open_type": "robin",
        "h_cm_s": 1.0e-5,
        "Ceq_cm3": 0.0,
        "dirichlet_value_cm3": 0.0,
        "anneal_use_arrhenius": False,
        "D_cm2_s": 1.0e-14,
        "arrhenius_D0_cm2_s": 1.0e-3,
        "arrhenius_Ea_eV": 3.5,
        "arrhenius_T_C": 1000.0,
        "oxide_D_scale": 0.1,
        "cap_eps_um": 0.0,
        "total_t_s": 0.4,
        "dt_s": 0.2,
        "record_history": True,
        "history_every_s": 0.2,
        "history_save_csv": True,
        "history_save_png": False,
        "outdir": str(outdir),
        "formats": ["npy", "csv", "vtk"],
        "export_vtk": False,
        "plot_log10": True,
        "plot_vmin": 1.0e14,
        "plot_vmax": 1.0e20,
        "linecut_x_um": 0.5,
        "linecut_y_um": 0.05,
        "compute_metrics": True,
        "export_tox_csv": True,
        "export_tox_png": False,
        "zip_outputs": False,
        "store_full_c": False,
    }


def test_deck_gui_parity_for_shared_service(tmp_path: Path) -> None:
    gui_out = tmp_path / "gui_out"
    deck_out = tmp_path / "deck_out"
    params = _sample_gui_params(gui_out)

    gui_result = run_simulation(params)

    cfg = GuiRunConfig.from_mapping(params)
    deck_payload = build_deck_from_gui_config(cfg)
    deck_path = tmp_path / "from_gui.yaml"
    deck_path.write_text(yaml.safe_dump(deck_payload, sort_keys=False), encoding="utf-8")
    deck_state = run_deck(deck_path, out_override=deck_out)

    assert deck_state.grid.shape == gui_result["grid"].shape
    np.testing.assert_allclose(gui_result["C"], deck_state.C, rtol=1e-10, atol=1.0e3)
    np.testing.assert_allclose(gui_result["tox_um"], deck_state.tox_um, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(gui_result["materials"], deck_state.materials)

    assert len(gui_result["history"]) == len(deck_state.history)
    assert bool(gui_result["metrics"]) == bool(deck_state.metrics)

    gui_manifest = {Path(p).name for p in gui_result["written"]}
    deck_manifest = {Path(p).name for p in deck_state.exports}
    assert gui_manifest == deck_manifest


def test_manifest_includes_expected_outputs(tmp_path: Path) -> None:
    outdir = tmp_path / "gui_out"
    params = _sample_gui_params(outdir)
    result = run_simulation(params)

    written_names = [Path(p).name for p in result["written"]]
    expected = {
        "C.npy",
        "C.vtk",
        "material.vtk",
        "C_log10.vtk",
        "history.csv",
        "metrics.json",
        "metrics.csv",
        "sheet_dose_vs_x.csv",
        "tox_vs_x.csv",
    }
    assert expected.issubset(set(written_names))
    assert len(written_names) == len(set(written_names))
