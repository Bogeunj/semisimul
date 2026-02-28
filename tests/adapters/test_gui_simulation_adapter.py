"""Adapter tests for GUI simulation adapter path."""

from __future__ import annotations

from pathlib import Path

import pytest

from proc2d.gui_app import run_simulation


pytestmark = pytest.mark.adapter


def test_gui_simulation_adapter_smoke(tmp_path: Path) -> None:
    params = {
        "Lx_um": 1.0,
        "Ly_um": 0.3,
        "Nx": 31,
        "Ny": 21,
        "background_doping_cm3": 1.0e15,
        "openings_um": [[0.2, 0.8]],
        "sigma_lat_um": 0.01,
        "dose_cm2": 3.0e12,
        "Rp_um": 0.04,
        "dRp_um": 0.015,
        "oxidation_enable": False,
        "oxidation_time_s": 1.0,
        "oxidation_A_um": 0.08,
        "oxidation_B_um2_s": 0.005,
        "oxidation_gamma": 2.27,
        "oxidation_apply_on": "all",
        "oxidation_consume_dopants": True,
        "oxidation_update_materials": True,
        "top_open_type": "neumann",
        "h_cm_s": 0.0,
        "Ceq_cm3": 0.0,
        "dirichlet_value_cm3": 0.0,
        "anneal_use_arrhenius": False,
        "D_cm2_s": 1.0e-14,
        "arrhenius_D0_cm2_s": 1.0e-3,
        "arrhenius_Ea_eV": 3.5,
        "arrhenius_T_C": 1000.0,
        "oxide_D_scale": 0.0,
        "cap_eps_um": 0.0,
        "total_t_s": 0.2,
        "dt_s": 0.1,
        "record_history": False,
        "history_every_s": 0.1,
        "history_save_csv": False,
        "history_save_png": False,
        "outdir": str(tmp_path / "out"),
        "formats": ["npy", "csv"],
        "export_vtk": False,
        "plot_log10": True,
        "plot_vmin": 1.0e14,
        "plot_vmax": 1.0e20,
        "linecut_x_um": 0.5,
        "linecut_y_um": 0.05,
        "compute_metrics": False,
        "export_tox_csv": False,
        "export_tox_png": False,
        "zip_outputs": False,
        "store_full_c": False,
    }

    result = run_simulation(params)

    assert result["grid"].shape == (21, 31)
    assert isinstance(result["runtime_s"], float)
    assert result["written"]
    assert (tmp_path / "out" / "C.npy").exists()
