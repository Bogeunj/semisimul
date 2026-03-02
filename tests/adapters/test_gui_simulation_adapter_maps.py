"""Adapter tests for GUI physical map payloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from proc2d.gui_app import run_simulation


pytestmark = pytest.mark.adapter


def _params(tmp_path: Path) -> dict[str, object]:
    return {
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
        "oxidation_enable": True,
        "oxidation_time_s": 1.0,
        "oxidation_A_um": 0.08,
        "oxidation_B_um2_s": 0.005,
        "oxidation_gamma": 2.27,
        "oxidation_apply_on": "all",
        "oxidation_consume_dopants": True,
        "oxidation_update_materials": True,
        "deposition_enable": True,
        "deposition_thickness_um": 0.01,
        "deposition_apply_on": "all",
        "etch_enable": False,
        "etch_thickness_um": 0.0,
        "etch_apply_on": "all",
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
        "electrical_enable": False,
        "electrical_model": "mosfet_long_channel",
        "electrical_Vgs_V": 2.5,
        "electrical_Vds_V": 0.1,
        "electrical_W_um": 1.0,
        "electrical_L_um": 0.5,
        "export_tox_csv": False,
        "export_tox_png": False,
        "zip_outputs": False,
        "store_full_c": False,
    }


def test_gui_result_contains_physical_maps(tmp_path: Path) -> None:
    result = run_simulation(_params(tmp_path))

    assert "physical_maps" in result
    maps = result["physical_maps"]
    expected = {
        "structure",
        "concentration_cm3",
        "potential_V",
        "electric_field_mag_V_cm",
        "conductivity_S_cm",
    }
    assert expected.issubset(set(maps.keys()))
    for key in expected:
        arr = np.asarray(maps[key])
        assert arr.shape == result["grid"].shape
        assert np.all(np.isfinite(arr))


def test_gui_physical_maps_are_consistent(tmp_path: Path) -> None:
    result = run_simulation(_params(tmp_path))
    maps = result["physical_maps"]

    np.testing.assert_array_equal(maps["concentration_cm3"], result["C"])
    np.testing.assert_array_equal(maps["structure"], result["materials"])

    phi = np.asarray(maps["potential_V"], dtype=float)
    e_mag = np.asarray(maps["electric_field_mag_V_cm"], dtype=float)
    gy, gx = np.gradient(phi, result["grid"].dy_cm, result["grid"].dx_cm)
    e_ref = np.sqrt(gx * gx + gy * gy)
    np.testing.assert_allclose(
        e_mag[1:-1, 1:-1], e_ref[1:-1, 1:-1], rtol=1e-5, atol=1e-8
    )
