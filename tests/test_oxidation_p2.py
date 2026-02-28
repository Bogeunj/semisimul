"""P2 oxidation and oxide-aware process tests."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from proc2d.deck import _arrhenius_D, run_deck
from proc2d.grid import Grid2D
from proc2d.implant import implant_2d_gaussian
from proc2d.mask import full_open_mask
from proc2d.oxidation import apply_surface_outward_shift, deal_grove_tox_update


pytestmark = pytest.mark.integration


def test_deal_grove_tox_update_monotonic() -> None:
    tox0 = np.asarray([0.0, 0.01, 0.03], dtype=float)
    tox1 = deal_grove_tox_update(tox0, time_s=20.0, A_um=0.1, B_um2_s=0.01)
    assert np.all(tox1 >= tox0)
    assert np.all(np.isfinite(tox1))


def test_surface_outward_shift_on_linear_profile() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=1.0, Nx=3, Ny=11)
    C0 = np.repeat(grid.y_um[:, None], grid.Nx, axis=1)
    delta = np.full(grid.Nx, 0.2, dtype=float)
    C1 = apply_surface_outward_shift(C0, grid, delta)

    assert np.allclose(C1[0, :], 0.0)
    mid = grid.nearest_y_index(0.5)
    assert np.allclose(C1[mid, :], grid.y_um[mid] - 0.2, atol=1e-12)


def test_implant_peak_shifts_with_oxide_thickness() -> None:
    grid = Grid2D.from_domain(Lx_um=0.2, Ly_um=0.3, Nx=2, Ny=301)
    mask = full_open_mask(grid.Nx)
    tox = np.asarray([0.0, 0.03], dtype=float)

    dC = implant_2d_gaussian(
        grid=grid,
        dose_cm2=1.0e13,
        Rp_um=0.05,
        dRp_um=0.01,
        mask_eff=mask,
        tox_um=tox,
    )

    y0 = float(grid.y_um[int(np.argmax(dC[:, 0]))])
    y1 = float(grid.y_um[int(np.argmax(dC[:, 1]))])
    assert abs((y1 - y0) - 0.03) <= grid.dy_um + 1.0e-12


def test_arrhenius_diffusivity_increases_with_temperature() -> None:
    D_low = _arrhenius_D(D0_cm2_s=1.0e-3, Ea_eV=3.5, T_C=900.0)
    D_high = _arrhenius_D(D0_cm2_s=1.0e-3, Ea_eV=3.5, T_C=1050.0)
    assert D_high > D_low > 0.0


def test_deck_oxidation_exports_tox_and_material(tmp_path) -> None:
    deck = {
        "domain": {"Lx_um": 1.0, "Ly_um": 0.4, "Nx": 81, "Ny": 81},
        "background_doping_cm3": 1.0e15,
        "steps": [
            {"type": "mask", "openings_um": [[0.4, 0.6]], "sigma_lat_um": 0.02},
            {
                "type": "oxidation",
                "model": "deal_grove",
                "time_s": 5.0,
                "A_um": 0.1,
                "B_um2_s": 0.01,
                "apply_on": "all",
            },
            {"type": "implant", "dose_cm2": 5.0e12, "Rp_um": 0.04, "dRp_um": 0.015},
            {
                "type": "anneal",
                "D_cm2_s": 1.0e-14,
                "total_t_s": 1.0,
                "dt_s": 0.25,
                "oxide": {"D_scale": 0.0},
            },
            {
                "type": "export",
                "outdir": "outputs/p2_test",
                "formats": ["vtk", "png"],
                "extra": {"tox_csv": True, "tox_png": True},
            },
        ],
    }
    deck_path = tmp_path / "deck_p2_test.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False), encoding="utf-8")

    outdir = tmp_path / "out"
    state = run_deck(deck_path, out_override=outdir)

    assert state.tox_um is not None
    assert float(np.max(state.tox_um)) > 0.0
    assert (outdir / "material.vtk").exists()
    assert (outdir / "tox_vs_x.csv").exists()
    assert (outdir / "tox_vs_x.png").exists()
