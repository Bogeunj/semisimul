from __future__ import annotations

import numpy as np

from proc2d.diffusion import TopBCConfig, top_flux_out, top_open_fraction_with_cap
from proc2d.grid import Grid2D


def test_top_open_fraction_with_cap_hard_and_exp() -> None:
    tox: np.ndarray = np.array([0.0, 0.01, 0.05], dtype=float)
    mask: np.ndarray = np.ones(3, dtype=float)

    hard = top_open_fraction_with_cap(mask, tox, cap_eps_um=0.02, cap_model="hard")
    np.testing.assert_allclose(hard, np.array([1.0, 1.0, 0.0], dtype=float), rtol=0.0, atol=1e-12)

    exp = top_open_fraction_with_cap(mask, tox, cap_eps_um=0.0, cap_model="exp", cap_len_um=0.01)
    assert exp[0] > exp[1] > exp[2] > 0.0


def test_exp_cap_reduces_robin_flux_as_tox_grows() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.2, Nx=21, Ny=11)
    C = np.full(grid.shape, 1.0e15, dtype=float)
    top_bc = TopBCConfig(open_type="robin", blocked_type="neumann", h_cm_s=1.0e-5, Ceq_cm3=0.0)

    mask = np.ones(grid.Nx, dtype=float)
    open_low = top_open_fraction_with_cap(mask, np.zeros(grid.Nx), cap_eps_um=0.0, cap_model="exp", cap_len_um=0.01)
    open_high = top_open_fraction_with_cap(
        mask,
        np.full(grid.Nx, 0.05, dtype=float),
        cap_eps_um=0.0,
        cap_model="exp",
        cap_len_um=0.01,
    )

    flux_low = top_flux_out(C, grid, top_bc=top_bc, mask_eff=open_low)
    flux_high = top_flux_out(C, grid, top_bc=top_bc, mask_eff=open_high)
    assert flux_high < flux_low
