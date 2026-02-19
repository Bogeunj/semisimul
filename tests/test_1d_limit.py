"""1D limit behavior test."""

from __future__ import annotations

import numpy as np

from proc2d.diffusion import TopBCConfig, anneal_implicit
from proc2d.grid import Grid2D
from proc2d.implant import implant_2d_gaussian
from proc2d.mask import full_open_mask


def test_full_open_uniform_bc_keeps_x_uniformity() -> None:
    """If mask and BC are x-uniform, concentration should remain nearly x-uniform."""
    grid = Grid2D.from_domain(Lx_um=2.0, Ly_um=0.5, Nx=121, Ny=101)
    mask_eff = full_open_mask(grid.Nx)

    C0 = np.full(grid.shape, 5.0e14, dtype=float)
    C0 += implant_2d_gaussian(
        grid=grid,
        dose_cm2=2.0e13,
        Rp_um=0.06,
        dRp_um=0.02,
        mask_eff=mask_eff,
    )

    C1 = anneal_implicit(
        C0=C0,
        grid=grid,
        D_cm2_s=8.0e-15,
        total_t_s=1.5,
        dt_s=0.15,
        top_bc=TopBCConfig(open_type="robin", blocked_type="neumann", h_cm_s=2.0e-5, Ceq_cm3=0.0),
        mask_eff=mask_eff,
    )

    x_spread = np.max(np.ptp(C1, axis=1))
    scale = max(1.0, float(np.max(np.abs(C1))))
    assert x_spread / scale < 1.0e-9
