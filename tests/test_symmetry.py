"""Symmetry test for centered mask/opening."""

from __future__ import annotations

import numpy as np

from proc2d.diffusion import TopBCConfig, anneal_implicit
from proc2d.grid import Grid2D
from proc2d.implant import implant_2d_gaussian
from proc2d.mask import build_mask_1d, smooth_mask_1d


def test_center_opening_preserves_x_symmetry() -> None:
    """A centered opening should produce x-mirror symmetric concentration."""
    grid = Grid2D.from_domain(Lx_um=2.0, Ly_um=0.5, Nx=101, Ny=81)
    mask_raw = build_mask_1d(grid.x_um, openings_um=[[0.8, 1.2]])
    mask_eff = smooth_mask_1d(mask_raw, sigma_lat_um=0.03, dx_um=grid.dx_um)

    C = np.full(grid.shape, 1.0e15, dtype=float)
    C += implant_2d_gaussian(
        grid=grid,
        dose_cm2=1.0e13,
        Rp_um=0.05,
        dRp_um=0.02,
        mask_eff=mask_eff,
    )

    Cn = anneal_implicit(
        C0=C,
        grid=grid,
        D_cm2_s=1.0e-14,
        total_t_s=2.0,
        dt_s=0.2,
        top_bc=TopBCConfig(open_type="robin", blocked_type="neumann", h_cm_s=1.0e-5, Ceq_cm3=0.0),
        mask_eff=mask_eff,
    )

    diff = np.max(np.abs(Cn - Cn[:, ::-1]))
    scale = max(1.0, float(np.max(np.abs(Cn))))
    assert diff / scale < 1.0e-9
