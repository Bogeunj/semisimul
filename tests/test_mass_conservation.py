"""Mass conservation tests."""

from __future__ import annotations

import numpy as np

from proc2d.diffusion import TopBCConfig, anneal_implicit, total_amount
from proc2d.grid import Grid2D


def test_mass_conservation_neumann_everywhere() -> None:
    """Total amount should be conserved when all boundaries are Neumann."""
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.4, Nx=51, Ny=41)
    rng = np.random.default_rng(42)
    C0 = 1.0e15 + 1.0e13 * rng.random(grid.shape)

    C1 = anneal_implicit(
        C0=C0,
        grid=grid,
        D_cm2_s=2.0e-14,
        total_t_s=5.0,
        dt_s=0.2,
        top_bc=TopBCConfig(open_type="neumann", blocked_type="neumann"),
        mask_eff=None,
    )

    m0 = total_amount(C0, grid)
    m1 = total_amount(C1, grid)
    rel = abs(m1 - m0) / max(abs(m0), 1.0)
    assert rel < 1.0e-10
