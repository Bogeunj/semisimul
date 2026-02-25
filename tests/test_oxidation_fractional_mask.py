from __future__ import annotations

import numpy as np

from proc2d.grid import Grid2D
from proc2d.oxidation import apply_oxidation


def test_oxidation_binary_and_time_scale_match_for_binary_mask() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.4, Nx=6, Ny=21)
    C0 = np.full(grid.shape, 1.0e15, dtype=float)
    tox0 = np.zeros(grid.Nx, dtype=float)
    mask = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=float)

    _, tox_bin, _, _ = apply_oxidation(
        C0,
        grid,
        tox0,
        mask,
        time_s=5.0,
        A_um=0.1,
        B_um2_s=0.01,
        apply_on="open",
        mask_weighting="binary",
        open_threshold=0.5,
    )

    _, tox_ts, _, _ = apply_oxidation(
        C0,
        grid,
        tox0,
        mask,
        time_s=5.0,
        A_um=0.1,
        B_um2_s=0.01,
        apply_on="open",
        mask_weighting="time_scale",
    )

    np.testing.assert_allclose(tox_bin, tox_ts, rtol=0.0, atol=1e-12)


def test_oxidation_time_scale_reflects_fractional_edge() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.4, Nx=5, Ny=21)
    C0 = np.full(grid.shape, 1.0e15, dtype=float)
    tox0 = np.zeros(grid.Nx, dtype=float)
    mask = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)

    _, tox_bin, _, _ = apply_oxidation(
        C0,
        grid,
        tox0,
        mask,
        time_s=8.0,
        A_um=0.1,
        B_um2_s=0.01,
        apply_on="open",
        mask_weighting="binary",
        open_threshold=0.5,
    )

    _, tox_ts, _, _ = apply_oxidation(
        C0,
        grid,
        tox0,
        mask,
        time_s=8.0,
        A_um=0.1,
        B_um2_s=0.01,
        apply_on="open",
        mask_weighting="time_scale",
    )

    assert np.all(tox_ts >= tox0)
    assert tox_ts[1] > tox_bin[1]
    assert tox_ts[2] > tox_bin[2]
    assert tox_ts[-1] >= tox_bin[-1] - 1e-12
