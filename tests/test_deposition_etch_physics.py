"""Unit tests for deposition/etch physical coupling."""

from __future__ import annotations

import numpy as np
import pytest

from proc2d.deposition import apply_deposition
from proc2d.etch import apply_etch
from proc2d.grid import Grid2D
from proc2d.implant import implant_2d_gaussian
from proc2d.mask import build_mask_1d, full_open_mask, smooth_mask_1d


pytestmark = pytest.mark.unit


def test_deposition_all_increases_tox_and_updates_materials() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.4, Nx=81, Ny=81)
    C0 = np.full(grid.shape, 1.0e15, dtype=float)
    tox0 = np.zeros(grid.Nx, dtype=float)
    mask = full_open_mask(grid.Nx)

    C1, tox1, materials1, _ = apply_deposition(
        C0,
        grid,
        tox0,
        mask,
        thickness_um=0.03,
        apply_on="all",
    )

    assert np.all(tox1 >= tox0)
    np.testing.assert_allclose(
        tox1, np.full(grid.Nx, 0.03, dtype=float), rtol=0.0, atol=1e-12
    )
    assert np.count_nonzero(materials1 == 1) > 0
    assert C1.shape == C0.shape


def test_etch_clamps_tox_nonnegative() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.4, Nx=61, Ny=61)
    C0 = np.full(grid.shape, 1.0e15, dtype=float)
    tox0 = np.zeros(grid.Nx, dtype=float)
    mask = full_open_mask(grid.Nx)

    C1, tox1, _, _ = apply_deposition(C0, grid, tox0, mask, thickness_um=0.01)
    C2, tox2, _, _ = apply_etch(C1, grid, tox1, mask, thickness_um=0.05)

    assert np.all(np.isfinite(C2))
    assert np.all(tox2 >= 0.0)
    np.testing.assert_allclose(tox2, np.zeros_like(tox2), rtol=0.0, atol=1e-12)


def test_deposition_open_time_scale_uses_fractional_mask() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.4, Nx=121, Ny=81)
    C0 = np.full(grid.shape, 1.0e15, dtype=float)
    tox0 = np.zeros(grid.Nx, dtype=float)

    mask_raw = build_mask_1d(grid.x_um, [[0.45, 0.55]])
    mask = smooth_mask_1d(mask_raw, sigma_lat_um=0.03, dx_um=grid.dx_um)

    _, tox_bin, _, _ = apply_deposition(
        C0,
        grid,
        tox0,
        mask,
        thickness_um=0.02,
        apply_on="open",
        mask_weighting="binary",
        open_threshold=0.5,
    )
    _, tox_ts, _, _ = apply_deposition(
        C0,
        grid,
        tox0,
        mask,
        thickness_um=0.02,
        apply_on="open",
        mask_weighting="time_scale",
    )

    edge = np.where((mask > 0.05) & (mask < 0.45))[0]
    assert edge.size > 0
    i_edge = int(edge[0])
    assert tox_ts[i_edge] > tox_bin[i_edge]


def test_etch_open_binary_respects_threshold() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.4, Nx=101, Ny=81)
    C0 = np.full(grid.shape, 1.0e15, dtype=float)
    tox0 = np.zeros(grid.Nx, dtype=float)
    mask = build_mask_1d(grid.x_um, [[0.4, 0.6]])

    C1, tox1, _, _ = apply_deposition(
        C0, grid, tox0, full_open_mask(grid.Nx), thickness_um=0.03
    )
    _, tox2, _, _ = apply_etch(
        C1,
        grid,
        tox1,
        mask,
        thickness_um=0.02,
        apply_on="open",
        mask_weighting="binary",
        open_threshold=0.5,
    )

    open_idx = mask > 0.5
    blocked_idx = mask <= 0.5
    assert float(np.mean(tox2[open_idx])) < float(np.mean(tox2[blocked_idx]))
    np.testing.assert_allclose(
        tox2[blocked_idx], tox1[blocked_idx], rtol=0.0, atol=1e-12
    )


def test_deposition_shifts_implant_peak_depth() -> None:
    grid = Grid2D.from_domain(Lx_um=0.4, Ly_um=0.3, Nx=41, Ny=301)
    C0 = np.full(grid.shape, 1.0e15, dtype=float)
    tox0 = np.zeros(grid.Nx, dtype=float)
    mask = full_open_mask(grid.Nx)

    _, tox_dep, _, _ = apply_deposition(
        C0, grid, tox0, mask, thickness_um=0.03, apply_on="all"
    )

    dC0 = implant_2d_gaussian(
        grid=grid,
        dose_cm2=1.0e13,
        Rp_um=0.05,
        dRp_um=0.01,
        mask_eff=mask,
        tox_um=tox0,
    )
    dC1 = implant_2d_gaussian(
        grid=grid,
        dose_cm2=1.0e13,
        Rp_um=0.05,
        dRp_um=0.01,
        mask_eff=mask,
        tox_um=tox_dep,
    )

    iy0 = int(np.argmax(dC0[:, grid.Nx // 2]))
    iy1 = int(np.argmax(dC1[:, grid.Nx // 2]))
    y_shift = float(grid.y_um[iy1] - grid.y_um[iy0])
    assert abs(y_shift - 0.03) <= grid.dy_um + 1.0e-12
