"""Tests for analysis/metrics utilities."""

from __future__ import annotations

import numpy as np

from proc2d.grid import Grid2D
from proc2d.metrics import (
    junction_depth,
    junction_depth_1d,
    lateral_extents_at_y,
    peak_info,
    sheet_dose_vs_x,
    total_mass,
)


def test_metrics_basic_values() -> None:
    grid = Grid2D.from_domain(Lx_um=2.0, Ly_um=1.0, Nx=3, Ny=2)
    C = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)

    mass = total_mass(C, grid)
    expected_mass = float(np.sum(C) * grid.dx_cm * grid.dy_cm)
    assert abs(mass - expected_mass) < 1e-18

    peak = peak_info(C, grid)
    assert peak["peak_cm3"] == 6.0
    assert peak["i"] == 2
    assert peak["j"] == 1
    assert abs(float(peak["x_um"]) - 2.0) < 1e-12
    assert abs(float(peak["y_um"]) - 1.0) < 1e-12

    sd = sheet_dose_vs_x(C, grid)
    expected_sd = np.array([5.0, 7.0, 9.0]) * grid.dy_cm
    np.testing.assert_allclose(sd, expected_sd, rtol=0.0, atol=1e-18)


def test_junction_depth_and_lateral_extents() -> None:
    y_um = np.array([0.0, 0.1], dtype=float)
    profile = np.array([1.0e18, 1.0e16], dtype=float)
    depth = junction_depth_1d(profile, y_um, threshold_cm3=1.0e17)
    assert depth is not None
    assert abs(depth - 0.0909090909) < 1e-6

    grid = Grid2D.from_domain(Lx_um=2.0, Ly_um=0.3, Nx=5, Ny=4)
    C = np.zeros(grid.shape, dtype=float)
    C[:, :] = 1.0e16
    C[1, 1:4] = 2.0e17

    jd = junction_depth(C, grid, x_um=1.0, threshold_cm3=1.0e17)
    assert jd is not None
    assert jd >= 0.0

    lat = lateral_extents_at_y(C, grid, y_um=float(grid.y_um[1]), threshold_cm3=1.0e17)
    assert lat["width_um_total"] > 0.0
    assert len(lat["segments_um"]) == 1
