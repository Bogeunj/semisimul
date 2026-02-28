from __future__ import annotations

import numpy as np

from proc2d.grid import Grid2D
from proc2d.metrics import iso_contour_area


def test_iso_area_cell_count_and_tri_linear_on_uniform_fields() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=1.0, Nx=11, Ny=11)

    low = np.zeros(grid.shape, dtype=float)
    high = np.full(grid.shape, 10.0, dtype=float)

    assert iso_contour_area(low, grid, threshold_cm3=1.0, method="cell_count") == 0.0
    assert iso_contour_area(low, grid, threshold_cm3=1.0, method="tri_linear") == 0.0

    full_area_cell_count = float(grid.Nx * grid.Ny * grid.dx_um * grid.dy_um)
    full_area_tri_linear = float(grid.Lx_um * grid.Ly_um)
    assert abs(iso_contour_area(high, grid, threshold_cm3=1.0, method="cell_count") - full_area_cell_count) < 1e-12
    assert abs(iso_contour_area(high, grid, threshold_cm3=1.0, method="tri_linear") - full_area_tri_linear) < 1e-12


def test_iso_area_tri_linear_is_accurate_for_linear_ramp() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=1.0, Nx=41, Ny=21)
    C = np.repeat(grid.x_um[None, :], grid.Ny, axis=0)

    area = iso_contour_area(C, grid, threshold_cm3=0.5, method="tri_linear")
    assert abs(area - 0.5) < 5e-3


def test_iso_area_tri_linear_refines_with_grid() -> None:
    grid_coarse = Grid2D.from_domain(Lx_um=1.0, Ly_um=1.0, Nx=11, Ny=11)
    grid_fine = Grid2D.from_domain(Lx_um=1.0, Ly_um=1.0, Nx=81, Ny=81)

    C_coarse = np.repeat(grid_coarse.x_um[None, :], grid_coarse.Ny, axis=0)
    C_fine = np.repeat(grid_fine.x_um[None, :], grid_fine.Ny, axis=0)

    a_coarse = iso_contour_area(C_coarse, grid_coarse, threshold_cm3=0.37, method="tri_linear")
    a_fine = iso_contour_area(C_fine, grid_fine, threshold_cm3=0.37, method="tri_linear")

    expected = 0.63
    e_coarse = abs(a_coarse - expected)
    e_fine = abs(a_fine - expected)
    assert e_fine <= e_coarse + 1e-12
