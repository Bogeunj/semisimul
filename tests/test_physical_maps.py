"""Unit tests for derived physical map calculations."""

from __future__ import annotations

import numpy as np
import pytest

from proc2d.field_maps import (
    build_structure_map,
    compute_conductivity_map,
    compute_physical_maps,
    electric_field_magnitude,
    solve_potential_map,
)
from proc2d.grid import Grid2D


pytestmark = pytest.mark.unit


def test_compute_physical_maps_returns_expected_keys_and_shapes() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.3, Nx=41, Ny=31)
    C = np.full(grid.shape, 1.0e16, dtype=float)
    materials = np.zeros(grid.shape, dtype=np.int8)

    maps = compute_physical_maps(
        C=C,
        grid=grid,
        materials=materials,
        background_doping_cm3=1.0e15,
    )

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
        assert arr.shape == grid.shape
        assert np.all(np.isfinite(arr))


def test_structure_map_matches_materials_exactly() -> None:
    grid = Grid2D.from_domain(Lx_um=0.8, Ly_um=0.2, Nx=21, Ny=17)
    materials = np.zeros(grid.shape, dtype=np.int8)
    materials[:4, :] = 1

    structure = build_structure_map(materials)
    np.testing.assert_array_equal(structure, materials)


def test_structure_map_preserves_large_material_ids() -> None:
    materials = np.array([[0.0, 1.0, 255.0]], dtype=float)
    structure = build_structure_map(materials)
    np.testing.assert_array_equal(structure, np.array([[0, 1, 255]], dtype=np.int32))


def test_electric_field_magnitude_matches_linear_potential_gradient() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.5, Nx=31, Ny=27)
    xx, yy = np.meshgrid(grid.x_cm, grid.y_cm)
    phi = 0.7 * xx - 1.3 * yy

    emag = electric_field_magnitude(phi, grid)
    target = np.sqrt(0.7**2 + 1.3**2)
    core = emag[1:-1, 1:-1]
    np.testing.assert_allclose(core, target, rtol=1e-4, atol=1e-8)


def test_solve_potential_map_responds_to_charge_density() -> None:
    grid = Grid2D.from_domain(Lx_um=0.8, Ly_um=0.25, Nx=41, Ny=31)
    materials = np.zeros(grid.shape, dtype=np.int8)
    C_bg = np.full(grid.shape, 1.0e15, dtype=float)
    C_hi = np.full(grid.shape, 5.0e17, dtype=float)

    phi_bg = solve_potential_map(
        C=C_bg,
        grid=grid,
        materials=materials,
        background_doping_cm3=1.0e15,
        iterations=120,
    )
    phi_hi = solve_potential_map(
        C=C_hi,
        grid=grid,
        materials=materials,
        background_doping_cm3=1.0e15,
        iterations=120,
    )

    assert float(np.max(np.abs(phi_bg))) <= 1.0e-6
    assert float(np.max(np.abs(phi_hi))) > 1.0e-4


def test_compute_conductivity_map_increases_with_concentration_in_si() -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.2, Nx=41, Ny=17)
    x = np.linspace(1.0e15, 5.0e17, grid.Nx, dtype=float)
    C = np.tile(x[None, :], (grid.Ny, 1))
    materials = np.zeros(grid.shape, dtype=np.int8)

    sigma = compute_conductivity_map(
        C=C,
        materials=materials,
        background_doping_cm3=1.0e15,
    )

    line = sigma[grid.Ny // 2, :]
    assert np.all(np.diff(line) >= 0.0)
