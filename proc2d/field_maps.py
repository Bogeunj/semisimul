"""Derived physical map calculations for 2D process states."""

from __future__ import annotations

import numpy as np

from .grid import Grid2D
from .units import ensure_positive

_Q_C = 1.602176634e-19
_EPS0_F_CM = 8.8541878128e-14


def _as_2d_array(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got ndim={arr.ndim}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _material_masks(materials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = _as_2d_array("materials", materials)
    mat_i = np.rint(mat).astype(np.int32)
    is_oxide = mat_i == 1
    is_silicon = ~is_oxide
    return is_silicon, is_oxide


def _interface_permittivity(eps_a: float, eps_b: float) -> float:
    """Return interface permittivity using harmonic averaging."""
    if eps_a <= 0.0 or eps_b <= 0.0:
        return 0.0
    return 2.0 * eps_a * eps_b / (eps_a + eps_b)


def _validate_same_shape(
    name_a: str, a: np.ndarray, name_b: str, b: np.ndarray
) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have identical shapes, got {a.shape} and {b.shape}."
        )


def _grid_spacing_cm(grid: Grid2D) -> tuple[float, float]:
    dx_cm = float(grid.dx_cm)
    dy_cm = float(grid.dy_cm)
    ensure_positive("grid.dx_cm", dx_cm)
    ensure_positive("grid.dy_cm", dy_cm)
    return dx_cm, dy_cm


def build_structure_map(materials: np.ndarray) -> np.ndarray:
    """Return a sanitized copy of the material structure map."""
    arr = _as_2d_array("materials", materials)
    return np.rint(arr).astype(np.int32)


def compute_conductivity_map(
    C: np.ndarray,
    materials: np.ndarray,
    background_doping_cm3: float,
    mobility_si_cm2_Vs: float = 250.0,
    sigma_oxide_S_cm: float = 1e-18,
) -> np.ndarray:
    """Compute conductivity map using sigma=q*mu*n in Si, near-zero in oxide."""
    conc = _as_2d_array("C", C)
    is_silicon, _ = _material_masks(materials)
    _validate_same_shape("C", conc, "materials", is_silicon)

    bg = float(background_doping_cm3)
    mu = float(mobility_si_cm2_Vs)
    sigma_ox = float(sigma_oxide_S_cm)
    ensure_positive("background_doping_cm3", bg, allow_zero=True)
    ensure_positive("mobility_si_cm2_Vs", mu)
    ensure_positive("sigma_oxide_S_cm", sigma_ox, allow_zero=True)

    n_eff = np.maximum(conc - bg, 0.0)
    sigma = np.full(conc.shape, sigma_ox, dtype=float)
    sigma[is_silicon] = _Q_C * mu * n_eff[is_silicon]

    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
    return sigma


def solve_potential_map(
    C: np.ndarray,
    grid: Grid2D,
    materials: np.ndarray,
    background_doping_cm3: float,
    iterations: int = 120,
    omega: float = 1.7,
    eps_si_rel: float = 11.7,
    eps_ox_rel: float = 3.9,
) -> np.ndarray:
    """Solve variable-permittivity Poisson equation via SOR with zero-phi boundaries."""
    conc = _as_2d_array("C", C)
    structure = build_structure_map(materials)
    _validate_same_shape("C", conc, "materials", structure)
    if conc.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {conc.shape}.")

    bg = float(background_doping_cm3)
    ensure_positive("background_doping_cm3", bg, allow_zero=True)

    n_iter = int(iterations)
    if n_iter < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}.")

    w = float(omega)
    if not (0.0 < w < 2.0):
        raise ValueError(f"omega must satisfy 0 < omega < 2, got {omega}.")

    eps_si = float(eps_si_rel)
    eps_ox = float(eps_ox_rel)
    ensure_positive("eps_si_rel", eps_si)
    ensure_positive("eps_ox_rel", eps_ox)
    dx_cm, dy_cm = _grid_spacing_cm(grid)
    dx2 = dx_cm * dx_cm
    dy2 = dy_cm * dy_cm

    is_silicon = structure != 1
    eps_rel = np.where(is_silicon, eps_si, eps_ox)
    eps_abs = _EPS0_F_CM * eps_rel

    rho = np.zeros_like(conc, dtype=float)
    rho[is_silicon] = _Q_C * (conc[is_silicon] - bg)

    phi = np.zeros_like(conc, dtype=float)
    ny, nx = phi.shape
    for _ in range(n_iter):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                eps_w = _interface_permittivity(eps_abs[j, i], eps_abs[j, i - 1])
                eps_n = _interface_permittivity(eps_abs[j, i], eps_abs[j - 1, i])
                eps_s = _interface_permittivity(eps_abs[j, i], eps_abs[j + 1, i])

                eps_e = _interface_permittivity(eps_abs[j, i], eps_abs[j, i + 1])

                denom = (eps_e + eps_w) / dx2 + (eps_n + eps_s) / dy2
                if denom <= 0.0:
                    continue
                numer = (
                    (eps_e * phi[j, i + 1] + eps_w * phi[j, i - 1]) / dx2
                    + (eps_n * phi[j - 1, i] + eps_s * phi[j + 1, i]) / dy2
                    + rho[j, i]
                )
                phi_star = numer / denom
                phi[j, i] = (1.0 - w) * phi[j, i] + w * phi_star

    phi = np.nan_to_num(
        phi, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max
    )
    return phi


def electric_field_magnitude(potential_V: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Return electric-field magnitude from potential map in V/cm."""
    phi = _as_2d_array("potential_V", potential_V)
    if phi.shape != grid.shape:
        raise ValueError(f"potential_V shape must be {grid.shape}, got {phi.shape}.")
    dx_cm, dy_cm = _grid_spacing_cm(grid)

    grad_y, grad_x = np.gradient(phi, dy_cm, dx_cm)
    e_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    e_mag = np.nan_to_num(e_mag, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
    return e_mag


def compute_physical_maps(
    C: np.ndarray,
    grid: Grid2D,
    materials: np.ndarray,
    background_doping_cm3: float,
) -> dict[str, np.ndarray]:
    """Compute structure, concentration, potential, field, and conductivity maps."""
    conc = _as_2d_array("C", C)
    structure = build_structure_map(materials)
    _validate_same_shape("C", conc, "materials", structure)
    if conc.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {conc.shape}.")

    potential = solve_potential_map(
        C=conc,
        grid=grid,
        materials=structure,
        background_doping_cm3=background_doping_cm3,
    )
    e_mag = electric_field_magnitude(potential, grid)
    conductivity = compute_conductivity_map(
        C=conc,
        materials=structure,
        background_doping_cm3=background_doping_cm3,
    )

    return {
        "structure": structure,
        "concentration_cm3": conc.copy(),
        "potential_V": potential,
        "electric_field_mag_V_cm": e_mag,
        "conductivity_S_cm": conductivity,
    }


__all__ = [
    "build_structure_map",
    "compute_conductivity_map",
    "solve_potential_map",
    "electric_field_magnitude",
    "compute_physical_maps",
]
