"""Oxidation helpers for proc2d.

This module implements a minimal Deal-Grove oxidation workflow and related
geometry/material updates for 2D cross-section simulations.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid2D
from .units import ensure_positive


def build_material_map(grid: Grid2D, tox_um: np.ndarray) -> np.ndarray:
    """Build material map from oxide thickness profile.

    Returns int map with shape (Ny, Nx):
    - 0: Si
    - 1: SiO2
    """
    tox = np.asarray(tox_um, dtype=float)
    if tox.shape != (grid.Nx,):
        raise ValueError(f"tox_um must have shape ({grid.Nx},), got {tox.shape}.")
    if np.any(tox < 0.0):
        raise ValueError("tox_um must be non-negative.")

    y = grid.y_um[:, None]
    ox = y < tox[None, :]
    return ox.astype(np.int8)


def deal_grove_tox_update(
    tox_old_um: np.ndarray,
    time_s: float | np.ndarray,
    A_um: float,
    B_um2_s: float,
) -> np.ndarray:
    """Compute updated oxide thickness via Deal-Grove incremental formula.

    Formula per x:
      tau = (tox_old^2 + A*tox_old)/B
      tox_new = (-A + sqrt(A^2 + 4*B*(time + tau)))/2
    """
    ensure_positive("A_um", float(A_um), allow_zero=True)
    ensure_positive("B_um2_s", float(B_um2_s), allow_zero=True)

    tox_old = np.asarray(tox_old_um, dtype=float)
    if np.any(tox_old < 0.0):
        raise ValueError("tox_old_um must be non-negative.")

    time_arr = np.asarray(time_s, dtype=float)
    if time_arr.ndim == 0:
        ensure_positive("time_s", float(time_arr), allow_zero=True)
    else:
        if time_arr.shape != tox_old.shape:
            raise ValueError(f"time_s must have shape {tox_old.shape}, got {time_arr.shape}.")
        if np.any(time_arr < 0.0):
            raise ValueError("time_s must be non-negative.")

    B = float(B_um2_s)
    A = float(A_um)
    if B == 0.0:
        return tox_old.copy()

    tau = (tox_old * tox_old + A * tox_old) / B
    discr = A * A + 4.0 * B * (time_arr + tau)
    discr = np.maximum(discr, 0.0)
    tox_new = (-A + np.sqrt(discr)) / 2.0
    tox_new = np.maximum(tox_new, tox_old)
    return tox_new


def apply_surface_outward_shift(C: np.ndarray, grid: Grid2D, delta_out_um: np.ndarray) -> np.ndarray:
    """Apply column-wise surface outward shift using linear interpolation.

    For each x-column:
      C_new(y) = C_old(y - delta_out_um[x])

    Values above surface after shift (y - delta_out < 0) are filled with 0.
    """
    arr = np.asarray(C, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {arr.shape}.")

    delta = np.asarray(delta_out_um, dtype=float)
    if delta.shape != (grid.Nx,):
        raise ValueError(f"delta_out_um must have shape ({grid.Nx},), got {delta.shape}.")

    y = np.asarray(grid.y_um, dtype=float)
    Cn = np.empty_like(arr)
    for i in range(grid.Nx):
        y_src = y - float(delta[i])
        Cn[:, i] = np.interp(y_src, y, arr[:, i], left=0.0, right=float(arr[-1, i]))
    return Cn


def apply_oxidation(
    C: np.ndarray,
    grid: Grid2D,
    tox_old_um: np.ndarray,
    mask_eff: np.ndarray,
    *,
    time_s: float,
    A_um: float,
    B_um2_s: float,
    gamma: float = 2.27,
    apply_on: str = "all",
    consume_dopants: bool = True,
    mask_weighting: str = "binary",
    open_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply one oxidation update and return updated fields.

    Returns
    -------
    tuple
        (C_new, tox_new_um, materials, delta_out_um)
    """
    arr = np.asarray(C, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {arr.shape}.")
    tox_old = np.asarray(tox_old_um, dtype=float)
    if tox_old.shape != (grid.Nx,):
        raise ValueError(f"tox_old_um must have shape ({grid.Nx},), got {tox_old.shape}.")

    ensure_positive("gamma", float(gamma))
    apply_on_l = str(apply_on).lower()
    if apply_on_l not in {"all", "open", "blocked"}:
        raise ValueError("apply_on must be one of: all, open, blocked")

    weighting = str(mask_weighting).lower()
    if weighting not in {"binary", "time_scale"}:
        raise ValueError("mask_weighting must be one of: binary, time_scale")
    thresh = float(open_threshold)
    if thresh < 0.0 or thresh > 1.0:
        raise ValueError("open_threshold must be within [0, 1].")

    mask = np.asarray(mask_eff, dtype=float)
    if mask.shape != (grid.Nx,):
        raise ValueError(f"mask_eff must have shape ({grid.Nx},), got {mask.shape}.")

    if weighting == "binary":
        is_open = mask > thresh
        if apply_on_l == "all":
            active = np.ones(grid.Nx, dtype=bool)
        elif apply_on_l == "open":
            active = is_open
        else:
            active = ~is_open

        tox_all = deal_grove_tox_update(tox_old, time_s=float(time_s), A_um=float(A_um), B_um2_s=float(B_um2_s))
        tox_new = tox_old.copy()
        tox_new[active] = tox_all[active]
    else:
        mask_frac = np.clip(mask, 0.0, 1.0)
        if apply_on_l == "all":
            weights = np.ones(grid.Nx, dtype=float)
        elif apply_on_l == "open":
            weights = mask_frac
        else:
            weights = 1.0 - mask_frac

        tox_new = deal_grove_tox_update(
            tox_old,
            time_s=float(time_s) * weights,
            A_um=float(A_um),
            B_um2_s=float(B_um2_s),
        )

    tox_new = np.maximum(tox_new, tox_old)

    delta_tox = tox_new - tox_old
    delta_out_um = delta_tox * (1.0 - 1.0 / float(gamma))

    C_shifted = apply_surface_outward_shift(arr, grid, delta_out_um)
    materials = build_material_map(grid, tox_new)

    if consume_dopants:
        C_shifted = C_shifted.copy()
        C_shifted[materials == 1] = 0.0

    return C_shifted, tox_new, materials, delta_out_um
