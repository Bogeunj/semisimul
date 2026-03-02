"""Deposition helpers for process simulation."""

from __future__ import annotations

import numpy as np

from .grid import Grid2D
from .oxidation import apply_surface_outward_shift, build_material_map
from .units import ensure_positive


def apply_deposition(
    C: np.ndarray,
    grid: Grid2D,
    tox_old_um: np.ndarray,
    mask_eff: np.ndarray,
    *,
    thickness_um: float,
    apply_on: str = "all",
    mask_weighting: str = "binary",
    open_threshold: float = 0.5,
    consume_dopants: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply conformal oxide deposition and return updated fields."""
    arr = np.asarray(C, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {arr.shape}.")

    tox_old = np.asarray(tox_old_um, dtype=float)
    if tox_old.shape != (grid.Nx,):
        raise ValueError(
            f"tox_old_um must have shape ({grid.Nx},), got {tox_old.shape}."
        )
    if np.any(tox_old < 0.0):
        raise ValueError("tox_old_um must be non-negative.")

    mask = np.asarray(mask_eff, dtype=float)
    if mask.shape != (grid.Nx,):
        raise ValueError(f"mask_eff must have shape ({grid.Nx},), got {mask.shape}.")

    dep = float(thickness_um)
    ensure_positive("thickness_um", dep, allow_zero=True)

    apply_on_l = str(apply_on).lower()
    if apply_on_l not in {"all", "open", "blocked"}:
        raise ValueError("apply_on must be one of: all, open, blocked")

    weighting = str(mask_weighting).lower()
    if weighting not in {"binary", "time_scale"}:
        raise ValueError("mask_weighting must be one of: binary, time_scale")

    thresh = float(open_threshold)
    if thresh < 0.0 or thresh > 1.0:
        raise ValueError("open_threshold must be within [0, 1].")

    if weighting == "binary":
        is_open = mask > thresh
        if apply_on_l == "all":
            delta_tox = np.full(grid.Nx, dep, dtype=float)
        elif apply_on_l == "open":
            delta_tox = np.where(is_open, dep, 0.0)
        else:
            delta_tox = np.where(is_open, 0.0, dep)
    else:
        mask_frac = np.clip(mask, 0.0, 1.0)
        if apply_on_l == "all":
            weights = np.ones(grid.Nx, dtype=float)
        elif apply_on_l == "open":
            weights = mask_frac
        else:
            weights = 1.0 - mask_frac
        delta_tox = dep * weights

    tox_new = tox_old + delta_tox
    Cn = apply_surface_outward_shift(arr, grid, delta_tox)
    materials = build_material_map(grid, tox_new)
    if consume_dopants:
        Cn = Cn.copy()
        Cn[materials == 1] = 0.0
    return Cn, tox_new, materials, delta_tox


__all__ = ["apply_deposition"]
