"""Gaussian implant model."""

from __future__ import annotations

import numpy as np

from .grid import Grid2D
from .units import ensure_positive, um_to_cm


def gaussian_depth_profile(
    y_cm: np.ndarray,
    dose_cm2: float,
    Rp_cm: float,
    dRp_cm: float,
) -> np.ndarray:
    """Return 1D Gaussian depth profile g(y) in cm^-3.

    g(y) = dose / (sqrt(2*pi)*dRp) * exp(-0.5*((y - Rp)/dRp)^2)
    """
    ensure_positive("dose_cm2", float(dose_cm2))
    ensure_positive("dRp_cm", float(dRp_cm))

    pref = float(dose_cm2) / (np.sqrt(2.0 * np.pi) * float(dRp_cm))
    z = (np.asarray(y_cm, dtype=float) - float(Rp_cm)) / float(dRp_cm)
    return pref * np.exp(-0.5 * z * z)


def implant_2d_gaussian(
    grid: Grid2D,
    dose_cm2: float,
    Rp_um: float,
    dRp_um: float,
    mask_eff: np.ndarray,
) -> np.ndarray:
    """Return 2D implant increment dC(y,x) = g(y)*mask_eff(x)."""
    ensure_positive("dose_cm2", float(dose_cm2))
    ensure_positive("dRp_um", float(dRp_um))

    if mask_eff.shape != (grid.Nx,):
        raise ValueError(
            f"mask_eff must have shape ({grid.Nx},), got {mask_eff.shape}."
        )

    Rp_cm = float(um_to_cm(float(Rp_um)))
    dRp_cm = float(um_to_cm(float(dRp_um)))
    g_y = gaussian_depth_profile(grid.y_cm, dose_cm2=dose_cm2, Rp_cm=Rp_cm, dRp_cm=dRp_cm)
    return g_y[:, None] * np.asarray(mask_eff, dtype=float)[None, :]
