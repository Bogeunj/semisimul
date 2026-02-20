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
    tox_um: np.ndarray | None = None,
) -> np.ndarray:
    """Return 2D implant increment with optional oxide depth shift.

    Base model:
      dC(y,x) = g(y_eff) * mask_eff(x)

    where ``y_eff = y_um - tox_um[x]`` (silicon-relative depth). For
    ``y_eff < 0`` the implant is set to zero.
    """
    ensure_positive("dose_cm2", float(dose_cm2))
    ensure_positive("dRp_um", float(dRp_um))

    if mask_eff.shape != (grid.Nx,):
        raise ValueError(
            f"mask_eff must have shape ({grid.Nx},), got {mask_eff.shape}."
        )

    Rp_cm = float(um_to_cm(float(Rp_um)))
    dRp_cm = float(um_to_cm(float(dRp_um)))

    if tox_um is None:
        tox = np.zeros(grid.Nx, dtype=float)
    else:
        tox = np.asarray(tox_um, dtype=float)
        if tox.shape != (grid.Nx,):
            raise ValueError(f"tox_um must have shape ({grid.Nx},), got {tox.shape}.")

    y_eff_um = grid.y_um[:, None] - tox[None, :]
    y_eff_cm = np.asarray(um_to_cm(y_eff_um), dtype=float)

    pref = float(dose_cm2) / (np.sqrt(2.0 * np.pi) * float(dRp_cm))
    z = (y_eff_cm - float(Rp_cm)) / float(dRp_cm)
    g2d = pref * np.exp(-0.5 * z * z)
    g2d[y_eff_cm < 0.0] = 0.0

    return g2d * np.asarray(mask_eff, dtype=float)[None, :]
