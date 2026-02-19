"""Mask generation and lateral smoothing."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .units import ensure_positive


def build_mask_1d(x_um: np.ndarray, openings_um: Sequence[Sequence[float]]) -> np.ndarray:
    """Build a binary 1D mask over x from opening intervals in um.

    Parameters
    ----------
    x_um:
        1D x coordinates in um.
    openings_um:
        Iterable of [x_start_um, x_end_um]. Points inside any interval are open=1.
    """
    if len(x_um.shape) != 1:
        raise ValueError("x_um must be a 1D array.")

    mask = np.zeros_like(x_um, dtype=float)
    x_min = float(x_um[0])
    x_max = float(x_um[-1])

    for k, opening in enumerate(openings_um):
        if len(opening) != 2:
            raise ValueError(f"openings_um[{k}] must contain exactly 2 values.")

        x0 = float(opening[0])
        x1 = float(opening[1])
        if x0 >= x1:
            raise ValueError(
                f"openings_um[{k}] start must be smaller than end, got [{x0}, {x1}]."
            )
        if x1 < x_min or x0 > x_max:
            continue

        left = max(x0, x_min)
        right = min(x1, x_max)
        mask[(x_um >= left) & (x_um <= right)] = 1.0

    return mask


def smooth_mask_1d(mask: np.ndarray, sigma_lat_um: float, dx_um: float) -> np.ndarray:
    """Apply Gaussian smoothing to mask edges and clip to [0, 1]."""
    if mask.ndim != 1:
        raise ValueError("mask must be a 1D array.")
    ensure_positive("dx_um", float(dx_um))
    if sigma_lat_um < 0.0:
        raise ValueError(f"sigma_lat_um must be >= 0, got {sigma_lat_um}.")

    if sigma_lat_um == 0.0:
        return np.clip(mask.astype(float), 0.0, 1.0)

    sigma_px = float(sigma_lat_um) / float(dx_um)
    smoothed = gaussian_filter1d(mask.astype(float), sigma=sigma_px, mode="nearest")
    return np.clip(smoothed, 0.0, 1.0)


def full_open_mask(nx: int) -> np.ndarray:
    """Return an all-open mask of length nx."""
    if nx <= 0:
        raise ValueError(f"nx must be > 0, got {nx}.")
    return np.ones(nx, dtype=float)


def validate_mask(mask_eff: np.ndarray, nx: int) -> None:
    """Validate mask dimensions and range."""
    if mask_eff.ndim != 1 or mask_eff.shape[0] != nx:
        raise ValueError(f"mask_eff must be shape ({nx},), got {mask_eff.shape}.")
    if np.any(mask_eff < 0.0) or np.any(mask_eff > 1.0):
        raise ValueError("mask_eff values must be within [0, 1].")


def openings_from_any(value: Iterable[Sequence[float]]) -> list[list[float]]:
    """Normalize opening intervals into a list of float pairs."""
    openings: list[list[float]] = []
    for item in value:
        if len(item) != 2:
            raise ValueError("Each opening interval must contain two values.")
        openings.append([float(item[0]), float(item[1])])
    return openings
