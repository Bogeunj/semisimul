"""Quantitative metrics for proc2d concentration fields.

All concentration values use cm-based units:
- C: cm^-3
- x, y queries from user: um
- integrals use grid spacings converted in ``Grid2D``
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .grid import Grid2D


def total_mass(C: np.ndarray, grid: Grid2D) -> float:
    """Return total amount integral in cm^-1.

    Uses the 2D cross-section convention with implicit z-thickness of 1 cm:

    ``M = sum(C) * dx_cm * dy_cm``
    """
    arr = np.asarray(C, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {arr.shape}.")
    return float(np.sum(arr) * grid.dx_cm * grid.dy_cm)


def peak_info(C: np.ndarray, grid: Grid2D) -> dict[str, float | int]:
    """Return peak concentration and location information.

    Returns
    -------
    dict
        {
          "peak_cm3": float,
          "x_um": float,
          "y_um": float,
          "i": int,
          "j": int,
        }
    """
    arr = np.asarray(C, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {arr.shape}.")

    flat_idx = int(np.argmax(arr))
    j, i = np.unravel_index(flat_idx, arr.shape)
    return {
        "peak_cm3": float(arr[j, i]),
        "x_um": float(grid.x_um[i]),
        "y_um": float(grid.y_um[j]),
        "i": int(i),
        "j": int(j),
    }


def sheet_dose_vs_x(C: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Return sheet dose along x in cm^-2.

    ``Sd[i] = sum_j C[j, i] * dy_cm``
    """
    arr = np.asarray(C, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {arr.shape}.")
    return np.sum(arr, axis=0) * grid.dy_cm


def junction_depth_1d(
    profile_cm3: np.ndarray,
    y_um: np.ndarray,
    threshold_cm3: float,
    mode: Literal["first", "last"] = "first",
) -> float | None:
    """Find 1D threshold crossing depth in um via linear interpolation.

    Parameters
    ----------
    profile_cm3:
        Concentration profile versus depth.
    y_um:
        Depth coordinates in um (monotonic increasing).
    threshold_cm3:
        Threshold concentration [cm^-3].
    mode:
        ``"first"`` returns the first crossing from small y to large y,
        ``"last"`` returns the deepest crossing.

    Returns
    -------
    float | None
        Crossing depth in um, or ``None`` if no crossing exists.
    """
    p = np.asarray(profile_cm3, dtype=float)
    y = np.asarray(y_um, dtype=float)
    if p.ndim != 1 or y.ndim != 1:
        raise ValueError("profile_cm3 and y_um must be 1D arrays.")
    if p.shape[0] != y.shape[0]:
        raise ValueError("profile_cm3 and y_um must have same length.")
    if p.shape[0] < 2:
        return None
    if mode not in ("first", "last"):
        raise ValueError(f"Unsupported mode '{mode}'. Use 'first' or 'last'.")

    th = float(threshold_cm3)
    crossings: list[float] = []

    for k in range(p.shape[0] - 1):
        p0 = float(p[k])
        p1 = float(p[k + 1])
        y0 = float(y[k])
        y1 = float(y[k + 1])

        if p0 == th:
            crossings.append(y0)

        d0 = p0 - th
        d1 = p1 - th
        has_cross = (d0 * d1 < 0.0) or (p1 == th)
        if not has_cross:
            continue

        if p1 == p0:
            y_cross = y0
        else:
            frac = (th - p0) / (p1 - p0)
            y_cross = y0 + frac * (y1 - y0)
        crossings.append(float(y_cross))

    if not crossings:
        return None

    # Preserve order while removing near-duplicates from exact threshold points.
    uniq: list[float] = []
    for val in crossings:
        if not uniq or abs(val - uniq[-1]) > 1e-12:
            uniq.append(val)

    if mode == "first":
        return float(uniq[0])
    return float(uniq[-1])


def junction_depth(C: np.ndarray, grid: Grid2D, x_um: float, threshold_cm3: float) -> float | None:
    """Return junction depth at nearest x index for a threshold [cm^-3]."""
    arr = np.asarray(C, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {arr.shape}.")

    ix = grid.nearest_x_index(float(x_um))
    profile = arr[:, ix]
    return junction_depth_1d(profile, grid.y_um, threshold_cm3=float(threshold_cm3), mode="first")


def lateral_extents_at_y(C: np.ndarray, grid: Grid2D, y_um: float, threshold_cm3: float) -> dict[str, object]:
    """Return lateral segments where ``C >= threshold`` at a given y-slice.

    The segments are reported in um as ``[[x0, x1], ...]`` and include a
    total width estimate in um.
    """
    arr = np.asarray(C, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {arr.shape}.")

    iy = grid.nearest_y_index(float(y_um))
    row = arr[iy, :]
    active = row >= float(threshold_cm3)

    segments_idx: list[tuple[int, int]] = []
    start: int | None = None
    for i, is_on in enumerate(active):
        if is_on and start is None:
            start = i
        elif (not is_on) and (start is not None):
            segments_idx.append((start, i - 1))
            start = None
    if start is not None:
        segments_idx.append((start, grid.Nx - 1))

    dx = float(grid.dx_um)
    x_min = float(grid.x_um[0])
    x_max = float(grid.x_um[-1])
    segments_um: list[list[float]] = []
    width_um_total = 0.0
    for i0, i1 in segments_idx:
        x0 = max(x_min, float(grid.x_um[i0]) - 0.5 * dx)
        x1 = min(x_max, float(grid.x_um[i1]) + 0.5 * dx)
        if x1 < x0:
            x1 = x0
        segments_um.append([x0, x1])
        width_um_total += x1 - x0

    return {
        "y_um_requested": float(y_um),
        "y_um_used": float(grid.y_um[iy]),
        "width_um_total": float(width_um_total),
        "segments_um": segments_um,
    }


def iso_contour_area(C: np.ndarray, grid: Grid2D, threshold_cm3: float) -> float:
    """Approximate area of region with ``C >= threshold`` in um^2.

    Cell-counting approximation:
    ``area_um2 = n_cells * dx_um * dy_um``
    """
    arr = np.asarray(C, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {arr.shape}.")
    n_cells = int(np.count_nonzero(arr >= float(threshold_cm3)))
    return float(n_cells * grid.dx_um * grid.dy_um)
