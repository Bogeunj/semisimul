"""Unit conversion helpers.

The simulator receives geometry inputs in um but solves in cm-based units.
"""

from __future__ import annotations

import numpy as np

ArrayLike = float | np.ndarray

UM_TO_CM = 1.0e-4
CM_TO_UM = 1.0e4


def _to_scalar_if_needed(value: np.ndarray) -> ArrayLike:
    if value.ndim == 0:
        return float(value)
    return value


def um_to_cm(value: ArrayLike) -> ArrayLike:
    """Convert um to cm."""
    arr = np.asarray(value, dtype=float) * UM_TO_CM
    return _to_scalar_if_needed(arr)


def cm_to_um(value: ArrayLike) -> ArrayLike:
    """Convert cm to um."""
    arr = np.asarray(value, dtype=float) * CM_TO_UM
    return _to_scalar_if_needed(arr)


def ensure_positive(name: str, value: float, allow_zero: bool = False) -> None:
    """Validate that a scalar is positive (or non-negative if allow_zero)."""
    if allow_zero:
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}.")
        return
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def ensure_nonnegative_array(name: str, value: np.ndarray) -> None:
    """Validate array contains only non-negative values."""
    if np.any(value < 0.0):
        raise ValueError(f"{name} must be non-negative.")
