"""Structured 2D grid definition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .units import ensure_positive, um_to_cm


@dataclass(frozen=True)
class Grid2D:
    """Structured 2D grid.

    Coordinates follow process convention:
    - x: lateral direction
    - y: depth direction, with y=0 at the top surface
    """

    Lx_um: float
    Ly_um: float
    Nx: int
    Ny: int
    x_um: np.ndarray
    y_um: np.ndarray
    x_cm: np.ndarray
    y_cm: np.ndarray
    dx_um: float
    dy_um: float
    dx_cm: float
    dy_cm: float

    @classmethod
    def from_domain(cls, Lx_um: float, Ly_um: float, Nx: int, Ny: int) -> "Grid2D":
        """Construct a grid from domain size and number of points."""
        ensure_positive("Lx_um", float(Lx_um))
        ensure_positive("Ly_um", float(Ly_um))
        if Nx < 2:
            raise ValueError(f"Nx must be >= 2, got {Nx}.")
        if Ny < 2:
            raise ValueError(f"Ny must be >= 2, got {Ny}.")

        x_um = np.linspace(0.0, float(Lx_um), int(Nx), dtype=float)
        y_um = np.linspace(0.0, float(Ly_um), int(Ny), dtype=float)
        dx_um = float(x_um[1] - x_um[0])
        dy_um = float(y_um[1] - y_um[0])

        x_cm = np.asarray(um_to_cm(x_um), dtype=float)
        y_cm = np.asarray(um_to_cm(y_um), dtype=float)
        dx_cm = float(x_cm[1] - x_cm[0])
        dy_cm = float(y_cm[1] - y_cm[0])

        return cls(
            Lx_um=float(Lx_um),
            Ly_um=float(Ly_um),
            Nx=int(Nx),
            Ny=int(Ny),
            x_um=x_um,
            y_um=y_um,
            x_cm=x_cm,
            y_cm=y_cm,
            dx_um=dx_um,
            dy_um=dy_um,
            dx_cm=dx_cm,
            dy_cm=dy_cm,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return field shape as (Ny, Nx)."""
        return (self.Ny, self.Nx)

    @property
    def size(self) -> int:
        """Return total number of grid points."""
        return self.Nx * self.Ny

    def flat_index(self, j: int, i: int) -> int:
        """Convert (j, i) index to flattened row-major index."""
        return j * self.Nx + i

    def nearest_x_index(self, x_um: float) -> int:
        """Nearest x index for a requested x in um."""
        return int(np.argmin(np.abs(self.x_um - float(x_um))))

    def nearest_y_index(self, y_um: float) -> int:
        """Nearest y index for a requested y in um."""
        return int(np.argmin(np.abs(self.y_um - float(y_um))))
