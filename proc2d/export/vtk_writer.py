"""Legacy VTK writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..grid import Grid2D


def _ensure_outdir(outdir: str | Path) -> Path:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_scalar_vtk_structured_points(
    field: np.ndarray,
    grid: Grid2D,
    outdir: str | Path,
    filename: str,
    scalar_name: str,
) -> Path:
    arr = np.asarray(field)
    if arr.shape != grid.shape:
        raise ValueError(f"field shape must be {grid.shape}, got {arr.shape}.")

    out = _ensure_outdir(outdir)
    path = out / filename
    flat = arr.reshape(-1)
    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("proc2d scalar field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {grid.Nx} {grid.Ny} 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write(f"SPACING {grid.dx_um:.12g} {grid.dy_um:.12g} 1\n")
        f.write(f"POINT_DATA {grid.Nx * grid.Ny}\n")
        f.write(f"SCALARS {scalar_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for k, value in enumerate(flat):
            f.write(f"{float(value):.8e}")
            if (k + 1) % 6 == 0:
                f.write("\n")
            else:
                f.write(" ")
        if (flat.size % 6) != 0:
            f.write("\n")
    return path


def save_vtk_structured_points(
    C: np.ndarray,
    grid: Grid2D,
    outdir: str | Path,
    filename: str = "C.vtk",
    scalar_name: str = "doping",
    use_log10: bool = False,
    log_floor_cm3: float | None = None,
) -> Path:
    """Save concentration field as legacy VTK ASCII STRUCTURED_POINTS."""
    arr = np.asarray(C, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {arr.shape}.")

    write_arr = arr
    if use_log10:
        floor = 1.0e10
        if log_floor_cm3 is not None:
            floor = max(floor, float(log_floor_cm3))
        write_arr = np.log10(np.clip(arr, floor, None))

    return _save_scalar_vtk_structured_points(
        field=write_arr,
        grid=grid,
        outdir=outdir,
        filename=filename,
        scalar_name=scalar_name,
    )
