"""Export manager orchestrating format-specific writers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ..grid import Grid2D
from .csv_writer import save_linecuts_csv, save_tox_vs_x_csv
from .npy_writer import save_field_npy
from .png_writer import save_heatmap_png, save_tox_vs_x_png
from .vtk_writer import _save_scalar_vtk_structured_points, save_vtk_structured_points


def _ensure_outdir(outdir: str | Path) -> Path:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_results(
    C: np.ndarray,
    grid: Grid2D,
    outdir: str | Path,
    formats: Iterable[str],
    linecuts: Iterable[dict] | None = None,
    plot_cfg: dict | None = None,
    tox_um: np.ndarray | None = None,
    materials: np.ndarray | None = None,
    extra: dict | None = None,
) -> list[Path]:
    """Export current state based on requested formats."""
    out = _ensure_outdir(outdir)
    requested = {str(fmt).lower() for fmt in formats}
    valid = {"npy", "csv", "png", "vtk"}
    unknown = requested - valid
    if unknown:
        raise ValueError(f"Unsupported export format(s): {sorted(unknown)}")

    written: list[Path] = []

    if "npy" in requested:
        written.append(save_field_npy(C, out, filename="C.npy"))

    if "csv" in requested:
        cuts = list(linecuts or [])
        if cuts:
            written.extend(save_linecuts_csv(C, grid, cuts, out))
        else:
            default_cut = [{"kind": "vertical", "x_um": float(grid.x_um[grid.Nx // 2])}]
            written.extend(save_linecuts_csv(C, grid, default_cut, out))

    if "png" in requested:
        written.append(save_heatmap_png(C, grid, out, filename="C.png", plot_cfg=plot_cfg))

    if "vtk" in requested:
        written.append(save_vtk_structured_points(C, grid, out, filename="C.vtk", scalar_name="doping"))
        if materials is not None:
            mat = np.asarray(materials)
            if mat.shape != grid.shape:
                raise ValueError(f"materials must have shape {grid.shape}, got {mat.shape}.")
            written.append(
                _save_scalar_vtk_structured_points(
                    field=mat.astype(float),
                    grid=grid,
                    outdir=out,
                    filename="material.vtk",
                    scalar_name="material",
                )
            )
        cfg = dict(plot_cfg or {})
        if bool(cfg.get("log10", False)):
            log_floor = float(cfg.get("vmin", 1.0e10)) if cfg.get("vmin") is not None else 1.0e10
            written.append(
                save_vtk_structured_points(
                    C,
                    grid,
                    out,
                    filename="C_log10.vtk",
                    scalar_name="doping_log10",
                    use_log10=True,
                    log_floor_cm3=log_floor,
                )
            )

    extra_cfg = dict(extra or {})
    if tox_um is not None and bool(extra_cfg.get("tox_csv", False)):
        written.append(save_tox_vs_x_csv(tox_um, grid, out, filename="tox_vs_x.csv"))
    if tox_um is not None and bool(extra_cfg.get("tox_png", False)):
        written.append(save_tox_vs_x_png(tox_um, grid, out, filename="tox_vs_x.png"))

    return written
