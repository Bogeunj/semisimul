"""CSV writers for linecuts and profiles."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from ..grid import Grid2D


def _ensure_outdir(outdir: str | Path) -> Path:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_linecuts_csv(
    C: np.ndarray,
    grid: Grid2D,
    linecuts: Iterable[dict],
    outdir: str | Path,
) -> list[Path]:
    """Save requested linecuts as CSV files."""
    out = _ensure_outdir(outdir)
    paths: list[Path] = []

    for idx, spec in enumerate(linecuts):
        if not isinstance(spec, dict):
            raise ValueError(f"linecuts[{idx}] must be a mapping.")

        kind = str(spec.get("kind", "")).lower()
        if kind == "vertical":
            if "x_um" not in spec:
                raise ValueError(f"linecuts[{idx}] vertical cut requires x_um.")
            x_req = float(spec["x_um"])
            ix = grid.nearest_x_index(x_req)
            x_used = float(grid.x_um[ix])
            y_vals = grid.y_um
            c_vals = np.asarray(C[:, ix], dtype=float)
            path = out / f"linecut_vertical_x{str(x_used).replace('.', 'p')}um.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["kind", "vertical"])
                writer.writerow(["x_um_requested", f"{x_req:.12g}"])
                writer.writerow(["x_um_used", f"{x_used:.12g}"])
                writer.writerow(["y_um", "C_cm3"])
                for y, c in zip(y_vals, c_vals):
                    writer.writerow([f"{float(y):.12g}", f"{float(c):.12g}"])
            paths.append(path)

        elif kind == "horizontal":
            if "y_um" not in spec:
                raise ValueError(f"linecuts[{idx}] horizontal cut requires y_um.")
            y_req = float(spec["y_um"])
            iy = grid.nearest_y_index(y_req)
            y_used = float(grid.y_um[iy])
            x_vals = grid.x_um
            c_vals = np.asarray(C[iy, :], dtype=float)
            path = out / f"linecut_horizontal_y{str(y_used).replace('.', 'p')}um.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["kind", "horizontal"])
                writer.writerow(["y_um_requested", f"{y_req:.12g}"])
                writer.writerow(["y_um_used", f"{y_used:.12g}"])
                writer.writerow(["x_um", "C_cm3"])
                for x, c in zip(x_vals, c_vals):
                    writer.writerow([f"{float(x):.12g}", f"{float(c):.12g}"])
            paths.append(path)

        else:
            raise ValueError(
                f"linecuts[{idx}].kind must be 'vertical' or 'horizontal', got '{kind}'."
            )

    return paths


def save_sheet_dose_vs_x_csv(
    x_um: np.ndarray,
    sheet_dose_cm2: np.ndarray,
    outdir: str | Path,
    filename: str = "sheet_dose_vs_x.csv",
) -> Path:
    """Save x-wise sheet dose profile CSV."""
    x_arr = np.asarray(x_um, dtype=float)
    sd_arr = np.asarray(sheet_dose_cm2, dtype=float)
    if x_arr.shape != sd_arr.shape:
        raise ValueError(
            f"x_um and sheet_dose_cm2 must have same shape, got {x_arr.shape} and {sd_arr.shape}."
        )

    path = _ensure_outdir(outdir) / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_um", "sheet_dose_cm2"])
        for x, sd in zip(x_arr, sd_arr):
            writer.writerow([f"{float(x):.12g}", f"{float(sd):.12g}"])
    return path


def save_tox_vs_x_csv(tox_um: np.ndarray, grid: Grid2D, outdir: str | Path, filename: str = "tox_vs_x.csv") -> Path:
    """Save oxide thickness profile tox(x) as CSV."""
    tox = np.asarray(tox_um, dtype=float)
    if tox.shape != (grid.Nx,):
        raise ValueError(f"tox_um must have shape ({grid.Nx},), got {tox.shape}.")

    path = _ensure_outdir(outdir) / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_um", "tox_um"])
        for x, t in zip(grid.x_um, tox):
            writer.writerow([f"{float(x):.12g}", f"{float(t):.12g}"])
    return path
