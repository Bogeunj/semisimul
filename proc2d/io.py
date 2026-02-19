"""Output helpers for simulation results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .grid import Grid2D


def ensure_outdir(outdir: str | Path) -> Path:
    """Create output directory if needed and return resolved path."""
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_field_npy(C: np.ndarray, outdir: str | Path, filename: str = "C.npy") -> Path:
    """Save concentration field as .npy."""
    path = ensure_outdir(outdir) / filename
    np.save(path, np.asarray(C, dtype=float))
    return path


def save_linecuts_csv(
    C: np.ndarray,
    grid: Grid2D,
    linecuts: Iterable[dict],
    outdir: str | Path,
) -> list[Path]:
    """Save requested linecuts as CSV files."""
    out = ensure_outdir(outdir)
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


def save_heatmap_png(
    C: np.ndarray,
    grid: Grid2D,
    outdir: str | Path,
    filename: str = "C.png",
    plot_cfg: dict | None = None,
) -> Path:
    """Save 2D concentration heatmap as PNG."""
    out = ensure_outdir(outdir)
    path = out / filename

    cfg = dict(plot_cfg or {})
    use_log10 = bool(cfg.get("log10", False))
    vmin = cfg.get("vmin")
    vmax = cfg.get("vmax")

    arr = np.asarray(C, dtype=float)
    if use_log10:
        floor = 1e10
        if vmin is not None:
            floor = max(floor, float(vmin))
        plot_arr = np.log10(np.clip(arr, floor, None))
        vmin_plot = np.log10(float(vmin)) if vmin is not None else None
        vmax_plot = np.log10(float(vmax)) if vmax is not None else None
        cbar_label = "log10(C [cm^-3])"
    else:
        plot_arr = arr
        vmin_plot = float(vmin) if vmin is not None else None
        vmax_plot = float(vmax) if vmax is not None else None
        cbar_label = "C [cm^-3]"

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    im = ax.imshow(
        plot_arr,
        extent=[float(grid.x_um[0]), float(grid.x_um[-1]), float(grid.y_um[-1]), float(grid.y_um[0])],
        aspect="auto",
        origin="upper",
        cmap="inferno",
        vmin=vmin_plot,
        vmax=vmax_plot,
    )
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title("Dopant concentration")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def export_results(
    C: np.ndarray,
    grid: Grid2D,
    outdir: str | Path,
    formats: Iterable[str],
    linecuts: Iterable[dict] | None = None,
    plot_cfg: dict | None = None,
) -> list[Path]:
    """Export current state based on requested formats."""
    out = ensure_outdir(outdir)
    requested = {str(fmt).lower() for fmt in formats}
    valid = {"npy", "csv", "png"}
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

    return written
