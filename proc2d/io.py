"""Output helpers for simulation results."""

from __future__ import annotations

import csv
import json
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


def _flatten_metrics(prefix: str, value: object, rows: list[tuple[str, str]]) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _flatten_metrics(name, sub, rows)
        return
    if isinstance(value, list):
        rows.append((prefix, json.dumps(value, ensure_ascii=False)))
        return
    rows.append((prefix, f"{value}"))


def save_metrics_json(metrics: dict, outdir: str | Path, filename: str = "metrics.json") -> Path:
    """Save analysis metrics in JSON format."""
    path = ensure_outdir(outdir) / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


def save_metrics_csv(metrics: dict, outdir: str | Path, filename: str = "metrics.csv") -> Path:
    """Save analysis metrics in flattened key-value CSV format."""
    path = ensure_outdir(outdir) / filename
    rows: list[tuple[str, str]] = []
    _flatten_metrics("", metrics, rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in rows:
            writer.writerow([key, value])
    return path


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

    path = ensure_outdir(outdir) / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_um", "sheet_dose_cm2"])
        for x, sd in zip(x_arr, sd_arr):
            writer.writerow([f"{float(x):.12g}", f"{float(sd):.12g}"])
    return path


def save_history_csv(history: list[dict[str, float]], outdir: str | Path, filename: str = "history.csv") -> Path:
    """Save anneal history list into CSV."""
    path = ensure_outdir(outdir) / filename
    fieldnames: list[str] = []
    for row in history:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["time_s", "mass", "peak_cm3", "flux_out", "residual"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    return path


def save_history_png(history: list[dict[str, float]], outdir: str | Path, filename: str = "history.png") -> Path:
    """Save anneal history plot (mass / flux / residual vs time)."""
    if not history:
        raise ValueError("History is empty; nothing to plot.")

    t = np.asarray([float(row.get("time_s", np.nan)) for row in history], dtype=float)
    mass = np.asarray([float(row.get("mass", np.nan)) for row in history], dtype=float)
    flux = np.asarray([float(row.get("flux_out", np.nan)) for row in history], dtype=float)
    residual = np.asarray([float(row.get("residual", np.nan)) for row in history], dtype=float)

    path = ensure_outdir(outdir) / filename
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7.2, 6.0), dpi=140, sharex=True)

    axes[0].plot(t, mass, color="#1f77b4", lw=1.8)
    axes[0].set_ylabel("mass [cm^-1]")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, flux, color="#ff7f0e", lw=1.6)
    axes[1].set_ylabel("flux_out [cm^-1 s^-1]")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, residual, color="#2ca02c", lw=1.6)
    axes[2].set_ylabel("residual")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Anneal history", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_tox_vs_x_csv(tox_um: np.ndarray, grid: Grid2D, outdir: str | Path, filename: str = "tox_vs_x.csv") -> Path:
    """Save oxide thickness profile tox(x) as CSV."""
    tox = np.asarray(tox_um, dtype=float)
    if tox.shape != (grid.Nx,):
        raise ValueError(f"tox_um must have shape ({grid.Nx},), got {tox.shape}.")

    path = ensure_outdir(outdir) / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_um", "tox_um"])
        for x, t in zip(grid.x_um, tox):
            writer.writerow([f"{float(x):.12g}", f"{float(t):.12g}"])
    return path


def save_tox_vs_x_png(tox_um: np.ndarray, grid: Grid2D, outdir: str | Path, filename: str = "tox_vs_x.png") -> Path:
    """Save oxide thickness profile tox(x) as PNG."""
    tox = np.asarray(tox_um, dtype=float)
    if tox.shape != (grid.Nx,):
        raise ValueError(f"tox_um must have shape ({grid.Nx},), got {tox.shape}.")

    path = ensure_outdir(outdir) / filename
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=140)
    ax.plot(grid.x_um, tox, lw=2.0)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("tox [um]")
    ax.set_title("Oxide thickness profile")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
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

    out = ensure_outdir(outdir)
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
    out = ensure_outdir(outdir)
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
