"""GUI simulation adapter independent from Streamlit runtime."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...config import GuiRunConfig, build_deck_from_gui_config
from ...deck import run_simulation_payload
from ...domain.grid import Grid2D
from ...mask import full_open_mask

_compute_physical_maps: Callable[..., dict[str, np.ndarray]] | None
try:
    from ...field_maps import compute_physical_maps as _compute_physical_maps
except ImportError:  # pragma: no cover - optional module support
    _compute_physical_maps = None


def _vertical_linecut(
    C: np.ndarray, grid: Grid2D, x_um: float
) -> tuple[np.ndarray, np.ndarray, float, int]:
    ix = grid.nearest_x_index(float(x_um))
    return grid.y_um, np.asarray(C[:, ix], dtype=float), float(grid.x_um[ix]), ix


def _horizontal_linecut(
    C: np.ndarray, grid: Grid2D, y_um: float
) -> tuple[np.ndarray, np.ndarray, float, int]:
    iy = grid.nearest_y_index(float(y_um))
    return grid.x_um, np.asarray(C[iy, :], dtype=float), float(grid.y_um[iy]), iy


def _make_zip(
    paths: list[Path], outdir: Path, filename: str = "outputs_bundle.zip"
) -> Path:
    zip_path = outdir / filename
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if path.exists() and path.is_file():
                zf.write(path, arcname=path.name)
    return zip_path


def _extract_electrical_metrics(metrics: Any) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    electrical = metrics.get("electrical")
    if isinstance(electrical, dict):
        return dict(electrical)
    return None


def _fallback_physical_maps(
    C: np.ndarray,
    grid: Grid2D,
    materials: np.ndarray,
    background_doping_cm3: float,
) -> dict[str, np.ndarray]:
    structure = np.asarray(materials)
    concentration = np.asarray(C, dtype=float)
    bg = max(float(background_doping_cm3), 1.0)
    ratio = np.clip(concentration, 1.0, None) / bg
    potential = 0.02585 * np.log(ratio)
    potential = potential - float(np.mean(potential[0, :]))

    gy, gx = np.gradient(potential, grid.dy_cm, grid.dx_cm)
    e_mag = np.sqrt(gx * gx + gy * gy)

    q_C = 1.602176634e-19
    mu_n_cm2_V_s = 1350.0
    sigma = q_C * mu_n_cm2_V_s * np.clip(concentration, 0.0, None)
    sigma = np.where(structure != 1, sigma, 1.0e-18)

    return {
        "structure": structure,
        "concentration_cm3": concentration,
        "potential_V": potential,
        "electric_field_mag_V_cm": e_mag,
        "conductivity_S_cm": sigma,
    }


def _compute_maps_from_state(
    *, C: np.ndarray, grid: Grid2D, materials: np.ndarray, background_doping_cm3: float
) -> dict[str, np.ndarray]:
    if _compute_physical_maps is not None:
        maps = _compute_physical_maps(
            C=C,
            grid=grid,
            materials=materials,
            background_doping_cm3=background_doping_cm3,
        )
        if isinstance(maps, dict):
            required = {
                "structure",
                "concentration_cm3",
                "potential_V",
                "electric_field_mag_V_cm",
                "conductivity_S_cm",
            }
            if required.issubset(maps.keys()):
                return {
                    "structure": np.asarray(maps["structure"]),
                    "concentration_cm3": np.asarray(
                        maps["concentration_cm3"], dtype=float
                    ),
                    "potential_V": np.asarray(maps["potential_V"], dtype=float),
                    "electric_field_mag_V_cm": np.asarray(
                        maps["electric_field_mag_V_cm"], dtype=float
                    ),
                    "conductivity_S_cm": np.asarray(
                        maps["conductivity_S_cm"], dtype=float
                    ),
                }

    return _fallback_physical_maps(
        C=C,
        grid=grid,
        materials=materials,
        background_doping_cm3=background_doping_cm3,
    )


def run_simulation(params: dict[str, Any]) -> dict[str, Any]:
    """Run one simulation from GUI parameters via shared simulation service."""
    t0 = time.perf_counter()

    cfg = GuiRunConfig.from_mapping(params)
    deck = build_deck_from_gui_config(cfg)

    outdir = Path(cfg.outdir)
    if not outdir.is_absolute():
        outdir = (Path.cwd() / outdir).resolve()

    state = run_simulation_payload(
        deck,
        deck_path=Path("__gui__.yaml"),
        out_override=outdir,
    )

    grid = state.grid
    C = np.asarray(state.C, dtype=float)
    mask_eff = (
        np.asarray(state.mask_eff, dtype=float)
        if state.mask_eff is not None
        else full_open_mask(grid.Nx)
    )
    tox_um = (
        np.asarray(state.tox_um, dtype=float)
        if state.tox_um is not None
        else np.zeros(grid.Nx, dtype=float)
    )

    cap_eps_um = float(cfg.cap_eps_um)
    if cap_eps_um <= 0.0:
        cap_eps_um = 0.5 * float(grid.dy_um)
    bc_gate = (tox_um <= cap_eps_um).astype(float)
    mask_eff_bc = np.asarray(mask_eff, dtype=float) * bc_gate

    plot_cfg = {
        "log10": bool(cfg.plot_log10),
        "vmin": float(cfg.plot_vmin) if cfg.plot_vmin is not None else None,
        "vmax": float(cfg.plot_vmax) if cfg.plot_vmax is not None else None,
    }

    written = list(state.exports)
    written_names = {Path(p).name for p in written}

    metrics_json_path = (
        outdir / "metrics.json" if "metrics.json" in written_names else None
    )
    metrics_csv_path = (
        outdir / "metrics.csv" if "metrics.csv" in written_names else None
    )
    sheet_dose_csv_path = (
        outdir / "sheet_dose_vs_x.csv"
        if "sheet_dose_vs_x.csv" in written_names
        else None
    )

    history_csv_path = (
        outdir / "history.csv" if "history.csv" in written_names else None
    )
    history_png_path = (
        outdir / "history.png" if "history.png" in written_names else None
    )

    vtk_path = outdir / "C.vtk" if "C.vtk" in written_names else None
    material_vtk_path = (
        outdir / "material.vtk" if "material.vtk" in written_names else None
    )
    tox_csv_path = outdir / "tox_vs_x.csv" if "tox_vs_x.csv" in written_names else None
    tox_png_path = outdir / "tox_vs_x.png" if "tox_vs_x.png" in written_names else None

    zip_path: Path | None = None
    if bool(cfg.zip_outputs):
        dedup_written = []
        seen: set[str] = set()
        for p in written:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            dedup_written.append(p)
        zip_path = _make_zip(dedup_written, outdir, filename="outputs_bundle.zip")
        written.append(zip_path)

    v_coord, v_values, x_used, ix = _vertical_linecut(C, grid, float(cfg.linecut_x_um))
    h_coord, h_values, y_used, iy = _horizontal_linecut(
        C, grid, float(cfg.linecut_y_um)
    )

    runtime_s = float(time.perf_counter() - t0)
    electrical_metrics = _extract_electrical_metrics(state.metrics)
    materials = (
        np.asarray(state.materials)
        if state.materials is not None
        else np.zeros(grid.shape, dtype=np.int8)
    )
    physical_maps = _compute_maps_from_state(
        C=C,
        grid=grid,
        materials=materials,
        background_doping_cm3=float(cfg.background_doping_cm3),
    )

    return {
        "grid": grid,
        "C": C,
        "mask_eff": mask_eff,
        "mask_eff_bc": mask_eff_bc,
        "tox_um": tox_um,
        "materials": materials,
        "physical_maps": physical_maps,
        "outdir": outdir,
        "written": written,
        "plot_cfg": plot_cfg,
        "metrics": state.metrics,
        "electrical_metrics": electrical_metrics,
        "metrics_json_path": metrics_json_path,
        "metrics_csv_path": metrics_csv_path,
        "sheet_dose_csv_path": sheet_dose_csv_path,
        "history": state.history,
        "history_csv_path": history_csv_path,
        "history_png_path": history_png_path,
        "vtk_path": vtk_path,
        "material_vtk_path": material_vtk_path,
        "tox_csv_path": tox_csv_path,
        "tox_png_path": tox_png_path,
        "zip_path": zip_path,
        "runtime_s": runtime_s,
        "linecuts": {
            "vertical": {
                "requested_x_um": float(cfg.linecut_x_um),
                "x_um_used": float(x_used),
                "ix": int(ix),
                "coord_um": v_coord,
                "values_cm3": v_values,
            },
            "horizontal": {
                "requested_y_um": float(cfg.linecut_y_um),
                "y_um_used": float(y_used),
                "iy": int(iy),
                "coord_um": h_coord,
                "values_cm3": h_values,
            },
        },
    }


__all__ = ["run_simulation"]
