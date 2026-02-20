"""Streamlit GUI for proc2d simulation."""

from __future__ import annotations

import csv
import datetime as dt
import time
import zipfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yaml

from .diffusion import TopBCConfig, anneal_implicit, anneal_implicit_with_history
from .grid import Grid2D
from .implant import implant_2d_gaussian
from .io import (
    export_results,
    save_history_csv,
    save_history_png,
    save_metrics_csv,
    save_metrics_json,
    save_sheet_dose_vs_x_csv,
)
from .mask import build_mask_1d, full_open_mask, openings_from_any, smooth_mask_1d, validate_mask
from .metrics import junction_depth, lateral_extents_at_y, peak_info, sheet_dose_vs_x, total_mass
from .oxidation import apply_oxidation, build_material_map

KB_EV_K = 8.617333262145e-5


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _example_deck_path() -> Path:
    return _project_root() / "examples" / "deck_basic.yaml"


def _first_step(steps: list[dict[str, Any]], stype: str) -> dict[str, Any] | None:
    for step in steps:
        if isinstance(step, dict) and str(step.get("type", "")).lower() == stype:
            return step
    return None


def _optional_mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping.")
    return value


def load_default_params() -> dict[str, Any]:
    """Load defaults from examples/deck_basic.yaml when available."""
    defaults: dict[str, Any] = {
        "Lx_um": 2.0,
        "Ly_um": 0.5,
        "Nx": 401,
        "Ny": 201,
        "background_doping_cm3": 1.0e15,
        "openings_um": [[0.8, 1.2]],
        "sigma_lat_um": 0.03,
        "dose_cm2": 1.0e13,
        "Rp_um": 0.05,
        "dRp_um": 0.02,
        "D_cm2_s": 1.0e-14,
        "total_t_s": 10.0,
        "dt_s": 0.5,
        "top_open_type": "robin",
        "h_cm_s": 1.0e-5,
        "Ceq_cm3": 0.0,
        "dirichlet_value_cm3": 0.0,
        "outdir": "outputs/gui_run",
        "formats": ["npy", "csv", "png"],
        "linecut_x_um": 1.0,
        "linecut_y_um": 0.05,
        "plot_log10": True,
        "plot_vmin": 1.0e14,
        "plot_vmax": 1.0e20,
        "compute_metrics": True,
        "record_history": False,
        "history_every_s": 0.5,
        "history_save_png": True,
        "history_save_csv": True,
        "export_vtk": False,
        "export_tox_csv": True,
        "export_tox_png": True,
        "zip_outputs": False,
        "store_full_c": False,
        "oxidation_enable": False,
        "oxidation_time_s": 5.0,
        "oxidation_A_um": 0.1,
        "oxidation_B_um2_s": 0.01,
        "oxidation_gamma": 2.27,
        "oxidation_apply_on": "all",
        "oxidation_consume_dopants": True,
        "oxidation_update_materials": True,
        "anneal_use_arrhenius": False,
        "arrhenius_D0_cm2_s": 1.0e-3,
        "arrhenius_Ea_eV": 3.5,
        "arrhenius_T_C": 1000.0,
        "oxide_D_scale": 0.0,
        "cap_eps_um": 0.0,
    }

    deck_path = _example_deck_path()
    if not deck_path.exists():
        return defaults

    try:
        with deck_path.open("r", encoding="utf-8") as f:
            deck = yaml.safe_load(f) or {}
    except Exception:
        return defaults

    if not isinstance(deck, dict):
        return defaults

    domain = deck.get("domain", {})
    if isinstance(domain, dict):
        defaults["Lx_um"] = float(domain.get("Lx_um", defaults["Lx_um"]))
        defaults["Ly_um"] = float(domain.get("Ly_um", defaults["Ly_um"]))
        defaults["Nx"] = int(domain.get("Nx", defaults["Nx"]))
        defaults["Ny"] = int(domain.get("Ny", defaults["Ny"]))
        defaults["background_doping_cm3"] = float(
            deck.get("background_doping_cm3", domain.get("background_doping_cm3", defaults["background_doping_cm3"]))
        )

    raw_steps = deck.get("steps", [])
    steps = raw_steps if isinstance(raw_steps, list) else []

    mask_step = _first_step(steps, "mask")
    if mask_step:
        defaults["openings_um"] = mask_step.get("openings_um", defaults["openings_um"])
        defaults["sigma_lat_um"] = float(mask_step.get("sigma_lat_um", defaults["sigma_lat_um"]))

    implant_step = _first_step(steps, "implant")
    if implant_step:
        defaults["dose_cm2"] = float(implant_step.get("dose_cm2", defaults["dose_cm2"]))
        defaults["Rp_um"] = float(implant_step.get("Rp_um", defaults["Rp_um"]))
        defaults["dRp_um"] = float(implant_step.get("dRp_um", defaults["dRp_um"]))

    oxidation_step = _first_step(steps, "oxidation")
    if oxidation_step:
        defaults["oxidation_enable"] = True
        defaults["oxidation_time_s"] = float(oxidation_step.get("time_s", defaults["oxidation_time_s"]))
        defaults["oxidation_A_um"] = float(oxidation_step.get("A_um", defaults["oxidation_A_um"]))
        defaults["oxidation_B_um2_s"] = float(oxidation_step.get("B_um2_s", defaults["oxidation_B_um2_s"]))
        defaults["oxidation_gamma"] = float(oxidation_step.get("gamma", defaults["oxidation_gamma"]))
        defaults["oxidation_apply_on"] = str(oxidation_step.get("apply_on", defaults["oxidation_apply_on"]))
        defaults["oxidation_consume_dopants"] = bool(
            oxidation_step.get("consume_dopants", defaults["oxidation_consume_dopants"])
        )
        defaults["oxidation_update_materials"] = bool(
            oxidation_step.get("update_materials", defaults["oxidation_update_materials"])
        )

    anneal_step = _first_step(steps, "anneal")
    if anneal_step:
        if "D_cm2_s" in anneal_step:
            defaults["D_cm2_s"] = float(anneal_step.get("D_cm2_s", defaults["D_cm2_s"]))
        defaults["total_t_s"] = float(anneal_step.get("total_t_s", defaults["total_t_s"]))
        defaults["dt_s"] = float(anneal_step.get("dt_s", defaults["dt_s"]))

        diff_cfg = anneal_step.get("diffusivity", {})
        if isinstance(diff_cfg, dict) and str(diff_cfg.get("model", "")).lower() == "arrhenius":
            defaults["anneal_use_arrhenius"] = True
            defaults["arrhenius_D0_cm2_s"] = float(diff_cfg.get("D0_cm2_s", defaults["arrhenius_D0_cm2_s"]))
            defaults["arrhenius_Ea_eV"] = float(diff_cfg.get("Ea_eV", defaults["arrhenius_Ea_eV"]))
            if "T_C" in diff_cfg:
                defaults["arrhenius_T_C"] = float(diff_cfg["T_C"])

        oxide_cfg = anneal_step.get("oxide", {})
        if isinstance(oxide_cfg, dict):
            defaults["oxide_D_scale"] = float(oxide_cfg.get("D_scale", defaults["oxide_D_scale"]))

        defaults["cap_eps_um"] = float(anneal_step.get("cap_eps_um", defaults["cap_eps_um"]))
        top_bc = anneal_step.get("top_bc", {})
        if isinstance(top_bc, dict):
            open_cfg = top_bc.get("open", {})
            if isinstance(open_cfg, dict):
                defaults["top_open_type"] = str(open_cfg.get("type", defaults["top_open_type"]))
                defaults["h_cm_s"] = float(open_cfg.get("h_cm_s", defaults["h_cm_s"]))
                defaults["Ceq_cm3"] = float(open_cfg.get("Ceq_cm3", defaults["Ceq_cm3"]))
                defaults["dirichlet_value_cm3"] = float(
                    open_cfg.get("value_cm3", defaults["dirichlet_value_cm3"])
                )

    export_step = _first_step(steps, "export")
    if export_step:
        defaults["outdir"] = str(export_step.get("outdir", defaults["outdir"]))
        raw_formats = export_step.get("formats", defaults["formats"])
        if isinstance(raw_formats, list) and raw_formats:
            defaults["formats"] = [str(x).lower() for x in raw_formats]
        linecuts = export_step.get("linecuts", [])
        if isinstance(linecuts, list):
            for linecut in linecuts:
                if not isinstance(linecut, dict):
                    continue
                kind = str(linecut.get("kind", "")).lower()
                if kind == "vertical" and "x_um" in linecut:
                    defaults["linecut_x_um"] = float(linecut["x_um"])
                if kind == "horizontal" and "y_um" in linecut:
                    defaults["linecut_y_um"] = float(linecut["y_um"])
        plot_cfg = export_step.get("plot", {})
        if isinstance(plot_cfg, dict):
            defaults["plot_log10"] = bool(plot_cfg.get("log10", defaults["plot_log10"]))
            if "vmin" in plot_cfg:
                defaults["plot_vmin"] = float(plot_cfg["vmin"])
            if "vmax" in plot_cfg:
                defaults["plot_vmax"] = float(plot_cfg["vmax"])

        extra_cfg = export_step.get("extra", {})
        if isinstance(extra_cfg, dict):
            defaults["export_tox_csv"] = bool(extra_cfg.get("tox_csv", defaults["export_tox_csv"]))
            defaults["export_tox_png"] = bool(extra_cfg.get("tox_png", defaults["export_tox_png"]))

    return defaults


def _parse_openings_text(openings_text: str) -> list[list[float]]:
    if openings_text.strip() == "":
        return []
    try:
        parsed = yaml.safe_load(openings_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"openings_um parsing failed: {exc}") from exc

    if parsed is None:
        return []
    if not isinstance(parsed, list):
        raise ValueError("openings_um must be a list, for example: [[0.8, 1.2]]")
    return openings_from_any(parsed)


def _make_top_bc(params: dict[str, Any]) -> TopBCConfig:
    open_type = str(params["top_open_type"]).lower()
    if open_type == "robin":
        return TopBCConfig(
            open_type="robin",
            blocked_type="neumann",
            h_cm_s=float(params["h_cm_s"]),
            Ceq_cm3=float(params["Ceq_cm3"]),
        )
    if open_type == "dirichlet":
        return TopBCConfig(
            open_type="dirichlet",
            blocked_type="neumann",
            dirichlet_value_cm3=float(params["dirichlet_value_cm3"]),
        )
    return TopBCConfig(open_type="neumann", blocked_type="neumann")


def _mask_segments(mask_eff: np.ndarray, x_um: np.ndarray, threshold: float = 0.5) -> list[tuple[float, float]]:
    open_mask = np.asarray(mask_eff, dtype=float) >= float(threshold)
    segments: list[tuple[float, float]] = []
    start: int | None = None
    for i, is_open in enumerate(open_mask):
        if is_open and start is None:
            start = i
        elif (not is_open) and start is not None:
            segments.append((float(x_um[start]), float(x_um[i - 1])))
            start = None
    if start is not None:
        segments.append((float(x_um[start]), float(x_um[-1])))
    return segments


def _heatmap_figure(
    C: np.ndarray,
    grid: Grid2D,
    mask_eff: np.ndarray | None,
    tox_um: np.ndarray | None,
    log10: bool,
    vmin: float | None,
    vmax: float | None,
):
    arr = np.asarray(C, dtype=float)
    if log10:
        floor = 1.0e10
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

    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=140)
    im = ax.imshow(
        plot_arr,
        extent=[float(grid.x_um[0]), float(grid.x_um[-1]), float(grid.y_um[-1]), float(grid.y_um[0])],
        origin="upper",
        aspect="auto",
        cmap="inferno",
        vmin=vmin_plot,
        vmax=vmax_plot,
    )
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title("Final concentration")

    if mask_eff is not None:
        for x0, x1 in _mask_segments(mask_eff, grid.x_um):
            ax.plot([x0, x1], [0.0, 0.0], color="#4dd0e1", lw=4.0, solid_capstyle="butt")

    if tox_um is not None:
        tox = np.asarray(tox_um, dtype=float)
        if tox.shape == (grid.Nx,):
            ax.plot(grid.x_um, tox, color="#00e5ff", lw=1.6, alpha=0.9, label="tox")
            ax.legend(loc="upper right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig


def _vertical_linecut(C: np.ndarray, grid: Grid2D, x_um: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    ix = grid.nearest_x_index(float(x_um))
    return grid.y_um, np.asarray(C[:, ix], dtype=float), float(grid.x_um[ix]), ix


def _horizontal_linecut(C: np.ndarray, grid: Grid2D, y_um: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    iy = grid.nearest_y_index(float(y_um))
    return grid.x_um, np.asarray(C[iy, :], dtype=float), float(grid.y_um[iy]), iy


def _linecut_csv_text(
    kind: str,
    req_value: float,
    used_value: float,
    coord_name: str,
    coord_values: np.ndarray,
    c_values: np.ndarray,
) -> str:
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["kind", kind])
    if kind == "vertical":
        writer.writerow(["x_um_requested", f"{req_value:.12g}"])
        writer.writerow(["x_um_used", f"{used_value:.12g}"])
    else:
        writer.writerow(["y_um_requested", f"{req_value:.12g}"])
        writer.writerow(["y_um_used", f"{used_value:.12g}"])
    writer.writerow([coord_name, "C_cm3"])
    for coord, c in zip(coord_values, c_values):
        writer.writerow([f"{float(coord):.12g}", f"{float(c):.12g}"])
    return buf.getvalue()


def _history_figure(history: list[dict[str, float]]):
    def _safe_float(value: object) -> float:
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        try:
            return float(str(value))
        except Exception:
            return float("nan")

    t = np.asarray([_safe_float(row.get("time_s", np.nan)) for row in history], dtype=float)
    mass = np.asarray([_safe_float(row.get("mass", np.nan)) for row in history], dtype=float)
    flux = np.asarray([_safe_float(row.get("flux_out", np.nan)) for row in history], dtype=float)
    residual = np.asarray([_safe_float(row.get("residual", np.nan)) for row in history], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.0), dpi=140, sharex=True)
    axes[0].plot(t, mass, lw=1.8)
    axes[0].set_ylabel("mass [cm^-1]")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, flux, lw=1.6, color="#ff7f0e")
    axes[1].set_ylabel("flux_out")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, residual, lw=1.6, color="#2ca02c")
    axes[2].set_ylabel("residual")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def _scalarize_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_scalarize_metrics(value, prefix=name))
        elif isinstance(value, (int, float, np.floating)):
            out[name] = float(value)
    return out


def _make_zip(paths: list[Path], outdir: Path, filename: str = "outputs_bundle.zip") -> Path:
    zip_path = outdir / filename
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if path.exists() and path.is_file():
                zf.write(path, arcname=path.name)
    return zip_path


def _compute_metrics_report(C: np.ndarray, grid: Grid2D, x_ref_um: float, y_ref_um: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "total_mass_cm1": float(total_mass(C, grid)),
        "peak": peak_info(C, grid),
    }
    junctions: list[dict[str, Any]] = []
    for threshold in (1.0e17, 1.0e18):
        depth = junction_depth(C, grid, x_um=float(x_ref_um), threshold_cm3=float(threshold))
        junctions.append(
            {
                "x_um_requested": float(x_ref_um),
                "threshold_cm3": float(threshold),
                "depth_um": None if depth is None else float(depth),
            }
        )
    report["junctions"] = junctions
    report["lateral_at_y"] = lateral_extents_at_y(C, grid, y_um=float(y_ref_um), threshold_cm3=1.0e17)

    sd = sheet_dose_vs_x(C, grid)
    report["sheet_dose_summary"] = {
        "min_cm2": float(np.min(sd)),
        "max_cm2": float(np.max(sd)),
        "mean_cm2": float(np.mean(sd)),
    }
    return report


def run_simulation(params: dict[str, Any]) -> dict[str, Any]:
    """Run one simulation from GUI parameters."""
    t0 = time.perf_counter()

    grid = Grid2D.from_domain(
        Lx_um=float(params["Lx_um"]),
        Ly_um=float(params["Ly_um"]),
        Nx=int(params["Nx"]),
        Ny=int(params["Ny"]),
    )

    C = np.full(grid.shape, float(params["background_doping_cm3"]), dtype=float)
    tox_um = np.zeros(grid.Nx, dtype=float)
    materials = build_material_map(grid, tox_um)

    openings_um = params["openings_um"]
    if openings_um:
        mask_raw = build_mask_1d(grid.x_um, openings_um)
        mask_eff = smooth_mask_1d(
            mask_raw,
            sigma_lat_um=float(params["sigma_lat_um"]),
            dx_um=grid.dx_um,
        )
    else:
        mask_eff = full_open_mask(grid.Nx)
    validate_mask(mask_eff, grid.Nx)

    if bool(params.get("oxidation_enable", False)):
        C, tox_new, materials_new, _ = apply_oxidation(
            C,
            grid,
            tox_um,
            mask_eff,
            time_s=float(params["oxidation_time_s"]),
            A_um=float(params["oxidation_A_um"]),
            B_um2_s=float(params["oxidation_B_um2_s"]),
            gamma=float(params["oxidation_gamma"]),
            apply_on=str(params["oxidation_apply_on"]),
            consume_dopants=bool(params["oxidation_consume_dopants"]),
        )
        if np.any(tox_new > float(grid.Ly_um) + 1e-12):
            raise ValueError("oxidation: tox exceeds domain Ly_um")
        tox_um = tox_new
        if bool(params.get("oxidation_update_materials", True)):
            materials = materials_new

    C += implant_2d_gaussian(
        grid=grid,
        dose_cm2=float(params["dose_cm2"]),
        Rp_um=float(params["Rp_um"]),
        dRp_um=float(params["dRp_um"]),
        mask_eff=mask_eff,
        tox_um=tox_um,
    )

    top_bc = _make_top_bc(params)
    cap_eps_um = float(params.get("cap_eps_um", 0.0))
    if cap_eps_um <= 0.0:
        cap_eps_um = 0.5 * float(grid.dy_um)
    bc_gate = (tox_um <= cap_eps_um).astype(float)
    mask_eff_bc = np.asarray(mask_eff, dtype=float) * bc_gate

    if bool(params.get("anneal_use_arrhenius", False)):
        D0 = float(params["arrhenius_D0_cm2_s"])
        Ea = float(params["arrhenius_Ea_eV"])
        T_C = float(params["arrhenius_T_C"])
        T_K = T_C + 273.15
        if T_K <= 0.0:
            raise ValueError("Arrhenius temperature must satisfy T_C > -273.15")
        D_si = D0 * float(np.exp(-Ea / (KB_EV_K * T_K)))
    else:
        D_si = float(params["D_cm2_s"])

    oxide_D_scale = float(params.get("oxide_D_scale", 0.0))
    if oxide_D_scale < 0.0:
        raise ValueError("oxide_D_scale must be >= 0")
    D_field = np.full(grid.shape, D_si, dtype=float)
    D_field[np.asarray(materials) == 1] *= oxide_D_scale

    history: list[dict[str, float]] = []
    if bool(params["record_history"]):
        C, history = anneal_implicit_with_history(
            C0=C,
            grid=grid,
            D_cm2_s=D_field,
            total_t_s=float(params["total_t_s"]),
            dt_s=float(params["dt_s"]),
            top_bc=top_bc,
            mask_eff=mask_eff_bc,
            record_enable=True,
            record_every_s=float(params["history_every_s"]),
        )
    else:
        C = anneal_implicit(
            C0=C,
            grid=grid,
            D_cm2_s=D_field,
            total_t_s=float(params["total_t_s"]),
            dt_s=float(params["dt_s"]),
            top_bc=top_bc,
            mask_eff=mask_eff_bc,
        )

    outdir = Path(str(params["outdir"]))
    if not outdir.is_absolute():
        outdir = (Path.cwd() / outdir).resolve()

    formats = [str(x).lower() for x in params["formats"]]
    if bool(params["export_vtk"]) and "vtk" not in formats:
        formats.append("vtk")
    plot_cfg = {
        "log10": bool(params["plot_log10"]),
        "vmin": float(params["plot_vmin"]) if params["plot_vmin"] is not None else None,
        "vmax": float(params["plot_vmax"]) if params["plot_vmax"] is not None else None,
    }
    linecuts = [
        {"kind": "vertical", "x_um": float(params["linecut_x_um"])},
        {"kind": "horizontal", "y_um": float(params["linecut_y_um"])},
    ]

    written = export_results(
        C=C,
        grid=grid,
        outdir=outdir,
        formats=formats,
        linecuts=linecuts,
        plot_cfg=plot_cfg,
        tox_um=tox_um,
        materials=materials,
        extra={
            "tox_csv": bool(params.get("export_tox_csv", False)),
            "tox_png": bool(params.get("export_tox_png", False)),
        },
    )

    metrics_report: dict[str, Any] | None = None
    metrics_json_path: Path | None = None
    metrics_csv_path: Path | None = None
    sheet_dose_csv_path: Path | None = None
    if bool(params["compute_metrics"]):
        metrics_report = _compute_metrics_report(
            C,
            grid,
            x_ref_um=float(params["linecut_x_um"]),
            y_ref_um=float(params["linecut_y_um"]),
        )
        metrics_json_path = save_metrics_json(metrics_report, outdir, filename="metrics.json")
        metrics_csv_path = save_metrics_csv(metrics_report, outdir, filename="metrics.csv")
        sd = sheet_dose_vs_x(C, grid)
        sheet_dose_csv_path = save_sheet_dose_vs_x_csv(grid.x_um, sd, outdir, filename="sheet_dose_vs_x.csv")
        written.extend([metrics_json_path, metrics_csv_path, sheet_dose_csv_path])

    history_csv_path: Path | None = None
    history_png_path: Path | None = None
    if history and bool(params["history_save_csv"]):
        history_csv_path = save_history_csv(history, outdir, filename="history.csv")
        written.append(history_csv_path)
    if history and bool(params["history_save_png"]):
        history_png_path = save_history_png(history, outdir, filename="history.png")
        written.append(history_png_path)

    vtk_path = outdir / "C.vtk"
    vtk_path = vtk_path if ("vtk" in formats and vtk_path.exists()) else None
    material_vtk_path = outdir / "material.vtk"
    material_vtk_path = material_vtk_path if material_vtk_path.exists() else None
    tox_csv_path = outdir / "tox_vs_x.csv"
    tox_csv_path = tox_csv_path if tox_csv_path.exists() else None
    tox_png_path = outdir / "tox_vs_x.png"
    tox_png_path = tox_png_path if tox_png_path.exists() else None

    zip_path: Path | None = None
    if bool(params["zip_outputs"]):
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

    v_coord, v_values, x_used, ix = _vertical_linecut(C, grid, float(params["linecut_x_um"]))
    h_coord, h_values, y_used, iy = _horizontal_linecut(C, grid, float(params["linecut_y_um"]))

    runtime_s = float(time.perf_counter() - t0)

    return {
        "grid": grid,
        "C": C,
        "mask_eff": mask_eff,
        "mask_eff_bc": mask_eff_bc,
        "tox_um": tox_um,
        "materials": materials,
        "outdir": outdir,
        "written": written,
        "plot_cfg": plot_cfg,
        "metrics": metrics_report,
        "metrics_json_path": metrics_json_path,
        "metrics_csv_path": metrics_csv_path,
        "sheet_dose_csv_path": sheet_dose_csv_path,
        "history": history,
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
                "requested_x_um": float(params["linecut_x_um"]),
                "x_um_used": float(x_used),
                "ix": int(ix),
                "coord_um": v_coord,
                "values_cm3": v_values,
            },
            "horizontal": {
                "requested_y_um": float(params["linecut_y_um"]),
                "y_um_used": float(y_used),
                "iy": int(iy),
                "coord_um": h_coord,
                "values_cm3": h_values,
            },
        },
    }


def _push_recent_run(result: dict[str, Any], params: dict[str, Any]) -> None:
    runs = st.session_state.get("proc2d_recent_runs", [])
    if not isinstance(runs, list):
        runs = []

    outdir = Path(result["outdir"])
    png_path = outdir / "C.png"
    png_written = any(Path(p).name == "C.png" for p in result.get("written", []))
    png_bytes = png_path.read_bytes() if (png_written and png_path.exists()) else b""

    store_full = bool(params.get("store_full_c", False))
    run_record = {
        "run_id": dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "params": dict(params),
        "plot_cfg": dict(result.get("plot_cfg", {})),
        "png_bytes": png_bytes,
        "metrics": result.get("metrics"),
        "linecuts": {
            "vertical": {
                "coord_um": np.asarray(result["linecuts"]["vertical"]["coord_um"], dtype=float),
                "values_cm3": np.asarray(result["linecuts"]["vertical"]["values_cm3"], dtype=float),
            },
            "horizontal": {
                "coord_um": np.asarray(result["linecuts"]["horizontal"]["coord_um"], dtype=float),
                "values_cm3": np.asarray(result["linecuts"]["horizontal"]["values_cm3"], dtype=float),
            },
        },
        "grid": {
            "Lx_um": float(result["grid"].Lx_um),
            "Ly_um": float(result["grid"].Ly_um),
            "Nx": int(result["grid"].Nx),
            "Ny": int(result["grid"].Ny),
        },
        "C": np.asarray(result["C"], dtype=float).copy() if store_full else None,
    }
    runs.append(run_record)
    runs = runs[-2:]
    st.session_state["proc2d_recent_runs"] = runs


def _linecut_plot(
    coord_um: np.ndarray,
    values_cm3: np.ndarray,
    x_label: str,
    title: str,
    scale: str,
    log_floor: float,
):
    y = np.asarray(values_cm3, dtype=float)
    if scale == "log10":
        y_plot = np.log10(np.clip(y, max(1e-30, float(log_floor)), None))
        y_label = "log10(C [cm^-3])"
    else:
        y_plot = y
        y_label = "C [cm^-3]"

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
    ax.plot(np.asarray(coord_um, dtype=float), y_plot, lw=2.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _compare_two_runs_tab() -> None:
    runs = st.session_state.get("proc2d_recent_runs", [])
    if not isinstance(runs, list) or len(runs) < 2:
        st.info("비교하려면 최소 2회 실행이 필요합니다.")
        return

    run_a = runs[-2]
    run_b = runs[-1]

    st.write(f"A: `{run_a['run_id']}` ({run_a['timestamp']})")
    st.write(f"B: `{run_b['run_id']}` ({run_b['timestamp']})")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Run A map")
        if run_a.get("C") is not None and run_b.get("C") is not None:
            # Render with same color scale when full fields are stored.
            C_a = np.asarray(run_a["C"], dtype=float)
            C_b = np.asarray(run_b["C"], dtype=float)
            vmin = min(float(np.min(C_a)), float(np.min(C_b)))
            vmax = max(float(np.max(C_a)), float(np.max(C_b)))
            fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=130)
            ax.imshow(C_a, origin="upper", aspect="auto", cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_title("A (same scale)")
            ax.set_axis_off()
            st.pyplot(fig)
            plt.close(fig)
        elif run_a.get("png_bytes"):
            st.image(run_a["png_bytes"], use_container_width=True)
        else:
            st.info("Run A image unavailable (PNG 미저장 + full C 미보관)")

    with col2:
        st.caption("Run B map")
        if run_a.get("C") is not None and run_b.get("C") is not None:
            C_a = np.asarray(run_a["C"], dtype=float)
            C_b = np.asarray(run_b["C"], dtype=float)
            vmin = min(float(np.min(C_a)), float(np.min(C_b)))
            vmax = max(float(np.max(C_a)), float(np.max(C_b)))
            fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=130)
            ax.imshow(C_b, origin="upper", aspect="auto", cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_title("B (same scale)")
            ax.set_axis_off()
            st.pyplot(fig)
            plt.close(fig)
        elif run_b.get("png_bytes"):
            st.image(run_b["png_bytes"], use_container_width=True)
        else:
            st.info("Run B image unavailable (PNG 미저장 + full C 미보관)")

    linecut_kind = st.radio(
        "Compare linecut",
        ["Vertical", "Horizontal"],
        horizontal=True,
        key="compare-linecut-kind",
    )
    if linecut_kind == "Vertical":
        a_x = np.asarray(run_a["linecuts"]["vertical"]["coord_um"], dtype=float)
        a_y = np.asarray(run_a["linecuts"]["vertical"]["values_cm3"], dtype=float)
        b_x = np.asarray(run_b["linecuts"]["vertical"]["coord_um"], dtype=float)
        b_y = np.asarray(run_b["linecuts"]["vertical"]["values_cm3"], dtype=float)
        x_label = "y [um]"
    else:
        a_x = np.asarray(run_a["linecuts"]["horizontal"]["coord_um"], dtype=float)
        a_y = np.asarray(run_a["linecuts"]["horizontal"]["values_cm3"], dtype=float)
        b_x = np.asarray(run_b["linecuts"]["horizontal"]["coord_um"], dtype=float)
        b_y = np.asarray(run_b["linecuts"]["horizontal"]["values_cm3"], dtype=float)
        x_label = "x [um]"

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=140)
    ax.plot(a_x, a_y, lw=1.8, label=f"A ({run_a['run_id']})")
    ax.plot(b_x, b_y, lw=1.8, label=f"B ({run_b['run_id']})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("C [cm^-3]")
    ax.set_title("Linecut overlay")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    metrics_a = run_a.get("metrics") or {}
    metrics_b = run_b.get("metrics") or {}
    if metrics_a and metrics_b:
        scalar_a = _scalarize_metrics(metrics_a)
        scalar_b = _scalarize_metrics(metrics_b)
        rows: list[dict[str, str | float]] = []
        for key in sorted(set(scalar_a) & set(scalar_b)):
            a_val = scalar_a[key]
            b_val = scalar_b[key]
            rows.append({
                "metric": key,
                "A": f"{a_val:.6g}",
                "B": f"{b_val:.6g}",
                "delta(B-A)": f"{(b_val - a_val):.6g}",
            })
        st.subheader("Metrics difference")
        st.table(rows)


def run_gui() -> None:
    """Render Streamlit app."""
    st.set_page_config(page_title="proc2d GUI", layout="wide")
    st.title("proc2d: Process 2D Cross-Section GUI")
    st.caption(
        "예제 deck 기본값을 로드해 파라미터를 조정하고, 같은 화면에서 "
        "맵/라인컷/메트릭/히스토리/비교를 확인할 수 있습니다."
    )

    defaults = load_default_params()
    openings_text_default = yaml.safe_dump(defaults["openings_um"], sort_keys=False).strip()

    with st.sidebar:
        st.header("Simulation Parameters")
        with st.form("proc2d-form"):
            st.subheader("Domain")
            Lx_um = st.number_input("Lx_um", min_value=1e-6, value=float(defaults["Lx_um"]), format="%.6g")
            Ly_um = st.number_input("Ly_um", min_value=1e-6, value=float(defaults["Ly_um"]), format="%.6g")
            Nx = st.number_input("Nx", min_value=2, value=int(defaults["Nx"]), step=1)
            Ny = st.number_input("Ny", min_value=2, value=int(defaults["Ny"]), step=1)
            background_doping_cm3 = st.number_input(
                "background_doping_cm3",
                min_value=0.0,
                value=float(defaults["background_doping_cm3"]),
                format="%.6g",
            )

            st.subheader("Mask")
            openings_text = st.text_area(
                "openings_um (YAML/JSON list)",
                value=openings_text_default,
                height=90,
                help="예: [[0.8, 1.2], [1.6, 1.8]]",
            )
            sigma_lat_um = st.number_input("sigma_lat_um", min_value=0.0, value=float(defaults["sigma_lat_um"]), format="%.6g")

            st.subheader("Implant")
            dose_cm2 = st.number_input("dose_cm2", min_value=0.0, value=float(defaults["dose_cm2"]), format="%.6g")
            Rp_um = st.number_input("Rp_um", value=float(defaults["Rp_um"]), format="%.6g")
            dRp_um = st.number_input("dRp_um", min_value=1e-12, value=float(defaults["dRp_um"]), format="%.6g")

            st.subheader("Oxidation (optional)")
            oxidation_enable = st.checkbox("Enable oxidation step", value=bool(defaults["oxidation_enable"]))
            oxidation_time_s = st.number_input(
                "oxidation.time_s",
                min_value=0.0,
                value=float(defaults["oxidation_time_s"]),
                format="%.6g",
                disabled=not oxidation_enable,
            )
            oxidation_A_um = st.number_input(
                "oxidation.A_um",
                min_value=0.0,
                value=float(defaults["oxidation_A_um"]),
                format="%.6g",
                disabled=not oxidation_enable,
            )
            oxidation_B_um2_s = st.number_input(
                "oxidation.B_um2_s",
                min_value=0.0,
                value=float(defaults["oxidation_B_um2_s"]),
                format="%.6g",
                disabled=not oxidation_enable,
            )
            oxidation_gamma = st.number_input(
                "oxidation.gamma",
                min_value=1e-12,
                value=float(defaults["oxidation_gamma"]),
                format="%.6g",
                disabled=not oxidation_enable,
            )
            apply_on_options = ["all", "open", "blocked"]
            default_apply_on = str(defaults["oxidation_apply_on"]).lower()
            if default_apply_on not in apply_on_options:
                default_apply_on = "all"
            oxidation_apply_on = st.selectbox(
                "oxidation.apply_on",
                options=apply_on_options,
                index=apply_on_options.index(default_apply_on),
                disabled=not oxidation_enable,
            )
            oxidation_consume_dopants = st.checkbox(
                "oxidation.consume_dopants",
                value=bool(defaults["oxidation_consume_dopants"]),
                disabled=not oxidation_enable,
            )
            oxidation_update_materials = st.checkbox(
                "oxidation.update_materials",
                value=bool(defaults["oxidation_update_materials"]),
                disabled=not oxidation_enable,
            )

            st.subheader("Anneal")
            D_cm2_s = st.number_input("D_cm2_s", min_value=0.0, value=float(defaults["D_cm2_s"]), format="%.6g")
            total_t_s = st.number_input("total_t_s", min_value=0.0, value=float(defaults["total_t_s"]), format="%.6g")
            dt_s = st.number_input("dt_s", min_value=1e-12, value=float(defaults["dt_s"]), format="%.6g")
            anneal_use_arrhenius = st.checkbox(
                "anneal.diffusivity.model=arrhenius",
                value=bool(defaults["anneal_use_arrhenius"]),
            )
            arrhenius_D0_cm2_s = st.number_input(
                "arrhenius.D0_cm2_s",
                min_value=0.0,
                value=float(defaults["arrhenius_D0_cm2_s"]),
                format="%.6g",
                disabled=not anneal_use_arrhenius,
            )
            arrhenius_Ea_eV = st.number_input(
                "arrhenius.Ea_eV",
                min_value=0.0,
                value=float(defaults["arrhenius_Ea_eV"]),
                format="%.6g",
                disabled=not anneal_use_arrhenius,
            )
            arrhenius_T_C = st.number_input(
                "arrhenius.T_C",
                value=float(defaults["arrhenius_T_C"]),
                format="%.6g",
                disabled=not anneal_use_arrhenius,
            )
            oxide_D_scale = st.number_input(
                "anneal.oxide.D_scale",
                min_value=0.0,
                value=float(defaults["oxide_D_scale"]),
                format="%.6g",
            )
            cap_eps_um = st.number_input(
                "anneal.cap_eps_um",
                min_value=0.0,
                value=float(defaults["cap_eps_um"]),
                format="%.6g",
            )

            open_types = ["robin", "neumann", "dirichlet"]
            default_open_type = str(defaults["top_open_type"]).lower()
            open_type_idx = open_types.index(default_open_type) if default_open_type in open_types else 0
            top_open_type = st.selectbox("top_bc.open.type", options=open_types, index=open_type_idx)
            h_cm_s = 0.0
            Ceq_cm3 = 0.0
            dirichlet_value_cm3 = 0.0
            if top_open_type == "robin":
                h_cm_s = st.number_input("h_cm_s", min_value=0.0, value=float(defaults["h_cm_s"]), format="%.6g")
                Ceq_cm3 = st.number_input("Ceq_cm3", value=float(defaults["Ceq_cm3"]), format="%.6g")
            if top_open_type == "dirichlet":
                dirichlet_value_cm3 = st.number_input(
                    "dirichlet value_cm3",
                    value=float(defaults["dirichlet_value_cm3"]),
                    format="%.6g",
                )

            st.subheader("Export")
            outdir = st.text_input("outdir", value=str(defaults["outdir"]))
            all_formats = ["npy", "csv", "png"]
            default_formats = [f for f in defaults["formats"] if f in all_formats]
            formats = st.multiselect("formats", options=all_formats, default=default_formats)
            linecut_x_um = st.number_input(
                "linecut x_um",
                min_value=0.0,
                max_value=float(Lx_um),
                value=float(min(max(defaults["linecut_x_um"], 0.0), Lx_um)),
                format="%.6g",
            )
            linecut_y_um = st.number_input(
                "linecut y_um",
                min_value=0.0,
                max_value=float(Ly_um),
                value=float(min(max(defaults["linecut_y_um"], 0.0), Ly_um)),
                format="%.6g",
            )
            plot_log10 = st.checkbox("plot.log10", value=bool(defaults["plot_log10"]))
            plot_vmin = st.number_input("plot.vmin", min_value=1e-30, value=float(defaults["plot_vmin"]), format="%.6g")
            plot_vmax = st.number_input("plot.vmax", min_value=1e-30, value=float(defaults["plot_vmax"]), format="%.6g")

            st.subheader("Run options")
            compute_metrics = st.checkbox("Compute metrics (analyze)", value=bool(defaults["compute_metrics"]))
            record_history = st.checkbox("Record anneal history", value=bool(defaults["record_history"]))
            history_every_s = st.number_input(
                "history every_s",
                min_value=1e-12,
                value=float(defaults["history_every_s"]),
                format="%.6g",
            )
            history_save_csv = st.checkbox("history save CSV", value=bool(defaults["history_save_csv"]))
            history_save_png = st.checkbox("history save PNG", value=bool(defaults["history_save_png"]))
            export_vtk = st.checkbox("Export VTK", value=bool(defaults["export_vtk"]))
            export_tox_csv = st.checkbox("Export tox profile CSV", value=bool(defaults["export_tox_csv"]))
            export_tox_png = st.checkbox("Export tox profile PNG", value=bool(defaults["export_tox_png"]))
            zip_outputs = st.checkbox("Download all outputs as ZIP", value=bool(defaults["zip_outputs"]))
            store_full_c = st.checkbox(
                "Store full C in session (for high-fidelity compare)",
                value=bool(defaults["store_full_c"]),
            )

            submitted = st.form_submit_button("Run Simulation")

    if submitted:
        try:
            openings_um = _parse_openings_text(openings_text)
            if float(plot_vmin) >= float(plot_vmax):
                raise ValueError("plot.vmin must be smaller than plot.vmax")
            if bool(plot_log10) and (float(plot_vmin) <= 0.0 or float(plot_vmax) <= 0.0):
                raise ValueError("plot.vmin and plot.vmax must be > 0 when plot.log10 is enabled")
            params = {
                "Lx_um": float(Lx_um),
                "Ly_um": float(Ly_um),
                "Nx": int(Nx),
                "Ny": int(Ny),
                "background_doping_cm3": float(background_doping_cm3),
                "openings_um": openings_um,
                "sigma_lat_um": float(sigma_lat_um),
                "dose_cm2": float(dose_cm2),
                "Rp_um": float(Rp_um),
                "dRp_um": float(dRp_um),
                "oxidation_enable": bool(oxidation_enable),
                "oxidation_time_s": float(oxidation_time_s),
                "oxidation_A_um": float(oxidation_A_um),
                "oxidation_B_um2_s": float(oxidation_B_um2_s),
                "oxidation_gamma": float(oxidation_gamma),
                "oxidation_apply_on": str(oxidation_apply_on),
                "oxidation_consume_dopants": bool(oxidation_consume_dopants),
                "oxidation_update_materials": bool(oxidation_update_materials),
                "D_cm2_s": float(D_cm2_s),
                "total_t_s": float(total_t_s),
                "dt_s": float(dt_s),
                "anneal_use_arrhenius": bool(anneal_use_arrhenius),
                "arrhenius_D0_cm2_s": float(arrhenius_D0_cm2_s),
                "arrhenius_Ea_eV": float(arrhenius_Ea_eV),
                "arrhenius_T_C": float(arrhenius_T_C),
                "oxide_D_scale": float(oxide_D_scale),
                "cap_eps_um": float(cap_eps_um),
                "top_open_type": top_open_type,
                "h_cm_s": float(h_cm_s),
                "Ceq_cm3": float(Ceq_cm3),
                "dirichlet_value_cm3": float(dirichlet_value_cm3),
                "outdir": str(outdir),
                "formats": list(formats),
                "linecut_x_um": float(linecut_x_um),
                "linecut_y_um": float(linecut_y_um),
                "plot_log10": bool(plot_log10),
                "plot_vmin": float(plot_vmin),
                "plot_vmax": float(plot_vmax),
                "compute_metrics": bool(compute_metrics),
                "record_history": bool(record_history),
                "history_every_s": float(history_every_s),
                "history_save_csv": bool(history_save_csv),
                "history_save_png": bool(history_save_png),
                "export_vtk": bool(export_vtk),
                "export_tox_csv": bool(export_tox_csv),
                "export_tox_png": bool(export_tox_png),
                "zip_outputs": bool(zip_outputs),
                "store_full_c": bool(store_full_c),
            }
            with st.spinner("Running simulation..."):
                result = run_simulation(params)
            st.session_state["proc2d_last_result"] = result
            _push_recent_run(result, params)
            st.success(f"Simulation complete in {result['runtime_s']:.3f} s. Outputs: {result['outdir']}")
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

    result = st.session_state.get("proc2d_last_result")
    if result is None:
        st.info("왼쪽에서 파라미터를 조정하고 'Run Simulation'을 눌러 실행하세요.")
        return

    grid: Grid2D = result["grid"]
    C: np.ndarray = result["C"]
    mask_eff: np.ndarray = result["mask_eff"]
    outdir_path: Path = result["outdir"]
    plot_cfg = result["plot_cfg"]

    tabs = st.tabs(["Map", "Linecuts", "Metrics", "History", "Compare", "Artifacts"])

    with tabs[0]:
        st.subheader("2D Map")
        fig_map = _heatmap_figure(
            C,
            grid,
            mask_eff=mask_eff,
            tox_um=result.get("tox_um"),
            log10=bool(plot_cfg.get("log10", False)),
            vmin=plot_cfg.get("vmin"),
            vmax=plot_cfg.get("vmax"),
        )
        st.pyplot(fig_map)
        plt.close(fig_map)
        st.caption("cyan at y=0: mask open; thin cyan curve: tox(x)")

    with tabs[1]:
        st.subheader("Linecuts")
        linecut_kind = st.radio(
            "Linecut type",
            ["Vertical: C(y) at x", "Horizontal: C(x) at y"],
            horizontal=True,
        )
        linecut_scale = st.radio("Scale", ["linear", "log10"], horizontal=True)
        linecut_log_floor = st.number_input("log10 floor (cm^-3)", min_value=1e-30, value=1.0e10, format="%.6g")

        if linecut_kind.startswith("Vertical"):
            x_req = st.slider(
                "x_um",
                min_value=0.0,
                max_value=float(grid.Lx_um),
                value=float(min(max(result["linecuts"]["vertical"]["requested_x_um"], 0.0), grid.Lx_um)),
                step=float(grid.dx_um),
            )
            coord, values, x_used, ix = _vertical_linecut(C, grid, x_req)
            fig = _linecut_plot(
                coord,
                values,
                x_label="y [um]",
                title=f"Vertical linecut at x={x_used:.6g} um (ix={ix})",
                scale=linecut_scale,
                log_floor=float(linecut_log_floor),
            )
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"requested x={x_req:.6g} um, snapped to grid x={x_used:.6g} um")

            csv_text = _linecut_csv_text(
                kind="vertical",
                req_value=x_req,
                used_value=x_used,
                coord_name="y_um",
                coord_values=coord,
                c_values=values,
            )
            st.download_button(
                label="Download vertical linecut CSV",
                data=csv_text,
                file_name=f"linecut_vertical_x{f'{x_used:.6g}'.replace('.', 'p')}um.csv",
                mime="text/csv",
            )
        else:
            y_req = st.slider(
                "y_um",
                min_value=0.0,
                max_value=float(grid.Ly_um),
                value=float(min(max(result["linecuts"]["horizontal"]["requested_y_um"], 0.0), grid.Ly_um)),
                step=float(grid.dy_um),
            )
            coord, values, y_used, iy = _horizontal_linecut(C, grid, y_req)
            fig = _linecut_plot(
                coord,
                values,
                x_label="x [um]",
                title=f"Horizontal linecut at y={y_used:.6g} um (iy={iy})",
                scale=linecut_scale,
                log_floor=float(linecut_log_floor),
            )
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"requested y={y_req:.6g} um, snapped to grid y={y_used:.6g} um")

            csv_text = _linecut_csv_text(
                kind="horizontal",
                req_value=y_req,
                used_value=y_used,
                coord_name="x_um",
                coord_values=coord,
                c_values=values,
            )
            st.download_button(
                label="Download horizontal linecut CSV",
                data=csv_text,
                file_name=f"linecut_horizontal_y{f'{y_used:.6g}'.replace('.', 'p')}um.csv",
                mime="text/csv",
            )

    with tabs[2]:
        st.subheader("Metrics")
        metrics_report = result.get("metrics")
        if metrics_report:
            peak = metrics_report.get("peak", {}) if isinstance(metrics_report, dict) else {}
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Peak [cm^-3]", f"{float(peak.get('peak_cm3', np.nan)):.6g}")
            with col_m2:
                st.metric("Total mass [cm^-1]", f"{float(metrics_report.get('total_mass_cm1', np.nan)):.6g}")

            st.json(metrics_report)

            metrics_json_path = result.get("metrics_json_path")
            if metrics_json_path and Path(metrics_json_path).exists():
                st.download_button(
                    label="Download metrics.json",
                    data=Path(metrics_json_path).read_bytes(),
                    file_name="metrics.json",
                    mime="application/json",
                )
            metrics_csv_path = result.get("metrics_csv_path")
            if metrics_csv_path and Path(metrics_csv_path).exists():
                st.download_button(
                    label="Download metrics.csv",
                    data=Path(metrics_csv_path).read_bytes(),
                    file_name="metrics.csv",
                    mime="text/csv",
                )
            sheet_csv_path = result.get("sheet_dose_csv_path")
            if sheet_csv_path and Path(sheet_csv_path).exists():
                st.download_button(
                    label="Download sheet_dose_vs_x.csv",
                    data=Path(sheet_csv_path).read_bytes(),
                    file_name="sheet_dose_vs_x.csv",
                    mime="text/csv",
                )
        else:
            st.info("Compute metrics 옵션이 꺼져 있습니다.")

    with tabs[3]:
        st.subheader("Anneal History")
        history = result.get("history", [])
        if history:
            fig_hist = _history_figure(history)
            st.pyplot(fig_hist)
            plt.close(fig_hist)

            history_csv_path = result.get("history_csv_path")
            if history_csv_path and Path(history_csv_path).exists():
                st.download_button(
                    label="Download history.csv",
                    data=Path(history_csv_path).read_bytes(),
                    file_name="history.csv",
                    mime="text/csv",
                )
            history_png_path = result.get("history_png_path")
            if history_png_path and Path(history_png_path).exists():
                st.download_button(
                    label="Download history.png",
                    data=Path(history_png_path).read_bytes(),
                    file_name="history.png",
                    mime="image/png",
                )
        else:
            st.info("Record anneal history 옵션이 꺼져 있거나 기록 데이터가 없습니다.")

    with tabs[4]:
        st.subheader("Before/After Compare (Recent 2 runs)")
        _compare_two_runs_tab()

    with tabs[5]:
        st.subheader("Artifacts")
        st.write(f"Runtime: {result['runtime_s']:.3f} s")
        for path in result.get("written", []):
            st.write(f"- `{path}`")

        vtk_path = result.get("vtk_path")
        if vtk_path and Path(vtk_path).exists():
            st.download_button(
                label="Download C.vtk",
                data=Path(vtk_path).read_bytes(),
                file_name="C.vtk",
                mime="application/octet-stream",
            )
        material_vtk_path = result.get("material_vtk_path")
        if material_vtk_path and Path(material_vtk_path).exists():
            st.download_button(
                label="Download material.vtk",
                data=Path(material_vtk_path).read_bytes(),
                file_name="material.vtk",
                mime="application/octet-stream",
            )
        tox_csv_path = result.get("tox_csv_path")
        if tox_csv_path and Path(tox_csv_path).exists():
            st.download_button(
                label="Download tox_vs_x.csv",
                data=Path(tox_csv_path).read_bytes(),
                file_name="tox_vs_x.csv",
                mime="text/csv",
            )
        tox_png_path = result.get("tox_png_path")
        if tox_png_path and Path(tox_png_path).exists():
            st.download_button(
                label="Download tox_vs_x.png",
                data=Path(tox_png_path).read_bytes(),
                file_name="tox_vs_x.png",
                mime="image/png",
            )

        zip_path = result.get("zip_path")
        if zip_path and Path(zip_path).exists():
            st.download_button(
                label="Download outputs ZIP",
                data=Path(zip_path).read_bytes(),
                file_name=Path(zip_path).name,
                mime="application/zip",
            )

        c_png_path = outdir_path / "C.png"
        if c_png_path.exists():
            st.download_button(
                label="Download C.png",
                data=c_png_path.read_bytes(),
                file_name="C.png",
                mime="image/png",
            )
