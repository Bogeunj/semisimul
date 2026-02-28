"""Streamlit GUI entrypoint and GUI-specific simulation adapter."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from .app.gui.forms import load_default_params, render_sidebar_form
from .app.gui.session import get_last_result, push_recent_run, set_last_result
from .app.gui.tabs import horizontal_linecut, render_result_tabs, vertical_linecut
from .config import GuiRunConfig, build_deck_from_gui_config
from .deck import run_simulation_payload
from .mask import full_open_mask


def _make_zip(paths: list[Path], outdir: Path, filename: str = "outputs_bundle.zip") -> Path:
    zip_path = outdir / filename
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if path.exists() and path.is_file():
                zf.write(path, arcname=path.name)
    return zip_path


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
    mask_eff = np.asarray(state.mask_eff, dtype=float) if state.mask_eff is not None else full_open_mask(grid.Nx)
    tox_um = np.asarray(state.tox_um, dtype=float) if state.tox_um is not None else np.zeros(grid.Nx, dtype=float)

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

    metrics_json_path = outdir / "metrics.json" if "metrics.json" in written_names else None
    metrics_csv_path = outdir / "metrics.csv" if "metrics.csv" in written_names else None
    sheet_dose_csv_path = outdir / "sheet_dose_vs_x.csv" if "sheet_dose_vs_x.csv" in written_names else None

    history_csv_path = outdir / "history.csv" if "history.csv" in written_names else None
    history_png_path = outdir / "history.png" if "history.png" in written_names else None

    vtk_path = outdir / "C.vtk" if "C.vtk" in written_names else None
    material_vtk_path = outdir / "material.vtk" if "material.vtk" in written_names else None
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

    v_coord, v_values, x_used, ix = vertical_linecut(C, grid, float(cfg.linecut_x_um))
    h_coord, h_values, y_used, iy = horizontal_linecut(C, grid, float(cfg.linecut_y_um))

    runtime_s = float(time.perf_counter() - t0)

    return {
        "grid": grid,
        "C": C,
        "mask_eff": mask_eff,
        "mask_eff_bc": mask_eff_bc,
        "tox_um": tox_um,
        "materials": state.materials,
        "outdir": outdir,
        "written": written,
        "plot_cfg": plot_cfg,
        "metrics": state.metrics,
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


def run_gui() -> None:
    """Render Streamlit app."""
    st.set_page_config(page_title="proc2d GUI", layout="wide")
    st.title("proc2d: Process 2D Cross-Section GUI")
    st.caption(
        "예제 deck 기본값을 로드해 파라미터를 조정하고, 같은 화면에서 "
        "맵/라인컷/메트릭/히스토리/비교를 확인할 수 있습니다."
    )

    defaults = load_default_params()
    try:
        submitted, params = render_sidebar_form(defaults)
    except ValueError as exc:
        st.error(f"Simulation failed: {exc}")
        submitted, params = False, None

    if submitted:
        try:
            if params is None:
                raise ValueError("Invalid form parameters")
            with st.spinner("Running simulation..."):
                result = run_simulation(params)
            set_last_result(result)
            push_recent_run(result, params)
            st.success(f"Simulation complete in {result['runtime_s']:.3f} s. Outputs: {result['outdir']}")
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

    result = get_last_result()
    if result is None:
        st.info("왼쪽에서 파라미터를 조정하고 'Run Simulation'을 눌러 실행하세요.")
        return

    render_result_tabs(result)


__all__ = ["load_default_params", "run_gui", "run_simulation"]
