"""Streamlit GUI for proc2d simulation."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yaml

from .diffusion import TopBCConfig, anneal_implicit
from .grid import Grid2D
from .implant import implant_2d_gaussian
from .io import export_results
from .mask import build_mask_1d, full_open_mask, openings_from_any, smooth_mask_1d, validate_mask


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _example_deck_path() -> Path:
    return _project_root() / "examples" / "deck_basic.yaml"


def _first_step(steps: list[dict[str, Any]], stype: str) -> dict[str, Any] | None:
    for step in steps:
        if isinstance(step, dict) and str(step.get("type", "")).lower() == stype:
            return step
    return None


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

    anneal_step = _first_step(steps, "anneal")
    if anneal_step:
        defaults["D_cm2_s"] = float(anneal_step.get("D_cm2_s", defaults["D_cm2_s"]))
        defaults["total_t_s"] = float(anneal_step.get("total_t_s", defaults["total_t_s"]))
        defaults["dt_s"] = float(anneal_step.get("dt_s", defaults["dt_s"]))
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


def _heatmap_figure(
    C: np.ndarray,
    grid: Grid2D,
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

    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=140)
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


def run_simulation(params: dict[str, Any]) -> dict[str, Any]:
    """Run one simulation from GUI parameters."""
    grid = Grid2D.from_domain(
        Lx_um=float(params["Lx_um"]),
        Ly_um=float(params["Ly_um"]),
        Nx=int(params["Nx"]),
        Ny=int(params["Ny"]),
    )

    C = np.full(grid.shape, float(params["background_doping_cm3"]), dtype=float)

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

    C += implant_2d_gaussian(
        grid=grid,
        dose_cm2=float(params["dose_cm2"]),
        Rp_um=float(params["Rp_um"]),
        dRp_um=float(params["dRp_um"]),
        mask_eff=mask_eff,
    )

    top_bc = _make_top_bc(params)
    C = anneal_implicit(
        C0=C,
        grid=grid,
        D_cm2_s=float(params["D_cm2_s"]),
        total_t_s=float(params["total_t_s"]),
        dt_s=float(params["dt_s"]),
        top_bc=top_bc,
        mask_eff=mask_eff,
    )

    outdir = Path(str(params["outdir"]))
    if not outdir.is_absolute():
        outdir = (Path.cwd() / outdir).resolve()

    formats = [str(x).lower() for x in params["formats"]]
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
    )

    return {
        "grid": grid,
        "C": C,
        "mask_eff": mask_eff,
        "outdir": outdir,
        "written": written,
        "plot_cfg": plot_cfg,
        "requested_linecut_x_um": float(params["linecut_x_um"]),
        "requested_linecut_y_um": float(params["linecut_y_um"]),
    }


def run_gui() -> None:
    """Render Streamlit app."""
    st.set_page_config(page_title="proc2d GUI", layout="wide")
    st.title("proc2d: Process 2D Cross-Section GUI")
    st.caption(
        "기본값은 examples/deck_basic.yaml 에서 로드됩니다. "
        "Run을 누르면 동일 GUI에서 C.png와 라인컷 그래프를 확인할 수 있습니다."
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
            sigma_lat_um = st.number_input(
                "sigma_lat_um",
                min_value=0.0,
                value=float(defaults["sigma_lat_um"]),
                format="%.6g",
            )

            st.subheader("Implant")
            dose_cm2 = st.number_input("dose_cm2", min_value=0.0, value=float(defaults["dose_cm2"]), format="%.6g")
            Rp_um = st.number_input("Rp_um", value=float(defaults["Rp_um"]), format="%.6g")
            dRp_um = st.number_input("dRp_um", min_value=1e-12, value=float(defaults["dRp_um"]), format="%.6g")

            st.subheader("Anneal")
            D_cm2_s = st.number_input("D_cm2_s", min_value=0.0, value=float(defaults["D_cm2_s"]), format="%.6g")
            total_t_s = st.number_input("total_t_s", min_value=0.0, value=float(defaults["total_t_s"]), format="%.6g")
            dt_s = st.number_input("dt_s", min_value=1e-12, value=float(defaults["dt_s"]), format="%.6g")

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
            plot_vmin = st.number_input("plot.vmin", min_value=0.0, value=float(defaults["plot_vmin"]), format="%.6g")
            plot_vmax = st.number_input("plot.vmax", min_value=0.0, value=float(defaults["plot_vmax"]), format="%.6g")

            submitted = st.form_submit_button("Run Simulation")

    if submitted:
        try:
            openings_um = _parse_openings_text(openings_text)
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
                "D_cm2_s": float(D_cm2_s),
                "total_t_s": float(total_t_s),
                "dt_s": float(dt_s),
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
            }
            with st.spinner("Running simulation..."):
                result = run_simulation(params)
            st.session_state["proc2d_last_result"] = result
            st.success(f"Simulation complete. Outputs: {result['outdir']}")
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

    result = st.session_state.get("proc2d_last_result")
    if result is None:
        st.info("왼쪽에서 파라미터를 조정하고 'Run Simulation'을 눌러 실행하세요.")
        return

    grid: Grid2D = result["grid"]
    C: np.ndarray = result["C"]
    outdir_path: Path = result["outdir"]
    plot_cfg = result["plot_cfg"]

    col_left, col_right = st.columns([1.35, 1.0])

    with col_left:
        st.subheader("2D Concentration Map")
        png_path = outdir_path / "C.png"
        if png_path.exists():
            st.image(str(png_path), caption=f"C.png ({png_path})", use_container_width=True)
        else:
            fig_map = _heatmap_figure(
                C,
                grid,
                log10=bool(plot_cfg.get("log10", False)),
                vmin=plot_cfg.get("vmin"),
                vmax=plot_cfg.get("vmax"),
            )
            st.pyplot(fig_map)
            plt.close(fig_map)
        st.caption("위 이미지는 현재 파라미터로 실행된 결과입니다.")

    with col_right:
        st.subheader("Artifacts")
        for path in result.get("written", []):
            st.write(f"- `{path}`")

    st.subheader("Linecut Viewer")
    kind = st.radio(
        "Linecut type",
        ["Vertical: C(y) at x", "Horizontal: C(x) at y"],
        horizontal=True,
    )

    if kind.startswith("Vertical"):
        x_req = st.slider(
            "x_um",
            min_value=0.0,
            max_value=float(grid.Lx_um),
            value=float(min(max(result["requested_linecut_x_um"], 0.0), grid.Lx_um)),
            step=float(grid.dx_um),
        )
        coord, values, x_used, ix = _vertical_linecut(C, grid, x_req)
        fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=140)
        ax.plot(coord, values, lw=2.0)
        ax.set_xlabel("y [um]")
        ax.set_ylabel("C [cm^-3]")
        ax.set_title(f"Vertical linecut at x={x_used:.6g} um (ix={ix})")
        ax.grid(alpha=0.3)
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
            value=float(min(max(result["requested_linecut_y_um"], 0.0), grid.Ly_um)),
            step=float(grid.dy_um),
        )
        coord, values, y_used, iy = _horizontal_linecut(C, grid, y_req)
        fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=140)
        ax.plot(coord, values, lw=2.0)
        ax.set_xlabel("x [um]")
        ax.set_ylabel("C [cm^-3]")
        ax.set_title(f"Horizontal linecut at y={y_used:.6g} um (iy={iy})")
        ax.grid(alpha=0.3)
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
