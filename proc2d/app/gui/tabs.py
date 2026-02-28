"""Tab renderers and plotting helpers for GUI output."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ...grid import Grid2D
from .compare import render_compare_tab


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


def vertical_linecut(C: np.ndarray, grid: Grid2D, x_um: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return vertical linecut values snapped to nearest x-grid index."""
    ix = grid.nearest_x_index(float(x_um))
    return grid.y_um, np.asarray(C[:, ix], dtype=float), float(grid.x_um[ix]), ix


def horizontal_linecut(C: np.ndarray, grid: Grid2D, y_um: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return horizontal linecut values snapped to nearest y-grid index."""
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


def render_result_tabs(result: dict[str, Any]) -> None:
    """Render all result tabs for a completed simulation run."""
    tabs = st.tabs(["Map", "Linecuts", "Metrics", "History", "Compare", "Artifacts"])

    with tabs[0]:
        st.subheader("2D Map")
        fig_map = _heatmap_figure(
            np.asarray(result["C"], dtype=float),
            result["grid"],
            mask_eff=result["mask_eff"],
            tox_um=result.get("tox_um"),
            log10=bool(result["plot_cfg"].get("log10", False)),
            vmin=result["plot_cfg"].get("vmin"),
            vmax=result["plot_cfg"].get("vmax"),
        )
        st.pyplot(fig_map)
        plt.close(fig_map)
        st.caption("cyan at y=0: mask open; thin cyan curve: tox(x)")

    with tabs[1]:
        _render_linecuts_tab(result)

    with tabs[2]:
        _render_metrics_tab(result)

    with tabs[3]:
        _render_history_tab(result)

    with tabs[4]:
        st.subheader("Before/After Compare (Recent 2 runs)")
        render_compare_tab()

    with tabs[5]:
        _render_artifacts_tab(result)


def _render_linecuts_tab(result: dict[str, Any]) -> None:
    st.subheader("Linecuts")
    grid: Grid2D = result["grid"]
    C = np.asarray(result["C"], dtype=float)
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
        coord, values, x_used, ix = vertical_linecut(C, grid, x_req)
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
        coord, values, y_used, iy = horizontal_linecut(C, grid, y_req)
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


def _render_metrics_tab(result: dict[str, Any]) -> None:
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


def _render_history_tab(result: dict[str, Any]) -> None:
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


def _render_artifacts_tab(result: dict[str, Any]) -> None:
    st.subheader("Artifacts")
    outdir_path = Path(result["outdir"])
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
