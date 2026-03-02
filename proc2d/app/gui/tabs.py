"""Tab renderers and plotting helpers for GUI output."""

from __future__ import annotations

import csv
import html
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import streamlit as st

from ...grid import Grid2D
from .compare import render_compare_tab
from .storage import GuiStorageManager


@dataclass(frozen=True)
class MapTabSpec:
    """Descriptor for one map-style tab in the results UI."""

    key: str
    label: str
    supports_linecut: bool = True
    physical_map_key: str | None = None


@dataclass(frozen=True)
class MapStyle:
    """Pure map style descriptor used by GUI map renderers."""

    cmap_name: str
    low_color: str | None = None


def map_style_for_spec_key(key: str) -> MapStyle:
    """Return visual style for a map spec key."""
    normalized = str(key)
    if normalized.startswith("physical_"):
        normalized = normalized.removeprefix("physical_")

    if normalized in {"concentration", "concentration_cm3"}:
        return MapStyle(cmap_name="inferno", low_color="#b3b3b3")
    if normalized in {"potential", "potential_V"}:
        return MapStyle(cmap_name="bwr")
    if normalized in {"electric_field", "electric_field_mag_V_cm"}:
        return MapStyle(cmap_name="bwr")
    if normalized in {"conductivity", "conductivity_S_cm"}:
        return MapStyle(cmap_name="cividis")
    if normalized == "structure":
        return MapStyle(cmap_name="viridis")
    return MapStyle(cmap_name="viridis")


def structure_material_legend() -> list[tuple[int, str, str]]:
    """Return structure material ids with names and descriptions."""
    return [
        (
            0,
            "Si",
            "Semiconductor silicon region where active dopants and carriers exist.",
        ),
        (
            1,
            "SiO2",
            "Insulating silicon-dioxide oxide region used as mask and dielectric.",
        ),
    ]


def _structure_color_map() -> dict[int, str]:
    return {
        0: "#d9b44a",
        1: "#5aa8de",
    }


def _render_structure_legend() -> None:
    colors = _structure_color_map()
    chunks: list[str] = []
    for mat_id, name, desc in structure_material_legend():
        color = colors.get(int(mat_id), "#777777")
        chunks.append(
            '<span style="display:inline-flex;align-items:center;margin-right:16px;">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:{html.escape(color)};margin-right:6px;"></span>'
            f'<span title="{html.escape(desc)}"><b>{html.escape(name)}</b></span>'
            "</span>"
        )
    st.markdown("".join(chunks), unsafe_allow_html=True)


def _label_for_physical_map_key(map_key: str) -> str:
    labels = {
        "potential_V": "Potential",
        "electric_field_mag_V_cm": "Electric Field",
        "conductivity_S_cm": "Conductivity",
    }
    if map_key in labels:
        return labels[map_key]
    return map_key.replace("_", " ").title()


def build_map_tab_specs(result: dict[str, Any] | None = None) -> list[MapTabSpec]:
    """Build ordered map tab specs for GUI result rendering."""
    specs = [
        MapTabSpec(key="structure", label="Structure"),
        MapTabSpec(key="concentration", label="Concentration"),
    ]

    physical_maps_raw = (
        result.get("physical_maps") if isinstance(result, dict) else None
    )
    physical_maps = physical_maps_raw if isinstance(physical_maps_raw, dict) else {}

    if physical_maps:
        for map_key in physical_maps:
            if map_key in {"structure", "concentration_cm3"}:
                continue
            specs.append(
                MapTabSpec(
                    key=f"physical_{map_key}",
                    label=_label_for_physical_map_key(map_key),
                    physical_map_key=map_key,
                )
            )
        return specs

    specs.extend(
        [
            MapTabSpec(
                key="potential", label="Potential", physical_map_key="potential_V"
            ),
            MapTabSpec(
                key="electric_field",
                label="Electric Field",
                physical_map_key="electric_field_mag_V_cm",
            ),
            MapTabSpec(
                key="conductivity",
                label="Conductivity",
                physical_map_key="conductivity_S_cm",
            ),
        ]
    )
    return specs


def build_result_tab_labels(result: dict[str, Any] | None = None) -> list[str]:
    """Build ordered labels for all result tabs."""
    labels = [spec.label for spec in build_map_tab_specs(result)]
    labels.extend(["Metrics", "History", "Compare", "Artifacts", "Storage"])
    return labels


def _colormap_with_low_color(cmap_name: str, low_color: str | None):
    base = plt.get_cmap(cmap_name)
    if low_color is None:
        return base
    sampled = base(np.linspace(0.0, 1.0, 256))
    sampled[0] = mcolors.to_rgba(low_color)
    cmap = mcolors.ListedColormap(sampled)
    cmap.set_under(low_color)
    return cmap


def _mask_segments(
    mask_eff: np.ndarray, x_um: np.ndarray, threshold: float = 0.5
) -> list[tuple[float, float]]:
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
    cmap_name: str,
    low_color: str | None,
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
        extent=(
            float(grid.x_um[0]),
            float(grid.x_um[-1]),
            float(grid.y_um[-1]),
            float(grid.y_um[0]),
        ),
        origin="upper",
        aspect="auto",
        cmap=_colormap_with_low_color(cmap_name, low_color),
        vmin=vmin_plot,
        vmax=vmax_plot,
    )
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title("Final concentration")

    if mask_eff is not None:
        for x0, x1 in _mask_segments(mask_eff, grid.x_um):
            ax.plot(
                [x0, x1], [0.0, 0.0], color="#4dd0e1", lw=4.0, solid_capstyle="butt"
            )

    if tox_um is not None:
        tox = np.asarray(tox_um, dtype=float)
        if tox.shape == (grid.Nx,):
            ax.plot(grid.x_um, tox, color="#00e5ff", lw=1.6, alpha=0.9, label="tox")
            ax.legend(loc="upper right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig


def vertical_linecut(
    C: np.ndarray, grid: Grid2D, x_um: float
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return vertical linecut values snapped to nearest x-grid index."""
    ix = grid.nearest_x_index(float(x_um))
    return grid.y_um, np.asarray(C[:, ix], dtype=float), float(grid.x_um[ix]), ix


def horizontal_linecut(
    C: np.ndarray, grid: Grid2D, y_um: float
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return horizontal linecut values snapped to nearest y-grid index."""
    iy = grid.nearest_y_index(float(y_um))
    return grid.x_um, np.asarray(C[iy, :], dtype=float), float(grid.y_um[iy]), iy


def _linecut_csv_text(
    kind: str,
    req_value: float,
    used_value: float,
    coord_name: str,
    coord_values: np.ndarray,
    values: np.ndarray,
    value_col_name: str,
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
    writer.writerow([coord_name, value_col_name])
    for coord, value in zip(coord_values, values):
        writer.writerow([f"{float(coord):.12g}", f"{float(value):.12g}"])
    return buf.getvalue()


def _field_csv_text(field: np.ndarray, grid: Grid2D, value_col_name: str) -> str:
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["x_um", "y_um", value_col_name])
    arr = np.asarray(field, dtype=float)
    for iy, y_um in enumerate(grid.y_um):
        for ix, x_um in enumerate(grid.x_um):
            writer.writerow(
                [
                    f"{float(x_um):.12g}",
                    f"{float(y_um):.12g}",
                    f"{float(arr[iy, ix]):.12g}",
                ]
            )
    return buf.getvalue()


def _history_figure(history: list[dict[str, float]]):
    def _safe_float(value: object) -> float:
        try:
            return float(str(value))
        except Exception:
            return float("nan")

    t = np.asarray(
        [_safe_float(row.get("time_s", np.nan)) for row in history], dtype=float
    )
    mass = np.asarray(
        [_safe_float(row.get("mass", np.nan)) for row in history], dtype=float
    )
    flux = np.asarray(
        [_safe_float(row.get("flux_out", np.nan)) for row in history], dtype=float
    )
    residual = np.asarray(
        [_safe_float(row.get("residual", np.nan)) for row in history], dtype=float
    )

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
    values: np.ndarray,
    x_label: str,
    title: str,
    y_label: str,
    scale: str,
    log_floor: float,
):
    y = np.asarray(values, dtype=float)
    if scale == "log10":
        y_plot = np.log10(np.clip(y, max(1e-30, float(log_floor)), None))
        y_axis_label = f"log10({y_label})"
    else:
        y_plot = y
        y_axis_label = y_label

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
    ax.plot(np.asarray(coord_um, dtype=float), y_plot, lw=2.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _map_field_for_spec(
    result: dict[str, Any], spec: MapTabSpec
) -> tuple[np.ndarray, str, str, str, bool]:
    grid: Grid2D = result["grid"]
    C = np.asarray(result["C"], dtype=float)
    physical_maps_raw = result.get("physical_maps")
    physical_maps = physical_maps_raw if isinstance(physical_maps_raw, dict) else {}

    if spec.key == "structure":
        arr_src = physical_maps.get("structure", result.get("materials"))
        arr = (
            np.asarray(arr_src, dtype=float)
            if arr_src is not None
            else np.zeros(grid.shape, dtype=float)
        )
        return (
            arr,
            "Structure",
            "material_id",
            map_style_for_spec_key("structure").cmap_name,
            False,
        )
    if spec.key == "concentration":
        arr_src = physical_maps.get("concentration_cm3", C)
        style = map_style_for_spec_key("concentration")
        return (
            np.asarray(arr_src, dtype=float),
            "Final concentration",
            "C_cm3",
            style.cmap_name,
            True,
        )
    if spec.key == "potential" or spec.physical_map_key == "potential_V":
        arr_src = physical_maps.get("potential_V")
        style = map_style_for_spec_key("potential")
        if arr_src is not None:
            return (
                np.asarray(arr_src, dtype=float),
                "Potential",
                "potential_V",
                style.cmap_name,
                False,
            )
        c_log = np.log10(np.clip(C, 1.0e10, None))
        phi = c_log - float(np.nanmean(c_log))
        return phi, "Potential", "potential_V", style.cmap_name, False
    if (
        spec.key == "electric_field"
        or spec.physical_map_key == "electric_field_mag_V_cm"
    ):
        arr_src = physical_maps.get("electric_field_mag_V_cm")
        style = map_style_for_spec_key("electric_field")
        if arr_src is not None:
            return (
                np.asarray(arr_src, dtype=float),
                "Electric Field Magnitude",
                "E_mag_V_cm",
                style.cmap_name,
                False,
            )
        c_log = np.log10(np.clip(C, 1.0e10, None))
        phi = c_log - float(np.nanmean(c_log))
        dphi_dy, dphi_dx = np.gradient(phi, grid.dy_cm, grid.dx_cm)
        e_mag = np.sqrt(dphi_dx**2 + dphi_dy**2)
        return e_mag, "Electric Field Magnitude", "E_mag_V_cm", style.cmap_name, False
    if spec.key == "conductivity" or spec.physical_map_key == "conductivity_S_cm":
        arr_src = physical_maps.get("conductivity_S_cm")
        style = map_style_for_spec_key("conductivity")
        if arr_src is not None:
            return (
                np.asarray(arr_src, dtype=float),
                "Conductivity",
                "sigma_S_cm",
                style.cmap_name,
                False,
            )
        sigma = np.clip(C, 0.0, None) * 1.0e-20
        return sigma, "Conductivity", "sigma_S_cm", style.cmap_name, False

    if spec.physical_map_key is not None:
        arr_src = physical_maps.get(spec.physical_map_key)
        if arr_src is not None:
            return (
                np.asarray(arr_src, dtype=float),
                spec.label,
                spec.physical_map_key,
                "viridis",
                False,
            )

    return C, spec.label, "value", "viridis", False


def _render_map_linecuts(
    field: np.ndarray,
    grid: Grid2D,
    spec: MapTabSpec,
    value_col_name: str,
    linecut_defaults: dict[str, Any],
) -> None:
    st.markdown("**Linecuts**")
    linecut_kind = st.radio(
        "Linecut type",
        ["Vertical: value(y) at x", "Horizontal: value(x) at y"],
        horizontal=True,
        key=f"{spec.key}_linecut_kind",
    )
    linecut_scale = st.radio(
        "Scale",
        ["linear", "log10"],
        horizontal=True,
        key=f"{spec.key}_linecut_scale",
    )
    linecut_log_floor = st.number_input(
        "log10 floor",
        min_value=1e-30,
        value=1.0e-30,
        format="%.6g",
        key=f"{spec.key}_linecut_floor",
    )

    if linecut_kind.startswith("Vertical"):
        requested_x_um = float(
            linecut_defaults.get("vertical", {}).get("requested_x_um", grid.Lx_um * 0.5)
        )
        x_req = st.slider(
            "x_um",
            min_value=0.0,
            max_value=float(grid.Lx_um),
            value=float(min(max(requested_x_um, 0.0), grid.Lx_um)),
            step=float(grid.dx_um),
            key=f"{spec.key}_linecut_x",
        )
        coord, values, x_used, ix = vertical_linecut(field, grid, x_req)
        fig = _linecut_plot(
            coord,
            values,
            x_label="y [um]",
            title=f"{spec.label} vertical linecut at x={x_used:.6g} um (ix={ix})",
            y_label=value_col_name,
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
            values=values,
            value_col_name=value_col_name,
        )
        st.download_button(
            label=f"Download {spec.label.lower()} vertical linecut CSV",
            data=csv_text,
            file_name=(
                f"linecut_{spec.key}_vertical_x{f'{x_used:.6g}'.replace('.', 'p')}um.csv"
            ),
            mime="text/csv",
            key=f"{spec.key}_linecut_csv_vertical",
        )
        return

    requested_y_um = float(
        linecut_defaults.get("horizontal", {}).get("requested_y_um", grid.Ly_um * 0.5)
    )
    y_req = st.slider(
        "y_um",
        min_value=0.0,
        max_value=float(grid.Ly_um),
        value=float(min(max(requested_y_um, 0.0), grid.Ly_um)),
        step=float(grid.dy_um),
        key=f"{spec.key}_linecut_y",
    )
    coord, values, y_used, iy = horizontal_linecut(field, grid, y_req)
    fig = _linecut_plot(
        coord,
        values,
        x_label="x [um]",
        title=f"{spec.label} horizontal linecut at y={y_used:.6g} um (iy={iy})",
        y_label=value_col_name,
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
        values=values,
        value_col_name=value_col_name,
    )
    st.download_button(
        label=f"Download {spec.label.lower()} horizontal linecut CSV",
        data=csv_text,
        file_name=f"linecut_{spec.key}_horizontal_y{f'{y_used:.6g}'.replace('.', 'p')}um.csv",
        mime="text/csv",
        key=f"{spec.key}_linecut_csv_horizontal",
    )


def _render_map_tab(result: dict[str, Any], spec: MapTabSpec) -> None:
    grid: Grid2D = result["grid"]
    style = map_style_for_spec_key(spec.key)
    field, title, value_col_name, cmap_name, is_concentration = _map_field_for_spec(
        result, spec
    )
    st.subheader(spec.label)

    if spec.key == "structure":
        legend = structure_material_legend()
        colors = _structure_color_map()
        max_id = max(int(np.max(field)), max(item[0] for item in legend), 1)
        cmap_list = [colors.get(i, "#777777") for i in range(max_id + 1)]
        structure_cmap = mcolors.ListedColormap(cmap_list)
        fig_map, ax = plt.subplots(figsize=(8.5, 4.0), dpi=140)
        ax.imshow(
            np.asarray(field, dtype=float),
            extent=(
                float(grid.x_um[0]),
                float(grid.x_um[-1]),
                float(grid.y_um[-1]),
                float(grid.y_um[0]),
            ),
            origin="upper",
            aspect="auto",
            cmap=structure_cmap,
            interpolation="nearest",
            vmin=0.0,
            vmax=float(max_id),
        )
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
        ax.set_title("Structure")
        fig_map.tight_layout()
        st.pyplot(fig_map)
        plt.close(fig_map)
        st.caption("Material legend (hover each name for description)")
        _render_structure_legend()
    elif is_concentration:
        fig_map = _heatmap_figure(
            field,
            grid,
            mask_eff=result["mask_eff"],
            tox_um=result.get("tox_um"),
            log10=bool(result["plot_cfg"].get("log10", False)),
            vmin=result["plot_cfg"].get("vmin"),
            vmax=result["plot_cfg"].get("vmax"),
            cmap_name=style.cmap_name,
            low_color=style.low_color,
        )
        st.pyplot(fig_map)
        plt.close(fig_map)
        st.caption("cyan at y=0: mask open; thin cyan curve: tox(x)")
    else:
        arr = np.asarray(field, dtype=float)
        vmin: float | None = None
        vmax: float | None = None
        if spec.key == "potential" or spec.physical_map_key == "potential_V":
            max_abs = float(np.max(np.abs(arr)))
            if max_abs > 0.0:
                vmin = -max_abs
                vmax = max_abs
        fig_map, ax = plt.subplots(figsize=(8.5, 4.0), dpi=140)
        im = ax.imshow(
            arr,
            extent=(
                float(grid.x_um[0]),
                float(grid.x_um[-1]),
                float(grid.y_um[-1]),
                float(grid.y_um[0]),
            ),
            origin="upper",
            aspect="auto",
            cmap=cmap_name,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
        ax.set_title(title)
        cbar = fig_map.colorbar(im, ax=ax)
        cbar.set_label(value_col_name)
        fig_map.tight_layout()
        st.pyplot(fig_map)
        plt.close(fig_map)

    _render_map_linecuts(
        field=np.asarray(field, dtype=float),
        grid=grid,
        spec=spec,
        value_col_name=value_col_name,
        linecut_defaults=result.get("linecuts", {}),
    )
    st.download_button(
        label=f"Download {spec.label.lower()} map CSV",
        data=_field_csv_text(field, grid, value_col_name=value_col_name),
        file_name=f"{spec.key}_map.csv",
        mime="text/csv",
        key=f"{spec.key}_map_csv",
    )


def render_result_tabs(result: dict[str, Any]) -> None:
    """Render all result tabs for a completed simulation run."""
    map_specs = build_map_tab_specs(result)
    tabs = st.tabs(build_result_tab_labels(result))

    for idx, spec in enumerate(map_specs):
        with tabs[idx]:
            _render_map_tab(result, spec)

    tab_offset = len(map_specs)

    with tabs[tab_offset]:
        _render_metrics_tab(result)

    with tabs[tab_offset + 1]:
        _render_history_tab(result)

    with tabs[tab_offset + 2]:
        st.subheader("Before/After Compare (Recent 2 runs)")
        render_compare_tab()

    with tabs[tab_offset + 3]:
        _render_artifacts_tab(result)

    with tabs[tab_offset + 4]:
        _render_storage_tab(result)


def _format_size_bytes(size_bytes: int) -> str:
    size = float(max(0, int(size_bytes)))
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{int(size_bytes)} B"


def _current_run_artifact_total_size_bytes(result: dict[str, Any]) -> int:
    outdir = Path(result.get("outdir", Path.cwd()))
    total = 0
    for written_path in result.get("written", []):
        path = Path(written_path)
        if not path.is_absolute():
            path = outdir / path
        if not path.exists():
            continue
        if path.is_file():
            total += int(path.stat().st_size)
            continue
        if path.is_dir():
            total += sum(int(p.stat().st_size) for p in path.rglob("*") if p.is_file())
    return total


def _render_storage_tab(result: dict[str, Any]) -> None:
    st.subheader("Storage")
    outdir_path = Path(result["outdir"])
    storage_root = outdir_path.parent
    manager = GuiStorageManager(storage_root)

    total_storage_bytes = manager.get_total_size_bytes()
    current_run_total_bytes = _current_run_artifact_total_size_bytes(result)

    st.write(f"Root path: `{storage_root}`")
    st.write(
        f"Total storage size: {_format_size_bytes(total_storage_bytes)} ({total_storage_bytes} bytes)"
    )
    st.write(
        "Current run artifact total size: "
        f"{_format_size_bytes(current_run_total_bytes)} ({current_run_total_bytes} bytes)"
    )

    entries = manager.list_entries()
    if not entries:
        st.info("No storage entries found.")
        return

    st.markdown("**Entries**")
    st.dataframe(
        [
            {
                "rel path": entry.rel_path,
                "type": entry.entry_type,
                "size": _format_size_bytes(entry.size_bytes),
            }
            for entry in entries
        ],
        use_container_width=True,
        hide_index=True,
    )

    options = [entry.rel_path for entry in entries]
    selected_rel_path = st.selectbox(
        "Selected entry",
        options=options,
        index=0,
        key="storage_selected_entry",
    )

    st.markdown("**Actions**")
    confirm_delete = st.checkbox(
        "Confirm delete",
        value=False,
        key="storage_confirm_delete",
    )
    if st.button("Delete selected", key="storage_delete_button"):
        try:
            if not confirm_delete:
                st.error("Enable confirm delete before deleting an entry.")
            else:
                manager.delete(selected_rel_path)
                st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    rename_value = st.text_input(
        "Rename selected entry (new basename)",
        value=Path(selected_rel_path).name,
        key="storage_rename_basename",
    )
    if st.button("Rename selected", key="storage_rename_button"):
        try:
            manager.rename(selected_rel_path, rename_value.strip())
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    move_destination = st.text_input(
        "Move selected entry to destination directory (relative)",
        value="",
        key="storage_move_destination",
    )
    if st.button("Move selected", key="storage_move_button"):
        try:
            manager.move(selected_rel_path, move_destination.strip())
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


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
    linecut_log_floor = st.number_input(
        "log10 floor (cm^-3)", min_value=1e-30, value=1.0e10, format="%.6g"
    )

    if linecut_kind.startswith("Vertical"):
        x_req = st.slider(
            "x_um",
            min_value=0.0,
            max_value=float(grid.Lx_um),
            value=float(
                min(
                    max(result["linecuts"]["vertical"]["requested_x_um"], 0.0),
                    grid.Lx_um,
                )
            ),
            step=float(grid.dx_um),
        )
        coord, values, x_used, ix = vertical_linecut(C, grid, x_req)
        fig = _linecut_plot(
            coord,
            values,
            x_label="y [um]",
            title=f"Vertical linecut at x={x_used:.6g} um (ix={ix})",
            y_label="C [cm^-3]",
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
            values=values,
            value_col_name="C_cm3",
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
            value=float(
                min(
                    max(result["linecuts"]["horizontal"]["requested_y_um"], 0.0),
                    grid.Ly_um,
                )
            ),
            step=float(grid.dy_um),
        )
        coord, values, y_used, iy = horizontal_linecut(C, grid, y_req)
        fig = _linecut_plot(
            coord,
            values,
            x_label="x [um]",
            title=f"Horizontal linecut at y={y_used:.6g} um (iy={iy})",
            y_label="C [cm^-3]",
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
            values=values,
            value_col_name="C_cm3",
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
    electrical_metrics = result.get("electrical_metrics")
    if electrical_metrics is None and isinstance(metrics_report, dict):
        raw_electrical = metrics_report.get("electrical")
        if isinstance(raw_electrical, dict):
            electrical_metrics = raw_electrical

    if metrics_report:
        peak = (
            metrics_report.get("peak", {}) if isinstance(metrics_report, dict) else {}
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Peak [cm^-3]", f"{float(peak.get('peak_cm3', np.nan)):.6g}")
        with col_m2:
            st.metric(
                "Total mass [cm^-1]",
                f"{float(metrics_report.get('total_mass_cm1', np.nan)):.6g}",
            )

        if isinstance(electrical_metrics, dict) and electrical_metrics:
            st.markdown("**Electrical**")
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                st.metric(
                    "Id [A]", f"{float(electrical_metrics.get('Id_A', np.nan)):.6g}"
                )
            with col_e2:
                st.metric(
                    "Vth [V]", f"{float(electrical_metrics.get('Vth_V', np.nan)):.6g}"
                )
            with col_e3:
                st.metric(
                    "Cox [F/cm^2]",
                    f"{float(electrical_metrics.get('Cox_F_cm2', np.nan)):.6g}",
                )

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
