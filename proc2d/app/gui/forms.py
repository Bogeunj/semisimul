"""Streamlit sidebar forms and default parameter loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from ...mask import openings_from_any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def render_sidebar_form(defaults: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    """Render sidebar form and return (submitted, params)."""
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

    if not submitted:
        return False, None

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
    return True, params
