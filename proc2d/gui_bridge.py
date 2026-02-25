from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .presets import make_default_gui_analyze_step


def _parse_schedule_text(schedule_text: str) -> list[dict[str, float]] | None:
    text = str(schedule_text or "").strip()
    if not text:
        return None

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"arrhenius.schedule parsing failed: {exc}") from exc

    if not isinstance(parsed, list) or not parsed:
        raise ValueError("arrhenius.schedule must be a non-empty list.")

    out: list[dict[str, float]] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"arrhenius.schedule[{idx}] must be a mapping.")
        if "t_s" not in item or "T_C" not in item:
            raise ValueError(f"arrhenius.schedule[{idx}] requires t_s and T_C.")
        out.append(
            {
                "t_s": float(item["t_s"]),
                "T_C": float(item["T_C"]),
            }
        )
    return out


def build_deck_from_ui(params: dict[str, Any], *, outdir: str | Path) -> dict[str, Any]:
    out_path = Path(outdir)
    formats = [str(x).lower() for x in params["formats"]]
    if bool(params.get("export_vtk", False)) and "vtk" not in formats:
        formats.append("vtk")

    steps: list[dict[str, Any]] = []

    openings = list(params.get("openings_um", []))
    if openings:
        steps.append(
            {
                "type": "mask",
                "openings_um": openings,
                "sigma_lat_um": float(params["sigma_lat_um"]),
            }
        )

    if bool(params.get("oxidation_enable", False)):
        steps.append(
            {
                "type": "oxidation",
                "model": "deal_grove",
                "time_s": float(params["oxidation_time_s"]),
                "A_um": float(params["oxidation_A_um"]),
                "B_um2_s": float(params["oxidation_B_um2_s"]),
                "gamma": float(params["oxidation_gamma"]),
                "apply_on": str(params["oxidation_apply_on"]),
                "consume_dopants": bool(params["oxidation_consume_dopants"]),
                "update_materials": bool(params["oxidation_update_materials"]),
                "mask_weighting": str(params.get("oxidation_mask_weighting", "binary")),
                "open_threshold": float(params.get("oxidation_open_threshold", 0.5)),
            }
        )

    steps.append(
        {
            "type": "implant",
            "dose_cm2": float(params["dose_cm2"]),
            "Rp_um": float(params["Rp_um"]),
            "dRp_um": float(params["dRp_um"]),
        }
    )

    anneal_step: dict[str, Any] = {
        "type": "anneal",
        "total_t_s": float(params["total_t_s"]),
        "dt_s": float(params["dt_s"]),
        "oxide": {"D_scale": float(params.get("oxide_D_scale", 0.0))},
        "cap_eps_um": float(params.get("cap_eps_um", 0.0)),
        "cap_model": str(params.get("cap_model", "hard")),
    }

    if str(params.get("cap_model", "hard")).lower() == "exp":
        anneal_step["cap_len_um"] = float(params.get("cap_len_um", 0.01))

    open_type = str(params["top_open_type"]).lower()
    if open_type == "robin":
        anneal_step["top_bc"] = {
            "open": {
                "type": "robin",
                "h_cm_s": float(params["h_cm_s"]),
                "Ceq_cm3": float(params["Ceq_cm3"]),
            },
            "blocked": {"type": "neumann"},
        }
    elif open_type == "dirichlet":
        anneal_step["top_bc"] = {
            "open": {
                "type": "dirichlet",
                "value_cm3": float(params["dirichlet_value_cm3"]),
            },
            "blocked": {"type": "neumann"},
        }
    else:
        anneal_step["top_bc"] = {
            "open": {"type": "neumann"},
            "blocked": {"type": "neumann"},
        }

    if bool(params.get("anneal_use_arrhenius", False)):
        diff_cfg: dict[str, Any] = {
            "model": "arrhenius",
            "D0_cm2_s": float(params["arrhenius_D0_cm2_s"]),
            "Ea_eV": float(params["arrhenius_Ea_eV"]),
        }
        schedule = _parse_schedule_text(str(params.get("anneal_schedule_text", "")))
        if schedule is None:
            diff_cfg["T_C"] = float(params["arrhenius_T_C"])
        else:
            diff_cfg["schedule"] = schedule
        anneal_step["diffusivity"] = diff_cfg
    else:
        anneal_step["D_cm2_s"] = float(params["D_cm2_s"])

    if bool(params.get("record_history", False)):
        anneal_step["record"] = {
            "enable": True,
            "every_s": float(params["history_every_s"]),
            "save_csv": bool(params.get("history_save_csv", True)),
            "save_png": bool(params.get("history_save_png", True)),
        }

    steps.append(anneal_step)

    if bool(params.get("compute_metrics", False)):
        steps.append(
            make_default_gui_analyze_step(
                x_ref_um=float(params["linecut_x_um"]),
                y_ref_um=float(params["linecut_y_um"]),
            )
        )

    steps.append(
        {
            "type": "export",
            "outdir": str(out_path),
            "formats": formats,
            "linecuts": [
                {"kind": "vertical", "x_um": float(params["linecut_x_um"])},
                {"kind": "horizontal", "y_um": float(params["linecut_y_um"])},
            ],
            "plot": {
                "log10": bool(params["plot_log10"]),
                "vmin": float(params["plot_vmin"]),
                "vmax": float(params["plot_vmax"]),
            },
            "extra": {
                "tox_csv": bool(params.get("export_tox_csv", False)),
                "tox_png": bool(params.get("export_tox_png", False)),
            },
        }
    )

    return {
        "domain": {
            "Lx_um": float(params["Lx_um"]),
            "Ly_um": float(params["Ly_um"]),
            "Nx": int(params["Nx"]),
            "Ny": int(params["Ny"]),
        },
        "background_doping_cm3": float(params["background_doping_cm3"]),
        "steps": steps,
    }
