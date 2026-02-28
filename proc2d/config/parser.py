"""Config parsing and translation utilities."""

from __future__ import annotations

from typing import Any, Mapping

from .deck_models import DomainConfig
from .gui_models import GuiRunConfig


def _as_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping.")
    return value


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in {context}.")
    return mapping[key]


def _to_float(value: Any, key: str, context: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{context}.{key} must be a number, got {value!r}.") from exc


def _to_int(value: Any, key: str, context: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{context}.{key} must be an integer, got {value!r}.") from exc


def parse_steps(deck: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Parse and normalize deck steps."""
    raw_steps = _required(deck, "steps", "deck")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("deck.steps must be a non-empty list.")

    steps: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_steps):
        steps.append(_as_mapping(raw, f"steps[{idx}]"))
    return steps


def parse_domain_config(deck: Mapping[str, Any]) -> DomainConfig:
    """Extract strongly typed domain config from deck payload."""
    domain = _as_mapping(_required(deck, "domain", "deck"), "deck.domain")

    Lx_um = _to_float(_required(domain, "Lx_um", "deck.domain"), "Lx_um", "deck.domain")
    Ly_um = _to_float(_required(domain, "Ly_um", "deck.domain"), "Ly_um", "deck.domain")
    Nx = _to_int(_required(domain, "Nx", "deck.domain"), "Nx", "deck.domain")
    Ny = _to_int(_required(domain, "Ny", "deck.domain"), "Ny", "deck.domain")

    if "background_doping_cm3" in deck:
        bg = _to_float(deck["background_doping_cm3"], "background_doping_cm3", "deck")
    else:
        bg = _to_float(domain.get("background_doping_cm3", 0.0), "background_doping_cm3", "deck.domain")

    return DomainConfig(Lx_um=Lx_um, Ly_um=Ly_um, Nx=Nx, Ny=Ny, background_doping_cm3=bg)


def _top_bc_from_gui(cfg: GuiRunConfig) -> dict[str, Any]:
    open_type = str(cfg.top_open_type).lower()
    if open_type == "robin":
        return {
            "open": {"type": "robin", "h_cm_s": float(cfg.h_cm_s), "Ceq_cm3": float(cfg.Ceq_cm3)},
            "blocked": {"type": "neumann"},
        }
    if open_type == "dirichlet":
        return {
            "open": {"type": "dirichlet", "value_cm3": float(cfg.dirichlet_value_cm3)},
            "blocked": {"type": "neumann"},
        }
    return {"open": {"type": "neumann"}, "blocked": {"type": "neumann"}}


def build_deck_from_gui_config(cfg: GuiRunConfig) -> dict[str, Any]:
    """Convert GUI config into canonical deck payload."""
    deck: dict[str, Any] = {
        "domain": {
            "Lx_um": float(cfg.Lx_um),
            "Ly_um": float(cfg.Ly_um),
            "Nx": int(cfg.Nx),
            "Ny": int(cfg.Ny),
        },
        "background_doping_cm3": float(cfg.background_doping_cm3),
        "steps": [],
    }

    steps: list[dict[str, Any]] = deck["steps"]
    if cfg.openings_um:
        steps.append(
            {
                "type": "mask",
                "openings_um": [[float(a), float(b)] for a, b in cfg.openings_um],
                "sigma_lat_um": float(cfg.sigma_lat_um),
            }
        )

    if bool(cfg.oxidation_enable):
        steps.append(
            {
                "type": "oxidation",
                "model": "deal_grove",
                "time_s": float(cfg.oxidation_time_s),
                "A_um": float(cfg.oxidation_A_um),
                "B_um2_s": float(cfg.oxidation_B_um2_s),
                "gamma": float(cfg.oxidation_gamma),
                "apply_on": str(cfg.oxidation_apply_on),
                "consume_dopants": bool(cfg.oxidation_consume_dopants),
                "update_materials": bool(cfg.oxidation_update_materials),
            }
        )

    steps.append(
        {
            "type": "implant",
            "dose_cm2": float(cfg.dose_cm2),
            "Rp_um": float(cfg.Rp_um),
            "dRp_um": float(cfg.dRp_um),
        }
    )

    anneal_step: dict[str, Any] = {
        "type": "anneal",
        "total_t_s": float(cfg.total_t_s),
        "dt_s": float(cfg.dt_s),
        "top_bc": _top_bc_from_gui(cfg),
        "oxide": {"D_scale": float(cfg.oxide_D_scale)},
    }
    if float(cfg.cap_eps_um) > 0.0:
        anneal_step["cap_eps_um"] = float(cfg.cap_eps_um)

    if bool(cfg.anneal_use_arrhenius):
        diff_cfg: dict[str, Any] = {
            "model": "arrhenius",
            "D0_cm2_s": float(cfg.arrhenius_D0_cm2_s),
            "Ea_eV": float(cfg.arrhenius_Ea_eV),
        }
        if cfg.arrhenius_schedule:
            diff_cfg["schedule"] = [
                {"t_s": float(seg["t_s"]), "T_C": float(seg["T_C"])} for seg in cfg.arrhenius_schedule
            ]
        else:
            diff_cfg["T_C"] = float(cfg.arrhenius_T_C)
        anneal_step["diffusivity"] = diff_cfg
    else:
        anneal_step["D_cm2_s"] = float(cfg.D_cm2_s)

    if bool(cfg.record_history):
        anneal_step["record"] = {
            "enable": True,
            "every_s": float(cfg.history_every_s),
            "save_csv": bool(cfg.history_save_csv),
            "save_png": bool(cfg.history_save_png),
        }
    steps.append(anneal_step)

    if bool(cfg.compute_metrics):
        x_ref = float(cfg.linecut_x_um)
        y_ref = float(cfg.linecut_y_um)
        steps.append(
            {
                "type": "analyze",
                "junctions": [
                    {"x_um": x_ref, "threshold_cm3": 1.0e17},
                    {"x_um": x_ref, "threshold_cm3": 1.0e18},
                ],
                "laterals": [{"y_um": y_ref, "threshold_cm3": 1.0e17}],
                "sheet_dose": {"save_csv": True},
                "save": {"json": True, "csv": True},
            }
        )

    formats = [str(fmt).lower() for fmt in cfg.formats]
    if not formats:
        formats = ["npy"]
    if bool(cfg.export_vtk) and "vtk" not in formats:
        formats.append("vtk")

    steps.append(
        {
            "type": "export",
            "outdir": str(cfg.outdir),
            "formats": formats,
            "linecuts": [
                {"kind": "vertical", "x_um": float(cfg.linecut_x_um)},
                {"kind": "horizontal", "y_um": float(cfg.linecut_y_um)},
            ],
            "plot": {
                "log10": bool(cfg.plot_log10),
                "vmin": float(cfg.plot_vmin) if cfg.plot_vmin is not None else None,
                "vmax": float(cfg.plot_vmax) if cfg.plot_vmax is not None else None,
            },
            "extra": {
                "tox_csv": bool(cfg.export_tox_csv),
                "tox_png": bool(cfg.export_tox_png),
            },
        }
    )

    return deck
