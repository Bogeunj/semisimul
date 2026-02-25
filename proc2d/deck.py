"""YAML deck parser and execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .diffusion import (
    anneal_implicit,
    anneal_implicit_with_history,
    parse_top_bc_config,
    top_open_fraction_with_cap,
)
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
from .metrics import iso_contour_area, junction_depth, lateral_extents_at_y, peak_info, sheet_dose_vs_x, total_mass
from .oxidation import apply_oxidation, build_material_map
from .units import ensure_positive

KB_EV_K = 8.617333262145e-5


class DeckError(ValueError):
    """Raised when a deck is invalid or execution fails."""


@dataclass
class SimulationState:
    """In-memory state while running a process deck."""

    deck_path: Path
    grid: Grid2D
    C: np.ndarray
    out_override: str | Path | None = None
    default_export_outdir_step: str | None = None
    mask_eff: np.ndarray | None = None
    tox_um: np.ndarray | None = None
    materials: np.ndarray | None = None
    metrics: dict[str, Any] | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    exports: list[Path] = field(default_factory=list)


def _as_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeckError(f"{context} must be a mapping.")
    return value


def _opt_mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise DeckError(f"{context} must be a mapping.")
    return value


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise DeckError(f"Missing required key '{key}' in {context}.")
    return mapping[key]


def _to_float(value: Any, key: str, context: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise DeckError(f"{context}.{key} must be a number, got {value!r}.") from exc


def _to_int(value: Any, key: str, context: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise DeckError(f"{context}.{key} must be an integer, got {value!r}.") from exc


def load_deck(deck_path: str | Path) -> dict[str, Any]:
    """Load YAML deck from file."""
    path = Path(deck_path)
    if not path.exists():
        raise DeckError(f"Deck file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise DeckError(f"Failed to parse YAML deck: {path}") from exc

    if payload is None:
        raise DeckError(f"Deck is empty: {path}")
    return _as_mapping(payload, "deck")


def normalize_deck(deck: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _as_mapping(deck, "deck")

    raw_steps = _required(payload, "steps", "deck")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise DeckError("deck.steps must be a non-empty list.")

    steps: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_steps):
        steps.append(_as_mapping(raw, f"steps[{idx}]"))

    _as_mapping(_required(payload, "domain", "deck"), "deck.domain")

    normalized = dict(payload)
    normalized["steps"] = steps
    return normalized, steps


def _resolve_outdir(deck_path: Path, outdir_step: str | None, out_override: str | Path | None) -> Path:
    if out_override is not None:
        path = Path(out_override)
        if not path.is_absolute():
            path = path.resolve()
        return path

    path = Path("outputs/run") if outdir_step is None else Path(outdir_step)
    if not path.is_absolute():
        path = (deck_path.parent / path).resolve()
    return path


def _default_export_outdir_step(steps: list[dict[str, Any]]) -> str | None:
    for step in steps:
        if str(step.get("type", "")).lower() == "export":
            if "outdir" in step:
                return str(step["outdir"])
            return None
    return None


def _resolve_state_outdir(state: SimulationState, outdir_step: str | None = None) -> Path:
    target = outdir_step if outdir_step is not None else state.default_export_outdir_step
    return _resolve_outdir(state.deck_path, target, state.out_override)


def _ensure_state_oxide_fields(state: SimulationState) -> None:
    if state.tox_um is None:
        state.tox_um = np.zeros(state.grid.Nx, dtype=float)
    tox = np.asarray(state.tox_um, dtype=float)
    if tox.shape != (state.grid.Nx,):
        raise DeckError(f"state.tox_um must have shape ({state.grid.Nx},), got {tox.shape}")
    if np.any(tox < 0.0):
        raise DeckError("state.tox_um must be non-negative")
    state.tox_um = tox

    if state.materials is None:
        state.materials = build_material_map(state.grid, tox)
    else:
        mat = np.asarray(state.materials)
        if mat.shape != state.grid.shape:
            raise DeckError(f"state.materials must have shape {state.grid.shape}, got {mat.shape}")
        state.materials = mat.astype(np.int8)


def _build_initial_state(
    deck: dict[str, Any],
    deck_path: Path,
    steps: list[dict[str, Any]],
    out_override: str | Path | None,
) -> SimulationState:
    domain = _as_mapping(_required(deck, "domain", "deck"), "deck.domain")

    Lx_um = _to_float(_required(domain, "Lx_um", "deck.domain"), "Lx_um", "deck.domain")
    Ly_um = _to_float(_required(domain, "Ly_um", "deck.domain"), "Ly_um", "deck.domain")
    Nx = _to_int(_required(domain, "Nx", "deck.domain"), "Nx", "deck.domain")
    Ny = _to_int(_required(domain, "Ny", "deck.domain"), "Ny", "deck.domain")

    if "background_doping_cm3" in deck:
        background_raw = deck["background_doping_cm3"]
        background_ctx = "deck"
    else:
        background_raw = domain.get("background_doping_cm3", 0.0)
        background_ctx = "deck.domain"
    background = _to_float(background_raw, "background_doping_cm3", background_ctx)

    ensure_positive("background_doping_cm3", background, allow_zero=True)
    grid = Grid2D.from_domain(Lx_um=Lx_um, Ly_um=Ly_um, Nx=Nx, Ny=Ny)
    C0 = np.full(grid.shape, background, dtype=float)

    state = SimulationState(
        deck_path=deck_path,
        grid=grid,
        C=C0,
        out_override=out_override,
        default_export_outdir_step=_default_export_outdir_step(steps),
        tox_um=np.zeros(grid.Nx, dtype=float),
    )
    assert state.tox_um is not None
    state.materials = build_material_map(grid, state.tox_um)
    return state


def _run_mask_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    context = f"steps[{idx}] (mask)"
    openings_raw = _required(step, "openings_um", context)
    if not isinstance(openings_raw, list):
        raise DeckError(f"{context}.openings_um must be a list of [start, end] pairs.")

    sigma_lat_um = _to_float(step.get("sigma_lat_um", 0.0), "sigma_lat_um", context)
    if sigma_lat_um < 0.0:
        raise DeckError(f"{context}.sigma_lat_um must be >= 0.")

    openings = openings_from_any(openings_raw)
    mask_raw = build_mask_1d(state.grid.x_um, openings)
    state.mask_eff = smooth_mask_1d(mask_raw, sigma_lat_um=sigma_lat_um, dx_um=state.grid.dx_um)
    validate_mask(state.mask_eff, state.grid.Nx)


def _run_oxidation_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    context = f"steps[{idx}] (oxidation)"
    _ensure_state_oxide_fields(state)
    assert state.tox_um is not None

    model = str(step.get("model", "deal_grove")).lower()
    if model != "deal_grove":
        raise DeckError(f"{context}.model must be 'deal_grove', got '{model}'.")

    time_s = _to_float(_required(step, "time_s", context), "time_s", context)
    A_um = _to_float(_required(step, "A_um", context), "A_um", context)
    B_um2_s = _to_float(_required(step, "B_um2_s", context), "B_um2_s", context)
    gamma = _to_float(step.get("gamma", 2.27), "gamma", context)
    ensure_positive(f"{context}.time_s", time_s, allow_zero=True)
    ensure_positive(f"{context}.A_um", A_um, allow_zero=True)
    ensure_positive(f"{context}.B_um2_s", B_um2_s, allow_zero=True)
    ensure_positive(f"{context}.gamma", gamma)

    apply_on = str(step.get("apply_on", "all")).lower()
    consume_dopants = bool(step.get("consume_dopants", True))
    update_materials = bool(step.get("update_materials", True))
    mask_weighting = str(step.get("mask_weighting", "binary")).lower()
    open_threshold = _to_float(step.get("open_threshold", 0.5), "open_threshold", context)
    if open_threshold < 0.0 or open_threshold > 1.0:
        raise DeckError(f"{context}.open_threshold must be within [0, 1].")

    if "tox_init_um" in step and np.allclose(state.tox_um, 0.0):
        tox_init = _to_float(step["tox_init_um"], "tox_init_um", context)
        ensure_positive(f"{context}.tox_init_um", tox_init, allow_zero=True)
        state.tox_um = np.full(state.grid.Nx, tox_init, dtype=float)

    mask_eff = state.mask_eff if state.mask_eff is not None else full_open_mask(state.grid.Nx)

    Cn, tox_new, materials_new, _ = apply_oxidation(
        state.C,
        state.grid,
        state.tox_um,
        mask_eff,
        time_s=time_s,
        A_um=A_um,
        B_um2_s=B_um2_s,
        gamma=gamma,
        apply_on=apply_on,
        consume_dopants=consume_dopants,
        mask_weighting=mask_weighting,
        open_threshold=open_threshold,
    )

    if np.any(tox_new > float(state.grid.Ly_um) + 1e-12):
        raise DeckError(
            f"{context} failed: oxide thickness exceeds domain depth Ly_um={state.grid.Ly_um}"
        )

    state.C = Cn
    state.tox_um = tox_new
    if update_materials:
        state.materials = materials_new


def _run_implant_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    context = f"steps[{idx}] (implant)"
    dose_cm2 = _to_float(_required(step, "dose_cm2", context), "dose_cm2", context)
    Rp_um = _to_float(_required(step, "Rp_um", context), "Rp_um", context)
    dRp_um = _to_float(_required(step, "dRp_um", context), "dRp_um", context)

    ensure_positive(f"{context}.dose_cm2", dose_cm2)
    ensure_positive(f"{context}.dRp_um", dRp_um)
    _ensure_state_oxide_fields(state)

    mask_eff = state.mask_eff if state.mask_eff is not None else full_open_mask(state.grid.Nx)
    dC = implant_2d_gaussian(
        grid=state.grid,
        dose_cm2=dose_cm2,
        Rp_um=Rp_um,
        dRp_um=dRp_um,
        mask_eff=mask_eff,
        tox_um=state.tox_um,
    )
    state.C = state.C + dC


def _arrhenius_D(D0_cm2_s: float, Ea_eV: float, T_C: float) -> float:
    ensure_positive("D0_cm2_s", float(D0_cm2_s), allow_zero=True)
    ensure_positive("Ea_eV", float(Ea_eV), allow_zero=True)
    T_K = float(T_C) + 273.15
    ensure_positive("T_K", T_K)
    return float(D0_cm2_s) * float(np.exp(-float(Ea_eV) / (KB_EV_K * T_K)))


def _anneal_diffusivity_segments(step: dict[str, Any], context: str) -> tuple[list[tuple[float, float]], float]:
    dt_s = _to_float(_required(step, "dt_s", context), "dt_s", context)
    ensure_positive(f"{context}.dt_s", dt_s)

    if "D_cm2_s" in step:
        D = _to_float(_required(step, "D_cm2_s", context), "D_cm2_s", context)
        total_t = _to_float(_required(step, "total_t_s", context), "total_t_s", context)
        ensure_positive(f"{context}.D_cm2_s", D, allow_zero=True)
        ensure_positive(f"{context}.total_t_s", total_t, allow_zero=True)
        return [(float(total_t), float(D))], float(dt_s)

    diff_cfg = _opt_mapping(step.get("diffusivity"), f"{context}.diffusivity")
    model = str(diff_cfg.get("model", "")).lower()
    if model != "arrhenius":
        raise DeckError(f"{context}: provide either D_cm2_s or diffusivity.model='arrhenius'.")

    D0 = _to_float(_required(diff_cfg, "D0_cm2_s", f"{context}.diffusivity"), "D0_cm2_s", f"{context}.diffusivity")
    Ea = _to_float(_required(diff_cfg, "Ea_eV", f"{context}.diffusivity"), "Ea_eV", f"{context}.diffusivity")
    ensure_positive(f"{context}.diffusivity.D0_cm2_s", D0, allow_zero=True)
    ensure_positive(f"{context}.diffusivity.Ea_eV", Ea, allow_zero=True)

    if "schedule" in diff_cfg:
        schedule = diff_cfg["schedule"]
        if not isinstance(schedule, list) or not schedule:
            raise DeckError(f"{context}.diffusivity.schedule must be a non-empty list.")
        segments: list[tuple[float, float]] = []
        t_sum = 0.0
        for s_idx, seg in enumerate(schedule):
            if not isinstance(seg, dict):
                raise DeckError(f"{context}.diffusivity.schedule[{s_idx}] must be a mapping.")
            seg_t = _to_float(_required(seg, "t_s", f"{context}.diffusivity.schedule[{s_idx}]"), "t_s", f"{context}.diffusivity.schedule[{s_idx}]")
            seg_T = _to_float(_required(seg, "T_C", f"{context}.diffusivity.schedule[{s_idx}]"), "T_C", f"{context}.diffusivity.schedule[{s_idx}]")
            ensure_positive(f"{context}.diffusivity.schedule[{s_idx}].t_s", seg_t)
            D_seg = _arrhenius_D(D0, Ea, seg_T)
            segments.append((float(seg_t), float(D_seg)))
            t_sum += float(seg_t)

        if "total_t_s" in step:
            total_t = _to_float(step["total_t_s"], "total_t_s", context)
            if abs(total_t - t_sum) > max(1e-12, 1e-9 * max(total_t, t_sum, 1.0)):
                raise DeckError(
                    f"{context}.total_t_s ({total_t}) must match schedule duration sum ({t_sum})."
                )
        return segments, float(dt_s)

    total_t = _to_float(_required(step, "total_t_s", context), "total_t_s", context)
    T_C = _to_float(_required(diff_cfg, "T_C", f"{context}.diffusivity"), "T_C", f"{context}.diffusivity")
    ensure_positive(f"{context}.total_t_s", total_t, allow_zero=True)
    D = _arrhenius_D(D0, Ea, T_C)
    return [(float(total_t), float(D))], float(dt_s)


def _open_fraction_with_oxide_cap(
    state: SimulationState,
    base_mask: np.ndarray,
    cap_eps_um: float,
    cap_model: str = "hard",
    cap_len_um: float | None = None,
) -> np.ndarray:
    _ensure_state_oxide_fields(state)
    tox = np.asarray(state.tox_um, dtype=float)
    return top_open_fraction_with_cap(
        np.asarray(base_mask, dtype=float),
        tox,
        cap_eps_um=float(cap_eps_um),
        cap_model=str(cap_model),
        cap_len_um=cap_len_um,
    )


def _run_anneal_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    context = f"steps[{idx}] (anneal)"
    _ensure_state_oxide_fields(state)

    top_bc = parse_top_bc_config(step.get("top_bc"))
    segments, dt_s = _anneal_diffusivity_segments(step, context)

    mask_eff = state.mask_eff if state.mask_eff is not None else full_open_mask(state.grid.Nx)

    oxide_cfg = _opt_mapping(step.get("oxide"), f"{context}.oxide")
    D_scale = _to_float(oxide_cfg.get("D_scale", 0.0), "D_scale", f"{context}.oxide")
    ensure_positive(f"{context}.oxide.D_scale", D_scale, allow_zero=True)

    cap_eps_um = _to_float(step.get("cap_eps_um", 0.5 * state.grid.dy_um), "cap_eps_um", context)
    ensure_positive(f"{context}.cap_eps_um", cap_eps_um, allow_zero=True)

    cap_model = str(step.get("cap_model", "hard")).lower()
    cap_len_raw = step.get("cap_len_um")
    cap_len_um: float | None
    if cap_len_raw is None:
        cap_len_um = max(0.5 * state.grid.dy_um, 1e-12) if cap_model == "exp" else None
    else:
        cap_len_um = _to_float(cap_len_raw, "cap_len_um", context)
        ensure_positive(f"{context}.cap_len_um", cap_len_um)

    try:
        mask_eff_bc = _open_fraction_with_oxide_cap(
            state,
            mask_eff,
            cap_eps_um=cap_eps_um,
            cap_model=cap_model,
            cap_len_um=cap_len_um,
        )
    except ValueError as exc:
        raise DeckError(f"{context} invalid cap settings: {exc}") from exc

    record_cfg = _opt_mapping(step.get("record"), f"{context}.record")
    record_enable = bool(record_cfg.get("enable", False))
    record_every_s = _to_float(record_cfg.get("every_s", dt_s), "every_s", f"{context}.record")
    ensure_positive(f"{context}.record.every_s", record_every_s)

    C_curr = state.C
    history_all: list[dict[str, float]] = []
    t_offset = 0.0

    for s_idx, (seg_t, D_si) in enumerate(segments):
        ensure_positive(f"{context}.segment[{s_idx}].t_s", seg_t, allow_zero=True)
        ensure_positive(f"{context}.segment[{s_idx}].D_cm2_s", D_si, allow_zero=True)
        if seg_t == 0.0:
            continue

        D_field: float | np.ndarray
        if state.materials is not None:
            D_field_arr = np.full(state.grid.shape, float(D_si), dtype=float)
            D_field_arr[np.asarray(state.materials) == 1] *= float(D_scale)
            D_field = D_field_arr
        else:
            D_field = float(D_si)

        if record_enable:
            C_curr, seg_hist = anneal_implicit_with_history(
                C0=C_curr,
                grid=state.grid,
                D_cm2_s=D_field,
                total_t_s=seg_t,
                dt_s=dt_s,
                top_bc=top_bc,
                mask_eff=mask_eff_bc,
                record_enable=True,
                record_every_s=record_every_s,
            )
            for h_idx, row in enumerate(seg_hist):
                if s_idx > 0 and h_idx == 0:
                    continue
                merged = dict(row)
                merged["time_s"] = float(merged["time_s"]) + t_offset
                history_all.append(merged)
        else:
            C_curr = anneal_implicit(
                C0=C_curr,
                grid=state.grid,
                D_cm2_s=D_field,
                total_t_s=seg_t,
                dt_s=dt_s,
                top_bc=top_bc,
                mask_eff=mask_eff_bc,
            )

        t_offset += seg_t

    state.C = C_curr
    if record_enable:
        state.history = history_all
        hist_outdir = _resolve_state_outdir(state, outdir_step=record_cfg.get("outdir"))
        if bool(record_cfg.get("save_csv", True)):
            state.exports.append(save_history_csv(history_all, hist_outdir, filename="history.csv"))
        if bool(record_cfg.get("save_png", True)) and history_all:
            state.exports.append(save_history_png(history_all, hist_outdir, filename="history.png"))


def _run_analyze_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    context = f"steps[{idx}] (analyze)"
    outdir = _resolve_state_outdir(state, outdir_step=step.get("outdir"))

    silicon_only = bool(step.get("silicon_only", False))
    C_eval = np.asarray(state.C, dtype=float)
    if silicon_only:
        _ensure_state_oxide_fields(state)
        C_eval = C_eval.copy()
        C_eval[np.asarray(state.materials) == 1] = 0.0

    report: dict[str, Any] = {
        "silicon_only": silicon_only,
        "total_mass_cm1": float(total_mass(C_eval, state.grid)),
        "peak": peak_info(C_eval, state.grid),
    }

    junction_specs = step.get("junctions", [])
    if not isinstance(junction_specs, list):
        raise DeckError(f"{context}.junctions must be a list.")
    junction_results: list[dict[str, Any]] = []
    for j_idx, item in enumerate(junction_specs):
        if not isinstance(item, dict):
            raise DeckError(f"{context}.junctions[{j_idx}] must be a mapping.")
        x_um = _to_float(_required(item, "x_um", f"{context}.junctions[{j_idx}]"), "x_um", f"{context}.junctions[{j_idx}]")
        th = _to_float(
            _required(item, "threshold_cm3", f"{context}.junctions[{j_idx}]"),
            "threshold_cm3",
            f"{context}.junctions[{j_idx}]",
        )
        ensure_positive(f"{context}.junctions[{j_idx}].threshold_cm3", th)
        depth_um = junction_depth(C_eval, state.grid, x_um=x_um, threshold_cm3=th)
        junction_results.append(
            {
                "x_um_requested": float(x_um),
                "x_um_used": float(state.grid.x_um[state.grid.nearest_x_index(x_um)]),
                "threshold_cm3": float(th),
                "depth_um": None if depth_um is None else float(depth_um),
            }
        )
    if junction_results:
        report["junctions"] = junction_results

    lateral_specs = step.get("laterals", [])
    if not isinstance(lateral_specs, list):
        raise DeckError(f"{context}.laterals must be a list.")
    lateral_results: list[dict[str, Any]] = []
    for l_idx, item in enumerate(lateral_specs):
        if not isinstance(item, dict):
            raise DeckError(f"{context}.laterals[{l_idx}] must be a mapping.")
        y_um = _to_float(_required(item, "y_um", f"{context}.laterals[{l_idx}]"), "y_um", f"{context}.laterals[{l_idx}]")
        th = _to_float(
            _required(item, "threshold_cm3", f"{context}.laterals[{l_idx}]"),
            "threshold_cm3",
            f"{context}.laterals[{l_idx}]",
        )
        ensure_positive(f"{context}.laterals[{l_idx}].threshold_cm3", th)
        lateral_results.append(lateral_extents_at_y(C_eval, state.grid, y_um=y_um, threshold_cm3=th))
    if lateral_results:
        report["laterals"] = lateral_results

    if "iso_area" in step:
        iso_cfg = _opt_mapping(step.get("iso_area"), f"{context}.iso_area")
        iso_th = _to_float(
            _required(iso_cfg, "threshold_cm3", f"{context}.iso_area"),
            "threshold_cm3",
            f"{context}.iso_area",
        )
        ensure_positive(f"{context}.iso_area.threshold_cm3", iso_th)
        method = str(iso_cfg.get("method", "cell_count")).lower()
        report["iso_area_um2"] = float(iso_contour_area(C_eval, state.grid, iso_th, method=method))
        report["iso_area_method"] = method
    elif "iso_area_threshold_cm3" in step:
        iso_th = _to_float(step["iso_area_threshold_cm3"], "iso_area_threshold_cm3", context)
        ensure_positive(f"{context}.iso_area_threshold_cm3", iso_th)
        report["iso_area_um2"] = float(iso_contour_area(C_eval, state.grid, iso_th, method="cell_count"))

    sheet_cfg = _opt_mapping(step.get("sheet_dose"), f"{context}.sheet_dose")
    save_sheet_csv = bool(sheet_cfg.get("save_csv", False))
    if save_sheet_csv:
        sd = sheet_dose_vs_x(C_eval, state.grid)
        state.exports.append(save_sheet_dose_vs_x_csv(state.grid.x_um, sd, outdir, filename="sheet_dose_vs_x.csv"))
        report["sheet_dose_summary"] = {
            "min_cm2": float(np.min(sd)),
            "max_cm2": float(np.max(sd)),
            "mean_cm2": float(np.mean(sd)),
        }

    save_cfg = _opt_mapping(step.get("save"), f"{context}.save")
    save_json = bool(save_cfg.get("json", True))
    save_csv = bool(save_cfg.get("csv", True))
    if save_json:
        state.exports.append(save_metrics_json(report, outdir, filename="metrics.json"))
    if save_csv:
        state.exports.append(save_metrics_csv(report, outdir, filename="metrics.csv"))

    state.metrics = report


def _run_export_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    context = f"steps[{idx}] (export)"
    outdir = _resolve_state_outdir(state, outdir_step=step.get("outdir"))
    formats = step.get("formats", ["npy"])
    if not isinstance(formats, list) or not formats:
        raise DeckError(f"{context}.formats must be a non-empty list.")

    linecuts = step.get("linecuts", [])
    if not isinstance(linecuts, list):
        raise DeckError(f"{context}.linecuts must be a list.")

    plot_cfg = step.get("plot", {})
    if plot_cfg is None:
        plot_cfg = {}
    if not isinstance(plot_cfg, dict):
        raise DeckError(f"{context}.plot must be a mapping.")

    extra_cfg = _opt_mapping(step.get("extra"), f"{context}.extra")

    try:
        written = export_results(
            C=state.C,
            grid=state.grid,
            outdir=outdir,
            formats=formats,
            linecuts=linecuts,
            plot_cfg=plot_cfg,
            tox_um=state.tox_um,
            materials=state.materials,
            extra=extra_cfg,
        )
    except ValueError as exc:
        raise DeckError(f"{context} failed: {exc}") from exc
    state.exports.extend(written)


def run_deck_data(
    deck: dict[str, Any],
    *,
    deck_path: str | Path | None = None,
    out_override: str | Path | None = None,
) -> SimulationState:
    """Run all process steps from an in-memory deck mapping."""
    if deck_path is None:
        deck_path_obj = (Path.cwd() / "_inline_deck.yaml").resolve()
    else:
        deck_path_obj = Path(deck_path).resolve()

    normalized, steps = normalize_deck(deck)
    state = _build_initial_state(normalized, deck_path_obj, steps=steps, out_override=out_override)

    for idx, step in enumerate(steps):
        stype = str(_required(step, "type", f"steps[{idx}]"))
        stype_l = stype.lower()

        try:
            if stype_l == "mask":
                _run_mask_step(state, step, idx)
            elif stype_l == "oxidation":
                _run_oxidation_step(state, step, idx)
            elif stype_l == "implant":
                _run_implant_step(state, step, idx)
            elif stype_l == "anneal":
                _run_anneal_step(state, step, idx)
            elif stype_l == "analyze":
                _run_analyze_step(state, step, idx)
            elif stype_l == "export":
                _run_export_step(state, step, idx)
            else:
                raise DeckError(
                    f"steps[{idx}].type '{stype_l}' is not supported. "
                    "Use one of: mask, oxidation, implant, anneal, analyze, export."
                )
        except DeckError:
            raise
        except Exception as exc:
            raise DeckError(f"Step {idx} ('{stype_l}') failed: {exc}") from exc

    return state


def run_deck(deck_path: str | Path, out_override: str | Path | None = None) -> SimulationState:
    """Run all process steps from a YAML deck."""
    path = Path(deck_path).resolve()
    deck = load_deck(path)
    from .engine import run_deck_mapping

    out_path = None if out_override is None else Path(out_override)
    result = run_deck_mapping(
        deck,
        base_dir=path.parent,
        out_override=out_path,
    )
    return result.state
