"""Anneal step runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...domain.constants import KB_EV_K
from ...domain.state import SimulationState
from ...errors import DeckError
from ...io import save_history_csv, save_history_png
from ...physics.diffusion import anneal_implicit, anneal_implicit_with_history, parse_top_bc_config
from ...physics.mask import full_open_mask
from ...units import ensure_positive
from .common import opt_mapping, required, to_float


def arrhenius_diffusivity(D0_cm2_s: float, Ea_eV: float, T_C: float) -> float:
    """Compute Arrhenius diffusivity for given temperature in Celsius."""
    ensure_positive("D0_cm2_s", float(D0_cm2_s), allow_zero=True)
    ensure_positive("Ea_eV", float(Ea_eV), allow_zero=True)
    T_K = float(T_C) + 273.15
    ensure_positive("T_K", T_K)
    return float(D0_cm2_s) * float(np.exp(-float(Ea_eV) / (KB_EV_K * T_K)))


def _anneal_diffusivity_segments(step: dict[str, Any], context: str) -> tuple[list[tuple[float, float]], float]:
    dt_s = to_float(required(step, "dt_s", context), "dt_s", context)
    ensure_positive(f"{context}.dt_s", dt_s)

    if "D_cm2_s" in step:
        D = to_float(required(step, "D_cm2_s", context), "D_cm2_s", context)
        total_t = to_float(required(step, "total_t_s", context), "total_t_s", context)
        ensure_positive(f"{context}.D_cm2_s", D, allow_zero=True)
        ensure_positive(f"{context}.total_t_s", total_t, allow_zero=True)
        return [(float(total_t), float(D))], float(dt_s)

    diff_cfg = opt_mapping(step.get("diffusivity"), f"{context}.diffusivity")
    model = str(diff_cfg.get("model", "")).lower()
    if model != "arrhenius":
        raise DeckError(f"{context}: provide either D_cm2_s or diffusivity.model='arrhenius'.")

    D0 = to_float(required(diff_cfg, "D0_cm2_s", f"{context}.diffusivity"), "D0_cm2_s", f"{context}.diffusivity")
    Ea = to_float(required(diff_cfg, "Ea_eV", f"{context}.diffusivity"), "Ea_eV", f"{context}.diffusivity")
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
            seg_t = to_float(
                required(seg, "t_s", f"{context}.diffusivity.schedule[{s_idx}]"),
                "t_s",
                f"{context}.diffusivity.schedule[{s_idx}]",
            )
            seg_T = to_float(
                required(seg, "T_C", f"{context}.diffusivity.schedule[{s_idx}]"),
                "T_C",
                f"{context}.diffusivity.schedule[{s_idx}]",
            )
            ensure_positive(f"{context}.diffusivity.schedule[{s_idx}].t_s", seg_t)
            D_seg = arrhenius_diffusivity(D0, Ea, seg_T)
            segments.append((float(seg_t), float(D_seg)))
            t_sum += float(seg_t)

        if "total_t_s" in step:
            total_t = to_float(step["total_t_s"], "total_t_s", context)
            if abs(total_t - t_sum) > max(1e-12, 1e-9 * max(total_t, t_sum, 1.0)):
                raise DeckError(
                    f"{context}.total_t_s ({total_t}) must match schedule duration sum ({t_sum})."
                )
        return segments, float(dt_s)

    total_t = to_float(required(step, "total_t_s", context), "total_t_s", context)
    T_C = to_float(required(diff_cfg, "T_C", f"{context}.diffusivity"), "T_C", f"{context}.diffusivity")
    ensure_positive(f"{context}.total_t_s", total_t, allow_zero=True)
    D = arrhenius_diffusivity(D0, Ea, T_C)
    return [(float(total_t), float(D))], float(dt_s)


def _open_fraction_with_oxide_cap(state: SimulationState, base_mask: np.ndarray, cap_eps_um: float) -> np.ndarray:
    state.ensure_oxide_fields()
    tox = np.asarray(state.tox_um, dtype=float)
    cap_open = tox <= float(cap_eps_um)
    return np.asarray(base_mask, dtype=float) * cap_open.astype(float)


def run_anneal_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    """Run anneal with constant/Arrhenius diffusivity and optional history."""
    context = f"steps[{idx}] (anneal)"
    state.ensure_oxide_fields()

    top_bc = parse_top_bc_config(step.get("top_bc"))
    segments, dt_s = _anneal_diffusivity_segments(step, context)

    mask_eff = state.mask_eff if state.mask_eff is not None else full_open_mask(state.grid.Nx)

    oxide_cfg = opt_mapping(step.get("oxide"), f"{context}.oxide")
    D_scale = to_float(oxide_cfg.get("D_scale", 0.0), "D_scale", f"{context}.oxide")
    ensure_positive(f"{context}.oxide.D_scale", D_scale, allow_zero=True)

    cap_eps_um = to_float(step.get("cap_eps_um", 0.5 * state.grid.dy_um), "cap_eps_um", context)
    ensure_positive(f"{context}.cap_eps_um", cap_eps_um, allow_zero=True)
    mask_eff_bc = _open_fraction_with_oxide_cap(state, mask_eff, cap_eps_um=cap_eps_um)

    record_cfg = opt_mapping(step.get("record"), f"{context}.record")
    record_enable = bool(record_cfg.get("enable", False))
    record_every_s = to_float(record_cfg.get("every_s", dt_s), "every_s", f"{context}.record")
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
            D_field = np.full(state.grid.shape, float(D_si), dtype=float)
            D_field[np.asarray(state.materials) == 1] *= float(D_scale)
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
        hist_outdir = state.resolve_outdir(outdir_step=record_cfg.get("outdir"))
        if bool(record_cfg.get("save_csv", True)):
            state.exports.append(save_history_csv(history_all, hist_outdir, filename="history.csv"))
        if bool(record_cfg.get("save_png", True)) and history_all:
            state.exports.append(save_history_png(history_all, hist_outdir, filename="history.png"))
