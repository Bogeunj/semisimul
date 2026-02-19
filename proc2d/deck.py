"""YAML deck parser and execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .diffusion import anneal_implicit, parse_top_bc_config
from .grid import Grid2D
from .implant import implant_2d_gaussian
from .io import export_results
from .mask import build_mask_1d, full_open_mask, openings_from_any, smooth_mask_1d, validate_mask
from .units import ensure_positive


class DeckError(ValueError):
    """Raised when a deck is invalid or execution fails."""


@dataclass
class SimulationState:
    """In-memory state while running a process deck."""

    deck_path: Path
    grid: Grid2D
    C: np.ndarray
    mask_eff: np.ndarray | None = None
    exports: list[Path] = field(default_factory=list)


def _as_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeckError(f"{context} must be a mapping.")
    return value


def _required(step: dict[str, Any], key: str, context: str) -> Any:
    if key not in step:
        raise DeckError(f"Missing required key '{key}' in {context}.")
    return step[key]


def _to_float(value: Any, key: str, context: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise DeckError(f"{context}.{key} must be a number, got {value!r}.") from exc


def _to_int(value: Any, key: str, context: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise DeckError(f"{context}.{key} must be an integer, got {value!r}.") from exc
    return out


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


def _build_initial_state(deck: dict[str, Any], deck_path: Path) -> SimulationState:
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
    return SimulationState(deck_path=deck_path, grid=grid, C=C0)


def _resolve_outdir(
    deck_path: Path,
    outdir_step: str | None,
    out_override: str | Path | None,
) -> Path:
    if out_override is not None:
        path = Path(out_override)
        if not path.is_absolute():
            path = path.resolve()
        return path
    else:
        if outdir_step is None:
            path = Path("outputs/run")
        else:
            path = Path(outdir_step)

    if not path.is_absolute():
        path = (deck_path.parent / path).resolve()
    return path


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


def _run_implant_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    context = f"steps[{idx}] (implant)"
    dose_cm2 = _to_float(_required(step, "dose_cm2", context), "dose_cm2", context)
    Rp_um = _to_float(_required(step, "Rp_um", context), "Rp_um", context)
    dRp_um = _to_float(_required(step, "dRp_um", context), "dRp_um", context)

    ensure_positive(f"{context}.dose_cm2", dose_cm2)
    ensure_positive(f"{context}.dRp_um", dRp_um)

    mask_eff = state.mask_eff if state.mask_eff is not None else full_open_mask(state.grid.Nx)
    dC = implant_2d_gaussian(
        grid=state.grid,
        dose_cm2=dose_cm2,
        Rp_um=Rp_um,
        dRp_um=dRp_um,
        mask_eff=mask_eff,
    )
    state.C = state.C + dC


def _run_anneal_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    context = f"steps[{idx}] (anneal)"
    D_cm2_s = _to_float(_required(step, "D_cm2_s", context), "D_cm2_s", context)
    total_t_s = _to_float(_required(step, "total_t_s", context), "total_t_s", context)
    dt_s = _to_float(_required(step, "dt_s", context), "dt_s", context)
    ensure_positive(f"{context}.D_cm2_s", D_cm2_s, allow_zero=True)
    ensure_positive(f"{context}.total_t_s", total_t_s, allow_zero=True)
    ensure_positive(f"{context}.dt_s", dt_s)

    top_bc = parse_top_bc_config(step.get("top_bc"))
    mask_eff = state.mask_eff if state.mask_eff is not None else full_open_mask(state.grid.Nx)
    state.C = anneal_implicit(
        C0=state.C,
        grid=state.grid,
        D_cm2_s=D_cm2_s,
        total_t_s=total_t_s,
        dt_s=dt_s,
        top_bc=top_bc,
        mask_eff=mask_eff,
    )


def _run_export_step(
    state: SimulationState,
    step: dict[str, Any],
    idx: int,
    out_override: str | Path | None,
) -> None:
    context = f"steps[{idx}] (export)"
    outdir = _resolve_outdir(
        state.deck_path,
        outdir_step=step.get("outdir"),
        out_override=out_override,
    )
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

    try:
        written = export_results(
            C=state.C,
            grid=state.grid,
            outdir=outdir,
            formats=formats,
            linecuts=linecuts,
            plot_cfg=plot_cfg,
        )
    except ValueError as exc:
        raise DeckError(f"{context} failed: {exc}") from exc
    state.exports.extend(written)


def run_deck(deck_path: str | Path, out_override: str | Path | None = None) -> SimulationState:
    """Run all process steps from a YAML deck."""
    deck_path = Path(deck_path).resolve()
    deck = load_deck(deck_path)

    steps = _required(deck, "steps", "deck")
    if not isinstance(steps, list) or not steps:
        raise DeckError("deck.steps must be a non-empty list.")

    state = _build_initial_state(deck, deck_path)

    for idx, raw_step in enumerate(steps):
        step = _as_mapping(raw_step, f"steps[{idx}]")
        stype = str(_required(step, "type", f"steps[{idx}]")).lower()

        try:
            if stype == "mask":
                _run_mask_step(state, step, idx)
            elif stype == "implant":
                _run_implant_step(state, step, idx)
            elif stype == "anneal":
                _run_anneal_step(state, step, idx)
            elif stype == "export":
                _run_export_step(state, step, idx, out_override=out_override)
            else:
                raise DeckError(
                    f"steps[{idx}].type '{stype}' is not supported. "
                    "Use one of: mask, implant, anneal, export."
                )
        except DeckError:
            raise
        except Exception as exc:
            raise DeckError(f"Step {idx} ('{stype}') failed: {exc}") from exc

    return state
