"""Analyze step runner."""

from __future__ import annotations

from typing import Any

from ...analysis.reports import run_metrics_analysis
from ...domain.state import SimulationState
from ...errors import DeckError
from ...units import ensure_positive
from .common import opt_mapping, required, to_float


def _parse_junction_specs(step: dict[str, Any], context: str) -> list[dict[str, float]]:
    raw_specs = step.get("junctions", [])
    if not isinstance(raw_specs, list):
        raise DeckError(f"{context}.junctions must be a list.")
    out: list[dict[str, float]] = []
    for j_idx, item in enumerate(raw_specs):
        if not isinstance(item, dict):
            raise DeckError(f"{context}.junctions[{j_idx}] must be a mapping.")
        x_um = to_float(required(item, "x_um", f"{context}.junctions[{j_idx}]"), "x_um", f"{context}.junctions[{j_idx}]")
        th = to_float(
            required(item, "threshold_cm3", f"{context}.junctions[{j_idx}]"),
            "threshold_cm3",
            f"{context}.junctions[{j_idx}]",
        )
        ensure_positive(f"{context}.junctions[{j_idx}].threshold_cm3", th)
        out.append({"x_um": float(x_um), "threshold_cm3": float(th)})
    return out


def _parse_lateral_specs(step: dict[str, Any], context: str) -> list[dict[str, float]]:
    raw_specs = step.get("laterals", [])
    if not isinstance(raw_specs, list):
        raise DeckError(f"{context}.laterals must be a list.")
    out: list[dict[str, float]] = []
    for l_idx, item in enumerate(raw_specs):
        if not isinstance(item, dict):
            raise DeckError(f"{context}.laterals[{l_idx}] must be a mapping.")
        y_um = to_float(required(item, "y_um", f"{context}.laterals[{l_idx}]"), "y_um", f"{context}.laterals[{l_idx}]")
        th = to_float(
            required(item, "threshold_cm3", f"{context}.laterals[{l_idx}]"),
            "threshold_cm3",
            f"{context}.laterals[{l_idx}]",
        )
        ensure_positive(f"{context}.laterals[{l_idx}].threshold_cm3", th)
        out.append({"y_um": float(y_um), "threshold_cm3": float(th)})
    return out


def run_analyze_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    """Run analysis report generation and artifact persistence."""
    context = f"steps[{idx}] (analyze)"
    outdir = state.resolve_outdir(outdir_step=step.get("outdir"))

    silicon_only = bool(step.get("silicon_only", False))
    if silicon_only:
        state.ensure_oxide_fields()

    junction_specs = _parse_junction_specs(step, context)
    lateral_specs = _parse_lateral_specs(step, context)

    iso_area_threshold_cm3: float | None = None
    if "iso_area_threshold_cm3" in step:
        iso_th = to_float(step["iso_area_threshold_cm3"], "iso_area_threshold_cm3", context)
        ensure_positive(f"{context}.iso_area_threshold_cm3", iso_th)
        iso_area_threshold_cm3 = float(iso_th)

    sheet_cfg = opt_mapping(step.get("sheet_dose"), f"{context}.sheet_dose")
    save_cfg = opt_mapping(step.get("save"), f"{context}.save")

    artifacts = run_metrics_analysis(
        C=state.C,
        grid=state.grid,
        silicon_only=silicon_only,
        materials=state.materials,
        junction_specs=junction_specs,
        lateral_specs=lateral_specs,
        iso_area_threshold_cm3=iso_area_threshold_cm3,
        outdir=outdir,
        save_json=bool(save_cfg.get("json", True)),
        save_csv=bool(save_cfg.get("csv", True)),
        save_sheet_csv=bool(sheet_cfg.get("save_csv", False)),
    )
    state.exports.extend(artifacts.written)
    state.metrics = artifacts.report
