"""Metric report generation and persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.grid import Grid2D
from ..io import save_metrics_csv, save_metrics_json, save_sheet_dose_vs_x_csv
from .metrics import iso_contour_area, junction_depth, lateral_extents_at_y, peak_info, sheet_dose_vs_x, total_mass


@dataclass(frozen=True)
class AnalyzeArtifacts:
    """Report payload and persisted artifact paths."""

    report: dict[str, Any]
    written: list[Path]


def _silicon_eval_field(
    C: np.ndarray,
    materials: np.ndarray | None,
    *,
    silicon_only: bool,
) -> np.ndarray:
    arr = np.asarray(C, dtype=float)
    if not silicon_only:
        return arr
    if materials is None:
        raise ValueError("materials is required when silicon_only=True")
    out = arr.copy()
    out[np.asarray(materials) == 1] = 0.0
    return out


def build_metrics_report(
    *,
    C: np.ndarray,
    grid: Grid2D,
    silicon_only: bool,
    materials: np.ndarray | None,
    junction_specs: list[dict[str, float]],
    lateral_specs: list[dict[str, float]],
    iso_area_threshold_cm3: float | None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Build metrics report dictionary from validated analysis specs."""
    C_eval = _silicon_eval_field(C, materials, silicon_only=silicon_only)

    report: dict[str, Any] = {
        "silicon_only": silicon_only,
        "total_mass_cm1": float(total_mass(C_eval, grid)),
        "peak": peak_info(C_eval, grid),
    }

    if junction_specs:
        junction_results: list[dict[str, Any]] = []
        for spec in junction_specs:
            x_um = float(spec["x_um"])
            th = float(spec["threshold_cm3"])
            depth_um = junction_depth(C_eval, grid, x_um=x_um, threshold_cm3=th)
            junction_results.append(
                {
                    "x_um_requested": x_um,
                    "x_um_used": float(grid.x_um[grid.nearest_x_index(x_um)]),
                    "threshold_cm3": th,
                    "depth_um": None if depth_um is None else float(depth_um),
                }
            )
        report["junctions"] = junction_results

    if lateral_specs:
        lateral_results: list[dict[str, Any]] = []
        for spec in lateral_specs:
            lateral_results.append(
                lateral_extents_at_y(
                    C_eval,
                    grid,
                    y_um=float(spec["y_um"]),
                    threshold_cm3=float(spec["threshold_cm3"]),
                )
            )
        report["laterals"] = lateral_results

    if iso_area_threshold_cm3 is not None:
        report["iso_area_um2"] = float(iso_contour_area(C_eval, grid, float(iso_area_threshold_cm3)))

    return report, C_eval


def save_metrics_artifacts(
    *,
    report: dict[str, Any],
    C_eval: np.ndarray,
    grid: Grid2D,
    outdir: str | Path,
    save_json: bool,
    save_csv: bool,
    save_sheet_csv: bool,
) -> list[Path]:
    """Persist report and derived sheet-dose artifacts."""
    written: list[Path] = []
    if save_sheet_csv:
        sd = sheet_dose_vs_x(C_eval, grid)
        written.append(save_sheet_dose_vs_x_csv(grid.x_um, sd, outdir, filename="sheet_dose_vs_x.csv"))
        report["sheet_dose_summary"] = {
            "min_cm2": float(np.min(sd)),
            "max_cm2": float(np.max(sd)),
            "mean_cm2": float(np.mean(sd)),
        }

    if save_json:
        written.append(save_metrics_json(report, outdir, filename="metrics.json"))
    if save_csv:
        written.append(save_metrics_csv(report, outdir, filename="metrics.csv"))
    return written


def run_metrics_analysis(
    *,
    C: np.ndarray,
    grid: Grid2D,
    silicon_only: bool,
    materials: np.ndarray | None,
    junction_specs: list[dict[str, float]],
    lateral_specs: list[dict[str, float]],
    iso_area_threshold_cm3: float | None,
    outdir: str | Path,
    save_json: bool,
    save_csv: bool,
    save_sheet_csv: bool,
) -> AnalyzeArtifacts:
    """Run metrics analysis and persist requested artifacts."""
    report, C_eval = build_metrics_report(
        C=C,
        grid=grid,
        silicon_only=silicon_only,
        materials=materials,
        junction_specs=junction_specs,
        lateral_specs=lateral_specs,
        iso_area_threshold_cm3=iso_area_threshold_cm3,
    )
    written = save_metrics_artifacts(
        report=report,
        C_eval=C_eval,
        grid=grid,
        outdir=outdir,
        save_json=save_json,
        save_csv=save_csv,
        save_sheet_csv=save_sheet_csv,
    )
    return AnalyzeArtifacts(report=report, written=written)
