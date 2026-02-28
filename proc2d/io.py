"""Backward-compatible output helpers delegating to export modules."""

from __future__ import annotations

from pathlib import Path

from .export.csv_writer import save_linecuts_csv, save_sheet_dose_vs_x_csv, save_tox_vs_x_csv
from .export.history_writer import save_history_csv, save_history_png
from .export.manager import export_results
from .export.metrics_writer import _flatten_metrics, save_metrics_csv, save_metrics_json
from .export.npy_writer import save_field_npy
from .export.png_writer import save_heatmap_png, save_tox_vs_x_png
from .export.vtk_writer import _save_scalar_vtk_structured_points, save_vtk_structured_points


def ensure_outdir(outdir: str | Path) -> Path:
    """Create output directory if needed and return resolved path."""
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


__all__ = [
    "ensure_outdir",
    "save_field_npy",
    "save_linecuts_csv",
    "save_heatmap_png",
    "save_metrics_json",
    "save_metrics_csv",
    "_flatten_metrics",
    "save_sheet_dose_vs_x_csv",
    "save_history_csv",
    "save_history_png",
    "save_tox_vs_x_csv",
    "save_tox_vs_x_png",
    "_save_scalar_vtk_structured_points",
    "save_vtk_structured_points",
    "export_results",
]
