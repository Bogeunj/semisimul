"""Unit tests for export manager behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from proc2d.export.manager import export_results
from proc2d.grid import Grid2D


pytestmark = pytest.mark.unit


def test_export_results_rejects_unknown_format(tmp_path: Path) -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.2, Nx=5, Ny=3)
    C = np.ones(grid.shape, dtype=float)

    with pytest.raises(ValueError, match=r"Unsupported export format\(s\)"):
        export_results(C, grid, tmp_path, formats=["npy", "badfmt"])


def test_export_results_writes_default_csv_linecut_when_none_requested(
    tmp_path: Path,
) -> None:
    grid = Grid2D.from_domain(Lx_um=1.0, Ly_um=0.2, Nx=5, Ny=3)
    C = np.arange(grid.Nx * grid.Ny, dtype=float).reshape(grid.shape)

    written = export_results(C, grid, tmp_path, formats=["csv"], linecuts=None)
    assert len(written) == 1

    linecut_path = written[0]
    assert linecut_path.exists()
    assert linecut_path.name.startswith("linecut_vertical_x")
    assert linecut_path.suffix == ".csv"

    text = linecut_path.read_text(encoding="utf-8")
    assert "kind,vertical" in text
    assert "y_um,C_cm3" in text
