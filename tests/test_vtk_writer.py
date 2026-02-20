"""Tests for legacy VTK export writer."""

from __future__ import annotations

import numpy as np

from proc2d.grid import Grid2D
from proc2d.io import save_vtk_structured_points


def test_save_vtk_structured_points_smoke(tmp_path) -> None:
    grid = Grid2D.from_domain(Lx_um=2.0, Ly_um=1.0, Nx=3, Ny=2)
    C = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=float,
    )

    vtk_path = save_vtk_structured_points(C, grid, outdir=tmp_path, filename="C.vtk")
    text = vtk_path.read_text(encoding="utf-8")

    assert "DATASET STRUCTURED_POINTS" in text
    assert "DIMENSIONS 3 2 1" in text
    assert "SCALARS doping float 1" in text
    assert "LOOKUP_TABLE default" in text

    lines = text.splitlines()
    idx = lines.index("LOOKUP_TABLE default")
    data_tokens = " ".join(lines[idx + 1 :]).split()
    assert len(data_tokens) == grid.Nx * grid.Ny
    assert abs(float(data_tokens[0]) - 1.0) < 1e-12
    assert abs(float(data_tokens[-1]) - 6.0) < 1e-12
