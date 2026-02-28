"""Simulation runtime state and initialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config.deck_models import DomainConfig
from ..errors import DeckError
from ..oxidation import build_material_map
from .grid import Grid2D
from .units import ensure_positive


def resolve_outdir(
    deck_path: Path,
    outdir_step: str | None,
    out_override: str | Path | None,
) -> Path:
    """Resolve output directory from override/step/default values."""
    if out_override is not None:
        path = Path(out_override)
        if not path.is_absolute():
            path = path.resolve()
        return path

    path = Path("outputs/run") if outdir_step is None else Path(outdir_step)
    if not path.is_absolute():
        path = (deck_path.parent / path).resolve()
    return path


def default_export_outdir_step(steps: list[dict[str, Any]]) -> str | None:
    """Find default outdir from first export step, if present."""
    for step in steps:
        if str(step.get("type", "")).lower() == "export":
            if "outdir" in step:
                return str(step["outdir"])
            return None
    return None


@dataclass
class SimulationState:
    """In-memory simulation state shared across step runners."""

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

    def resolve_outdir(self, outdir_step: str | None = None) -> Path:
        """Resolve outdir using state deck path/defaults/override."""
        target = outdir_step if outdir_step is not None else self.default_export_outdir_step
        return resolve_outdir(self.deck_path, target, self.out_override)

    def ensure_oxide_fields(self) -> None:
        """Ensure tox/material fields exist and have compatible shapes."""
        if self.tox_um is None:
            self.tox_um = np.zeros(self.grid.Nx, dtype=float)
        tox = np.asarray(self.tox_um, dtype=float)
        if tox.shape != (self.grid.Nx,):
            raise DeckError(f"state.tox_um must have shape ({self.grid.Nx},), got {tox.shape}")
        if np.any(tox < 0.0):
            raise DeckError("state.tox_um must be non-negative")
        self.tox_um = tox

        if self.materials is None:
            self.materials = build_material_map(self.grid, tox)
        else:
            mat = np.asarray(self.materials)
            if mat.shape != self.grid.shape:
                raise DeckError(f"state.materials must have shape {self.grid.shape}, got {mat.shape}")
            self.materials = mat.astype(np.int8)


def build_initial_state(
    *,
    domain: DomainConfig,
    deck_path: Path,
    steps: list[dict[str, Any]],
    out_override: str | Path | None,
) -> SimulationState:
    """Construct simulation state from validated domain config."""
    ensure_positive("background_doping_cm3", domain.background_doping_cm3, allow_zero=True)
    grid = Grid2D.from_domain(Lx_um=domain.Lx_um, Ly_um=domain.Ly_um, Nx=domain.Nx, Ny=domain.Ny)
    C0 = np.full(grid.shape, domain.background_doping_cm3, dtype=float)

    state = SimulationState(
        deck_path=deck_path,
        grid=grid,
        C=C0,
        out_override=out_override,
        default_export_outdir_step=default_export_outdir_step(steps),
        tox_um=np.zeros(grid.Nx, dtype=float),
    )
    assert state.tox_um is not None
    state.materials = build_material_map(grid, state.tox_um)
    return state
