"""Typed models for deck-level configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class DomainConfig:
    """Simulation domain configuration."""

    Lx_um: float
    Ly_um: float
    Nx: int
    Ny: int
    background_doping_cm3: float = 0.0


@dataclass(frozen=True)
class MaskStepConfig:
    """Typed mask step config."""

    type: Literal["mask"] = "mask"
    openings_um: list[list[float]] = field(default_factory=list)
    sigma_lat_um: float = 0.0


@dataclass(frozen=True)
class OxidationStepConfig:
    """Typed oxidation step config."""

    type: Literal["oxidation"] = "oxidation"
    model: Literal["deal_grove"] = "deal_grove"
    time_s: float = 0.0
    A_um: float = 0.0
    B_um2_s: float = 0.0
    gamma: float = 2.27
    apply_on: Literal["all", "open", "blocked"] = "all"
    consume_dopants: bool = True
    update_materials: bool = True


@dataclass(frozen=True)
class ImplantStepConfig:
    """Typed implant step config."""

    type: Literal["implant"] = "implant"
    dose_cm2: float = 0.0
    Rp_um: float = 0.0
    dRp_um: float = 0.0


@dataclass(frozen=True)
class AnnealStepConfig:
    """Typed anneal step config."""

    type: Literal["anneal"] = "anneal"
    total_t_s: float = 0.0
    dt_s: float = 0.0


@dataclass(frozen=True)
class AnalyzeStepConfig:
    """Typed analyze step config."""

    type: Literal["analyze"] = "analyze"
    silicon_only: bool = False


@dataclass(frozen=True)
class ExportStepConfig:
    """Typed export step config."""

    type: Literal["export"] = "export"
    outdir: str = "outputs/run"
    formats: list[str] = field(default_factory=lambda: ["npy"])
