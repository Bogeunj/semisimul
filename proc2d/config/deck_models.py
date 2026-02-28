"""Typed models for deck-level configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainConfig:
    """Simulation domain configuration."""

    Lx_um: float
    Ly_um: float
    Nx: int
    Ny: int
    background_doping_cm3: float = 0.0
