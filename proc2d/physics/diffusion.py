"""Diffusion physics aliases."""

from __future__ import annotations

from ..diffusion import (
    TopBCConfig,
    anneal_implicit,
    anneal_implicit_with_history,
    parse_top_bc_config,
    top_flux_out,
    top_open_fraction_with_cap,
    total_amount,
)

__all__ = [
    "TopBCConfig",
    "anneal_implicit",
    "anneal_implicit_with_history",
    "parse_top_bc_config",
    "top_open_fraction_with_cap",
    "top_flux_out",
    "total_amount",
]
