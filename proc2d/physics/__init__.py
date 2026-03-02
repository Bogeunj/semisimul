"""Physics-layer module aliases."""

from .diffusion import (
    TopBCConfig,
    anneal_implicit,
    anneal_implicit_with_history,
    parse_top_bc_config,
    total_amount,
)
from .deposition import apply_deposition
from .electrical import channel_doping_from_state, estimate_mosfet_metrics
from .etch import apply_etch
from .implant import implant_2d_gaussian
from .mask import (
    build_mask_1d,
    full_open_mask,
    openings_from_any,
    smooth_mask_1d,
    validate_mask,
)
from .oxidation import (
    apply_oxidation,
    apply_surface_outward_shift,
    build_material_map,
    deal_grove_tox_update,
)

__all__ = [
    "anneal_implicit",
    "anneal_implicit_with_history",
    "apply_deposition",
    "apply_etch",
    "apply_oxidation",
    "apply_surface_outward_shift",
    "build_mask_1d",
    "build_material_map",
    "channel_doping_from_state",
    "deal_grove_tox_update",
    "full_open_mask",
    "implant_2d_gaussian",
    "estimate_mosfet_metrics",
    "openings_from_any",
    "parse_top_bc_config",
    "smooth_mask_1d",
    "TopBCConfig",
    "total_amount",
    "validate_mask",
]
