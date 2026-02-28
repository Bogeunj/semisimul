"""Physics-layer module aliases."""

from .diffusion import anneal_implicit, anneal_implicit_with_history, parse_top_bc_config
from .implant import implant_2d_gaussian
from .mask import build_mask_1d, full_open_mask, openings_from_any, smooth_mask_1d, validate_mask
from .oxidation import apply_oxidation, build_material_map

__all__ = [
    "anneal_implicit",
    "anneal_implicit_with_history",
    "apply_oxidation",
    "build_mask_1d",
    "build_material_map",
    "full_open_mask",
    "implant_2d_gaussian",
    "openings_from_any",
    "parse_top_bc_config",
    "smooth_mask_1d",
    "validate_mask",
]
