"""Domain-layer types and constants."""

from .constants import KB_EV_K
from .grid import Grid2D
from .state import SimulationState, build_initial_state
from .units import ensure_positive

__all__ = ["Grid2D", "KB_EV_K", "SimulationState", "build_initial_state", "ensure_positive"]
