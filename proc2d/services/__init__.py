"""Service-layer entry points for proc2d."""

from .compare_service import build_metric_delta_rows, scalarize_metrics
from .simulation_service import (
    SimulationService,
    UnsupportedStepTypeError,
    build_default_simulation_service,
    build_simulation_service,
)

__all__ = [
    "SimulationService",
    "UnsupportedStepTypeError",
    "build_default_simulation_service",
    "build_metric_delta_rows",
    "build_simulation_service",
    "scalarize_metrics",
]
