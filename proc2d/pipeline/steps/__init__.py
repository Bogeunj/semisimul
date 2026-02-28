"""Default step runner registry mapping."""

from __future__ import annotations

from ..step_base import StepRunner
from .analyze_step import run_analyze_step
from .anneal_step import arrhenius_diffusivity, run_anneal_step
from .export_step import run_export_step
from .implant_step import run_implant_step
from .mask_step import run_mask_step
from .oxidation_step import run_oxidation_step


def build_default_step_handlers() -> dict[str, StepRunner]:
    """Return default step-type -> runner mapping."""
    return {
        "mask": run_mask_step,
        "oxidation": run_oxidation_step,
        "implant": run_implant_step,
        "anneal": run_anneal_step,
        "analyze": run_analyze_step,
        "export": run_export_step,
    }


__all__ = ["arrhenius_diffusivity", "build_default_step_handlers"]
