"""Export step runner."""

from __future__ import annotations

from typing import Any

from ...domain.state import SimulationState
from ...errors import DeckError
from ...io import export_results
from .common import opt_mapping


def run_export_step(state: SimulationState, step: dict[str, Any], idx: int) -> None:
    """Persist outputs in configured formats."""
    context = f"steps[{idx}] (export)"
    outdir = state.resolve_outdir(outdir_step=step.get("outdir"))
    formats = step.get("formats", ["npy"])
    if not isinstance(formats, list) or not formats:
        raise DeckError(f"{context}.formats must be a non-empty list.")

    linecuts = step.get("linecuts", [])
    if not isinstance(linecuts, list):
        raise DeckError(f"{context}.linecuts must be a list.")

    plot_cfg = step.get("plot", {})
    if plot_cfg is None:
        plot_cfg = {}
    if not isinstance(plot_cfg, dict):
        raise DeckError(f"{context}.plot must be a mapping.")

    extra_cfg = opt_mapping(step.get("extra"), f"{context}.extra")

    try:
        written = export_results(
            C=state.C,
            grid=state.grid,
            outdir=outdir,
            formats=formats,
            linecuts=linecuts,
            plot_cfg=plot_cfg,
            tox_um=state.tox_um,
            materials=state.materials,
            extra=extra_cfg,
        )
    except ValueError as exc:
        raise DeckError(f"{context} failed: {exc}") from exc
    state.exports.extend(written)
