"""Integration tests for pipeline/service execution path."""

from __future__ import annotations

from pathlib import Path

import pytest

from proc2d.services import build_default_simulation_service


pytestmark = pytest.mark.integration


def test_service_run_payload_smoke(tmp_path: Path) -> None:
    deck = {
        "domain": {"Lx_um": 1.0, "Ly_um": 0.3, "Nx": 31, "Ny": 21},
        "background_doping_cm3": 1.0e15,
        "steps": [
            {"type": "implant", "dose_cm2": 2.0e12, "Rp_um": 0.03, "dRp_um": 0.01},
            {"type": "anneal", "D_cm2_s": 1.0e-14, "total_t_s": 0.2, "dt_s": 0.1},
            {
                "type": "export",
                "outdir": "outputs/it_engine",
                "formats": ["npy"],
            },
        ],
    }

    service = build_default_simulation_service()
    state = service.run_payload(deck, deck_path=tmp_path / "inmem.yaml", out_override=tmp_path / "out")

    assert state.grid.shape == (21, 31)
    assert state.exports
    assert (tmp_path / "out" / "C.npy").exists()
