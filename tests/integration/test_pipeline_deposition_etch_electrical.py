"""Integration tests for deposition/etch/electrical process coupling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from proc2d.deck import run_deck


pytestmark = pytest.mark.integration


def _run_steps(tmp_path: Path, name: str, steps: list[dict[str, object]]):
    deck = {
        "domain": {"Lx_um": 1.0, "Ly_um": 0.3, "Nx": 81, "Ny": 61},
        "background_doping_cm3": 1.0e15,
        "steps": [
            *steps,
            {"type": "export", "outdir": f"outputs/{name}", "formats": ["npy"]},
        ],
    }
    deck_path = tmp_path / f"{name}.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False), encoding="utf-8")
    return run_deck(deck_path, out_override=tmp_path / name)


def _id_from_state(state) -> float:
    assert state.metrics is not None
    assert "electrical" in state.metrics
    return float(state.metrics["electrical"]["Id_A"])


def test_deposition_etch_electrical_end_to_end_smoke(tmp_path: Path) -> None:
    steps: list[dict[str, object]] = [
        {"type": "mask", "openings_um": [[0.35, 0.65]], "sigma_lat_um": 0.02},
        {
            "type": "deposition",
            "thickness_um": 0.03,
            "apply_on": "open",
            "mask_weighting": "time_scale",
        },
        {"type": "implant", "dose_cm2": 1.0e13, "Rp_um": 0.04, "dRp_um": 0.015},
        {"type": "anneal", "D_cm2_s": 1.0e-14, "total_t_s": 0.3, "dt_s": 0.1},
        {"type": "etch", "thickness_um": 0.01, "apply_on": "open"},
        {
            "type": "electrical",
            "model": "mosfet_long_channel",
            "Vgs_V": 2.5,
            "Vds_V": 0.1,
            "W_um": 1.0,
            "L_um": 0.5,
        },
    ]

    state = _run_steps(tmp_path, "dep_etch_elec", steps)
    assert state.tox_um is not None
    assert float(np.max(state.tox_um)) > 0.0
    assert state.materials is not None
    assert _id_from_state(state) >= 0.0
    assert (tmp_path / "dep_etch_elec" / "C.npy").exists()


def test_process_sequence_changes_id_trend(tmp_path: Path) -> None:
    common: list[dict[str, object]] = [
        {"type": "mask", "openings_um": [[0.35, 0.65]], "sigma_lat_um": 0.015},
        {"type": "implant", "dose_cm2": 1.0e13, "Rp_um": 0.04, "dRp_um": 0.015},
        {
            "type": "anneal",
            "D_cm2_s": 1.0e-14,
            "total_t_s": 0.4,
            "dt_s": 0.1,
            "top_bc": {
                "open": {"type": "robin", "h_cm_s": 1.0e-5, "Ceq_cm3": 0.0},
                "blocked": {"type": "neumann"},
            },
        },
        {
            "type": "electrical",
            "model": "mosfet_long_channel",
            "Vgs_V": 2.5,
            "Vds_V": 0.1,
            "W_um": 1.0,
            "L_um": 0.5,
        },
    ]

    baseline = _run_steps(tmp_path, "base_id", common)
    with_dep = _run_steps(
        tmp_path,
        "dep_id",
        [
            {"type": "deposition", "thickness_um": 0.04, "apply_on": "all"},
            *common,
        ],
    )
    with_dep_etch = _run_steps(
        tmp_path,
        "dep_etch_id",
        [
            {"type": "deposition", "thickness_um": 0.04, "apply_on": "all"},
            {"type": "etch", "thickness_um": 0.03, "apply_on": "all"},
            *common,
        ],
    )

    id_base = _id_from_state(baseline)
    id_dep = _id_from_state(with_dep)
    id_dep_etch = _id_from_state(with_dep_etch)

    assert id_dep < id_base
    assert id_dep_etch > id_dep
