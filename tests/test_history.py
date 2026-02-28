"""Anneal history integration tests."""

from __future__ import annotations

import csv

import pytest
import yaml

from proc2d.deck import run_deck


pytestmark = pytest.mark.integration


def test_anneal_history_csv_created_and_monotonic(tmp_path) -> None:
    deck = {
        "domain": {
            "Lx_um": 1.0,
            "Ly_um": 0.3,
            "Nx": 41,
            "Ny": 31,
        },
        "background_doping_cm3": 1.0e15,
        "steps": [
            {
                "type": "implant",
                "dose_cm2": 8.0e12,
                "Rp_um": 0.04,
                "dRp_um": 0.015,
            },
            {
                "type": "anneal",
                "D_cm2_s": 1.0e-14,
                "total_t_s": 1.0,
                "dt_s": 0.25,
                "top_bc": {
                    "open": {"type": "robin", "h_cm_s": 1.0e-5, "Ceq_cm3": 0.0},
                    "blocked": {"type": "neumann"},
                },
                "record": {
                    "enable": True,
                    "every_s": 0.25,
                    "save_csv": True,
                    "save_png": False,
                },
            },
            {
                "type": "export",
                "outdir": "outputs/test_history",
                "formats": ["npy"],
            },
        ],
    }
    deck_path = tmp_path / "deck_history.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False), encoding="utf-8")

    outdir = tmp_path / "out"
    state = run_deck(deck_path, out_override=outdir)

    history_csv = outdir / "history.csv"
    assert history_csv.exists()
    assert len(state.history) >= 2

    with history_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert rows
    assert "time_s" in rows[0]
    assert "mass" in rows[0]

    times = [float(r["time_s"]) for r in rows]
    for i in range(len(times) - 1):
        assert times[i] <= times[i + 1] + 1e-15
