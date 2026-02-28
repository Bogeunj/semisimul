from __future__ import annotations

import yaml

from proc2d.deck import run_deck


def test_analyze_iso_area_tri_linear_option(tmp_path) -> None:
    deck = {
        "domain": {"Lx_um": 1.0, "Ly_um": 0.3, "Nx": 81, "Ny": 61},
        "background_doping_cm3": 1.0e15,
        "steps": [
            {
                "type": "implant",
                "dose_cm2": 1.0e13,
                "Rp_um": 0.04,
                "dRp_um": 0.015,
            },
            {
                "type": "analyze",
                "iso_area": {
                    "threshold_cm3": 1.0e17,
                    "method": "tri_linear",
                },
                "save": {"json": True, "csv": True},
            },
            {
                "type": "export",
                "outdir": "outputs/iso_area",
                "formats": ["npy"],
            },
        ],
    }

    deck_path = tmp_path / "deck_iso_area.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False), encoding="utf-8")
    state = run_deck(deck_path, out_override=tmp_path / "out")
    assert state.metrics is not None
    assert "iso_area_um2" in state.metrics
    assert state.metrics.get("iso_area_method") == "tri_linear"
