"""Adapter tests for CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from proc2d.app.cli import main


pytestmark = pytest.mark.adapter


def test_cli_run_command_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    deck = {
        "domain": {"Lx_um": 0.5, "Ly_um": 0.2, "Nx": 21, "Ny": 11},
        "background_doping_cm3": 1.0e15,
        "steps": [
            {"type": "implant", "dose_cm2": 1.0e12, "Rp_um": 0.02, "dRp_um": 0.01},
            {"type": "anneal", "D_cm2_s": 1.0e-14, "total_t_s": 0.2, "dt_s": 0.1},
            {"type": "export", "outdir": "outputs/adapter_cli", "formats": ["npy"]},
        ],
    }
    deck_path = tmp_path / "deck_cli.yaml"
    deck_path.write_text(yaml.safe_dump(deck, sort_keys=False), encoding="utf-8")

    rc = main(["run", str(deck_path), "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Done. Grid=" in captured.out
    assert "Output:" in captured.out
    assert (tmp_path / "out" / "C.npy").exists()
