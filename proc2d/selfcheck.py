from __future__ import annotations

import importlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .engine import run_deck_mapping


@dataclass
class CheckRow:
    name: str
    ok: bool
    detail: str


@dataclass
class SelfCheckReport:
    rows: list[CheckRow]

    @property
    def ok(self) -> bool:
        return all(row.ok for row in self.rows)

    def to_text(self) -> str:
        lines: list[str] = []
        for row in self.rows:
            status = "OK" if row.ok else "FAIL"
            lines.append(f"[{status}] {row.name}: {row.detail}")
        lines.append(f"overall: {'OK' if self.ok else 'FAIL'}")
        return "\n".join(lines)


def run_selfcheck(*, smoke: bool = True) -> SelfCheckReport:
    rows: list[CheckRow] = []

    for module_name in ("numpy", "scipy", "yaml", "matplotlib"):
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            rows.append(CheckRow(module_name, True, f"version={version}"))
        except Exception as exc:
            rows.append(CheckRow(module_name, False, str(exc)))

    if smoke and all(row.ok for row in rows):
        deck = {
            "domain": {
                "Lx_um": 0.4,
                "Ly_um": 0.2,
                "Nx": 41,
                "Ny": 31,
            },
            "background_doping_cm3": 1.0e15,
            "steps": [
                {
                    "type": "mask",
                    "openings_um": [[0.15, 0.25]],
                    "sigma_lat_um": 0.01,
                },
                {
                    "type": "implant",
                    "dose_cm2": 1.0e13,
                    "Rp_um": 0.03,
                    "dRp_um": 0.01,
                },
                {
                    "type": "anneal",
                    "D_cm2_s": 1.0e-14,
                    "total_t_s": 0.2,
                    "dt_s": 0.1,
                    "top_bc": {
                        "open": {"type": "neumann"},
                        "blocked": {"type": "neumann"},
                    },
                },
                {
                    "type": "analyze",
                    "junctions": [{"x_um": 0.2, "threshold_cm3": 1.0e17}],
                    "save": {"json": True, "csv": True},
                },
                {
                    "type": "export",
                    "outdir": "outputs/selfcheck",
                    "formats": ["npy", "csv", "png"],
                },
            ],
        }

        try:
            with tempfile.TemporaryDirectory(prefix="proc2d-selfcheck-") as tmp:
                outdir = Path(tmp) / "out"
                result = run_deck_mapping(deck, base_dir=Path(tmp), out_override=outdir)
                expected = [
                    outdir / "C.npy",
                    outdir / "C.png",
                    outdir / "metrics.json",
                    outdir / "metrics.csv",
                ]
                missing = [str(path.name) for path in expected if not path.exists()]
                if missing:
                    rows.append(CheckRow("smoke", False, f"missing artifacts: {', '.join(missing)}"))
                else:
                    rows.append(
                        CheckRow(
                            "smoke",
                            True,
                            f"grid={result.state.grid.shape}, exports={len(result.state.exports)}",
                        )
                    )
        except Exception as exc:
            rows.append(CheckRow("smoke", False, str(exc)))

    return SelfCheckReport(rows=rows)
