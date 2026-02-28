"""Streamlit session-state helpers for proc2d GUI."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

LAST_RESULT_KEY = "proc2d_last_result"
RECENT_RUNS_KEY = "proc2d_recent_runs"
COMPARE_LINECUT_KEY = "compare-linecut-kind"


def get_last_result() -> dict[str, Any] | None:
    """Return last simulation result from session state."""
    result = st.session_state.get(LAST_RESULT_KEY)
    if isinstance(result, dict):
        return result
    return None


def set_last_result(result: dict[str, Any]) -> None:
    """Store last simulation result in session state."""
    st.session_state[LAST_RESULT_KEY] = result


def get_recent_runs() -> list[dict[str, Any]]:
    """Return recent run records used by compare tab."""
    runs = st.session_state.get(RECENT_RUNS_KEY, [])
    if isinstance(runs, list):
        return runs
    return []


def push_recent_run(result: dict[str, Any], params: dict[str, Any]) -> None:
    """Append run snapshot to session for 2-run compare."""
    runs = get_recent_runs()

    outdir = Path(result["outdir"])
    png_path = outdir / "C.png"
    png_written = any(Path(p).name == "C.png" for p in result.get("written", []))
    png_bytes = png_path.read_bytes() if (png_written and png_path.exists()) else b""

    store_full = bool(params.get("store_full_c", False))
    run_record = {
        "run_id": dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "params": dict(params),
        "plot_cfg": dict(result.get("plot_cfg", {})),
        "png_bytes": png_bytes,
        "metrics": result.get("metrics"),
        "linecuts": {
            "vertical": {
                "coord_um": np.asarray(result["linecuts"]["vertical"]["coord_um"], dtype=float),
                "values_cm3": np.asarray(result["linecuts"]["vertical"]["values_cm3"], dtype=float),
            },
            "horizontal": {
                "coord_um": np.asarray(result["linecuts"]["horizontal"]["coord_um"], dtype=float),
                "values_cm3": np.asarray(result["linecuts"]["horizontal"]["values_cm3"], dtype=float),
            },
        },
        "grid": {
            "Lx_um": float(result["grid"].Lx_um),
            "Ly_um": float(result["grid"].Ly_um),
            "Nx": int(result["grid"].Nx),
            "Ny": int(result["grid"].Ny),
        },
        "C": np.asarray(result["C"], dtype=float).copy() if store_full else None,
    }
    runs.append(run_record)
    st.session_state[RECENT_RUNS_KEY] = runs[-2:]
