"""NumPy field writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_field_npy(C: np.ndarray, outdir: str | Path, filename: str = "C.npy") -> Path:
    """Save concentration field as .npy."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    np.save(path, np.asarray(C, dtype=float))
    return path
