"""Anneal history writers."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def _ensure_outdir(outdir: str | Path) -> Path:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_history_csv(history: list[dict[str, float]], outdir: str | Path, filename: str = "history.csv") -> Path:
    """Save anneal history list into CSV."""
    path = _ensure_outdir(outdir) / filename
    fieldnames: list[str] = []
    for row in history:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["time_s", "mass", "peak_cm3", "flux_out", "residual"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    return path


def save_history_png(history: list[dict[str, float]], outdir: str | Path, filename: str = "history.png") -> Path:
    """Save anneal history plot (mass / flux / residual vs time)."""
    if not history:
        raise ValueError("History is empty; nothing to plot.")

    t = np.asarray([float(row.get("time_s", np.nan)) for row in history], dtype=float)
    mass = np.asarray([float(row.get("mass", np.nan)) for row in history], dtype=float)
    flux = np.asarray([float(row.get("flux_out", np.nan)) for row in history], dtype=float)
    residual = np.asarray([float(row.get("residual", np.nan)) for row in history], dtype=float)

    path = _ensure_outdir(outdir) / filename
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7.2, 6.0), dpi=140, sharex=True)

    axes[0].plot(t, mass, color="#1f77b4", lw=1.8)
    axes[0].set_ylabel("mass [cm^-1]")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, flux, color="#ff7f0e", lw=1.6)
    axes[1].set_ylabel("flux_out [cm^-1 s^-1]")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, residual, color="#2ca02c", lw=1.6)
    axes[2].set_ylabel("residual")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Anneal history", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
