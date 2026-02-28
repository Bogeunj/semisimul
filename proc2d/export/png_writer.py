"""PNG writers for heatmap and tox profile."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from ..grid import Grid2D


def _ensure_outdir(outdir: str | Path) -> Path:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_heatmap_png(
    C: np.ndarray,
    grid: Grid2D,
    outdir: str | Path,
    filename: str = "C.png",
    plot_cfg: dict | None = None,
) -> Path:
    """Save 2D concentration heatmap as PNG."""
    out = _ensure_outdir(outdir)
    path = out / filename

    cfg = dict(plot_cfg or {})
    use_log10 = bool(cfg.get("log10", False))
    vmin = cfg.get("vmin")
    vmax = cfg.get("vmax")

    arr = np.asarray(C, dtype=float)
    if use_log10:
        floor = 1e10
        if vmin is not None:
            floor = max(floor, float(vmin))
        plot_arr = np.log10(np.clip(arr, floor, None))
        vmin_plot = np.log10(float(vmin)) if vmin is not None else None
        vmax_plot = np.log10(float(vmax)) if vmax is not None else None
        cbar_label = "log10(C [cm^-3])"
    else:
        plot_arr = arr
        vmin_plot = float(vmin) if vmin is not None else None
        vmax_plot = float(vmax) if vmax is not None else None
        cbar_label = "C [cm^-3]"

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    im = ax.imshow(
        plot_arr,
        extent=(float(grid.x_um[0]), float(grid.x_um[-1]), float(grid.y_um[-1]), float(grid.y_um[0])),
        aspect="auto",
        origin="upper",
        cmap="inferno",
        vmin=vmin_plot,
        vmax=vmax_plot,
    )
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title("Dopant concentration")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_tox_vs_x_png(tox_um: np.ndarray, grid: Grid2D, outdir: str | Path, filename: str = "tox_vs_x.png") -> Path:
    """Save oxide thickness profile tox(x) as PNG."""
    tox = np.asarray(tox_um, dtype=float)
    if tox.shape != (grid.Nx,):
        raise ValueError(f"tox_um must have shape ({grid.Nx},), got {tox.shape}.")

    path = _ensure_outdir(outdir) / filename
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=140)
    ax.plot(grid.x_um, tox, lw=2.0)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("tox [um]")
    ax.set_title("Oxide thickness profile")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
