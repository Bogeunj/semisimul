"""Simple electrical metric estimators."""

from __future__ import annotations

import math

import numpy as np

from .units import ensure_positive, um_to_cm


def estimate_mosfet_metrics(
    Nch_cm3: float,
    tox_um: float,
    mobility_cm2_Vs: float,
    Vgs_V: float,
    Vds_V: float,
    W_um: float,
    L_um: float,
    eps_ox_rel: float = 3.9,
    t_ref_um: float = 0.01,
    Nref_cm3: float = 1e17,
    phi0_V: float = 0.7,
) -> dict[str, float]:
    """Estimate long-channel MOSFET metrics from coarse process features."""
    ensure_positive("Nch_cm3", float(Nch_cm3), allow_zero=True)
    ensure_positive("tox_um", float(tox_um), allow_zero=True)
    ensure_positive("mobility_cm2_Vs", float(mobility_cm2_Vs))
    ensure_positive("W_um", float(W_um))
    ensure_positive("L_um", float(L_um))
    ensure_positive("eps_ox_rel", float(eps_ox_rel))
    ensure_positive("t_ref_um", float(t_ref_um))
    ensure_positive("Nref_cm3", float(Nref_cm3))
    ensure_positive("phi0_V", float(phi0_V), allow_zero=True)

    eps0_F_cm = 8.8541878128e-14
    tox_eff_um = max(float(tox_um), 1.0e-4)
    tox_cm = float(um_to_cm(tox_eff_um))
    Cox_F_cm2 = float(eps_ox_rel) * eps0_F_cm / tox_cm

    nch_eff = max(float(Nch_cm3), 1.0e8)
    log_term = math.log10(nch_eff / float(Nref_cm3))
    tox_ratio = tox_eff_um / float(t_ref_um)
    Vth_V = float(phi0_V) + 0.12 * log_term + 0.25 * (tox_ratio - 1.0)

    Vgs = float(Vgs_V)
    Vds = float(Vds_V)
    Vov = max(Vgs - Vth_V, 0.0)
    beta = float(mobility_cm2_Vs) * Cox_F_cm2 * (float(W_um) / float(L_um))

    if Vov <= 0.0 or Vds <= 0.0:
        Id_A = 0.0
    elif Vds < Vov:
        Id_A = beta * (Vov * Vds - 0.5 * Vds * Vds)
    else:
        Id_A = 0.5 * beta * Vov * Vov

    return {
        "Nch_cm3": float(Nch_cm3),
        "tox_um": float(tox_eff_um),
        "mobility_cm2_Vs": float(mobility_cm2_Vs),
        "Cox_F_cm2": float(Cox_F_cm2),
        "Vth_V": float(Vth_V),
        "Id_A": float(max(Id_A, 0.0)),
        "Vgs_V": float(Vgs),
        "Vds_V": float(Vds),
        "W_um": float(W_um),
        "L_um": float(L_um),
        "beta_A_V2": float(beta),
        "Vov_V": float(Vov),
    }


def channel_doping_from_state(C: np.ndarray, tox_um: np.ndarray, dy_um: float) -> float:
    """Estimate channel doping using a shallow silicon window."""
    arr = np.asarray(C, dtype=float)
    if arr.ndim != 2:
        raise ValueError("C must be a 2D array.")
    tox = np.asarray(tox_um, dtype=float)
    if tox.shape != (arr.shape[1],):
        raise ValueError(f"tox_um must have shape ({arr.shape[1]},), got {tox.shape}.")

    j_surface = np.floor(np.maximum(tox, 0.0) / float(dy_um)).astype(int)
    window = max(1, int(round(0.02 / max(float(dy_um), 1.0e-12))))

    samples: list[float] = []
    ny = arr.shape[0]
    for i in range(arr.shape[1]):
        j0 = min(max(int(j_surface[i]), 0), ny - 1)
        j1 = min(j0 + window, ny)
        samples.append(float(np.mean(arr[j0:j1, i])))

    return float(np.median(np.asarray(samples, dtype=float)))
