"""Implicit diffusion solver and boundary conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve

from .grid import Grid2D
from .units import ensure_positive


@dataclass(frozen=True)
class TopBCConfig:
    """Top boundary configuration.

    open_type supports:
      - "neumann"
      - "robin"
      - "dirichlet" (demo fallback)
    blocked_type currently supports only "neumann".
    """

    open_type: str = "neumann"
    blocked_type: str = "neumann"
    h_cm_s: float = 0.0
    Ceq_cm3: float = 0.0
    dirichlet_value_cm3: float = 0.0


def parse_top_bc_config(top_bc: dict | None) -> TopBCConfig:
    """Parse top BC configuration from deck step."""
    if top_bc is None:
        return TopBCConfig()

    if not isinstance(top_bc, dict):
        raise ValueError("top_bc must be a mapping.")

    open_cfg = top_bc.get("open", {"type": "neumann"})
    blocked_cfg = top_bc.get("blocked", {"type": "neumann"})

    if not isinstance(open_cfg, dict):
        raise ValueError("top_bc.open must be a mapping.")
    if not isinstance(blocked_cfg, dict):
        raise ValueError("top_bc.blocked must be a mapping.")

    open_type = str(open_cfg.get("type", "neumann")).lower()
    blocked_type = str(blocked_cfg.get("type", "neumann")).lower()

    if blocked_type != "neumann":
        raise ValueError(
            "MVP/P2 supports only top_bc.blocked.type='neumann'. "
            f"Got '{blocked_type}'."
        )

    if open_type == "robin":
        h_cm_s = float(open_cfg.get("h_cm_s", 0.0))
        Ceq_cm3 = float(open_cfg.get("Ceq_cm3", 0.0))
        ensure_positive("top_bc.open.h_cm_s", h_cm_s, allow_zero=True)
        return TopBCConfig(
            open_type="robin",
            blocked_type="neumann",
            h_cm_s=h_cm_s,
            Ceq_cm3=Ceq_cm3,
        )

    if open_type == "dirichlet":
        value = float(open_cfg.get("value_cm3", 0.0))
        return TopBCConfig(
            open_type="dirichlet",
            blocked_type="neumann",
            dirichlet_value_cm3=value,
        )

    if open_type == "neumann":
        return TopBCConfig(open_type="neumann", blocked_type="neumann")

    raise ValueError(
        "top_bc.open.type must be one of: neumann, robin, dirichlet. "
        f"Got '{open_type}'."
    )


def _as_open_fraction(mask_eff: np.ndarray | None, nx: int) -> np.ndarray:
    if mask_eff is None:
        return np.ones(nx, dtype=float)
    arr = np.asarray(mask_eff, dtype=float)
    if arr.shape != (nx,):
        raise ValueError(f"mask_eff must have shape ({nx},), got {arr.shape}.")
    return np.clip(arr, 0.0, 1.0)


def _as_D_field(D_cm2_s: float | np.ndarray, grid: Grid2D) -> np.ndarray:
    if np.isscalar(D_cm2_s):
        D = float(np.asarray(D_cm2_s, dtype=float).item())
        ensure_positive("D_cm2_s", D, allow_zero=True)
        return np.full(grid.shape, D, dtype=float)

    arr = np.asarray(D_cm2_s, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"D field must have shape {grid.shape}, got {arr.shape}.")
    if np.any(arr < 0.0):
        raise ValueError("D field must be non-negative.")
    return arr


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def assemble_diffusion_operator(
    grid: Grid2D,
    D_cm2_s: float | np.ndarray,
    top_bc: TopBCConfig | None = None,
    mask_eff: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble ``dC/dt = A*C + b`` for scalar or field diffusivity.

    Spatial discretization:
    - regular grid finite-volume style
    - variable-D faces use harmonic mean
    - left/right/bottom: Neumann(0-flux)
    - top: mask-mixed open/blocked BC
    """
    if top_bc is None:
        top_bc = TopBCConfig()

    nx, ny = grid.Nx, grid.Ny
    n = nx * ny
    dx = float(grid.dx_cm)
    dy = float(grid.dy_cm)

    D = _as_D_field(D_cm2_s, grid)
    open_frac = _as_open_fraction(mask_eff, nx)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(n, dtype=float)

    for j in range(ny):
        for i in range(nx):
            p = grid.flat_index(j, i)
            diag = 0.0
            Dp = float(D[j, i])

            if i > 0:
                D_w = _harmonic_mean(Dp, float(D[j, i - 1]))
                c_w = D_w / (dx * dx)
                if c_w != 0.0:
                    q = grid.flat_index(j, i - 1)
                    rows.append(p)
                    cols.append(q)
                    data.append(c_w)
                    diag -= c_w

            if i < nx - 1:
                D_e = _harmonic_mean(Dp, float(D[j, i + 1]))
                c_e = D_e / (dx * dx)
                if c_e != 0.0:
                    q = grid.flat_index(j, i + 1)
                    rows.append(p)
                    cols.append(q)
                    data.append(c_e)
                    diag -= c_e

            if j > 0:
                D_n = _harmonic_mean(Dp, float(D[j - 1, i]))
                c_n = D_n / (dy * dy)
                if c_n != 0.0:
                    q = grid.flat_index(j - 1, i)
                    rows.append(p)
                    cols.append(q)
                    data.append(c_n)
                    diag -= c_n
            else:
                frac = float(open_frac[i])
                if top_bc.open_type == "robin":
                    h_eff = frac * float(top_bc.h_cm_s)
                    if h_eff != 0.0:
                        diag -= h_eff / dy
                        b[p] += (h_eff * float(top_bc.Ceq_cm3)) / dy
                elif top_bc.open_type == "dirichlet":
                    if frac != 0.0 and Dp != 0.0:
                        coeff = frac * (2.0 * Dp / (dy * dy))
                        diag -= coeff
                        b[p] += coeff * float(top_bc.dirichlet_value_cm3)
                elif top_bc.open_type == "neumann":
                    pass
                else:
                    raise ValueError(f"Unsupported top open BC type: {top_bc.open_type}")

            if j < ny - 1:
                D_s = _harmonic_mean(Dp, float(D[j + 1, i]))
                c_s = D_s / (dy * dy)
                if c_s != 0.0:
                    q = grid.flat_index(j + 1, i)
                    rows.append(p)
                    cols.append(q)
                    data.append(c_s)
                    diag -= c_s

            rows.append(p)
            cols.append(p)
            data.append(diag)

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    return A, b


def implicit_step(
    C: np.ndarray,
    A: sparse.csr_matrix,
    b: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Single backward-Euler step.

    ``(I - dt*A) C_{n+1} = C_n + dt*b``
    """
    ensure_positive("dt_s", float(dt_s))

    n = C.size
    I = sparse.identity(n, format="csc", dtype=float)
    lhs = I - float(dt_s) * A.tocsc()
    rhs = C.reshape(-1) + float(dt_s) * b
    Cn1 = spsolve(lhs, rhs)
    return Cn1.reshape(C.shape)


def top_flux_out(
    C: np.ndarray,
    grid: Grid2D,
    top_bc: TopBCConfig | None = None,
    mask_eff: np.ndarray | None = None,
) -> float:
    """Return outward top flux integrated along x [cm^-1 s^-1].

    For Robin top BC:
      ``flux_out = sum_i h_eff[i] * (C_surface[i] - Ceq) * dx_cm``
    """
    if top_bc is None or top_bc.open_type != "robin":
        return 0.0

    arr = np.asarray(C, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"C shape must be {grid.shape}, got {arr.shape}.")

    open_frac = _as_open_fraction(mask_eff, grid.Nx)
    h_eff = open_frac * float(top_bc.h_cm_s)
    c_surface = arr[0, :]
    flux_density = h_eff * (c_surface - float(top_bc.Ceq_cm3))
    return float(np.sum(flux_density) * grid.dx_cm)


def anneal_implicit_with_history(
    C0: np.ndarray,
    grid: Grid2D,
    D_cm2_s: float | np.ndarray,
    total_t_s: float,
    dt_s: float,
    top_bc: TopBCConfig | None = None,
    mask_eff: np.ndarray | None = None,
    record_enable: bool = False,
    record_every_s: float | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Run implicit diffusion anneal and optionally collect history records."""
    ensure_positive("total_t_s", float(total_t_s), allow_zero=True)
    ensure_positive("dt_s", float(dt_s))

    if record_every_s is None:
        record_every_s = float(dt_s)
    ensure_positive("record_every_s", float(record_every_s))

    C = np.asarray(C0, dtype=float)
    if C.shape != grid.shape:
        raise ValueError(f"C0 shape must be {grid.shape}, got {C.shape}.")

    history: list[dict[str, float]] = []
    is_robin = top_bc is not None and top_bc.open_type == "robin"

    def _row(time_s: float, mass: float, peak_cm3: float, flux_out_val: float, residual: float) -> dict[str, float]:
        return {
            "time_s": float(time_s),
            "mass": float(mass),
            "peak_cm3": float(peak_cm3),
            "flux_out": float(flux_out_val),
            "residual": float(residual),
        }

    t = 0.0
    mass_prev_step = total_amount(C, grid)
    if record_enable:
        flux0 = top_flux_out(C, grid, top_bc=top_bc, mask_eff=mask_eff) if is_robin else 0.0
        history.append(_row(0.0, mass_prev_step, float(np.max(C)), flux0, np.nan))

    if total_t_s == 0.0:
        return C.copy(), history

    A, b = assemble_diffusion_operator(grid=grid, D_cm2_s=D_cm2_s, top_bc=top_bc, mask_eff=mask_eff)

    n_full = int(np.floor(float(total_t_s) / float(dt_s) + 1.0e-12))
    rem = float(total_t_s - n_full * dt_s)
    n = C.size

    Cflat = C.reshape(-1)
    A_csc = A.tocsc()
    I = sparse.identity(n, format="csc", dtype=float)

    lu = None
    if n_full > 0:
        lhs = I - float(dt_s) * A_csc
        lu = splu(lhs)

    next_record_t = float(record_every_s)

    def _record_if_needed(dt_now: float, mass_now: float) -> None:
        nonlocal next_record_t
        if not record_enable:
            return
        should_record = (t + 1e-12 >= next_record_t) or (abs(t - float(total_t_s)) <= 1e-12)
        if not should_record:
            return

        C_view = np.asarray(Cflat, dtype=float).reshape(grid.shape)
        flux_now = top_flux_out(C_view, grid, top_bc=top_bc, mask_eff=mask_eff) if is_robin else 0.0
        residual_now = ((mass_now - mass_prev_step) / float(dt_now) + flux_now) if is_robin else np.nan
        history.append(_row(t, mass_now, float(np.max(C_view)), flux_now, residual_now))
        while next_record_t <= t + 1e-12:
            next_record_t += float(record_every_s)

    for _ in range(n_full):
        rhs = Cflat + float(dt_s) * b
        if lu is None:
            raise RuntimeError("Internal solver setup error: LU not initialized.")
        Cflat = lu.solve(rhs)
        t += float(dt_s)
        if record_enable:
            mass_now = total_amount(np.asarray(Cflat).reshape(grid.shape), grid)
            _record_if_needed(float(dt_s), mass_now)
            mass_prev_step = mass_now

    if rem > 1e-15:
        lhs_r = I - rem * A_csc
        rhs_r = Cflat + rem * b
        Cflat = spsolve(lhs_r, rhs_r)
        t += rem
        if record_enable:
            mass_now = total_amount(np.asarray(Cflat).reshape(grid.shape), grid)
            _record_if_needed(rem, mass_now)
            mass_prev_step = mass_now

    if record_enable and (not history or abs(history[-1]["time_s"] - float(total_t_s)) > 1e-12):
        C_view = np.asarray(Cflat, dtype=float).reshape(grid.shape)
        mass_now = total_amount(C_view, grid)
        flux_now = top_flux_out(C_view, grid, top_bc=top_bc, mask_eff=mask_eff) if is_robin else 0.0
        dt_last = max(float(record_every_s), 1e-30)
        residual_now = ((mass_now - mass_prev_step) / dt_last + flux_now) if is_robin else np.nan
        history.append(_row(float(total_t_s), mass_now, float(np.max(C_view)), flux_now, residual_now))

    return np.asarray(Cflat, dtype=float).reshape(grid.shape), history


def anneal_implicit(
    C0: np.ndarray,
    grid: Grid2D,
    D_cm2_s: float | np.ndarray,
    total_t_s: float,
    dt_s: float,
    top_bc: TopBCConfig | None = None,
    mask_eff: np.ndarray | None = None,
) -> np.ndarray:
    """Run implicit diffusion anneal from C0."""
    C_final, _ = anneal_implicit_with_history(
        C0=C0,
        grid=grid,
        D_cm2_s=D_cm2_s,
        total_t_s=total_t_s,
        dt_s=dt_s,
        top_bc=top_bc,
        mask_eff=mask_eff,
        record_enable=False,
    )
    return C_final


def total_amount(C: np.ndarray, grid: Grid2D) -> float:
    """Compute integral(C dA) in units of cm^-1 for this 2D cross-section."""
    return float(np.sum(np.asarray(C, dtype=float)) * grid.dx_cm * grid.dy_cm)
