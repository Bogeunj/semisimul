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
    blocked_type currently supports only "neumann" for MVP.
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
            "MVP supports only top_bc.blocked.type='neumann'. "
            f"Got '{blocked_type}'."
        )

    cfg = TopBCConfig(open_type=open_type, blocked_type=blocked_type)

    if open_type == "robin":
        h_cm_s = float(open_cfg.get("h_cm_s", 0.0))
        Ceq_cm3 = float(open_cfg.get("Ceq_cm3", 0.0))
        ensure_positive("top_bc.open.h_cm_s", h_cm_s, allow_zero=True)
        cfg = TopBCConfig(
            open_type="robin",
            blocked_type="neumann",
            h_cm_s=h_cm_s,
            Ceq_cm3=Ceq_cm3,
        )
    elif open_type == "dirichlet":
        value = float(open_cfg.get("value_cm3", 0.0))
        cfg = TopBCConfig(
            open_type="dirichlet",
            blocked_type="neumann",
            dirichlet_value_cm3=value,
        )
    elif open_type == "neumann":
        cfg = TopBCConfig(open_type="neumann", blocked_type="neumann")
    else:
        raise ValueError(
            "top_bc.open.type must be one of: neumann, robin, dirichlet. "
            f"Got '{open_type}'."
        )

    return cfg


def _as_open_fraction(mask_eff: np.ndarray | None, nx: int) -> np.ndarray:
    if mask_eff is None:
        return np.ones(nx, dtype=float)
    if mask_eff.shape != (nx,):
        raise ValueError(f"mask_eff must have shape ({nx},), got {mask_eff.shape}.")
    return np.clip(np.asarray(mask_eff, dtype=float), 0.0, 1.0)


def assemble_diffusion_operator(
    grid: Grid2D,
    D_cm2_s: float,
    top_bc: TopBCConfig | None = None,
    mask_eff: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble dC/dt = A*C + b for constant D.

    Spatial discretization is finite-volume-like on a regular grid:
    - interior: standard 5-point Laplacian
    - left/right/bottom boundaries: Neumann(0-flux)
    - top: mixed by mask_eff (open fraction)
      * open fraction -> open BC
      * blocked fraction -> Neumann(0-flux)
    """
    ensure_positive("D_cm2_s", float(D_cm2_s), allow_zero=True)

    if top_bc is None:
        top_bc = TopBCConfig()

    nx, ny = grid.Nx, grid.Ny
    n = nx * ny
    dx = grid.dx_cm
    dy = grid.dy_cm

    gx = float(D_cm2_s) / (dx * dx)
    gy = float(D_cm2_s) / (dy * dy)

    open_frac = _as_open_fraction(mask_eff, nx)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(n, dtype=float)

    for j in range(ny):
        for i in range(nx):
            p = grid.flat_index(j, i)
            diag = 0.0

            if i > 0:
                q = grid.flat_index(j, i - 1)
                rows.append(p)
                cols.append(q)
                data.append(gx)
                diag -= gx

            if i < nx - 1:
                q = grid.flat_index(j, i + 1)
                rows.append(p)
                cols.append(q)
                data.append(gx)
                diag -= gx

            if j > 0:
                q = grid.flat_index(j - 1, i)
                rows.append(p)
                cols.append(q)
                data.append(gy)
                diag -= gy
            else:
                frac = float(open_frac[i])
                if top_bc.open_type == "robin":
                    h_eff = frac * float(top_bc.h_cm_s)
                    if h_eff != 0.0:
                        diag -= h_eff / dy
                        b[p] += (h_eff * float(top_bc.Ceq_cm3)) / dy
                elif top_bc.open_type == "dirichlet":
                    if frac != 0.0:
                        coeff = frac * (2.0 * float(D_cm2_s) / (dy * dy))
                        diag -= coeff
                        b[p] += coeff * float(top_bc.dirichlet_value_cm3)
                elif top_bc.open_type == "neumann":
                    pass
                else:
                    raise ValueError(f"Unsupported top open BC type: {top_bc.open_type}")

            if j < ny - 1:
                q = grid.flat_index(j + 1, i)
                rows.append(p)
                cols.append(q)
                data.append(gy)
                diag -= gy

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

    (I - dt*A) C_{n+1} = C_n + dt*b
    """
    ensure_positive("dt_s", float(dt_s))

    n = C.size
    I = sparse.identity(n, format="csc", dtype=float)
    lhs = I - float(dt_s) * A.tocsc()
    rhs = C.reshape(-1) + float(dt_s) * b
    Cn1 = spsolve(lhs, rhs)
    return Cn1.reshape(C.shape)


def anneal_implicit(
    C0: np.ndarray,
    grid: Grid2D,
    D_cm2_s: float,
    total_t_s: float,
    dt_s: float,
    top_bc: TopBCConfig | None = None,
    mask_eff: np.ndarray | None = None,
) -> np.ndarray:
    """Run implicit diffusion anneal from C0.

    Uses fixed dt for all full steps and one final remainder step if needed.
    """
    ensure_positive("total_t_s", float(total_t_s), allow_zero=True)
    ensure_positive("dt_s", float(dt_s))

    C = np.asarray(C0, dtype=float)
    if C.shape != grid.shape:
        raise ValueError(f"C0 shape must be {grid.shape}, got {C.shape}.")
    if total_t_s == 0.0:
        return C.copy()

    A, b = assemble_diffusion_operator(
        grid=grid,
        D_cm2_s=D_cm2_s,
        top_bc=top_bc,
        mask_eff=mask_eff,
    )

    n_full = int(np.floor(float(total_t_s) / float(dt_s) + 1.0e-12))
    rem = float(total_t_s - n_full * dt_s)
    n = C.size

    Cflat = C.reshape(-1)
    A_csc = A.tocsc()
    I = sparse.identity(n, format="csc", dtype=float)

    if n_full > 0:
        lhs = I - float(dt_s) * A_csc
        lu = splu(lhs)
        for _ in range(n_full):
            rhs = Cflat + float(dt_s) * b
            Cflat = lu.solve(rhs)

    if rem > 1e-15:
        lhs_r = I - rem * A_csc
        rhs_r = Cflat + rem * b
        Cflat = spsolve(lhs_r, rhs_r)

    return np.asarray(Cflat, dtype=float).reshape(grid.shape)


def total_amount(C: np.ndarray, grid: Grid2D) -> float:
    """Compute integral(C dA) in units of cm^-1 for this 2D cross-section."""
    return float(np.sum(np.asarray(C, dtype=float)) * grid.dx_cm * grid.dy_cm)
