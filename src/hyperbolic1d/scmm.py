"""Split Coefficient Matrix Method (SCMM) solvers for single-phase 1-D
compressible flow of a calorically perfect gas.

This module is a faithful Python reconstruction of the numerical method of
Gessner (2010), "Modelagem numerica do escoamento anular gas-liquido
transiente pelo metodo da divisao da matriz de coeficientes", M.Sc. thesis,
UFSC (advisor: J. R. Barbosa Jr.), Chapter 3, and of the companion Fortran
implementation style (P1.FOR + GBAND).

Method summary
--------------
The quasi-linear system  dU/dt + G dU/dx = C  is split via the eigen-
decomposition  G = T Lambda T^-1  into

    G+ = T Lambda+ T^-1,   G- = T Lambda- T^-1,   Lambda+- = (Lambda +- |Lambda|)/2

and discretized in finite differences with backward (upwind) differences on
the G+ term and forward differences on the G- term; first order (2-point)
or second order (3-point) one-sided stencils are supported. Time stepping is
implicit (backward Euler); within each time step the coefficient matrices
are refreshed and the banded linear system re-solved until the fields stop
changing (Picard iteration), exactly as in the original Fortran codes.

Two formulations are provided:

* ``nonconservative`` : unknowns U = [P, u, s]      (thesis Sec. 3.1.1)
* ``conservative``    : unknowns V = [rho, G, E]    (thesis Sec. 3.1.2),
  where G = rho*u and E = rho*(e + u^2/2). Note that the "conservative"
  SCMM still discretizes the *quasi-linear* form A(V) dV/dx, not the flux
  divergence dF/dx, so discrete conservation is not exactly enforced; this
  is a property of the original method that the Phase-2 finite-volume
  solvers address.

The banded systems are solved with LAPACK ``gbsv`` via
``scipy.linalg.solve_banded``. The original GBAND routine (Aziz & Settari,
1979) performs the same banded Gaussian elimination but *without* pivoting;
partial pivoting can only improve robustness and does not alter converged
results.

Deliberate deviations from the original implementation are confined to:

1. the linear solver pivoting noted above;
2. the shock-entropy correction for the non-conservative formulation
   (thesis Eq. 3.61): the thesis states that the shock front was tracked
   and the entropy field corrected "where necessary" without giving the
   algorithm; ``shock_entropy_correction`` implements a documented
   interpretation (see :func:`_apply_shock_entropy_correction`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from scipy.linalg import solve_banded

__all__ = ["PerfectGas", "SCMMResult", "solve_scmm_shock_tube"]


# ----------------------------------------------------------------------------
# Gas model
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PerfectGas:
    """Calorically perfect gas. Defaults: air."""

    R: float = 287.0
    gamma: float = 1.4
    T_ref: float = 300.0       # entropy reference temperature (as in P1.FOR)
    p_ref: float = 101325.0    # entropy reference pressure (as in P1.FOR)

    @property
    def cp(self) -> float:
        return self.gamma * self.R / (self.gamma - 1.0)

    @property
    def cv(self) -> float:
        return self.R / (self.gamma - 1.0)

    def entropy(self, p: np.ndarray, T: np.ndarray) -> np.ndarray:
        return self.cp * np.log(T / self.T_ref) - self.R * np.log(p / self.p_ref)

    def temperature_from_ps(self, p: np.ndarray, s: np.ndarray) -> np.ndarray:
        # T = T_ref * exp(s/cp) * (p/p_ref)^(R/cp)   (as in P1.FOR CalcPROP)
        return self.T_ref * np.exp(s / self.cp) * (p / self.p_ref) ** (self.R / self.cp)


# ----------------------------------------------------------------------------
# Split matrices
# ----------------------------------------------------------------------------
def _split_weights(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (lambda+, lambda-) = ((lam +- |lam|)/2)."""
    return 0.5 * (lam + np.abs(lam)), 0.5 * (lam - np.abs(lam))


def _split_nonconservative(P, u, s, gas: PerfectGas):
    """G+/- for unknowns [P, u, s]; vectorized over nodes.

    Eigenstructure (thesis Sec. 3.2.1): lambda = u+a, u-a, u with
    T = [[1, 1, 0], [1/(rho a), -1/(rho a), 0], [0, 0, 1]],
    giving rank-one blocks G1, G2 and the entropy-advection block G3.
    """
    T = gas.temperature_from_ps(P, s)
    rho = P / (gas.R * T)
    a = np.sqrt(gas.gamma * gas.R * T)
    n = P.size

    l1p, l1m = _split_weights(u + a)
    l2p, l2m = _split_weights(u - a)
    l3p, l3m = _split_weights(u)

    def block(l1, l2, l3):
        Gk = np.zeros((n, 3, 3))
        # G1 contribution: (l1/2) * [[1, rho a, 0],[1/(rho a), 1, 0],[0,0,0]]
        Gk[:, 0, 0] += 0.5 * l1
        Gk[:, 0, 1] += 0.5 * l1 * rho * a
        Gk[:, 1, 0] += 0.5 * l1 / (rho * a)
        Gk[:, 1, 1] += 0.5 * l1
        # G2 contribution: (l2/2) * [[1, -rho a, 0],[-1/(rho a), 1, 0],[0,0,0]]
        Gk[:, 0, 0] += 0.5 * l2
        Gk[:, 0, 1] -= 0.5 * l2 * rho * a
        Gk[:, 1, 0] -= 0.5 * l2 / (rho * a)
        Gk[:, 1, 1] += 0.5 * l2
        # G3 contribution: entropy advection
        Gk[:, 2, 2] += l3
        return Gk

    return block(l1p, l2p, l3p), block(l1m, l2m, l3m)


def _split_conservative(V, gas: PerfectGas):
    """H+/- for unknowns [rho, rho*u, E]; vectorized over nodes.

    H = J G J^-1 coincides with the Euler flux Jacobian dF/dV for a
    perfect gas; the split uses the standard eigenvectors
    R = [[1,1,1],[u-a,u,u+a],[H-ua, u^2/2, H+ua]].
    """
    g = gas.gamma
    rho = V[:, 0]
    u = V[:, 1] / rho
    E = V[:, 2]
    p = (g - 1.0) * (E - 0.5 * rho * u * u)
    a = np.sqrt(g * p / rho)
    H = (E + p) / rho
    n = rho.size

    Rm = np.zeros((n, 3, 3))
    Rm[:, 0, :] = 1.0
    Rm[:, 1, 0] = u - a
    Rm[:, 1, 1] = u
    Rm[:, 1, 2] = u + a
    Rm[:, 2, 0] = H - u * a
    Rm[:, 2, 1] = 0.5 * u * u
    Rm[:, 2, 2] = H + u * a
    Lm = np.linalg.inv(Rm)

    lam = np.stack([u - a, u, u + a], axis=1)  # (n, 3)
    lp, lm = _split_weights(lam)

    Hp = np.einsum("nij,nj,njk->nik", Rm, lp, Lm)
    Hm = np.einsum("nij,nj,njk->nik", Rm, lm, Lm)
    return Hp, Hm


# ----------------------------------------------------------------------------
# Banded assembly
# ----------------------------------------------------------------------------
def _assemble_banded(Gp, Gm, dt, dx, order, bc_var: int):
    """Assemble the block-banded implicit SCMM matrix.

    Rows follow thesis Eqs. (2.36)-(2.38) [first order] and (2.40)-(2.44)
    [second order, uniform grid: beta = phi = 2], with the boundary
    treatment of Sec. 2.5.1.2/3.4.2:

    * node 0   : G+ derivative dropped (no upstream nodes);
    * node 1   : backward difference limited to first order;
    * node N-2 : forward difference limited to first order;
    * node N-1 : G- derivative dropped;
    * the velocity/mass-flux unknown ``bc_var`` at nodes 0 and N-1 is
      replaced by the Dirichlet row  x = 0  (closed tube ends).

    Returns (ab, m) ready for ``scipy.linalg.solve_banded``.
    """
    n = Gp.shape[0]
    nv = 3
    nun = n * nv
    m = nv * order + (nv - 1)          # scalar bandwidth: 5 (1st), 8 (2nd)
    ab = np.zeros((2 * m + 1, nun))

    # Per-node block coefficients for offsets -2..+2
    Bm2 = np.zeros_like(Gp)
    Bm1 = np.zeros_like(Gp)
    D = np.zeros_like(Gp)
    Fp1 = np.zeros_like(Gp)
    Fp2 = np.zeros_like(Gp)

    idx = np.arange(n)
    eye = np.eye(nv)[None, :, :]
    D += eye / dt

    if order == 1:
        back2 = np.zeros(n, dtype=bool)
        back1 = idx >= 1
        fwd1 = idx <= n - 2
        fwd2 = np.zeros(n, dtype=bool)
    else:
        back2 = idx >= 2
        back1 = idx == 1
        fwd1 = idx == n - 2
        fwd2 = idx <= n - 3

    # first-order one-sided stencils
    Bm1[back1] += -Gp[back1] / dx
    D[back1] += Gp[back1] / dx
    Fp1[fwd1] += Gm[fwd1] / dx
    D[fwd1] += -Gm[fwd1] / dx
    # second-order one-sided stencils (uniform grid)
    Bm2[back2] += Gp[back2] / (2 * dx)
    Bm1[back2] += -2.0 * Gp[back2] / dx
    D[back2] += 1.5 * Gp[back2] / dx
    Fp2[fwd2] += -Gm[fwd2] / (2 * dx)
    Fp1[fwd2] += 2.0 * Gm[fwd2] / dx
    D[fwd2] += -1.5 * Gm[fwd2] / dx

    # Dirichlet rows for the velocity/mass-flux unknown at the walls
    for node in (0, n - 1):
        for blk in (Bm2, Bm1, D, Fp1, Fp2):
            blk[node, bc_var, :] = 0.0
        D[node, bc_var, bc_var] = 1.0

    # Scatter blocks into LAPACK banded storage: ab[m + i - j, j] = A[i, j]
    offsets = ((-2, Bm2), (-1, Bm1), (0, D), (1, Fp1), (2, Fp2)) if order == 2 \
        else ((-1, Bm1), (0, D), (1, Fp1))
    for off, blk in offsets:
        ks = np.arange(max(0, -off), min(n, n - off))
        if ks.size == 0:
            continue
        for r in range(nv):
            for c in range(nv):
                i = nv * ks + r
                j = nv * (ks + off) + c
                ab[m + i - j, j] = blk[ks, r, c]
    return ab, m


# ----------------------------------------------------------------------------
# Shock entropy correction (non-conservative formulation only)
# ----------------------------------------------------------------------------
def _rankine_hugoniot_ds(pressure_ratio: float, gas: PerfectGas) -> float:
    """Entropy jump across a normal shock with p2/p1 = pressure_ratio
    (thesis Eq. 3.61 / Staedtke 2006, Eq. for ideal gas)."""
    Rp = pressure_ratio
    g = gas.gamma
    dr = ((g - 1.0) * Rp + (g + 1.0)) / ((g + 1.0) * Rp + (g - 1.0))
    return gas.cv * np.log(Rp * dr**g)


def _apply_shock_entropy_correction(x, P, u, s, s_init_right, x_contact,
                                    gas: PerfectGas):
    """Correct the entropy plateau between the contact and the shock.

    The non-conservative formulation transports s isentropically and
    therefore cannot generate the Rankine-Hugoniot entropy rise. Following
    the thesis prescription (track the shock, correct s where needed):

    * the shock front is located as the steepest compressive pressure drop
      right of the (tracked) contact surface;
    * the pressure ratio is taken from the plateaus adjacent to the smeared
      front;
    * nodes of right-chamber material origin between the contact surface
      ``x_contact`` (advected with the local fluid velocity by the caller)
      and the shock receive  s = s_right + delta_s(Rp).

    Left-chamber gas (between rarefaction and contact) remains isentropic
    and is never touched.
    """
    n = P.size
    i_c = int(np.searchsorted(x, x_contact))
    if i_c >= n - 4:
        return s
    dP = np.diff(P)
    i_sh = i_c + int(np.argmin(dP[i_c:]))
    if dP[i_sh] >= 0.0:
        return s
    off = 8
    i_up = min(n - 1, i_sh + off)       # ahead (pre-shock)
    i_dn = max(i_c, i_sh - off)         # behind (post-shock)
    Rp = P[i_dn] / P[i_up]
    if Rp <= 1.0 + 1e-9:
        return s
    ds = _rankine_hugoniot_ds(Rp, gas)
    s_new = s.copy()
    zone = np.arange(i_c, i_sh + 1)
    zone = zone[u[zone] > 0.05 * np.max(np.abs(u))]
    s_new[zone] = s_init_right + ds
    return s_new


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
@dataclass
class SCMMResult:
    x: np.ndarray
    t: float
    rho: np.ndarray
    u: np.ndarray
    p: np.ndarray
    T: np.ndarray
    s: np.ndarray
    iterations_per_step: list = field(default_factory=list)
    wall_time: float = 0.0
    formulation: str = ""
    order: int = 1


def solve_scmm_shock_tube(
    *,
    gas: PerfectGas = PerfectGas(),
    L: float = 4.0,
    x_diaphragm: float = 2.0,
    p_left: float = 5.0e5,
    T_left: float = 400.0,
    p_right: float = 1.0e5,
    T_right: float = 300.0,
    n_nodes: int = 4001,
    dt: float = 1.0e-6,
    t_end: float = 3.0e-3,
    formulation: Literal["conservative", "nonconservative"] = "conservative",
    order: Literal[1, 2] = 2,
    tolerances: dict | None = None,
    max_inner_iter: int = 100,
    shock_entropy_correction: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> SCMMResult:
    """Run the Gessner (2010) single-phase shock-tube benchmark with SCMM.

    Defaults reproduce thesis Sec. 3.4: air, 500 kPa / 400 K vs
    100 kPa / 300 K, diaphragm at midspan, 4001 nodes (dx = 1 mm),
    dt = 1 us, t_end = 3 ms. Tolerances default to thesis Table 3.2.
    """
    import time as _time

    x = np.linspace(0.0, L, n_nodes)
    dx = x[1] - x[0]
    n_steps = int(round(t_end / dt))

    if tolerances is None:
        tolerances = (
            {"P": 1e-1, "U": 1e-2, "s": 1e-2}
            if formulation == "nonconservative"
            else {"rho": 1e-6, "G": 1e-2, "E": 1e-2}
        )

    left = x <= x_diaphragm
    P = np.where(left, p_left, p_right).astype(float)
    T = np.where(left, T_left, T_right).astype(float)
    u = np.zeros(n_nodes)
    rho = P / (gas.R * T)
    s = gas.entropy(P, T)
    s_init_right = float(s[-1])

    t0 = _time.perf_counter()
    iters = []

    if formulation == "nonconservative":
        U = np.stack([P, u, s], axis=1)
        x_contact = x_diaphragm                 # material contact surface
        for step in range(n_steps):
            U0 = U.copy()
            Ui = U.copy()
            for it in range(max_inner_iter):
                Gp, Gm = _split_nonconservative(Ui[:, 0], Ui[:, 1], Ui[:, 2], gas)
                ab, m = _assemble_banded(Gp, Gm, dt, dx, order, bc_var=1)
                rhs = (U0 / dt).ravel()
                rhs[1] = 0.0                      # u(0) = 0
                rhs[-2] = 0.0                     # u(L) = 0
                Unew = solve_banded((m, m), ab, rhs).reshape(-1, 3)
                errP = np.max(np.abs(Unew[:, 0] - Ui[:, 0]))
                errU = np.max(np.abs(Unew[:, 1] - Ui[:, 1]))
                errS = np.max(np.abs(Unew[:, 2] - Ui[:, 2]))
                Ui = Unew
                if errP < tolerances["P"] and errU < tolerances["U"] and errS < tolerances["s"]:
                    break
            U = Ui
            # advect the contact surface with the local fluid velocity
            x_contact += dt * float(np.interp(x_contact, x, U[:, 1]))
            if shock_entropy_correction:
                U[:, 2] = _apply_shock_entropy_correction(
                    x, U[:, 0], U[:, 1], U[:, 2], s_init_right, x_contact, gas
                )
            iters.append(it + 1)
            if progress is not None:
                progress(step + 1, n_steps)
        P, u, s = U[:, 0], U[:, 1], U[:, 2]
        T = gas.temperature_from_ps(P, s)
        rho = P / (gas.R * T)
    else:
        E = P / (gas.gamma - 1.0) + 0.5 * rho * u * u
        V = np.stack([rho, rho * u, E], axis=1)
        for step in range(n_steps):
            V0 = V.copy()
            Vi = V.copy()
            for it in range(max_inner_iter):
                Hp, Hm = _split_conservative(Vi, gas)
                ab, m = _assemble_banded(Hp, Hm, dt, dx, order, bc_var=1)
                rhs = (V0 / dt).ravel()
                rhs[1] = 0.0                      # G(0) = 0
                rhs[-2] = 0.0                     # G(L) = 0
                Vnew = solve_banded((m, m), ab, rhs).reshape(-1, 3)
                errR = np.max(np.abs(Vnew[:, 0] - Vi[:, 0]))
                errG = np.max(np.abs(Vnew[:, 1] - Vi[:, 1]))
                errE = np.max(np.abs(Vnew[:, 2] - Vi[:, 2]))
                Vi = Vnew
                if errR < tolerances["rho"] and errG < tolerances["G"] and errE < tolerances["E"]:
                    break
            V = Vi
            iters.append(it + 1)
            if progress is not None:
                progress(step + 1, n_steps)
        rho = V[:, 0]
        u = V[:, 1] / rho
        E = V[:, 2]
        P = (gas.gamma - 1.0) * (E - 0.5 * rho * u * u)
        T = P / (gas.R * rho)
        s = gas.entropy(P, T)

    wall = _time.perf_counter() - t0
    return SCMMResult(
        x=x, t=n_steps * dt, rho=rho, u=u, p=P, T=T, s=s,
        iterations_per_step=iters, wall_time=wall,
        formulation=formulation, order=order,
    )
