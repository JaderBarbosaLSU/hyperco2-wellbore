"""Godunov-type finite-volume solvers for the 1-D Euler equations
(calorically perfect gas).

Phase-2 reference implementation to be compared against the legacy SCMM
scheme on identical shock-tube benchmarks.

Implemented:

* conservative finite-volume update  V_i^{n+1} = V_i^n - dt/dx (F_{i+1/2} - F_{i-1/2})
* Riemann fluxes: ``exact`` (Godunov's original method), ``hllc``, ``hll``
* first-order (piecewise-constant) reconstruction
* MUSCL-Hancock second-order extension with slope limiters
  (``minmod``, ``vanleer``) on primitive variables
* explicit time stepping under a CFL condition; reflective (solid-wall)
  boundaries matching the closed shock tube.

All updates are fully conservative by construction, which is the central
structural difference from the SCMM finite-difference scheme.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .exact_riemann import PrimitiveState, solve_riemann
from .scmm import PerfectGas

__all__ = ["GodunovResult", "solve_godunov_shock_tube"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _cons_to_prim(V: np.ndarray, g: float):
    rho = V[:, 0]
    u = V[:, 1] / rho
    p = (g - 1.0) * (V[:, 2] - 0.5 * rho * u * u)
    return rho, u, p


def _prim_to_cons(rho, u, p, g: float):
    V = np.empty((rho.size, 3))
    V[:, 0] = rho
    V[:, 1] = rho * u
    V[:, 2] = p / (g - 1.0) + 0.5 * rho * u * u
    return V


def _flux_from_prim(rho, u, p, g):
    E = p / (g - 1.0) + 0.5 * rho * u * u
    F = np.empty((np.size(rho), 3))
    F[:, 0] = rho * u
    F[:, 1] = rho * u * u + p
    F[:, 2] = u * (E + p)
    return F


# ----------------------------------------------------------------------------
# Riemann fluxes (vectorized over interfaces)
# ----------------------------------------------------------------------------
def _hll_speeds(rL, uL, pL, rR, uR, pR, g):
    """Wave-speed estimates: pressure-based (Toro Sec. 10.5-10.6)."""
    aL = np.sqrt(g * pL / rL)
    aR = np.sqrt(g * pR / rR)
    # PVRS star pressure estimate
    rbar = 0.5 * (rL + rR)
    abar = 0.5 * (aL + aR)
    p_pvrs = 0.5 * (pL + pR) - 0.5 * (uR - uL) * rbar * abar
    p_star = np.maximum(0.0, p_pvrs)
    qL = np.where(p_star <= pL, 1.0,
                  np.sqrt(1.0 + (g + 1.0) / (2.0 * g) * (p_star / pL - 1.0)))
    qR = np.where(p_star <= pR, 1.0,
                  np.sqrt(1.0 + (g + 1.0) / (2.0 * g) * (p_star / pR - 1.0)))
    SL = uL - aL * qL
    SR = uR + aR * qR
    return SL, SR, p_star


def _flux_hllc(rL, uL, pL, rR, uR, pR, g):
    SL, SR, _ = _hll_speeds(rL, uL, pL, rR, uR, pR, g)
    S_star = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (
        rL * (SL - uL) - rR * (SR - uR)
    )
    FL = _flux_from_prim(rL, uL, pL, g)
    FR = _flux_from_prim(rR, uR, pR, g)
    VL = _prim_to_cons(rL, uL, pL, g)
    VR = _prim_to_cons(rR, uR, pR, g)

    def star_state(rk, uk, pk, Vk, Sk):
        fac = rk * (Sk - uk) / (Sk - S_star)
        Vst = np.empty_like(Vk)
        Vst[:, 0] = fac
        Vst[:, 1] = fac * S_star
        Vst[:, 2] = fac * (Vk[:, 2] / rk + (S_star - uk) * (S_star + pk / (rk * (Sk - uk))))
        return Vst

    VstL = star_state(rL, uL, pL, VL, SL)
    VstR = star_state(rR, uR, pR, VR, SR)
    FstL = FL + SL[:, None] * (VstL - VL)
    FstR = FR + SR[:, None] * (VstR - VR)

    F = np.where((SL >= 0.0)[:, None], FL,
        np.where((S_star >= 0.0)[:, None], FstL,
        np.where((SR >= 0.0)[:, None], FstR, FR)))
    return F


def _flux_hll(rL, uL, pL, rR, uR, pR, g):
    SL, SR, _ = _hll_speeds(rL, uL, pL, rR, uR, pR, g)
    FL = _flux_from_prim(rL, uL, pL, g)
    FR = _flux_from_prim(rR, uR, pR, g)
    VL = _prim_to_cons(rL, uL, pL, g)
    VR = _prim_to_cons(rR, uR, pR, g)
    Fhll = (SR[:, None] * FL - SL[:, None] * FR
            + (SL * SR)[:, None] * (VR - VL)) / (SR - SL)[:, None]
    F = np.where((SL >= 0.0)[:, None], FL,
        np.where((SR <= 0.0)[:, None], FR, Fhll))
    return F


def _flux_exact(rL, uL, pL, rR, uR, pR, g):
    """Godunov flux from the exact Riemann solution sampled at xi = 0.

    Loop over interfaces (scalar Newton per interface); intended for
    verification and cost comparison rather than speed.
    """
    n = rL.size
    rho = np.empty(n)
    u = np.empty(n)
    p = np.empty(n)
    for i in range(n):
        sol = solve_riemann(
            PrimitiveState(rL[i], uL[i], pL[i]),
            PrimitiveState(rR[i], uR[i], pR[i]),
            gamma=g,
        )
        out = sol.sample(np.array([0.0]))
        rho[i], u[i], p[i] = out["rho"][0], out["u"][0], out["p"][0]
    return _flux_from_prim(rho, u, p, g)


_FLUXES = {"hllc": _flux_hllc, "hll": _flux_hll, "exact": _flux_exact}


# ----------------------------------------------------------------------------
# Limiters (on primitive-variable slopes)
# ----------------------------------------------------------------------------
def _limited_slopes(q: np.ndarray, limiter: str) -> np.ndarray:
    dq_m = np.zeros_like(q)
    dq_p = np.zeros_like(q)
    dq_m[1:] = q[1:] - q[:-1]
    dq_p[:-1] = q[1:] - q[:-1]
    if limiter == "minmod":
        s = np.where(
            dq_m * dq_p <= 0.0, 0.0,
            np.where(np.abs(dq_m) < np.abs(dq_p), dq_m, dq_p),
        )
    elif limiter == "vanleer":
        denom = dq_m + dq_p
        s = np.where(dq_m * dq_p <= 0.0, 0.0,
                     2.0 * dq_m * dq_p / np.where(denom == 0.0, 1.0, denom))
    else:
        raise ValueError(f"unknown limiter {limiter!r}")
    return s


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
@dataclass
class GodunovResult:
    x: np.ndarray
    t: float
    rho: np.ndarray
    u: np.ndarray
    p: np.ndarray
    T: np.ndarray
    n_steps: int = 0
    wall_time: float = 0.0
    scheme: str = ""
    mass_initial: float = 0.0
    mass_final: float = 0.0
    energy_initial: float = 0.0
    energy_final: float = 0.0


def solve_godunov_shock_tube(
    *,
    gas: PerfectGas = PerfectGas(),
    L: float = 4.0,
    x_diaphragm: float = 2.0,
    p_left: float = 5.0e5,
    T_left: float = 400.0,
    p_right: float = 1.0e5,
    T_right: float = 300.0,
    n_cells: int = 4000,
    t_end: float = 3.0e-3,
    cfl: float = 0.9,
    flux: Literal["hllc", "hll", "exact"] = "hllc",
    reconstruction: Literal["constant", "muscl"] = "constant",
    limiter: Literal["minmod", "vanleer"] = "vanleer",
) -> GodunovResult:
    """Finite-volume solution of the closed shock tube.

    ``n_cells = 4000`` gives dx = 1 mm, matching the 4001-node SCMM grid.
    First order: ``reconstruction='constant'``. Second order:
    ``reconstruction='muscl'`` (MUSCL-Hancock with the chosen limiter).
    """
    g = gas.gamma
    dx = L / n_cells
    xc = (np.arange(n_cells) + 0.5) * dx

    left = xc < x_diaphragm
    p = np.where(left, p_left, p_right).astype(float)
    T = np.where(left, T_left, T_right).astype(float)
    rho = p / (gas.R * T)
    u = np.zeros(n_cells)
    V = _prim_to_cons(rho, u, p, g)

    mass0 = float(np.sum(V[:, 0]) * dx)
    energy0 = float(np.sum(V[:, 2]) * dx)

    flux_fn = _FLUXES[flux]
    t = 0.0
    n_steps = 0
    t0 = _time.perf_counter()

    while t < t_end - 1e-15:
        rho, u, p = _cons_to_prim(V, g)
        a = np.sqrt(g * p / rho)
        dt = cfl * dx / np.max(np.abs(u) + a)
        dt = min(dt, t_end - t)

        # ghost cells: reflective walls (mirror with u -> -u)
        def with_ghosts(q, sign=1.0, n_ghost=2):
            return np.concatenate([sign * q[n_ghost - 1 :: -1], q,
                                   sign * q[: -n_ghost - 1 : -1]])

        rg = with_ghosts(rho)
        ug = with_ghosts(u, sign=-1.0)
        pg = with_ghosts(p)

        if reconstruction == "muscl":
            # limited slopes and half-step (Hancock) predictor
            sr = _limited_slopes(rg, limiter)
            su = _limited_slopes(ug, limiter)
            sp = _limited_slopes(pg, limiter)
            rL_face = rg + 0.5 * sr   # right face of each cell
            rR_face = rg - 0.5 * sr   # left face of each cell
            uL_face = ug + 0.5 * su
            uR_face = ug - 0.5 * su
            pL_face = pg + 0.5 * sp
            pR_face = pg - 0.5 * sp
            # Hancock half-step on primitive form
            dtdx2 = 0.5 * dt / dx
            drho = -(ug * sr + rg * su) * dtdx2
            du = -(ug * su + sp / rg) * dtdx2
            dp = -(g * pg * su + ug * sp) * dtdx2
            rL_face = np.maximum(rL_face + drho, 1e-10)
            rR_face = np.maximum(rR_face + drho, 1e-10)
            uL_face = uL_face + du
            uR_face = uR_face + du
            pL_face = np.maximum(pL_face + dp, 1e-6)
            pR_face = np.maximum(pR_face + dp, 1e-6)
            # interface i+1/2: left state = right face of cell i,
            #                  right state = left face of cell i+1
            rl, ul, pl = rL_face[:-1], uL_face[:-1], pL_face[:-1]
            rr, ur, pr = rR_face[1:], uR_face[1:], pR_face[1:]
        else:
            rl, ul, pl = rg[:-1], ug[:-1], pg[:-1]
            rr, ur, pr = rg[1:], ug[1:], pg[1:]

        F = flux_fn(rl, ul, pl, rr, ur, pr, g)
        # interior interfaces: indices 1..n_cells+2 in ghost framing; cells
        # occupy ghost indices 2..n_cells+1, their faces are F[1:-1]
        Fi = F[1:-1]
        V = V - dt / dx * (Fi[1 : n_cells + 1] - Fi[0:n_cells])
        t += dt
        n_steps += 1

    wall = _time.perf_counter() - t0
    rho, u, p = _cons_to_prim(V, g)
    scheme = f"{'MUSCL-Hancock/' + limiter if reconstruction == 'muscl' else 'first-order'}+{flux}"
    return GodunovResult(
        x=xc, t=t, rho=rho, u=u, p=p, T=p / (gas.R * rho),
        n_steps=n_steps, wall_time=wall, scheme=scheme,
        mass_initial=mass0, mass_final=float(np.sum(V[:, 0]) * dx),
        energy_initial=energy0, energy_final=float(np.sum(V[:, 2]) * dx),
    )
