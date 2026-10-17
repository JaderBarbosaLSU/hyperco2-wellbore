"""Exact Riemann solver for the 1-D Euler equations with a calorically
perfect gas.

Provides the analytical shock-tube solution used as the validation
reference for both the legacy SCMM solvers (Gessner, 2010) and the
Godunov-type finite-volume solvers.

Notation follows Toro, "Riemann Solvers and Numerical Methods for Fluid
Dynamics", 3rd ed., Ch. 4. The star-region pressure :math:`p^*` is found
with Newton iteration on the pressure function

.. math:: f(p) = f_L(p, W_L) + f_R(p, W_R) + (u_R - u_L)

where each :math:`f_K` is the shock (Rankine-Hugoniot) or rarefaction
(isentropic) branch depending on whether :math:`p > p_K`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PrimitiveState", "ExactRiemannSolution", "solve_riemann"]


@dataclass(frozen=True)
class PrimitiveState:
    """Primitive state (rho, u, p) of a calorically perfect gas."""

    rho: float
    u: float
    p: float

    def sound_speed(self, gamma: float) -> float:
        return float(np.sqrt(gamma * self.p / self.rho))

    @classmethod
    def from_pT(cls, p: float, T: float, u: float, R: float) -> "PrimitiveState":
        """Build a state from pressure and temperature via the ideal-gas law."""
        return cls(rho=p / (R * T), u=u, p=p)


def _f_branch(p: float, state: PrimitiveState, gamma: float) -> tuple[float, float]:
    """Pressure function f_K(p) and its derivative for one side."""
    pk, rhok = state.p, state.rho
    ak = state.sound_speed(gamma)
    if p > pk:  # shock branch
        Ak = 2.0 / ((gamma + 1.0) * rhok)
        Bk = (gamma - 1.0) / (gamma + 1.0) * pk
        sq = np.sqrt(Ak / (p + Bk))
        f = (p - pk) * sq
        df = sq * (1.0 - 0.5 * (p - pk) / (p + Bk))
    else:  # rarefaction branch
        f = 2.0 * ak / (gamma - 1.0) * ((p / pk) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        df = (1.0 / (rhok * ak)) * (p / pk) ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, df


@dataclass(frozen=True)
class ExactRiemannSolution:
    """Converged star-region data; callable sampler over xi = x/t."""

    left: PrimitiveState
    right: PrimitiveState
    gamma: float
    p_star: float
    u_star: float

    def sample(self, xi: np.ndarray) -> dict[str, np.ndarray]:
        """Sample the self-similar solution at speeds ``xi = x/t``.

        Returns arrays for rho, u, p.
        """
        g = self.gamma
        gm1, gp1 = g - 1.0, g + 1.0
        L, R = self.left, self.right
        aL, aR = L.sound_speed(g), R.sound_speed(g)
        ps, us = self.p_star, self.u_star

        xi = np.asarray(xi, dtype=float)
        rho = np.empty_like(xi)
        u = np.empty_like(xi)
        p = np.empty_like(xi)

        # --- left side of the contact -------------------------------------
        if ps > L.p:  # left shock
            rho_sL = L.rho * ((ps / L.p + gm1 / gp1) / (gm1 / gp1 * ps / L.p + 1.0))
            SL = L.u - aL * np.sqrt(gp1 / (2 * g) * ps / L.p + gm1 / (2 * g))
            mask = xi < SL
            rho[mask], u[mask], p[mask] = L.rho, L.u, L.p
            mask = (xi >= SL) & (xi < us)
            rho[mask], u[mask], p[mask] = rho_sL, us, ps
        else:  # left rarefaction
            rho_sL = L.rho * (ps / L.p) ** (1.0 / g)
            a_sL = aL * (ps / L.p) ** (gm1 / (2 * g))
            SHL = L.u - aL           # head
            STL = us - a_sL          # tail
            mask = xi < SHL
            rho[mask], u[mask], p[mask] = L.rho, L.u, L.p
            mask = (xi >= SHL) & (xi < STL)  # fan interior
            af = 2.0 / gp1 * (aL + gm1 / 2.0 * (L.u - xi[mask]))
            u[mask] = 2.0 / gp1 * (aL + gm1 / 2.0 * L.u + xi[mask])
            rho[mask] = L.rho * (af / aL) ** (2.0 / gm1)
            p[mask] = L.p * (af / aL) ** (2.0 * g / gm1)
            mask = (xi >= STL) & (xi < us)
            rho[mask], u[mask], p[mask] = rho_sL, us, ps

        # --- right side of the contact ------------------------------------
        if ps > R.p:  # right shock
            rho_sR = R.rho * ((ps / R.p + gm1 / gp1) / (gm1 / gp1 * ps / R.p + 1.0))
            SR = R.u + aR * np.sqrt(gp1 / (2 * g) * ps / R.p + gm1 / (2 * g))
            mask = (xi >= us) & (xi < SR)
            rho[mask], u[mask], p[mask] = rho_sR, us, ps
            mask = xi >= SR
            rho[mask], u[mask], p[mask] = R.rho, R.u, R.p
        else:  # right rarefaction
            rho_sR = R.rho * (ps / R.p) ** (1.0 / g)
            a_sR = aR * (ps / R.p) ** (gm1 / (2 * g))
            SHR = R.u + aR
            STR = us + a_sR
            mask = (xi >= us) & (xi < STR)
            rho[mask], u[mask], p[mask] = rho_sR, us, ps
            mask = (xi >= STR) & (xi < SHR)
            af = 2.0 / gp1 * (aR - gm1 / 2.0 * (R.u - xi[mask]))
            u[mask] = 2.0 / gp1 * (-aR + gm1 / 2.0 * R.u + xi[mask])
            rho[mask] = R.rho * (af / aR) ** (2.0 / gm1)
            p[mask] = R.p * (af / aR) ** (2.0 * g / gm1)
            mask = xi >= SHR
            rho[mask], u[mask], p[mask] = R.rho, R.u, R.p

        return {"rho": rho, "u": u, "p": p}

    def sample_xt(self, x: np.ndarray, t: float, x0: float = 0.0) -> dict[str, np.ndarray]:
        """Sample at positions ``x`` and time ``t`` with diaphragm at ``x0``."""
        if t <= 0.0:
            x = np.asarray(x, dtype=float)
            out = {
                "rho": np.where(x < x0, self.left.rho, self.right.rho),
                "u": np.where(x < x0, self.left.u, self.right.u),
                "p": np.where(x < x0, self.left.p, self.right.p),
            }
            return out
        return self.sample((np.asarray(x, dtype=float) - x0) / t)


def solve_riemann(
    left: PrimitiveState,
    right: PrimitiveState,
    gamma: float = 1.4,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> ExactRiemannSolution:
    """Solve the Riemann problem exactly for p* and u*."""
    aL = left.sound_speed(gamma)
    aR = right.sound_speed(gamma)

    # Vacuum check (pressure positivity condition, Toro Eq. 4.40)
    if 2.0 * (aL + aR) / (gamma - 1.0) <= right.u - left.u:
        raise ValueError("Initial states generate vacuum; solver not applicable.")

    # Initial guess: two-rarefaction approximation, clipped to be positive
    z = (gamma - 1.0) / (2.0 * gamma)
    p0 = ((aL + aR - 0.5 * (gamma - 1.0) * (right.u - left.u))
          / (aL / left.p ** z + aR / right.p ** z)) ** (1.0 / z)
    p = max(p0, 1e-8 * min(left.p, right.p))

    for _ in range(max_iter):
        fL, dfL = _f_branch(p, left, gamma)
        fR, dfR = _f_branch(p, right, gamma)
        f = fL + fR + (right.u - left.u)
        dp = f / (dfL + dfR)
        p_new = p - dp
        if p_new <= 0.0:
            p_new = 0.5 * p
        if abs(p_new - p) / (0.5 * (p_new + p)) < tol:
            p = p_new
            break
        p = p_new

    fL, _ = _f_branch(p, left, gamma)
    fR, _ = _f_branch(p, right, gamma)
    u_star = 0.5 * (left.u + right.u) + 0.5 * (fR - fL)
    return ExactRiemannSolution(left=left, right=right, gamma=gamma, p_star=p, u_star=u_star)
