"""Transient 1-D wellbore blowdown model — Godunov/HLLC finite volume.

Phase-3 architecture. Governing equations (z upward from the closed
bottom, z = 0, to the wellhead, z = L; constant flow area A):

    d(rho)/dt   + d(rho u)/dz           = 0
    d(rho u)/dt + d(rho u^2 + p)/dz     = -rho g - f_D rho u|u| / (2 D)
    d(E)/dt     + d(u (E + p))/dz       = -rho u g + (4/D) q_w''

with E = rho (e + u^2/2), q_w'' the wall heat flux into the fluid, and
f_D the Darcy friction factor. Wall shear enters only the momentum
equation: for a stationary wall it performs no net work, and the
associated dissipation is captured automatically by the conservative
total-energy formulation.

Numerics: MUSCL-Hancock reconstruction (van Leer limiter, primitive
variables) with the HLLC approximate Riemann solver at interior faces;
explicit time stepping under a CFL condition; sources integrated with a
Heun (average of begin/end-of-step) corrector and gravity included in the
Hancock half-step for formal second order.

Boundary conditions:

* bottom — solid wall (mirror ghost cells);
* wellhead — discharge through a valve/choke into a plenum at specified
  back pressure, coupled to the interior solution through the outgoing
  characteristic. See :func:`wellhead_face_state`.

The fluid enters only through the EOS interface of
:mod:`hyperbolic1d.eos`, so the ideal-gas nitrogen model can be replaced
by a real-fluid CO2 model without modifying this module.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from scipy.optimize import brentq

from .eos import IdealGasEOS, NITROGEN
from .godunov import _flux_hllc, _limited_slopes

__all__ = [
    "WellGeometry",
    "ChokeValve",
    "AdiabaticWall",
    "PrescribedWallTemperature",
    "WellboreModel",
    "BlowdownResult",
    "wellhead_face_state",
    "vessel_blowdown_ode",
]

G_EARTH = 9.80665  # m/s^2


# ----------------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class WellGeometry:
    """Vertical well of constant circular cross-section."""

    depth: float = 609.6          # m (2,000 ft)
    diameter: float = 0.100       # m (parameter; set to actual tubing/casing ID)
    roughness: float = 4.6e-5     # m (commercial steel)

    @property
    def area(self) -> float:
        return 0.25 * np.pi * self.diameter**2

    @property
    def volume(self) -> float:
        return self.area * self.depth


@dataclass(frozen=True)
class ChokeValve:
    """Wellhead restriction: isentropic-nozzle (orifice) model.

    ``diameter`` is the physical bore; ``cd`` the discharge coefficient.
    ``p_back`` is the downstream (atmospheric) pressure.
    """

    diameter: float = 0.0127      # m (1/2 in)
    cd: float = 0.84
    p_back: float = 101325.0      # Pa

    @property
    def effective_area(self) -> float:
        return self.cd * 0.25 * np.pi * self.diameter**2


class AdiabaticWall:
    """q_w'' = 0 everywhere (baseline)."""

    def heat_flux(self, z, t, T_fluid, rho, u, p, geom, eos):
        return np.zeros_like(T_fluid)


@dataclass
class PrescribedWallTemperature:
    """Simplified benchmark: fixed wall temperature T_wall(z), convective
    coupling with Dittus-Boelter h (constant transport properties)."""

    T_wall: float | np.ndarray = 300.0
    mu: float = 1.8e-5            # Pa s
    Pr: float = 0.72

    def heat_flux(self, z, t, T_fluid, rho, u, p, geom, eos):
        cp = eos.cp
        k = self.mu * cp / self.Pr
        Re = np.maximum(rho * np.abs(u) * geom.diameter / self.mu, 1.0)
        Nu = np.maximum(0.023 * Re**0.8 * self.Pr**0.4, 3.66)
        h = Nu * k / geom.diameter
        return h * (self.T_wall - T_fluid)


def darcy_friction_factor(Re: np.ndarray, rel_rough: float) -> np.ndarray:
    """Churchill (1977) correlation: all-regime explicit Darcy factor."""
    Re = np.maximum(Re, 1e-6)
    A = (2.457 * np.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * rel_rough))) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)


# ----------------------------------------------------------------------------
# Wellhead boundary condition
# ----------------------------------------------------------------------------
def wellhead_face_state(eos: IdealGasEOS, rho_i, u_i, p_i, choke: ChokeValve,
                        A_pipe: float):
    """Pipe-side state at the wellhead face for discharge through a choke.

    The face state (rho_f, u_f, p_f) must simultaneously satisfy:

    1. the outgoing (C+) characteristic from the interior:
       u_f = u_i + 2/(gamma-1) (a_i - a_f), with the face on the
       isentrope through the interior state (the exit flow between the
       last cell and the valve is smooth expansion);
    2. mass balance with the valve: rho_f u_f A_pipe = Cd A_t G(p0f, T0f, p_b),
       where (p0f, T0f) is the stagnation state of the face flow and G
       the isentropic-nozzle mass flux (choked or subcritical).

    The scalar unknown is the face pressure p_f, bracketed between the
    sonic face pressure (below which the exit itself would be supersonic,
    which the C+ coupling forbids) and the interior pressure. If the valve
    demands more than the sonic pipe flux, the exit runs sonic at pipe
    area (open-pipe choking, the A_t -> A_pipe limit). If the interior
    stagnation pressure falls below back pressure, the valve is treated
    as closed (check valve, no backflow).

    Returns (rho_f, u_f, p_f, mdot, choked_flag).
    """
    g = eos.gamma
    a_i = eos.sound_speed(p_i, rho_i)
    T_i = eos.temperature(p_i, rho_i)

    # interior stagnation state (for the no-flow check)
    h0 = eos.cp * T_i + 0.5 * u_i**2
    T0_i = h0 / eos.cp
    p0_i = p_i * (T0_i / T_i) ** (g / (g - 1.0))
    if p0_i <= choke.p_back * (1.0 + 1e-12) or choke.effective_area <= 0.0:
        return rho_i, 0.0, p_i, 0.0, False  # closed

    # sonic face state from the C+ invariant (lowest admissible p_f)
    a_son = (2.0 * a_i + (g - 1.0) * u_i) / (g + 1.0)
    p_son = p_i * (a_son / a_i) ** (2.0 * g / (g - 1.0))

    def face(pf):
        rho_f = eos.isentrope(pf, p_i, rho_i)
        u_f = eos.riemann_invariant_velocity(pf, p_i, rho_i, u_i)
        return rho_f, u_f

    def residual(pf):
        rho_f, u_f = face(pf)
        T_f = eos.temperature(pf, rho_f)
        T0 = T_f + 0.5 * u_f**2 / eos.cp
        p0 = pf * (T0 / T_f) ** (g / (g - 1.0))
        G, _ = eos.orifice_mass_flux(p0, T0, choke.p_back)
        return rho_f * u_f * A_pipe - choke.effective_area * G

    # pipe supply is maximal at the sonic face; if the valve still demands
    # more, the exit chokes at pipe area
    if residual(p_son) <= 0.0:
        rho_f, u_f = face(p_son)
        pf = p_son
    elif residual(p_i) >= 0.0:
        # pipe already supplies more than the valve passes even with no
        # expansion: the face must compress (pf > p_i). Upper bound: the
        # pressure at which the C+ invariant brings the face to rest
        # (u_f = 0), where the pipe supply vanishes.
        p_rest = p_i * ((a_i + 0.5 * (g - 1.0) * max(u_i, 0.0)) / a_i) ** (
            2.0 * g / (g - 1.0))
        pf = brentq(residual, p_i, p_rest * (1.0 + 1e-9), xtol=1e-6, rtol=1e-12)
        rho_f, u_f = face(pf)
    else:
        pf = brentq(residual, p_son, p_i, xtol=1e-6, rtol=1e-12)
        rho_f, u_f = face(pf)

    if u_f < 0.0:  # check-valve clamp
        return rho_i, 0.0, p_i, 0.0, False

    T_f = eos.temperature(pf, rho_f)
    T0 = T_f + 0.5 * u_f**2 / eos.cp
    p0 = pf * (T0 / T_f) ** (g / (g - 1.0))
    _, choked = eos.orifice_mass_flux(p0, T0, choke.p_back)
    mdot = rho_f * u_f * A_pipe
    return rho_f, u_f, pf, mdot, choked


# ----------------------------------------------------------------------------
# Model and driver
# ----------------------------------------------------------------------------
@dataclass
class BlowdownResult:
    z: np.ndarray
    snap_times: np.ndarray
    p: np.ndarray            # (n_snap, n_cells)
    T: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    t_hist: np.ndarray       # dense history
    mdot: np.ndarray
    mass: np.ndarray
    p_wellhead: np.ndarray
    T_wellhead: np.ndarray
    choked: np.ndarray
    energy_balance_error: float = np.nan
    mass_balance_error: float = np.nan
    n_steps: int = 0
    wall_time: float = 0.0


@dataclass
class WellboreModel:
    geometry: WellGeometry = field(default_factory=WellGeometry)
    choke: ChokeValve = field(default_factory=ChokeValve)
    eos: IdealGasEOS = NITROGEN
    gravity: bool = True
    friction: bool = False
    mu: float = 1.8e-5                       # viscosity for Re (friction)
    wall_model: object = field(default_factory=AdiabaticWall)
    n_cells: int = 400
    cfl: float = 0.8

    # ------------------------------------------------------------------
    def initial_state(self, p0: float, T0: float,
                      init: Literal["uniform", "hydrostatic"] = "uniform"):
        """Uniform (p0, T0) everywhere, or isothermal hydrostatic profile
        with p0 prescribed at the wellhead."""
        L = self.geometry.depth
        zc = (np.arange(self.n_cells) + 0.5) * (L / self.n_cells)
        if init == "hydrostatic" and self.gravity:
            # isothermal column: p(z) = p0 * exp(g (L - z) / (R T0))
            p = p0 * np.exp(G_EARTH * (L - zc) / (self.eos.R * T0))
        else:
            p = np.full(self.n_cells, p0)
        T = np.full(self.n_cells, T0)
        rho, e = self.eos.rho_e_from_pT(p, T)
        V = np.zeros((self.n_cells, 3))
        V[:, 0] = rho
        V[:, 2] = rho * e
        return zc, V

    # ------------------------------------------------------------------
    def _sources(self, V, zc, t):
        rho = V[:, 0]
        u = V[:, 1] / rho
        e = V[:, 2] / rho - 0.5 * u * u
        p, T, _ = self.eos.primitives(rho, np.maximum(e, 1.0))
        S = np.zeros_like(V)
        if self.gravity:
            S[:, 1] += -rho * G_EARTH
            # NOTE: the gravitational *energy* source (-rho u g) is NOT
            # added here. It is applied in solve() in the energy-consistent
            # face-flux form  -g * (F_mass,j-1/2 + F_mass,j+1/2)/2, which
            # makes the discrete total-energy + potential-energy budget
            # telescope exactly to the boundary flux.
        if self.friction:
            D = self.geometry.diameter
            Re = rho * np.abs(u) * D / self.mu
            fD = darcy_friction_factor(Re, self.geometry.roughness / D)
            S[:, 1] += -fD * rho * u * np.abs(u) / (2.0 * D)
        qw = self.wall_model.heat_flux(zc, t, T, rho, u, p, self.geometry, self.eos)
        S[:, 2] += 4.0 * qw / self.geometry.diameter
        return S

    # ------------------------------------------------------------------
    def solve(self, p0: float, T0: float, t_end: float,
              init: Literal["uniform", "hydrostatic"] = "uniform",
              n_snapshots: int = 9,
              hist_stride: int = 20,
              progress: Callable[[float, float], None] | None = None,
              ) -> BlowdownResult:
        eos = self.eos
        g = eos.gamma
        n = self.n_cells
        L = self.geometry.depth
        dz = L / n
        A = self.geometry.area

        zc, V = self.initial_state(p0, T0, init)
        snap_times = np.linspace(0.0, t_end, n_snapshots)
        snaps = {"p": [], "T": [], "rho": [], "u": []}
        snap_done = 0

        t = 0.0
        step = 0
        mass0 = float(np.sum(V[:, 0]) * dz * A)
        # total energy incl. potential (for the adiabatic budget)
        pe = lambda VV: float(np.sum(VV[:, 0] * zc) * dz * A) * G_EARTH \
            if self.gravity else 0.0
        etot0 = float(np.sum(V[:, 2]) * dz * A) + pe(V)
        mdot_int = 0.0           # integral of mdot dt
        hflux_int = 0.0          # integral of mdot * (h0 + g z_top) dt
        t_hist, mdot_hist, mass_hist, pwh_hist, Twh_hist, chk_hist = \
            [], [], [], [], [], []

        # record the initial state as the first snapshot
        if snap_times.size and snap_times[0] <= 1e-12:
            rho = V[:, 0]
            u = V[:, 1] / rho
            e = V[:, 2] / rho - 0.5 * u * u
            p, T, _ = eos.primitives(rho, e)
            for k, q in (("p", p), ("T", T), ("rho", rho), ("u", u)):
                snaps[k].append(q.copy())
            snap_done = 1

        wall0 = _time.perf_counter()
        while t < t_end - 1e-12:
            rho = V[:, 0]
            u = V[:, 1] / rho
            e = V[:, 2] / rho - 0.5 * u * u
            p, T, a = eos.primitives(rho, e)
            if np.any(p <= 0.0) or np.any(rho <= 0.0):
                raise FloatingPointError(
                    f"non-physical state at t={t:.4g}s, step {step}")

            dt = self.cfl * dz / np.max(np.abs(u) + a)
            dt = min(dt, t_end - t)

            # ghosts: mirror at bottom (solid wall), zero-gradient at top
            # (the top face flux is overridden by the choke BC anyway)
            def gh(q, sign=1.0):
                return np.concatenate([sign * q[1::-1], q, q[-1:], q[-1:]])

            rg, ug, pg = gh(rho), gh(u, -1.0), gh(p)

            # MUSCL-Hancock predictor (with gravity in the half step)
            sr = _limited_slopes(rg, "vanleer")
            su = _limited_slopes(ug, "vanleer")
            sp = _limited_slopes(pg, "vanleer")
            dtdx2 = 0.5 * dt / dz
            drho = -(ug * sr + rg * su) * dtdx2
            du = -(ug * su + sp / rg) * dtdx2
            if self.gravity:
                du = du - G_EARTH * 0.5 * dt
            dp = -(g * pg * su + ug * sp) * dtdx2
            rL = np.maximum(rg + 0.5 * sr + drho, 1e-10)
            rR = np.maximum(rg - 0.5 * sr + drho, 1e-10)
            uL = ug + 0.5 * su + du
            uR = ug - 0.5 * su + du
            pL = np.maximum(pg + 0.5 * sp + dp, 1e-3)
            pR = np.maximum(pg - 0.5 * sp + dp, 1e-3)

            # exact solid wall at the bottom face: left state = mirror of the
            # reconstructed right state (guarantees zero mass/energy flux)
            rL[1], uL[1], pL[1] = rR[2], -uR[2], pR[2]

            F = _flux_hllc(rL[:-1], uL[:-1], pL[:-1],
                           rR[1:], uR[1:], pR[1:], g)
            Fi = F[1:-1]                      # faces 0..n of the physical cells

            # wellhead face flux from the choke model (uses the last cell)
            rho_f, u_f, p_f, mdot, choked = wellhead_face_state(
                eos, rho[-1], u[-1], p[-1], self.choke, A)
            E_f = p_f / (g - 1.0) + 0.5 * rho_f * u_f**2
            Fi[n] = [rho_f * u_f, rho_f * u_f**2 + p_f, u_f * (E_f + p_f)]

            Vstar = V - dt / dz * (Fi[1:n + 1] - Fi[0:n])

            # energy-consistent gravitational work from face mass fluxes
            if self.gravity:
                Vstar[:, 2] += -G_EARTH * dt * 0.5 * (Fi[0:n, 0] + Fi[1:n + 1, 0])

            # Heun corrector for the remaining sources
            S0 = self._sources(V, zc, t)
            S1 = self._sources(np.maximum(Vstar, 1e-12), zc, t + dt)
            V = Vstar + dt * 0.5 * (S0 + S1)

            # budgets
            h0_f = eos.enthalpy(p_f, rho_f) + 0.5 * u_f**2
            mdot_int += mdot * dt
            hflux_int += mdot * (h0_f + (G_EARTH * L if self.gravity else 0.0)) * dt

            t += dt
            step += 1
            if step % hist_stride == 0 or t >= t_end - 1e-12:
                t_hist.append(t)
                mdot_hist.append(mdot)
                mass_hist.append(float(np.sum(V[:, 0]) * dz * A))
                pwh_hist.append(p[-1])
                Twh_hist.append(T[-1])
                chk_hist.append(choked)
                if progress is not None:
                    progress(t, t_end)
            while snap_done < snap_times.size and t >= snap_times[snap_done] - 1e-12:
                rho = V[:, 0]
                u = V[:, 1] / rho
                e = V[:, 2] / rho - 0.5 * u * u
                p, T, _ = eos.primitives(rho, e)
                for k, q in (("p", p), ("T", T), ("rho", rho), ("u", u)):
                    snaps[k].append(q.copy())
                snap_done += 1

        wall = _time.perf_counter() - wall0
        mass_end = float(np.sum(V[:, 0]) * dz * A)
        etot_end = float(np.sum(V[:, 2]) * dz * A) + pe(V)
        mass_err = (mass_end + mdot_int - mass0) / mass0
        # adiabatic budget: E(t) + integral of mdot*(h0 + g z_top) = E(0)
        adiabatic = isinstance(self.wall_model, AdiabaticWall) and not self.friction
        energy_err = (etot_end + hflux_int - etot0) / abs(etot0) if adiabatic else np.nan

        return BlowdownResult(
            z=zc, snap_times=snap_times[:snap_done],
            p=np.array(snaps["p"]), T=np.array(snaps["T"]),
            rho=np.array(snaps["rho"]), u=np.array(snaps["u"]),
            t_hist=np.array(t_hist), mdot=np.array(mdot_hist),
            mass=np.array(mass_hist), p_wellhead=np.array(pwh_hist),
            T_wellhead=np.array(Twh_hist), choked=np.array(chk_hist),
            mass_balance_error=mass_err, energy_balance_error=energy_err,
            n_steps=step, wall_time=wall,
        )


# ----------------------------------------------------------------------------
# 0-D reference: adiabatic vessel blowdown through a choke
# ----------------------------------------------------------------------------
def vessel_blowdown_ode(eos: IdealGasEOS, volume: float, choke: ChokeValve,
                        p0: float, T0: float, t_end: float, n_out: int = 400):
    """Semi-analytical reference: lumped adiabatic vessel, isentropic gas
    remaining in the vessel, isentropic-nozzle discharge.

        dM/dt = -Cd A_t G(p, T, p_b),   p/p0 = (rho/rho0)^gamma.

    Valid as the quasi-static limit of the 1-D well when the acoustic
    transit time is small compared with the discharge time and gravity is
    neglected. Returns dict with t, p, T, rho, mdot, mass.
    """
    from scipy.integrate import solve_ivp

    g = eos.gamma
    rho0, _ = eos.rho_e_from_pT(p0, T0)

    def rhs(t, y):
        (rho,) = y
        rho = max(rho, 1e-8)
        p = p0 * (rho / rho0) ** g
        T = eos.temperature(p, rho)
        G, _ = eos.orifice_mass_flux(p, T, choke.p_back)  # u=0 stagnation
        if p <= choke.p_back:
            G = 0.0
        return [-choke.effective_area * G / volume]

    sol = solve_ivp(rhs, (0.0, t_end), [rho0], dense_output=True,
                    rtol=1e-10, atol=1e-12, max_step=t_end / 200)
    t = np.linspace(0.0, t_end, n_out)
    rho = sol.sol(t)[0]
    p = p0 * (rho / rho0) ** g
    T = p / (eos.R * rho)
    mdot = np.array([
        choke.effective_area * eos.orifice_mass_flux(pp, TT, choke.p_back)[0]
        if pp > choke.p_back else 0.0
        for pp, TT in zip(p, T)
    ])
    return {"t": t, "p": p, "T": T, "rho": rho, "mdot": mdot,
            "mass": rho * volume}
