"""Real-fluid CO2 equation of state for the wellbore solver.

Property source: the Span & Wagner (1996) reference equation of state for
CO2, evaluated through CoolProp. Range of validity of the underlying EOS:
216.6-1100 K, up to 800 MPa; the tabulated envelope used here (gas phase,
~0.5-300 kg/m3, 220-340 K by default) is far inside it.

Coupling to the conservative finite-volume variables
----------------------------------------------------
The solver's inner loop requires the closure (rho, e) -> (p, T, a) for
every cell at every step. Direct CoolProp calls there would dominate the
run time, so the closure is served from precomputed tables on a uniform
(ln rho, T) grid storing e, p, a and the saturation margin:

* (rho, e) -> T by a vectorized Newton iteration on the (monotone in T)
  tabulated e(ln rho, T), followed by bilinear evaluation of p and a.
* Table resolution is a parameter; with the default 512x512 grid the
  interpolation error against direct Span-Wagner evaluation is quantified
  by :meth:`SpanWagnerCO2.table_verification` and reported in the tests
  (target < 0.1 % in p, T, a over the gas envelope).

Thermodynamic consistency: p, e, a in the table are each evaluated from
the same Span-Wagner Helmholtz form, so they are mutually consistent at
the table nodes; bilinear interpolation introduces the only (quantified)
inconsistency. The speed of sound is the full thermodynamic
a^2 = (dp/d rho)_s of the reference EOS -- not an ideal-gas surrogate --
so the HLLC wave speeds and the CFL constraint carry the real-fluid
acoustics (a is strongly reduced near the dome).

Boundary thermodynamics (choke): the outlet model requires the isentrope
through the interior state, the C+ Riemann-invariant velocity integral
u(p) = u_i + int_p^{p_i} dp'/(rho a) at constant s, the stagnation state
from (h0, s), and the isentropic-nozzle mass flux. These are evaluated
with scalar CoolProp calls (PSmass / HmassSmass inputs, Gauss-Legendre
quadrature for the invariant, Brent maximization of G = rho_t u_t over
the throat pressure for choking). They run once per choke update, not
per cell, so accuracy is kept at full EOS fidelity there.

Single-phase monitoring: :meth:`saturation_margin` returns T - T_sat(p)
per cell (positive = superheated gas); the solver tracks its minimum and
stops with a flagged event on first dome entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SpanWagnerCO2"]


def _CP():
    import CoolProp.CoolProp as CP
    return CP


# Gauss-Legendre nodes/weights (8-point) on [0, 1]
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_GL_X = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W


@dataclass
class SpanWagnerCO2:
    """Span-Wagner CO2 behind the solver's EOS interface (gas phase)."""

    fluid: str = "CO2"
    rho_min: float = 0.5          # kg/m^3
    rho_max: float = 300.0
    T_min: float = 220.0          # K
    T_max: float = 340.0
    n_rho: int = 512
    n_T: int = 512
    name: str = "CO2 (Span-Wagner via CoolProp)"
    # transport constants for friction / Dittus-Boelter (documented approx.)
    mu_ref: float = 1.5e-5        # Pa s (CO2 gas ~300 K)
    Pr_ref: float = 0.78
    # homogeneous-equilibrium (HEM) choke: allow the throat isentrope to
    # continue into the vapor dome with equilibrium (flashed) rho and h.
    # The WELLBORE remains single-phase (dome monitor unchanged); only the
    # algebraic nozzle expansion may flash. False restores the previous
    # behavior (flux evaluated at the single-phase dome bound).
    hem_choke: bool = True
    # two_phase=True builds HEM (homogeneous-equilibrium) tables: nodes
    # inside the vapor dome carry the equilibrium MIXTURE state (lever-rule
    # e and s, p = p_sat(T), and the full equilibrium two-phase sound speed
    # from saturation-line derivatives), and the density range is extended
    # to the saturated-liquid side. The (rho, e) -> T Newton of
    # :meth:`primitives` is unchanged: (de/dT)_rho = c_v,eq > 0 inside the
    # dome, so the inversion stays monotone. With the default False the
    # tables are gas-phase only with nonphysical extrapolated dome fill
    # (single-phase model; validity policed by the saturation monitor).
    two_phase: bool = False
    _built: bool = field(default=False, repr=False)

    def __post_init__(self):
        # extend the default density range to cover the liquid branch when
        # mixture states are representable (saturated liquid at 220 K is
        # ~1166 kg/m^3); an explicit user-set rho_max is respected
        if self.two_phase and self.rho_max == 300.0:
            self.rho_max = 1200.0

    # ------------------------------------------------------------------
    # table construction
    # ------------------------------------------------------------------
    def _build(self):
        CP = _CP()
        self.R = CP.PropsSI("gas_constant", self.fluid) / CP.PropsSI(
            "molar_mass", self.fluid)
        self.T_crit = CP.PropsSI("Tcrit", self.fluid)
        self.p_crit = CP.PropsSI("pcrit", self.fluid)
        self._cp_ref = CP.PropsSI("C", "P", 5e6, "T", 300.0, self.fluid)

        lr = np.linspace(np.log(self.rho_min), np.log(self.rho_max),
                         self.n_rho)
        Tg = np.linspace(self.T_min, self.T_max, self.n_T)
        self._lr, self._Tg = lr, Tg
        self._dlr = lr[1] - lr[0]
        self._dT = Tg[1] - Tg[0]

        st = CP.AbstractState("HEOS", self.fluid)
        shape = (self.n_rho, self.n_T)
        E = np.full(shape, np.nan)
        P = np.full(shape, np.nan)
        A = np.full(shape, np.nan)
        S = np.full(shape, np.nan)
        valid = np.zeros(shape, dtype=bool)
        iDT = CP.DmassT_INPUTS
        rhos = np.exp(lr)

        # saturation-line arrays on the T grid (two-phase mode): hold the
        # lever-rule endpoints and the saturation-line derivatives d(.)/dp
        # needed for the equilibrium sound speed. The dome is treated only
        # up to T_crit - 0.5 K; the sliver above that is handled as
        # single-phase by CoolProp directly (the dome is vanishingly thin
        # there and the lever-rule derivatives degenerate).
        if self.two_phase:
            Tdome = Tg[Tg < self.T_crit - 0.5]
            nd = Tdome.size
            sat = {k: np.empty(nd) for k in
                   ("p", "vl", "vv", "el", "ev", "sl", "sv")}
            for j, T in enumerate(Tdome):
                T = float(T)
                sat["p"][j] = CP.PropsSI("P", "T", T, "Q", 0, self.fluid)
                for q, tag in ((0, "l"), (1, "v")):
                    sat["v" + tag][j] = 1.0 / CP.PropsSI(
                        "D", "T", T, "Q", q, self.fluid)
                    sat["e" + tag][j] = CP.PropsSI(
                        "U", "T", T, "Q", q, self.fluid)
                    sat["s" + tag][j] = CP.PropsSI(
                        "S", "T", T, "Q", q, self.fluid)
            dps = {k: np.gradient(sat[k], sat["p"])
                   for k in ("vl", "vv", "sl", "sv")}
            self._sat_grid = (Tdome, sat)

        for j, T in enumerate(Tg):
            T = float(T)
            if T < self.T_crit:
                rho_v = CP.PropsSI("D", "T", T, "Q", 1.0, self.fluid)
                rho_l = (CP.PropsSI("D", "T", T, "Q", 0.0, self.fluid)
                         if self.two_phase else np.inf)
            else:
                rho_v, rho_l = np.inf, np.inf
            for i, rho in enumerate(rhos):
                in_dome = (self.two_phase and T < self.T_crit - 0.5
                           and rho > 0.999 * rho_v and rho < 1.001 * rho_l)
                if in_dome:
                    # equilibrium mixture node: lever rule on v, e, s and
                    # the full equilibrium sound speed
                    #   a^2 = -v^2 / (dv/dp)_s,
                    #   (dv/dp)_s = (1-x) vl' + x vv'
                    #             - (vv - vl) [(1-x) sl' + x sv']/(sv - sl)
                    # with ' = d/dp along the saturation line (includes
                    # equilibrium mass transfer; discontinuous at x=0,1).
                    jj = int(np.searchsorted(Tdome, T))
                    jj = min(max(jj, 0), nd - 1)
                    v = 1.0 / rho
                    vl, vv = sat["vl"][jj], sat["vv"][jj]
                    x = np.clip((v - vl) / (vv - vl), 0.0, 1.0)
                    E[i, j] = (1 - x) * sat["el"][jj] + x * sat["ev"][jj]
                    S[i, j] = (1 - x) * sat["sl"][jj] + x * sat["sv"][jj]
                    P[i, j] = sat["p"][jj]
                    dvdp_s = ((1 - x) * dps["vl"][jj] + x * dps["vv"][jj]
                              - (vv - vl)
                              * ((1 - x) * dps["sl"][jj] + x * dps["sv"][jj])
                              / (sat["sv"][jj] - sat["sl"][jj]))
                    A[i, j] = np.sqrt(-v * v / dvdp_s)
                    valid[i, j] = True
                    continue
                # single-phase node (gas; or liquid/supercritical in
                # two-phase mode). Narrow near-boundary exclusion bands
                # avoid CoolProp flash ambiguity at the dome edge.
                if not self.two_phase and rho > 0.999 * rho_v:
                    continue
                if self.two_phase and 0.999 * rho_v < rho < 1.001 * rho_l:
                    continue   # near-critical sliver band, filled below
                try:
                    st.update(iDT, float(rho), T)
                    E[i, j] = st.umass()
                    P[i, j] = st.p()
                    A[i, j] = st.speed_sound()
                    S[i, j] = st.smass()
                    valid[i, j] = True
                except ValueError:
                    continue
        # fill remaining invalid nodes by linear extrapolation in ln rho
        # from the last two valid nodes of each T column. In single-phase
        # mode these are the nonphysical dome placeholders that keep
        # bilinear lookups finite for boundary-grazing states (validity is
        # governed by the saturation monitor); in two-phase mode only the
        # thin near-critical sliver and flash-failure nodes remain.
        for j in range(self.n_T):
            col = valid[:, j]
            if col.all() or not col.any():
                continue
            iv = np.where(col)[0]
            if self.two_phase:
                # interior gaps (near-critical sliver, rare flash
                # failures): linear interpolation between valid neighbors;
                # constant continuation at the tails
                for F in (E, P, A, S):
                    F[:, j] = np.interp(lr, lr[iv], F[iv, j])
            else:
                # single-phase mode: linear ln-rho extrapolation from the
                # last two valid vapor-side nodes (unchanged legacy fill)
                ilast = iv[-1]
                if ilast < 1:
                    continue
                for F in (E, P, A, S):
                    slope = (F[ilast, j] - F[ilast - 1, j]) / self._dlr
                    for i in range(ilast + 1, self.n_rho):
                        F[i, j] = F[ilast, j] + slope * (lr[i] - lr[ilast])
        self._E, self._P, self._A, self._S = E, P, A, S
        self._valid = valid
        self._built = True

    def _ensure(self):
        if not self._built:
            self._build()

    # ------------------------------------------------------------------
    # bilinear helpers on the (ln rho, T) grid
    # ------------------------------------------------------------------
    def _bilin(self, F, lrho, T):
        i = np.clip(((lrho - self._lr[0]) / self._dlr).astype(int),
                    0, self.n_rho - 2)
        j = np.clip(((T - self._Tg[0]) / self._dT).astype(int),
                    0, self.n_T - 2)
        x = (lrho - self._lr[i]) / self._dlr
        y = (T - self._Tg[j]) / self._dT
        return ((1 - x) * (1 - y) * F[i, j] + x * (1 - y) * F[i + 1, j]
                + (1 - x) * y * F[i, j + 1] + x * y * F[i + 1, j + 1])

    # ------------------------------------------------------------------
    # solver interface
    # ------------------------------------------------------------------
    @property
    def cp(self):
        """Reference cp (5 MPa, 300 K) for the convection correlation
        only; the energy equation never uses it. Documented approximation."""
        self._ensure()
        return self._cp_ref

    def primitives(self, rho, e):
        """(p, T, a) from (rho, e): vectorized Newton on tabulated e(.,T)."""
        self._ensure()
        rho = np.asarray(rho, dtype=float)
        e = np.asarray(e, dtype=float)
        lr = np.log(np.clip(rho, self.rho_min, self.rho_max))
        # bisection-safeguarded Newton on the (strictly increasing)
        # tabulated e(., T). Plain Newton limit-cycles across the vapor-
        # dome boundary in two-phase mode, where (de/dT)_rho jumps by a
        # factor of several at the kink; the bracket makes the iteration
        # globally convergent for any monotone column.
        T = np.full_like(rho, 300.0)
        lo = np.full_like(rho, self.T_min)
        hi = np.full_like(rho, self.T_max)
        for _ in range(60):
            f = self._bilin(self._E, lr, T) - e
            lo = np.where(f < 0.0, np.maximum(lo, T), lo)
            hi = np.where(f > 0.0, np.minimum(hi, T), hi)
            dT = 2.0
            de = (self._bilin(self._E, lr, np.minimum(T + dT, self.T_max))
                  - self._bilin(self._E, lr, np.maximum(T - dT, self.T_min)))
            cv = de / (np.minimum(T + dT, self.T_max)
                       - np.maximum(T - dT, self.T_min))
            Tn = T - f / np.maximum(cv, 50.0)
            bad = ~np.isfinite(Tn) | (Tn <= lo) | (Tn >= hi)
            Tn = np.where(bad, 0.5 * (lo + hi), Tn)
            step = Tn - T
            T = Tn
            if np.max(np.abs(step)) < 1e-8:
                break
        p = self._bilin(self._P, lr, T)
        a = self._bilin(self._A, lr, T)
        return p, T, a

    def rho_e_from_pT(self, p, T):
        CP = _CP()
        p = np.atleast_1d(np.asarray(p, dtype=float))
        T = np.atleast_1d(np.asarray(T, dtype=float))
        if p.size == 1 and T.size > 1:
            p = np.full_like(T, p[0])
        if T.size == 1 and p.size > 1:
            T = np.full_like(p, T[0])
        rho = np.array([CP.PropsSI("D", "P", pp, "T", TT, self.fluid)
                        for pp, TT in zip(p, T)])
        e = np.array([CP.PropsSI("U", "P", pp, "T", TT, self.fluid)
                      for pp, TT in zip(p, T)])
        if rho.size == 1:
            return float(rho[0]), float(e[0])
        return rho, e

    def internal_energy(self, rho, p):
        """e from (rho, p): invert tabulated p(., T) in T, then e."""
        self._ensure()
        rho = np.asarray(rho, dtype=float)
        p = np.asarray(p, dtype=float)
        lr = np.log(np.clip(rho, self.rho_min, self.rho_max))
        # bisection-safeguarded Newton (see primitives(): the dome
        # boundary kinks the column and defeats plain Newton)
        T = np.full_like(rho, 300.0)
        lo = np.full_like(rho, self.T_min)
        hi = np.full_like(rho, self.T_max)
        for _ in range(60):
            f = self._bilin(self._P, lr, T) - p
            lo = np.where(f < 0.0, np.maximum(lo, T), lo)
            hi = np.where(f > 0.0, np.minimum(hi, T), hi)
            dT = 2.0
            dp = (self._bilin(self._P, lr, np.minimum(T + dT, self.T_max))
                  - self._bilin(self._P, lr, np.maximum(T - dT, self.T_min)))
            dpdT = dp / (np.minimum(T + dT, self.T_max)
                         - np.maximum(T - dT, self.T_min))
            Tn = T - f / np.maximum(dpdT, 1.0)
            bad = ~np.isfinite(Tn) | (Tn <= lo) | (Tn >= hi)
            Tn = np.where(bad, 0.5 * (lo + hi), Tn)
            step = Tn - T
            T = Tn
            if np.max(np.abs(step)) < 1e-8:
                break
        return self._bilin(self._E, lr, T)

    def enthalpy(self, p, rho, e=None):
        if e is None:
            e = self.internal_energy(rho, p)
        return e + p / rho

    def sound_speed(self, p, rho):
        self._ensure()
        e = self.internal_energy(rho, p)
        return self.primitives(rho, e)[2]

    def temperature(self, p, rho):
        e = self.internal_energy(rho, p)
        return self.primitives(rho, e)[1]

    # ------------------------------------------------------------------
    # saturation monitoring
    # ------------------------------------------------------------------
    def quality(self, rho, T):
        """Equilibrium vapor mass quality x(rho, T) by the lever rule on
        specific volume (two-phase tables required). Returns 1.0 for
        superheated gas and for T above the dome treatment limit
        (T_crit - 0.5 K), 0.0 for subcooled liquid. Vectorized."""
        self._ensure()
        if not self.two_phase:
            raise ValueError("quality() requires two_phase=True tables")
        rho = np.atleast_1d(np.asarray(rho, dtype=float))
        T = np.atleast_1d(np.asarray(T, dtype=float))
        Tdome, sat = self._sat_grid
        Tcl = np.clip(T, Tdome[0], Tdome[-1])
        vl = np.interp(Tcl, Tdome, sat["vl"])
        vv = np.interp(Tcl, Tdome, sat["vv"])
        x = np.clip((1.0 / rho - vl) / (vv - vl), 0.0, 1.0)
        x = np.where(T >= self.T_crit - 0.5, 1.0, x)
        return x if x.size > 1 else float(x[0])

    def void_fraction(self, rho, T):
        """Equilibrium vapor volume fraction alpha = x * v_v / v."""
        self._ensure()
        x = np.atleast_1d(self.quality(rho, T))
        rho = np.atleast_1d(np.asarray(rho, dtype=float))
        T = np.atleast_1d(np.asarray(T, dtype=float))
        Tdome, sat = self._sat_grid
        vv = np.interp(np.clip(T, Tdome[0], Tdome[-1]), Tdome, sat["vv"])
        a = np.clip(x * vv * rho, 0.0, 1.0)
        return a if a.size > 1 else float(a[0])

    def saturation_margin(self, p, T):
        """T - T_sat(p) [K]; positive = single-phase gas. Above p_crit
        the margin is taken against T_crit (dense/supercritical gate).
        Vectorized via a precomputed T_sat(ln p) table (200 nodes,
        interpolation error < 0.01 K)."""
        self._ensure()
        self._ensure_sat_tables()
        p = np.atleast_1d(np.asarray(p, dtype=float))
        T = np.atleast_1d(np.asarray(T, dtype=float))
        Ts = np.interp(np.log(np.clip(p, np.exp(self._lnp_sat[0]), None)),
                       self._lnp_sat, self._T_sat,
                       right=self.T_crit)
        Ts = np.where(p >= self.p_crit, self.T_crit, Ts)
        return T - Ts

    # ------------------------------------------------------------------
    # boundary (choke) thermodynamics -- scalar, full CoolProp fidelity
    # ------------------------------------------------------------------
    def _ensure_sat_tables(self):
        if hasattr(self, "_lnp_sat"):
            return
        self._ensure()
        CP = _CP()
        p_tr = CP.PropsSI("ptriple", self.fluid)
        pg = np.geomspace(p_tr * 1.001, self.p_crit * 0.99999, 200)
        self._lnp_sat = np.log(pg)
        self._T_sat = np.array([CP.PropsSI("T", "P", float(pp), "Q", 1.0,
                                           self.fluid) for pp in pg])
        # saturated-VAPOR entropy, monotone decreasing with pressure
        self._s_sat = np.array([CP.PropsSI("S", "P", float(pp), "Q", 1.0,
                                           self.fluid) for pp in pg])

    def saturation_pressure_for_entropy(self, s):
        """Pressure at which the isentrope s meets the saturated-vapor
        line (dome crossing from the gas side); None if it does not.
        Resolved from the precomputed s_vap(p) table (monotone), avoiding
        CoolProp's S-Q flash which is not robust over the whole dome."""
        self._ensure_sat_tables()
        s = float(s)
        # s_sat decreases with p: crossing exists for s in its range
        if s <= self._s_sat[-1] or s >= self._s_sat[0]:
            return None if s >= self._s_sat[0] else float(np.exp(
                self._lnp_sat[-1]))   # below critical-point entropy: treat
            # the whole sub-critical range as dome-side (conservative)
        lnp = np.interp(s, self._s_sat[::-1], self._lnp_sat[::-1])
        return float(np.exp(lnp))

    def _state_s(self, p, s):
        CP = _CP()
        rho = CP.PropsSI("D", "P", p, "S", s, self.fluid)
        a = CP.PropsSI("A", "P", p, "S", s, self.fluid)
        h = CP.PropsSI("H", "P", p, "S", s, self.fluid)
        return rho, a, h

    def _abstract_state(self):
        if not hasattr(self, "_st"):
            self._st = _CP().AbstractState("HEOS", self.fluid)
        return self._st

    def _rho_h_s(self, p, s):
        CP = _CP()
        st = self._abstract_state()
        st.update(CP.PSmass_INPUTS, float(p), float(s))
        return st.rhomass(), st.hmass()

    def entropy(self, p, rho):
        CP = _CP()
        st = self._abstract_state()
        st.update(CP.DmassP_INPUTS, float(rho), float(p))
        return st.smass()

    def isentrope(self, p, p_ref, rho_ref):
        s = self.entropy(p_ref, rho_ref)
        return self._rho_h_s(float(p), s)[0]

    def riemann_invariant_velocity(self, p, p_ref, rho_ref, u_ref):
        """u(p) = u_ref + int_p^{p_ref} dp'/(rho a) at s = s_ref
        (8-point Gauss-Legendre)."""
        s = self.entropy(p_ref, rho_ref)
        p = float(p)
        p_ref = float(p_ref)
        p_dome = self.saturation_pressure_for_entropy(s)
        if p_dome is not None:
            # integrate only over the gas side of the isentrope; the
            # solver's per-step dome monitor stops the run before this
            # clamp matters materially, but the boundary call must not
            # raise on a transiently grazing state
            lim = p_dome * 1.002
            p = max(p, lim)
            if p_ref < lim:
                # reference state at/inside the dome: acoustic estimate
                a_ref = max(self.primitives(
                    np.array([rho_ref]),
                    np.array([self.internal_energy(
                        np.array([rho_ref]), np.array([p_ref]))[0]]))[2][0],
                    50.0)
                return u_ref + (p_ref - p) / (rho_ref * a_ref)
        dp = p_ref - p
        acc = 0.0
        for x, w in zip(_GL_X, _GL_W):
            pp = p + x * dp
            rho, a, _ = self._state_s(pp, s)
            acc += w / (rho * a)
        return u_ref + acc * dp

    def stagnation_from_state(self, p, rho, u):
        """(p0, T0, s, h0) for the isentropic stagnation of (p, rho, u)."""
        CP = _CP()
        st = self._abstract_state()
        st.update(CP.DmassP_INPUTS, float(rho), float(p))
        s = st.smass()
        h0 = st.hmass() + 0.5 * u * u
        st.update(CP.HmassSmass_INPUTS, h0, s)
        return st.p(), st.T(), s, h0

    def orifice_mass_flux(self, p0, T0, p_back, s0=None, h0=None):
        """Real-gas isentropic-nozzle mass flux per unit throat area.

        G(p_t) = rho(p_t, s0) * sqrt(2 (h0 - h(p_t, s0))), maximized over
        the throat pressure; choked when the maximizer exceeds p_back.
        Returns (G, choked).

        With ``hem_choke`` (default) the throat expansion is allowed to
        enter the vapor dome: rho and h along the isentrope are then the
        homogeneous-equilibrium (flashed) mixture properties from the
        same Span-Wagner EOS, so G(p_t) is the standard HEM nozzle flux
        and the maximization finds the HEM critical (choked) flow.
        Reported on the instance after each call:
          last_choke_flashing        throat state inside the dome
          last_choke_throat_quality  HEM vapor quality at the throat
          last_choke_dome_limited    expansion truncated by the EOS
                                     validity boundary (sublimation /
                                     table range), not by the sonic
                                     condition -- with HEM enabled this
                                     is rare (deep vacuum discharge)
        In single-phase operation the upstream state is policed by the
        solver's per-step dome monitor. In HEM two-phase operation the
        stagnation state may itself be a saturated mixture, for which the
        (p0, T0) pair is degenerate (it sits on the saturation line):
        the caller then passes the stagnation entropy and enthalpy
        ``s0``/``h0`` directly (both are what the flux integrand actually
        uses) and T0 is ignored.
        """
        from scipy.optimize import minimize_scalar

        CP = _CP()
        p0, T0 = float(p0), float(T0)
        st = self._abstract_state()
        if s0 is None or h0 is None:
            st.update(CP.PT_INPUTS, p0, T0)
            s0 = st.smass()
            h0 = st.hmass()
        else:
            s0, h0 = float(s0), float(h0)

        def G(pt):
            """Nozzle mass flux; 0.0 where the throat state is outside
            the single-phase/EOS-validity region (e.g. the isentrope
            reaches the Span-Wagner Tmin = 216.6 K near the sublimation
            region during deep expansion to atmospheric back pressure)."""
            try:
                rho, h = self._rho_h_s(float(pt), s0)
            except ValueError:
                return 0.0
            dh = max(h0 - h, 0.0)
            return rho * np.sqrt(2.0 * dh)

        self.last_choke_dome_limited = False
        self.last_choke_flashing = False
        self.last_choke_throat_quality = None
        if p_back >= p0:
            return 0.0, False
        hem = getattr(self, "hem_choke", True)
        p_dome = self.saturation_pressure_for_entropy(s0)
        lo = max(p_back, 1.2e5, 0.05 * p0)
        if p_dome is not None and not hem:
            # single-phase scope (hem_choke=False): the throat expansion
            # is truncated at the dome crossing of the isentrope; the
            # returned flux is then the single-phase bound.
            lo = max(lo, p_dome * 1.001)
        if lo >= p0 * 0.999:
            # stagnation state itself at/inside the dome boundary
            self.last_choke_dome_limited = True
            return float(G(p0 * 0.999)), True
        # validity-guarded bracketed maximization: coarse log scan first
        # (G = 0 marks states outside the single-phase/EOS region), then
        # bounded refinement within the best valid bracket
        pts = np.geomspace(lo, p0 * 0.999, 15)
        Gs = np.array([G(pt) for pt in pts])
        imax = int(np.argmax(Gs))
        if Gs[imax] <= 0.0:
            self.last_choke_dome_limited = True
            return 0.0, False
        blo = pts[max(imax - 1, 0)]
        bhi = pts[min(imax + 1, pts.size - 1)]
        if (imax > 0 and Gs[imax - 1] == 0.0):
            # bordered by an invalid region: choke limited by the
            # phase/EOS boundary rather than the sonic condition
            self.last_choke_dome_limited = True
        res = minimize_scalar(lambda pt: -G(pt), bounds=(blo, bhi),
                              method="bounded",
                              options={"xatol": max(1e-4 * p0, 100.0)})
        p_star = float(res.x)
        if p_dome is not None and not hem and p_star <= lo * 1.002:
            self.last_choke_dome_limited = True
        choked = p_star > p_back * 1.001
        pt_eff = p_star if choked else float(p_back)
        if hem and p_dome is not None and pt_eff < p_dome * 0.999:
            # HEM throat state inside the dome: record flashing + quality
            self.last_choke_flashing = True
            try:
                CPm = _CP()
                st.update(CPm.PSmass_INPUTS, pt_eff, s0)
                q = st.Q()
                self.last_choke_throat_quality = (
                    float(q) if 0.0 <= q <= 1.0 else None)
            except ValueError:
                pass
        if choked:
            return float(-res.fun), True
        return float(G(float(p_back))), False

    # 0-D vessel support: pressure on the isentrope through (p_ref, rho_ref)
    def pressure_on_isentrope(self, rho, p_ref, rho_ref):
        CP = _CP()
        s = self.entropy(p_ref, rho_ref)
        return CP.PropsSI("P", "D", float(rho), "S", s, self.fluid)

    # ------------------------------------------------------------------
    # verification
    # ------------------------------------------------------------------
    def table_verification(self, n_samples=300, seed=7):
        """Random (rho, T) gas-phase states: closure (rho, e)->(p, T, a)
        vs direct CoolProp. Returns dict of max relative errors."""
        CP = _CP()
        self._ensure()
        rng = np.random.default_rng(seed)
        errs = {"p": [], "T": [], "a": []}
        n = 0
        while n < n_samples:
            rho = float(np.exp(rng.uniform(np.log(self.rho_min * 1.2),
                                           np.log(self.rho_max * 0.8))))
            T = float(rng.uniform(self.T_min + 5, self.T_max - 5))
            try:
                st = CP.AbstractState("HEOS", self.fluid)
                st.update(CP.DmassT_INPUTS, rho, T)
                if st.phase() not in (1, 2, 5):   # keep sc/gas only
                    continue
                e, p, a = st.umass(), st.p(), st.speed_sound()
            except ValueError:
                continue
            pt, Tt, at = self.primitives(np.array([rho]), np.array([e]))
            errs["p"].append(abs(pt[0] - p) / p)
            errs["T"].append(abs(Tt[0] - T) / T)
            errs["a"].append(abs(at[0] - a) / a)
            n += 1
        return {k: (float(np.max(v)), float(np.mean(v)))
                for k, v in errs.items()}
