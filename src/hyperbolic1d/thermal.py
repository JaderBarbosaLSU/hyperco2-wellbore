"""Coupled completion/formation radial heat-transfer model.

Implements the preferred (conjugate) thermal formulation: at every axial
cell of the wellbore model, a 1-D radial thermal network represents the
concentric completion layers and the surrounding formation, and is
advanced implicitly in time, coupled to the fluid through a convective
conductance. The wall heat flux returned to the wellbore energy equation
is the flux through the fluid/inner-casing interface.

Radial stack of the LSU CO2 research well (upper 2,000 ft; from the
facility design slides):

    fluid | 7-5/8" CRA inner casing | air annulus | 13-3/8" casing |
    cement | 20" surface casing | cement (26" borehole) | formation

* Solid shells carry thermal capacitance rho*cp and cylindrical-shell
  conduction resistances ln(r_out/r_in)/(2 pi k dz).
* The air annulus is modeled as a (nonlinear) conductance with no
  capacitance: natural convection via the Willhite (1967) wellbore
  treatment with the Dropkin & Somerscales correlation,
  k_eff = 0.049 (Gr Pr)^(1/3) Pr^0.074 k_air  (floored at k_air),
  in parallel with gray-surface radiation between concentric cylinders.
* The formation is discretized with logarithmically spaced shells out to
  ``r_far`` where the undisturbed geothermal temperature
  T_geo(z) = T_surface + gradient * depth is imposed. For blowdown time
  scales (minutes-hours) the thermal penetration depth into rock,
  sqrt(alpha t) ~ cm, is far smaller than r_far = 5 m.
* Fluid-side convection: Dittus-Boelter with a laminar floor.
* The 30" conductor pipe (top ~150 ft only) is neglected.

Time advance: backward-Euler on each radial network (tridiagonal Thomas
solve, vectorized across axial cells), with the fluid temperature and the
nonlinear annulus conductance evaluated at the previous coupling instant
(coefficient lagging). Network energy is accounted exactly, enabling a
discrete heat balance check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SolidLayer",
    "GasGap",
    "FormationCoupledWall",
    "lsu_co2_well_stack",
    "GEOTHERMAL_DEFAULT",
]

SIGMA_SB = 5.670374419e-8

IN = 0.0254  # m per inch


@dataclass(frozen=True)
class SolidLayer:
    """Annular solid shell [r_in, r_out] with conduction + capacitance."""

    r_in: float
    r_out: float
    k: float
    rho: float
    cp: float
    name: str = ""
    n_sub: int = 1          # radial sub-shells


@dataclass(frozen=True)
class GasGap:
    """Gas-filled annulus modeled as a massless nonlinear conductance:
    Willhite-type natural convection + concentric-cylinder radiation."""

    r_in: float
    r_out: float
    k: float = 0.026        # W/m K (air)
    nu: float = 1.6e-5      # m^2/s
    Pr: float = 0.71
    eps_in: float = 0.8     # emissivity of inner surface (casing OD)
    eps_out: float = 0.8    # emissivity of outer surface (casing ID)
    name: str = "air annulus"

    def conductance_per_length(self, T_in, T_out):
        """G' [W/m/K] such that Q' = G' (T_in - T_out). Vectorized."""
        T_in = np.asarray(T_in, dtype=float)
        T_out = np.asarray(T_out, dtype=float)
        Tm = 0.5 * (T_in + T_out)
        dT = np.abs(T_in - T_out)
        gap = self.r_out - self.r_in
        beta = 1.0 / Tm
        Gr = 9.80665 * beta * dT * gap**3 / self.nu**2
        GrPr = Gr * self.Pr
        # Dropkin & Somerscales via Willhite (1967); conduction floor
        k_eff = np.maximum(
            self.k, 0.049 * np.cbrt(GrPr) * self.Pr**0.074 * self.k)
        G_cond = 2.0 * np.pi * k_eff / np.log(self.r_out / self.r_in)
        # gray concentric-cylinder radiation, linearized conductance
        F = 1.0 / (1.0 / self.eps_in
                   + (self.r_in / self.r_out) * (1.0 / self.eps_out - 1.0))
        h_r = SIGMA_SB * F * (T_in**2 + T_out**2) * (T_in + T_out)
        G_rad = 2.0 * np.pi * self.r_in * h_r
        return G_cond + G_rad


@dataclass(frozen=True)
class Geothermal:
    T_surface: float = 294.3     # K (~70 F)
    gradient: float = 0.0255     # K/m (Gulf-coast ~1.4 F/100 ft)

    def T(self, depth):
        return self.T_surface + self.gradient * np.asarray(depth, dtype=float)


GEOTHERMAL_DEFAULT = Geothermal()


def lsu_co2_well_stack():
    """Radial stack of the LSU CO2 research well, upper 2,000 ft.

    Dimensions from the facility design slides (casing table) and API
    wall thicknesses: 7-5/8" 35.8# (wall 0.4595" from the ppf relation),
    13-3/8" 72# (wall 0.514"), 20" 133# (wall 0.635"), 26" borehole.
    Returns (r_fluid, layers, formation_layer_template, r_far).
    """
    r1 = 0.5 * 7.625 * IN - 0.4595 * IN      # 7-5/8 ID/2 = 85.17 mm
    r2 = 0.5 * 7.625 * IN                    # 96.84 mm
    r3 = 0.5 * 13.375 * IN - 0.514 * IN      # 13-3/8 ID/2 = 156.81 mm
    r4 = 0.5 * 13.375 * IN                   # 169.86 mm
    r5 = 0.5 * 20.0 * IN - 0.635 * IN        # 20" ID/2 = 237.87 mm
    r6 = 0.5 * 20.0 * IN                     # 254 mm
    r7 = 0.5 * 26.0 * IN                     # borehole 330.2 mm

    cr13 = dict(k=19.0, rho=7750.0, cp=460.0)      # 13% Cr steel
    steel = dict(k=45.0, rho=7850.0, cp=490.0)     # carbon steel
    cement = dict(k=0.9, rho=2000.0, cp=900.0)

    layers = [
        SolidLayer(r1, r2, **cr13, name="7-5/8 CRA casing"),
        GasGap(r2, r3),
        SolidLayer(r3, r4, **steel, name="13-3/8 casing"),
        SolidLayer(r4, r5, **cement, name="cement A", n_sub=2),
        SolidLayer(r5, r6, **steel, name="20 in casing"),
        SolidLayer(r6, r7, **cement, name="cement B", n_sub=2),
    ]
    formation = dict(k=2.1, rho=2300.0, cp=1000.0)  # Gulf-coast sediments
    return r1, layers, formation, 5.0


@dataclass
class FormationCoupledWall:
    """Conjugate wall model implementing the wellbore thermal interface.

    Build with the layer stack (default: LSU CO2 well) and attach to
    ``WellboreModel(wall_model=...)``. The wellbore solver calls
    :meth:`advance` once per time step; the returned array is the wall
    heat flux into the fluid, per unit inner-surface area.
    """

    layers: list | None = None
    r_fluid: float | None = None
    formation: dict | None = None
    r_far: float = 5.0
    n_formation: int = 16
    geothermal: Geothermal = field(default_factory=lambda: GEOTHERMAL_DEFAULT)
    # fluid-side transport properties for the convection correlation
    mu: float = 1.8e-5
    Pr_fluid: float = 0.72
    # populated on first advance()
    _init: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.layers is None:
            self.r_fluid, self.layers, self.formation, self.r_far = \
                lsu_co2_well_stack()

    # ------------------------------------------------------------------
    def _build(self, z, geom):
        """Assemble radial nodes (shared by all axial cells)."""
        shells = []     # (r_in, r_out, k, rhocp)
        gaps = {}       # index in connection list -> GasGap
        for lay in self.layers:
            if isinstance(lay, GasGap):
                gaps[len(shells) - 1] = lay      # between previous & next node
            else:
                r_edges = np.linspace(lay.r_in, lay.r_out, lay.n_sub + 1)
                for a, b in zip(r_edges[:-1], r_edges[1:]):
                    shells.append((a, b, lay.k, lay.rho * lay.cp))
        # formation shells, log-spaced
        r0 = shells[-1][1]
        r_edges = np.geomspace(r0, self.r_far, self.n_formation + 1)
        f = self.formation
        for a, b in zip(r_edges[:-1], r_edges[1:]):
            shells.append((a, b, f["k"], f["rho"] * f["cp"]))

        n = len(shells)
        self._r_in = np.array([s[0] for s in shells])
        self._r_out = np.array([s[1] for s in shells])
        self._r_c = np.sqrt(self._r_in * self._r_out)   # log-mean position
        self._k = np.array([s[2] for s in shells])
        # capacitance per unit length [J/m/K]
        self._Cp = np.array([s[3] for s in shells]) * np.pi * (
            self._r_out**2 - self._r_in**2)
        # half-shell resistances per unit length [m K / W]
        self._R_in_half = np.log(self._r_c / self._r_in) / (2 * np.pi * self._k)
        self._R_out_half = np.log(self._r_out / self._r_c) / (2 * np.pi * self._k)
        # static inter-node conductances (gaps handled separately)
        self._G_static = np.zeros(n - 1)
        self._gap_at = {}
        for i in range(n - 1):
            R = self._R_out_half[i] + self._R_in_half[i + 1]
            if i in gaps:
                self._gap_at[i] = gaps[i]
                self._R_solid_at_gap = R      # series solid halves
                self._G_static[i] = np.nan    # dynamic
            else:
                self._G_static[i] = 1.0 / R
        # outer boundary conductance to the undisturbed formation
        self._G_outer = 1.0 / self._R_out_half[-1]
        # temperatures: initialize at local geothermal
        nz = z.size
        depth = z[-1] + (z[1] - z[0]) / 2 - z   # depth below wellhead approx
        self._T_geo = self.geothermal.T(depth)
        self._T = np.tile(self._T_geo[:, None], (1, n))
        self._nz, self._nr = nz, n
        self._Q_into_fluid = 0.0      # J per unit? accounted in solver
        self._init = True

    # ------------------------------------------------------------------
    def _h_fluid(self, T, rho, u, p, geom, eos):
        cp = eos.cp
        k = self.mu * cp / self.Pr_fluid
        Re = np.maximum(rho * np.abs(u) * geom.diameter / self.mu, 1.0)
        Nu = np.maximum(0.023 * Re**0.8 * self.Pr_fluid**0.4, 3.66)
        return Nu * k / geom.diameter

    # ------------------------------------------------------------------
    def advance(self, z, t, dt, T_fluid, rho, u, p, geom, eos):
        """Advance the radial networks by dt; return q_w'' into the fluid
        [W/m^2] on the inner casing surface (area 2 pi r_fluid per length).
        """
        if not self._init:
            self._build(z, geom)
            if abs(geom.diameter - 2.0 * self.r_fluid) > 1e-6:
                import warnings
                warnings.warn(
                    f"WellGeometry.diameter={geom.diameter:.4f} m differs "
                    f"from the thermal stack bore {2*self.r_fluid:.4f} m; "
                    "heat-flux areas assume they match.")

        T = self._T
        n = self._nr

        # fluid-side conductance per unit length (incl. inner half-shell)
        h = self._h_fluid(T_fluid, rho, u, p, geom, eos)
        G_f = 1.0 / (1.0 / (h * 2 * np.pi * self.r_fluid) + self._R_in_half[0])

        # inter-node conductances; gap conductance lagged on current temps
        G = np.tile(self._G_static[None, :], (self._nz, 1))
        for i, gap in self._gap_at.items():
            Ggap = gap.conductance_per_length(T[:, i], T[:, i + 1])
            G[:, i] = 1.0 / (1.0 / Ggap + self._R_solid_at_gap)

        # assemble tridiagonal (vectorized over axial cells): backward Euler
        a = np.zeros((self._nz, n))      # sub-diagonal
        b = np.zeros((self._nz, n))      # diagonal
        c = np.zeros((self._nz, n))      # super-diagonal
        d = np.zeros((self._nz, n))      # rhs

        Cdt = self._Cp[None, :] / dt
        b[:] = Cdt
        d[:] = Cdt * T
        # fluid coupling at node 0 (T_fluid explicit)
        b[:, 0] += G_f
        d[:, 0] += G_f * T_fluid
        # interior connections
        a[:, 1:] = -G
        c[:, :-1] = -G
        b[:, 1:] += G
        b[:, :-1] += G
        # outer Dirichlet via conductance to T_geo
        b[:, -1] += self._G_outer
        d[:, -1] += self._G_outer * self._T_geo

        # vectorized Thomas
        cp_ = np.zeros_like(c)
        dp_ = np.zeros_like(d)
        cp_[:, 0] = c[:, 0] / b[:, 0]
        dp_[:, 0] = d[:, 0] / b[:, 0]
        for i in range(1, n):
            m = b[:, i] - a[:, i] * cp_[:, i - 1]
            cp_[:, i] = c[:, i] / m
            dp_[:, i] = (d[:, i] - a[:, i] * dp_[:, i - 1]) / m
        Tn = np.empty_like(T)
        Tn[:, -1] = dp_[:, -1]
        for i in range(n - 2, -1, -1):
            Tn[:, i] = dp_[:, i] - cp_[:, i] * Tn[:, i + 1]
        self._T = Tn

        # heat flux into the fluid on the inner surface, implicit in T_wall
        Qp = G_f * (Tn[:, 0] - T_fluid)            # W per unit length
        return Qp / (2 * np.pi * self.r_fluid)     # W/m^2

    # ------------------------------------------------------------------
    @property
    def radial_temperatures(self):
        """(nz, nr) node temperatures; node radii in ``radial_positions``."""
        return self._T

    @property
    def radial_positions(self):
        return self._r_c

    def stored_energy_per_length(self):
        """Sum of C'(T - T_geo) over nodes [J/m per axial cell]."""
        return np.sum(self._Cp[None, :] * (self._T - self._T_geo[:, None]),
                      axis=1)
