"""Verification tests for the coupled completion/formation thermal model."""

import numpy as np
import pytest

from hyperbolic1d.eos import NITROGEN
from hyperbolic1d.thermal import (FormationCoupledWall, GasGap, Geothermal,
                                  SolidLayer, lsu_co2_well_stack)
from hyperbolic1d.wellbore import WellGeometry


def _drive(model, geom, T_fluid, dt, n_steps, u=5.0, p=1e7):
    """Advance the network with constant fluid conditions; returns q''(t)."""
    z = np.linspace(1.0, geom.depth, 4)
    rho = np.full(z.size, p / (NITROGEN.R * T_fluid))
    uu = np.full(z.size, u)
    pp = np.full(z.size, p)
    Tf = np.full(z.size, T_fluid)
    q = []
    for k in range(n_steps):
        q.append(model.advance(z, k * dt, dt, Tf, rho, uu, pp, geom, NITROGEN))
    return np.array(q)


def test_steady_state_flux_uniformity():
    """At steady state the heat flow must be identical across every
    radial interface and match the fluid-side conductance expression."""
    r1, layers, formation, _ = lsu_co2_well_stack()
    geom = WellGeometry(diameter=2 * r1)
    geo = Geothermal(T_surface=300.0, gradient=0.0)
    m = FormationCoupledWall(layers=layers, r_fluid=r1, formation=formation,
                             r_far=2.0, n_formation=10, geothermal=geo)
    # drive long enough that the (finite, outer-Dirichlet) network settles
    q = _drive(m, geom, T_fluid=380.0, dt=2e4, n_steps=4000)
    T = m.radial_temperatures[0]
    # interface fluxes from the solved temperature field
    Gst = m._G_static.copy()
    fluxes = []
    for i in range(T.size - 1):
        if np.isnan(Gst[i]):
            gap = m._gap_at[i]
            G = 1.0 / (1.0 / gap.conductance_per_length(T[i], T[i + 1])
                       + m._R_solid_at_gap)
        else:
            G = Gst[i]
        fluxes.append(G * (T[i] - T[i + 1]))
    fluxes.append(m._G_outer * (T[-1] - m._T_geo[0]))
    fluxes = np.array(fluxes)
    assert np.allclose(fluxes, fluxes[0], rtol=1e-6)
    # and equals the flux delivered to the fluid (opposite sign)
    q_end = q[-1, 0] * 2 * np.pi * r1      # W/m into fluid
    assert q_end == pytest.approx(-fluxes[0], rel=1e-6)


def test_network_energy_conservation():
    """Insulated outer boundary: energy absorbed from the fluid equals
    the change of stored energy in the network (backward Euler exact)."""
    r1, layers, formation, _ = lsu_co2_well_stack()
    geom = WellGeometry(diameter=2 * r1)
    geo = Geothermal(T_surface=300.0, gradient=0.0)
    m = FormationCoupledWall(layers=layers, r_fluid=r1, formation=formation,
                             r_far=1.0, n_formation=8, geothermal=geo)
    # first advance builds the network; then insulate the outer boundary
    _drive(m, geom, T_fluid=300.0, dt=1.0, n_steps=1)
    m._G_outer = 0.0
    E0 = m.stored_energy_per_length()[0]
    dt, n = 50.0, 400
    q = _drive(m, geom, T_fluid=380.0, dt=dt, n_steps=n)
    E1 = m.stored_energy_per_length()[0]
    absorbed = -np.sum(q[:, 0]) * 2 * np.pi * r1 * dt   # J/m into network
    assert absorbed == pytest.approx(E1 - E0, rel=1e-10)


REFERENCE_CONSTANT_T_CYLINDER = {
    # tau : Q' / (2 pi k dT) for the constant-temperature cylinder in an
    # infinite medium. Computed with an INDEPENDENT reference solver
    # (Crank-Nicolson on a 6000-node log-radial grid out to r/r_w = 200;
    # scripts/make_thermal_reference.py). Note: simpler references were
    # found inadequate during verification: the leading-order log
    # asymptote is ~8-15% off in this tau range, Ramey/Hasan-Kabir-type
    # T_D correlations target the constant-FLUX boundary condition, and
    # naive quadrature of the exact Carslaw-Jaeger integral mis-handles
    # the singular u->0 region that dominates at large tau.
    10.0: 0.53391,
    50.0: 0.38818,
    200.0: 0.31080,
    1000.0: 0.25095,
}


def test_formation_only_vs_reference_cylinder():
    """Formation-only network vs the constant-wall-temperature cylinder
    (independent finite-difference reference, see table above)."""
    k, rho, cp = 2.1, 2300.0, 1000.0
    alpha = k / (rho * cp)
    r_w = 0.1
    geom = WellGeometry(diameter=2 * r_w)
    geo = Geothermal(T_surface=300.0, gradient=0.0)
    # near-massless, high-conductivity inner shell; fluid h >> any R
    shell = SolidLayer(r_w, r_w * 1.0001, k=1e4, rho=1.0, cp=1.0)
    m = FormationCoupledWall(layers=[shell], r_fluid=r_w,
                             formation=dict(k=k, rho=rho, cp=cp),
                             r_far=60.0, n_formation=220, geothermal=geo)
    m.mu = 1e3          # drives h to a huge value -> T_wall ~ T_fluid
    dT = 50.0
    # graded time steps: resolve the early transient, then stretch
    dts = np.concatenate([np.full(400, 5.0), np.full(400, 200.0),
                          np.full(400, 5.0e3), np.full(600, 2.0e4)])
    z = np.linspace(1.0, geom.depth, 4)
    rho_f = np.full(z.size, 1e7 / (NITROGEN.R * (300.0 + dT)))
    uu = np.full(z.size, 5.0)
    pp = np.full(z.size, 1e7)
    Tf = np.full(z.size, 300.0 + dT)
    t, ts, qs = 0.0, [], []
    for dt in dts:
        q = m.advance(z, t, dt, Tf, rho_f, uu, pp, geom, NITROGEN)
        t += dt
        ts.append(t)
        qs.append(q[0])
    ts, qs = np.array(ts), np.array(qs)
    tau = alpha * ts / r_w**2
    for tau_ref, qd_ref in REFERENCE_CONSTANT_T_CYLINDER.items():
        i = int(np.argmin(np.abs(tau - tau_ref)))
        qd_num = -qs[i] * 2 * np.pi * r_w / (2 * np.pi * k * dT)
        assert qd_num == pytest.approx(qd_ref, rel=0.01), f"tau={tau_ref}"


def test_radial_grid_convergence():
    """q''(t) converges as the formation discretization is refined."""
    r1, layers, formation, _ = lsu_co2_well_stack()
    geom = WellGeometry(diameter=2 * r1)
    geo = Geothermal(T_surface=300.0, gradient=0.0)
    out = {}
    for nf in (8, 16, 32):
        m = FormationCoupledWall(layers=list(layers), r_fluid=r1,
                                 formation=formation, r_far=5.0,
                                 n_formation=nf, geothermal=geo)
        q = _drive(m, geom, T_fluid=250.0, dt=5.0, n_steps=600)
        out[nf] = q[:, 0]
    e8 = np.abs(out[8] - out[32]).max() / np.abs(out[32]).max()
    e16 = np.abs(out[16] - out[32]).max() / np.abs(out[32]).max()
    assert e16 < e8
    assert e16 < 0.02


def test_gap_conductance_physical():
    """Annulus conductance: positive, increases with dT (convection),
    and bounded below by pure conduction + radiation."""
    gap = GasGap(0.0968, 0.1568)
    G1 = gap.conductance_per_length(305.0, 300.0)
    G2 = gap.conductance_per_length(350.0, 300.0)
    assert 0 < G1 < G2
    G_cond_floor = 2 * np.pi * gap.k / np.log(gap.r_out / gap.r_in)
    assert G1 > G_cond_floor  # radiation contributes


def test_conjugate_blowdown_budgets():
    from hyperbolic1d.wellbore import ChokeValve, WellboreModel

    r1, *_ = lsu_co2_well_stack()
    geom = WellGeometry(depth=609.6, diameter=2 * r1)
    m = WellboreModel(geometry=geom, choke=ChokeValve(), gravity=True,
                      n_cells=80, wall_model=FormationCoupledWall())
    r = m.solve(p0=1.379e7, T0=300.0, t_end=5.0, n_snapshots=2)
    assert abs(r.mass_balance_error) < 1e-12
    assert abs(r.energy_balance_error) < 1e-12
