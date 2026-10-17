"""Regression tests for the wellbore blowdown model (fast, coarse grids)."""

import numpy as np
import pytest

from hyperbolic1d.eos import NITROGEN
from hyperbolic1d.wellbore import (ChokeValve, WellboreModel, WellGeometry,
                                   vessel_blowdown_ode, wellhead_face_state)


def test_closed_well_stays_at_rest():
    m = WellboreModel(choke=ChokeValve(diameter=0.0), gravity=False, n_cells=64)
    r = m.solve(p0=1e7, T0=300.0, t_end=1.0, n_snapshots=2)
    assert np.abs(r.u).max() == 0.0
    assert abs(r.mass_balance_error) < 1e-13
    assert abs(r.energy_balance_error) < 1e-13


def test_budgets_machine_precision_with_gravity():
    m = WellboreModel(n_cells=100, gravity=True)
    r = m.solve(p0=1.379e7, T0=300.0, t_end=3.0, n_snapshots=2)
    assert abs(r.mass_balance_error) < 1e-12
    assert abs(r.energy_balance_error) < 1e-12


def test_choke_mass_flow_initial_value():
    """At t=0+ the choked orifice formula applies with near-stagnation
    upstream conditions; the 1-D face coupling must reproduce it."""
    eos = NITROGEN
    geom = WellGeometry()
    choke = ChokeValve()
    p0, T0 = 1.379e7, 300.0
    rho0, _ = eos.rho_e_from_pT(p0, T0)
    G, choked = eos.orifice_mass_flux(p0, T0, choke.p_back)
    mdot_orifice = choke.effective_area * G
    _, _, _, mdot_face, ch = wellhead_face_state(eos, rho0, 0.0, p0, choke,
                                                 geom.area)
    assert choked and ch
    # face coupling expands the flow slightly before the valve -> small
    # deviation from the pure stagnation-fed orifice value
    assert mdot_face == pytest.approx(mdot_orifice, rel=0.02)


def test_check_valve_no_backflow():
    eos = NITROGEN
    rho, _ = eos.rho_e_from_pT(5.0e4, 300.0)  # below back pressure
    _, u_f, _, mdot, _ = wellhead_face_state(eos, rho, 0.0, 5.0e4,
                                             ChokeValve(), 0.00785)
    assert mdot == 0.0 and u_f == 0.0


def test_vessel_ode_agreement_no_gravity():
    """Quasi-static limit: 1-D (no gravity) matches the 0-D vessel ODE."""
    geom = WellGeometry()
    choke = ChokeValve(diameter=0.0254)      # 1 in -> faster, shorter test
    m = WellboreModel(geometry=geom, choke=choke, gravity=False, n_cells=100)
    r = m.solve(p0=1.379e7, T0=300.0, t_end=30.0, n_snapshots=2)
    ref = vessel_blowdown_ode(NITROGEN, geom.volume, choke, 1.379e7, 300.0, 30.0)
    # quasi-static comparand is the mass inventory: the wellhead *pressure*
    # sits a few % below the column mean (local expansion toward the choke),
    # which the uniform 0-D model cannot represent.
    m_num = np.interp(ref["t"], r.t_hist, r.mass)
    rel_mass = np.abs(m_num - ref["mass"]) / ref["mass"]
    assert rel_mass.max() < 0.015
    p_num = np.interp(ref["t"][5:], r.t_hist, r.p_wellhead)
    rel_p = np.abs(p_num - ref["p"][5:]) / ref["p"][5:]
    assert rel_p.max() < 0.06


def test_open_pipe_limit_stable():
    geom = WellGeometry()
    m = WellboreModel(geometry=geom,
                      choke=ChokeValve(diameter=geom.diameter, cd=1.0),
                      gravity=True, n_cells=100)
    r = m.solve(p0=1.379e7, T0=300.0, t_end=2.0, n_snapshots=2)
    assert np.all(np.isfinite(r.mdot))
    assert abs(r.mass_balance_error) < 1e-12
    # open pipe passes far more than the 1/2-in choke
    assert r.mdot[0] > 50.0


def test_mdot_monotonic_in_choke_area():
    md = []
    for d in (0.00635, 0.0127, 0.0254):
        m = WellboreModel(choke=ChokeValve(diameter=d), gravity=True,
                          n_cells=64)
        r = m.solve(p0=1.379e7, T0=300.0, t_end=1.0, n_snapshots=2)
        md.append(r.mdot[-1])
    assert md[0] < md[1] < md[2]


def test_hydrostatic_init_near_equilibrium():
    m = WellboreModel(choke=ChokeValve(diameter=0.0), gravity=True,
                      n_cells=100)
    r = m.solve(p0=1e7, T0=300.0, t_end=5.0, init="hydrostatic",
                n_snapshots=2)
    # not a well-balanced scheme: residual velocities stay small but nonzero
    assert np.abs(r.u).max() < 0.5
