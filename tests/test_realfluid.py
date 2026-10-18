"""Verification of the Span-Wagner CO2 EOS coupling."""

import numpy as np
import pytest

from hyperbolic1d.eos import CO2_IDEAL, NITROGEN
from hyperbolic1d.realfluid import SpanWagnerCO2
from hyperbolic1d.thermal import lsu_co2_well_stack
from hyperbolic1d.wellbore import ChokeValve, WellboreModel, WellGeometry

pytest.importorskip("CoolProp")


@pytest.fixture(scope="module")
def co2():
    return SpanWagnerCO2()


@pytest.fixture(scope="module")
def lsu_geom():
    r1, *_ = lsu_co2_well_stack()
    return WellGeometry(diameter=2 * r1)


def test_table_accuracy_vs_coolprop(co2):
    """Closure (rho, e) -> (p, T, a) vs direct Span-Wagner evaluation."""
    errs = co2.table_verification(n_samples=200)
    for q, (emax, emean) in errs.items():
        assert emax < 1e-3, f"{q}: max rel err {emax}"
        assert emean < 1e-4, f"{q}: mean rel err {emean}"


def test_sound_speed_thermodynamic_consistency(co2):
    """a^2 must equal (dp/drho)_s: finite difference along the CoolProp
    isentrope vs the tabulated speed of sound."""
    import CoolProp.CoolProp as CP

    for (p, T) in [(4e6, 300.0), (1e6, 280.0), (2e6, 310.0)]:
        rho = CP.PropsSI("D", "P", p, "T", T, "CO2")
        s = CP.PropsSI("S", "P", p, "T", T, "CO2")
        dr = 1e-3 * rho
        pp = CP.PropsSI("P", "D", rho + dr, "S", s, "CO2")
        pm = CP.PropsSI("P", "D", rho - dr, "S", s, "CO2")
        a_fd = np.sqrt((pp - pm) / (2 * dr))
        e = CP.PropsSI("U", "P", p, "T", T, "CO2")
        a_tab = co2.primitives(np.array([rho]), np.array([e]))[2][0]
        assert a_tab == pytest.approx(a_fd, rel=2e-3)


def test_real_gas_orifice_exceeds_ideal(co2):
    """At 5 MPa/300 K (Z = 0.69) the real-gas critical flux exceeds the
    ideal-gas value (G ~ sqrt(rho0 p0) scaling), and reduces to it at
    low pressure."""
    G_real, choked = co2.orifice_mass_flux(5e6, 300.0, 101325.0)
    g = 1.289
    G_ideal = 5e6 * np.sqrt(g / (188.92 * 300.0)) * (2 / (g + 1)) ** (
        (g + 1) / (2 * (g - 1)))
    assert choked
    assert 1.05 * G_ideal < G_real < 1.5 * G_ideal
    # near-ideal limit at 0.3 MPa
    G_lo, _ = co2.orifice_mass_flux(3e5, 300.0, 101325.0)
    G_lo_ideal = G_ideal * 3e5 / 5e6
    assert G_lo == pytest.approx(G_lo_ideal, rel=0.04)


def test_low_pressure_blowdown_matches_ideal_gas(lsu_geom):
    """At 0.5 MPa CO2 is nearly ideal (Z ~ 0.98): the real-fluid path
    must agree with the ideal-gas CO2 model within a few percent."""
    kw = dict(geometry=lsu_geom, choke=ChokeValve(diameter=0.0254),
              gravity=True, friction=True, n_cells=60)
    m_real = WellboreModel(eos=SpanWagnerCO2(), **kw)
    r_real = m_real.solve(5e5, 300.0, 30.0, init="hydrostatic",
                          n_snapshots=2)
    m_id = WellboreModel(eos=CO2_IDEAL, **kw)
    r_id = m_id.solve(5e5, 300.0, 30.0, init="hydrostatic", n_snapshots=2)
    md_r = np.interp(r_id.t_hist, r_real.t_hist, r_real.mdot)
    rel = np.abs(md_r - r_id.mdot) / r_id.mdot.max()
    assert rel.max() < 0.05
    assert r_real.two_phase_entry is None


def test_general_path_matches_ideal_path_nitrogen(lsu_geom):
    """Real-fluid N2 through the general code path vs the validated
    ideal-gas path at 5 MPa (N2 nearly ideal there)."""
    eosN = SpanWagnerCO2(fluid="Nitrogen", rho_min=0.2, rho_max=120.0,
                         T_min=200.0, T_max=340.0)
    kw = dict(geometry=lsu_geom, choke=ChokeValve(), gravity=True,
              friction=True, n_cells=60)
    rg = WellboreModel(eos=eosN, **kw).solve(5e6, 300.0, 20.0,
                                             n_snapshots=2)
    ri = WellboreModel(eos=NITROGEN, **kw).solve(5e6, 300.0, 20.0,
                                                 n_snapshots=2)
    md = np.interp(ri.t_hist, rg.t_hist, rg.mdot)
    assert np.abs(md - ri.mdot).max() / ri.mdot.max() < 0.05
    assert abs(rg.mass_balance_error) < 1e-12
    assert abs(rg.energy_balance_error) < 1e-12


def test_two_phase_entry_flagged_low_margin(lsu_geom):
    """Uniform-pressure IC at 6.5 MPa (initial superheat only ~1.9 K):
    the settling expansion near the wellhead crosses the dome within a
    second; the solver must stop and report the first entry rather than
    continue with an invalid single-phase model."""
    m = WellboreModel(geometry=lsu_geom, choke=ChokeValve(),
                      eos=SpanWagnerCO2(), gravity=True, friction=True,
                      n_cells=80)
    r = m.solve(6.5e6, 300.0, 60.0, init="uniform", n_snapshots=2)
    e = r.two_phase_entry
    assert e is not None
    for key in ("t", "z", "depth", "p", "T", "rho", "T_sat"):
        assert key in e
    assert e["T"] < e["T_sat"]
    # entry near the wellhead, early in the transient
    assert e["depth"] < 50.0 and e["t"] < 10.0


def test_co2_budgets_machine_precision(lsu_geom):
    m = WellboreModel(geometry=lsu_geom, choke=ChokeValve(),
                      eos=SpanWagnerCO2(), gravity=True, friction=True,
                      n_cells=60)
    r = m.solve(5e6, 300.0, 10.0, init="hydrostatic", n_snapshots=2)
    assert abs(r.mass_balance_error) < 1e-12
    assert abs(r.energy_balance_error) < 1e-12
    assert r.sat_margin_min is not None and r.sat_margin_min.min() > 0
