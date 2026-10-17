import numpy as np
import pytest

from hyperbolic1d.exact_riemann import PrimitiveState, solve_riemann


def test_sod_star_state():
    """Toro (2009), Table 4.2, Test 1 (Sod)."""
    sol = solve_riemann(PrimitiveState(1.0, 0.0, 1.0),
                        PrimitiveState(0.125, 0.0, 0.1), gamma=1.4)
    assert sol.p_star == pytest.approx(0.30313, abs=1e-5)
    assert sol.u_star == pytest.approx(0.92745, abs=1e-5)


def test_toro_test2_123_problem():
    """Toro Test 2: two rarefactions, near-vacuum star region."""
    sol = solve_riemann(PrimitiveState(1.0, -2.0, 0.4),
                        PrimitiveState(1.0, 2.0, 0.4), gamma=1.4)
    assert sol.p_star == pytest.approx(0.00189, abs=2e-5)
    assert abs(sol.u_star) < 1e-10


def test_gessner_case_wave_structure():
    """Benchmark IC: left rarefaction, right shock, positive u*."""
    R = 287.0
    sol = solve_riemann(PrimitiveState.from_pT(5e5, 400.0, 0.0, R),
                        PrimitiveState.from_pT(1e5, 300.0, 0.0, R), gamma=1.4)
    assert sol.left.p > sol.p_star > sol.right.p
    assert sol.u_star > 0.0
    # sampled solution is continuous except at the three waves and bounded
    xi = np.linspace(-800, 800, 5001)
    out = sol.sample(xi)
    assert np.all(out["rho"] > 0) and np.all(out["p"] > 0)
    # far field recovers the initial states
    assert out["p"][0] == pytest.approx(5e5)
    assert out["p"][-1] == pytest.approx(1e5)
