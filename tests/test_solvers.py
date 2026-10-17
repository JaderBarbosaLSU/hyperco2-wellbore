"""Coarse-grid regression tests: fast proxies for the full benchmark.

Error bounds are set ~20% above values measured at the time the reference
implementation was frozen, so genuine regressions trip the tests while
platform-level numerical noise does not.
"""

import numpy as np
import pytest

from hyperbolic1d import solve_godunov_shock_tube, solve_scmm_shock_tube
from hyperbolic1d.benchmarks import mean_abs_errors

COARSE = dict(n_nodes=401, dt=1e-5, t_end=3e-3)


@pytest.mark.parametrize(
    "formulation,order,bounds",
    [
        ("conservative", 1, {"P": 5500, "U": 6.0, "rho": 0.05, "T": 4.5}),
        ("conservative", 2, {"P": 2700, "U": 3.2, "rho": 0.025, "T": 2.2}),
        ("nonconservative", 1, {"P": 7600, "U": 7.7, "rho": 0.062, "T": 4.5}),
        ("nonconservative", 2, {"P": 4600, "U": 5.5, "rho": 0.038, "T": 3.1}),
    ],
)
def test_scmm_coarse(formulation, order, bounds):
    r = solve_scmm_shock_tube(formulation=formulation, order=order, **COARSE)
    e = mean_abs_errors(r)
    for k, b in bounds.items():
        assert e[k] < b, f"{formulation}/o{order}: {k}={e[k]} >= {b}"


def test_scmm_hierarchy():
    """Conservative beats non-conservative; 2nd order beats 1st (thesis)."""
    errs = {}
    for f in ("conservative", "nonconservative"):
        for o in (1, 2):
            errs[(f, o)] = mean_abs_errors(
                solve_scmm_shock_tube(formulation=f, order=o, **COARSE))["P"]
    assert errs[("conservative", 2)] < errs[("conservative", 1)]
    assert errs[("nonconservative", 2)] < errs[("nonconservative", 1)]
    assert errs[("conservative", 2)] < errs[("nonconservative", 2)]


@pytest.mark.parametrize(
    "kwargs,bounds",
    [
        (dict(flux="hllc", reconstruction="constant"),
         {"P": 2500, "U": 2.3, "rho": 0.025, "T": 2.1}),
        (dict(flux="hll", reconstruction="constant"),
         {"P": 2900, "U": 2.6, "rho": 0.030, "T": 2.6}),
        (dict(flux="hllc", reconstruction="muscl", limiter="vanleer"),
         {"P": 500, "U": 0.6, "rho": 0.006, "T": 0.6}),
    ],
)
def test_godunov_coarse(kwargs, bounds):
    g = solve_godunov_shock_tube(n_cells=400, **kwargs)
    e = mean_abs_errors(g)
    for k, b in bounds.items():
        assert e[k] < b, f"{kwargs}: {k}={e[k]} >= {b}"


def test_godunov_exact_conservation():
    g = solve_godunov_shock_tube(n_cells=200, flux="hllc")
    assert abs(g.mass_final - g.mass_initial) / g.mass_initial < 1e-13
    assert abs(g.energy_final - g.energy_initial) / g.energy_initial < 1e-13


def test_godunov_positivity():
    g = solve_godunov_shock_tube(n_cells=400, flux="hllc",
                                 reconstruction="muscl")
    assert np.all(g.rho > 0) and np.all(g.p > 0)
