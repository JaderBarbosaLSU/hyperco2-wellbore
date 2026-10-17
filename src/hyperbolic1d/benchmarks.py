"""Benchmark definitions and error metrics.

The single-phase case reproduces Gessner (2010), Sec. 3.4 (Figure 3.1):
air, closed 4-m tube, diaphragm at midspan,
left chamber 500 kPa / 400 K, right chamber 100 kPa / 300 K, fluid at rest.
Reference: exact Riemann solution at t = 3 ms.

Thesis Table 3.3 mean-absolute-error targets (4001 nodes, dt = 1 us):

===========================  ========  =========  ============  ========
Formulation                  P [Pa]    U [m/s]    rho [kg/m3]   T [K]
===========================  ========  =========  ============  ========
Non-conservative, 1st order  3339      4.65       0.0253        2.14
Non-conservative, 2nd order  2713      3.87       0.0201        1.67
Conservative, 1st order      1555      2.02       0.0161        1.74
Conservative, 2nd order      311.6     0.331      0.00354       0.310
===========================  ========  =========  ============  ========
"""

from __future__ import annotations

import numpy as np

from .exact_riemann import PrimitiveState, solve_riemann
from .scmm import PerfectGas

GESSNER_SINGLE_PHASE = dict(
    L=4.0,
    x_diaphragm=2.0,
    p_left=5.0e5,
    T_left=400.0,
    p_right=1.0e5,
    T_right=300.0,
    t_end=3.0e-3,
)

THESIS_TABLE_3_3 = {
    ("nonconservative", 1): {"P": 3339.0, "U": 4.65, "rho": 0.0253, "T": 2.14, "cpu_s": 90},
    ("nonconservative", 2): {"P": 2713.0, "U": 3.87, "rho": 0.0201, "T": 1.67, "cpu_s": 203},
    ("conservative", 1): {"P": 1555.0, "U": 2.02, "rho": 0.0161, "T": 1.74, "cpu_s": 178},
    ("conservative", 2): {"P": 311.6, "U": 0.331, "rho": 0.00354, "T": 0.310, "cpu_s": 336},
}


def exact_reference(x: np.ndarray, t: float, gas: PerfectGas = PerfectGas(),
                    case: dict = GESSNER_SINGLE_PHASE) -> dict[str, np.ndarray]:
    """Exact solution of the benchmark sampled on ``x`` at time ``t``."""
    left = PrimitiveState.from_pT(case["p_left"], case["T_left"], 0.0, gas.R)
    right = PrimitiveState.from_pT(case["p_right"], case["T_right"], 0.0, gas.R)
    sol = solve_riemann(left, right, gamma=gas.gamma)
    out = sol.sample_xt(x, t, x0=case["x_diaphragm"])
    out["T"] = out["p"] / (gas.R * out["rho"])
    return out


def mean_abs_errors(result, gas: PerfectGas = PerfectGas(),
                    case: dict = GESSNER_SINGLE_PHASE) -> dict[str, float]:
    """Mean absolute error vs the exact solution, per thesis Table 3.3."""
    ref = exact_reference(result.x, result.t, gas, case)
    return {
        "P": float(np.mean(np.abs(result.p - ref["p"]))),
        "U": float(np.mean(np.abs(result.u - ref["u"]))),
        "rho": float(np.mean(np.abs(result.rho - ref["rho"]))),
        "T": float(np.mean(np.abs(result.T - ref["T"]))),
    }
