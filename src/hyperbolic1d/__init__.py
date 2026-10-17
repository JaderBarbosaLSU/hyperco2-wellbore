"""hyperbolic1d: transient compressible-flow solvers for hyperbolic PDEs.

Phase 1 - reconstruction of the SCMM solvers of Gessner (2010).
Phase 2 - Godunov-type finite-volume solvers for comparison.
"""

from .exact_riemann import ExactRiemannSolution, PrimitiveState, solve_riemann
from .godunov import GodunovResult, solve_godunov_shock_tube
from .scmm import PerfectGas, SCMMResult, solve_scmm_shock_tube

__version__ = "0.1.0"

__all__ = [
    "ExactRiemannSolution",
    "PrimitiveState",
    "solve_riemann",
    "PerfectGas",
    "SCMMResult",
    "solve_scmm_shock_tube",
    "GodunovResult",
    "solve_godunov_shock_tube",
]
