"""Generate the reference solution used by tests/test_thermal.py.

Constant-temperature cylinder (T = dT at r = r_w, undisturbed far field)
in an infinite medium: dimensionless wall heat flow Q'/(2 pi k dT) at
selected tau = alpha t / r_w^2.

Independent discretization: Crank-Nicolson on s = ln(r/r_w) with 6000
nodes out to r/r_w = 200, graded time steps. This replaces three
inadequate references identified during verification: (i) the
leading-order log asymptote (8-15% error in this tau range), (ii)
Ramey/Hasan-Kabir T_D correlations (constant-flux boundary condition,
not constant temperature), and (iii) naive quadrature of the exact
Carslaw-Jaeger integral (the singular u->0 region dominates at large
tau and is easily mis-integrated).
"""

import numpy as np
from scipy.linalg import solve_banded

S = np.log(200.0)
n = 6000
ds = S / n
s = np.linspace(0, S, n + 1)
T = np.zeros(n + 1)
T[0] = 1.0
w = np.exp(-2 * s)

taus_out = [10, 50, 200, 1000]
tau = 0.0
dtau = 1e-4
next_i = 0
while tau < taus_out[-1] + 1e-9:
    lam = w[1:-1] * dtau / (2 * ds * ds)
    ab = np.zeros((3, n - 1))
    ab[0, 1:] = -lam[:-1]
    ab[1, :] = 1 + 2 * lam
    ab[2, :-1] = -lam[1:]
    rhs = T[1:-1] + lam * (T[2:] - 2 * T[1:-1] + T[:-2])
    rhs[0] += lam[0] * 1.0          # Dirichlet T(0)=1 at the new level
    T[1:-1] = solve_banded((1, 1), ab, rhs)
    tau += dtau
    if dtau < 0.5:
        dtau = min(dtau * 1.02, 0.5)
    if next_i < len(taus_out) and tau >= taus_out[next_i] - 1e-9:
        dTds = (-1.5 * T[0] + 2 * T[1] - 0.5 * T[2]) / ds
        print(f"tau={taus_out[next_i]}: Q'/(2 pi k dT) = {-dTds:.5f}")
        next_i += 1
