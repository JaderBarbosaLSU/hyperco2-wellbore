"""Run the full-resolution Gessner (2010) single-phase shock-tube benchmark.

Reproduces thesis Table 3.3 (SCMM, both formulations and orders) and adds
the Phase-2 Godunov-type finite-volume solvers on the identical problem.
Saves fields to results/single_phase_fields.npz and the error table to
results/single_phase_errors.json.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from hyperbolic1d import solve_godunov_shock_tube, solve_scmm_shock_tube
from hyperbolic1d.benchmarks import THESIS_TABLE_3_3, mean_abs_errors

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

fields = {}
errors = {}

# --- SCMM (Phase 1) ---------------------------------------------------------
for form in ("nonconservative", "conservative"):
    for order in (1, 2):
        key = f"scmm_{form}_o{order}"
        print(f"[run] {key}", flush=True)
        r = solve_scmm_shock_tube(formulation=form, order=order)
        e = mean_abs_errors(r)
        e["wall_time_s"] = round(r.wall_time, 1)
        e["thesis"] = THESIS_TABLE_3_3[(form, order)]
        errors[key] = e
        fields[f"{key}_x"] = r.x
        for name in ("rho", "u", "p", "T"):
            fields[f"{key}_{name}"] = getattr(r, name)
        print(f"      P={e['P']:.1f} Pa  U={e['U']:.3f}  rho={e['rho']:.5f}  "
              f"T={e['T']:.3f}  ({r.wall_time:.0f}s)", flush=True)

# --- Godunov FV (Phase 2) ---------------------------------------------------
for scheme, kwargs in {
    "godunov_o1_hllc": dict(flux="hllc", reconstruction="constant"),
    "godunov_o1_exact": dict(flux="exact", reconstruction="constant"),
    "godunov_muscl_hllc": dict(flux="hllc", reconstruction="muscl", limiter="vanleer"),
    "godunov_muscl_minmod": dict(flux="hllc", reconstruction="muscl", limiter="minmod"),
}.items():
    print(f"[run] {scheme}", flush=True)
    g = solve_godunov_shock_tube(n_cells=4000, **kwargs)
    e = mean_abs_errors(g)
    e["wall_time_s"] = round(g.wall_time, 2)
    e["n_steps"] = g.n_steps
    e["mass_defect_rel"] = (g.mass_final - g.mass_initial) / g.mass_initial
    e["energy_defect_rel"] = (g.energy_final - g.energy_initial) / g.energy_initial
    errors[scheme] = e
    fields[f"{scheme}_x"] = g.x
    for name in ("rho", "u", "p", "T"):
        fields[f"{scheme}_{name}"] = getattr(g, name)
    print(f"      P={e['P']:.1f} Pa  U={e['U']:.3f}  rho={e['rho']:.5f}  "
          f"T={e['T']:.3f}  ({g.wall_time:.1f}s)", flush=True)

np.savez_compressed(RESULTS / "single_phase_fields.npz", **fields)
with open(RESULTS / "single_phase_errors.json", "w") as f:
    json.dump(errors, f, indent=2)
print("done")
