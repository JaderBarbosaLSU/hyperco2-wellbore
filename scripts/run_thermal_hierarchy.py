"""Thermal-hierarchy comparison on the LSU CO2 research well geometry.

N2 blowdown, 2,000 ft, 7-5/8" (35.8#) inner casing bore (ID 170.3 mm),
12.7 mm choke, 13.79 MPa / 300 K initial state, gravity on:

1. adiabatic wall (baseline),
2. prescribed wall temperature (local geothermal), Dittus-Boelter h,
3. coupled completion/formation radial model (air annulus with Willhite
   natural convection + radiation, cemented casings, formation).

Outputs -> results/thermal_hierarchy.png, results/thermal_radial.png,
results/thermal_hierarchy.json
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperbolic1d.thermal import (GEOTHERMAL_DEFAULT, FormationCoupledWall,
                                  lsu_co2_well_stack)
from hyperbolic1d.wellbore import (AdiabaticWall, ChokeValve,
                                   PrescribedWallTemperature, WellboreModel,
                                   WellGeometry)

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
RES.mkdir(exist_ok=True)

r1, layers, formation, r_far = lsu_co2_well_stack()
GEOM = WellGeometry(depth=609.6, diameter=2 * r1)
CHOKE = ChokeValve()
P0, T0, T_END = 1.379e7, 300.0, 1800.0
N = 300

# local geothermal profile for the prescribed-T case
zc = (np.arange(N) + 0.5) * (GEOM.depth / N)
T_geo = GEOTHERMAL_DEFAULT.T(GEOM.depth - zc)

cases = {
    "adiabatic": AdiabaticWall(),
    "prescribed T_w (geothermal)": PrescribedWallTemperature(T_wall=T_geo),
    "coupled formation model": FormationCoupledWall(),
}

report = {}
results = {}
for name, wall in cases.items():
    print(f"[run] {name}", flush=True)
    m = WellboreModel(geometry=GEOM, choke=CHOKE, gravity=True,
                      n_cells=N, wall_model=wall)
    r = m.solve(P0, T0, T_END, n_snapshots=7)
    results[name] = r
    report[name] = {
        "T_wellhead_min_K": float(r.T_wellhead.min()),
        "T_wellhead_end_K": float(r.T_wellhead[-1]),
        "mass_end_kg": float(r.mass[-1]),
        "mass_balance_error": r.mass_balance_error,
        "energy_balance_error": r.energy_balance_error,
        "wall_time_s": round(r.wall_time, 1),
    }
    print(f"    T_wh min={r.T_wellhead.min():.1f} K, end={r.T_wellhead[-1]:.1f} K;"
          f" budgets m={r.mass_balance_error:.1e} E={r.energy_balance_error:.1e};"
          f" {r.wall_time:.0f}s", flush=True)

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
colors = {"adiabatic": "tab:red",
          "prescribed T_w (geothermal)": "tab:green",
          "coupled formation model": "tab:blue"}
for name, r in results.items():
    c = colors[name]
    axes[0].plot(r.t_hist, r.T_wellhead, color=c, label=name)
    axes[1].plot(r.t_hist, r.mdot, color=c)
    axes[2].plot(r.t_hist, r.mass, color=c)
axes[0].set_ylabel("Wellhead fluid temperature [K]")
axes[1].set_ylabel("Wellhead mass flow rate [kg/s]")
axes[2].set_ylabel("Mass in well [kg]")
for ax in axes:
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("Thermal-model hierarchy — N$_2$ blowdown, LSU well geometry "
             "(ID 170.3 mm, 12.7 mm choke)")
fig.tight_layout()
fig.savefig(RES / "thermal_hierarchy.png", dpi=150)

# radial temperature field (coupled model) at three depths, end of run
wall = cases["coupled formation model"]
Tr = wall.radial_temperatures
rr = wall.radial_positions
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for frac, lab in ((0.95, "near wellhead"), (0.5, "mid depth"),
                  (0.05, "near bottom")):
    j = int(frac * (N - 1))
    ax.semilogx(rr, Tr[j], ".-", ms=3,
                label=f"{lab} (z = {zc[j]:.0f} m)")
for x, lab in ((r1, "bore"), (0.0968, "7-5/8 OD"), (0.1568, "13-3/8 ID"),
               (0.3302, "borehole")):
    ax.axvline(x, color="gray", lw=0.6, ls=":")
ax.set_xlabel("r [m]")
ax.set_ylabel("Temperature [K]")
ax.set_title(f"Radial temperature through completion and formation at "
             f"t = {T_END:.0f} s")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RES / "thermal_radial.png", dpi=150)

with open(RES / "thermal_hierarchy.json", "w") as f:
    json.dump(report, f, indent=2)
print("done", flush=True)
