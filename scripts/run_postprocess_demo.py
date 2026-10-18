"""Post-processing demonstration: fiber sensors, publication figures,
CSV export, and temperature animations on the LSU well conjugate case.

Main run: N2, LSU geometry (ID 170.3 mm), 12.7 mm choke, gravity +
friction + coupled completion/formation thermal model, 1800 s, with the
default fiber-optic sensor array (bore DTS points + external fiber on
the 13-3/8" casing at 50/500/1000/1500/1950 ft).

Early run: hydrostatic init, 8 s, for the acoustic-phase velocity panel.

Outputs:
  results/export/*.csv
  results/fig_sensors_T.png     T(t) at sensors (publication)
  results/fig_profiles.png      depth-oriented T,p profiles (publication)
  results/fig_velocity.png      early/late velocity panels (publication)
  results/fig_wellhead.png      wellhead histories (publication)
  results/temperature_sensors.mp4 / .gif
  results/blowdown_T_fields.mp4 (4-field animation with sensor markers)
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from hyperbolic1d.animate import animate_blowdown, animate_temperature
from hyperbolic1d.postprocess import (FT, export_csv, lsu_fiber_sensors,
                                      plot_profiles_at_times,
                                      plot_sensor_histories,
                                      plot_velocity_publication,
                                      plot_wellhead_histories)
from hyperbolic1d.thermal import FormationCoupledWall, lsu_co2_well_stack
from hyperbolic1d.wellbore import ChokeValve, WellboreModel, WellGeometry

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
RES.mkdir(exist_ok=True)

r1, *_ = lsu_co2_well_stack()
GEOM = WellGeometry(depth=609.6, diameter=2 * r1)
CHOKE = ChokeValve()
P0, T0 = 1.379e7, 300.0
DEPTHS_FT = (50, 500, 1000, 1500, 1950)

print("[1] main conjugate run (friction on, 1800 s, sensors)", flush=True)
sensors = lsu_fiber_sensors(DEPTHS_FT)
m = WellboreModel(geometry=GEOM, choke=CHOKE, gravity=True, friction=True,
                  n_cells=300, wall_model=FormationCoupledWall())
r = m.solve(P0, T0, 1800.0, n_snapshots=301, hist_stride=50,
            sensors=sensors)
print(f"    {r.n_steps} steps, {r.wall_time:.0f}s; budgets "
      f"m={r.mass_balance_error:.1e} E={r.energy_balance_error:.1e}",
      flush=True)

print("[2] early-time run (acoustic phase)", flush=True)
m2 = WellboreModel(geometry=GEOM, choke=CHOKE, gravity=True, friction=True,
                   n_cells=600)
r_early = m2.solve(P0, T0, 8.0, init="hydrostatic", n_snapshots=160,
                   hist_stride=5)

print("[3] CSV export", flush=True)
out = export_csv(r, RES / "export",
                 profile_times=(0, 60, 300, 900, 1800))
print(f"    -> {out}", flush=True)

print("[4] publication figures", flush=True)
plot_sensor_histories(r, RES / "fig_sensors_T.png", t_unit="min")
plot_profiles_at_times(r, times=(0, 60, 300, 900, 1800),
                       fields=("T", "p"),
                       filename=RES / "fig_profiles.png",
                       sensor_depths_ft=DEPTHS_FT)
plot_velocity_publication(r_early, r, RES / "fig_velocity.png",
                          early_times=(0.5, 1.2, 2.2, 3.0),
                          late_times=(60, 300, 900, 1800))
plot_wellhead_histories(r, RES / "fig_wellhead.png")
print("    4 figures written", flush=True)

print("[5] animations", flush=True)
animate_temperature(
    r, str(RES / "temperature_sensors.mp4"), fps=25,
    title="Fluid temperature field and fiber-optic sensor response — "
          "N$_2$ blowdown, LSU well, coupled thermal model")
animate_temperature(
    r, str(RES / "temperature_sensors.gif"), fps=14, dpi=70,
    title="Fluid temperature and fiber-optic sensors")
animate_blowdown(
    r, str(RES / "blowdown_T_fields.mp4"), fps=25,
    title="N$_2$ blowdown, LSU well — conjugate thermal model "
          "(dashed: fiber sensor depths)",
    time_label="t = {t:.0f} s",
    sensor_depths=[d * FT for d in DEPTHS_FT])
print("done", flush=True)
