"""Generate blowdown animations and the x-t wave-propagation diagram.

Two runs of the baseline N2 case (13.79 MPa / 300 K, 12.7 mm choke):

1. early time, 0-8 s, 400 snapshots -> wave propagation (expansion wave
   descending at ~a, bottom reflection, ringing);
2. full blowdown, 0-600 s, 600 snapshots -> bulk depressurization.

Outputs -> results/blowdown_waves.mp4/.gif, results/blowdown_full.mp4,
results/blowdown_xt.png
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from hyperbolic1d.animate import animate_blowdown, xt_diagram
from hyperbolic1d.wellbore import WellboreModel

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
RES.mkdir(exist_ok=True)

P0, T0 = 1.379e7, 300.0

print("[1] early-time run (wave propagation)", flush=True)
# hydrostatic initialization isolates the valve-opening wave: with a
# uniform initial column the gravitational settling transient (~0.45 MPa
# at the bottom) overlays and obscures the expansion wave.
m = WellboreModel(n_cells=600, gravity=True)
r_early = m.solve(P0, T0, t_end=8.0, init="hydrostatic",
                  n_snapshots=400, hist_stride=5)
print(f"    {r_early.n_steps} steps, {r_early.wall_time:.0f}s", flush=True)

p_init = r_early.p[0]          # hydrostatic reference profile
kw = dict(p_ref=p_init, p_label="$p - p_{init}(z)$ [kPa]", p_scale=1e-3,
          title="Expansion wave after valve opening — N$_2$, 12.7 mm choke"
                " (hydrostatic init)",
          time_label="t = {t:.3f} s")
animate_blowdown(r_early, str(RES / "blowdown_waves.mp4"), fps=25, **kw)
animate_blowdown(r_early, str(RES / "blowdown_waves.gif"), fps=16, dpi=70, **kw)
print("    waves animation written", flush=True)

import dataclasses

r_dp = dataclasses.replace(r_early, p=r_early.p - p_init[None, :])
xt_diagram(r_dp, str(RES / "blowdown_xt.png"), field="p", scale=1e-3,
           label="$p - p_{init}(z)$ [kPa]",
           title="x–t diagram: expansion wave and bottom reflections "
                 "(first 8 s, hydrostatic init)")
print("    x-t diagram written", flush=True)

print("[2] full-blowdown run", flush=True)
m = WellboreModel(n_cells=400, gravity=True)
r_full = m.solve(P0, T0, t_end=600.0, n_snapshots=600, hist_stride=50)
print(f"    {r_full.n_steps} steps, {r_full.wall_time:.0f}s", flush=True)

animate_blowdown(
    r_full, str(RES / "blowdown_full.mp4"), fps=30,
    title="Full blowdown — N$_2$, 2,000-ft well, 12.7 mm choke, adiabatic",
    time_label="t = {t:.0f} s")
print("done", flush=True)
