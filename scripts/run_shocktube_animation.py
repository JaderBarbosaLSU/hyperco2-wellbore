"""Animate the Gessner single-phase shock tube: exact Riemann solution vs
the FV MUSCL-Hancock/HLLC solution, frame-synchronized over 0-3 ms.

The exact solution is self-similar (a function of xi = (x - x0)/t), so it
is sampled analytically at every frame; the FV run stores snapshots at
the same instants. Outputs results/shocktube_exact_vs_fv.mp4/.gif.

Longer runs: --t-end-ms T extends the comparison window. The exact
Riemann solution holds only while no wave has reached a wall, so the
domain is scaled proportionally (L and x_diaphragm multiplied by
T / 3 ms) at FIXED cell size dx = 5 mm; the waves then never reach the
walls and the FV solution stays comparable to the exact one over the
whole window, while the structures spread over proportionally more
cells. Output filenames gain a _{T}ms suffix for non-default T.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from hyperbolic1d.benchmarks import GESSNER_SINGLE_PHASE, exact_reference
from hyperbolic1d.godunov import solve_godunov_shock_tube
from hyperbolic1d.scmm import PerfectGas

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
RES.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--t-end-ms", type=float, default=None,
                    help="comparison window in ms (default: benchmark 3 ms)")
args = parser.parse_args()

gas = PerfectGas()                 # air, matching the benchmark
case = dict(GESSNER_SINGLE_PHASE)
suffix = ""
if args.t_end_ms is not None:
    scale = args.t_end_ms * 1e-3 / case["t_end"]
    case["t_end"] = args.t_end_ms * 1e-3
    case["L"] *= scale                    # keep walls out of reach
    case["x_diaphragm"] *= scale
    suffix = f"_{args.t_end_ms:g}ms"
N_FRAMES = 180
T_END = case["t_end"]
DX = 4.0 / 800                            # benchmark cell size, 5 mm
n_cells = int(round(case["L"] / DX))

print(f"[1] FV run with snapshots (L={case['L']:.2f} m, "
      f"{n_cells} cells, t_end={T_END*1e3:g} ms)", flush=True)
fv = solve_godunov_shock_tube(
    gas=gas, n_cells=n_cells, flux="hllc", reconstruction="muscl",
    limiter="vanleer", t_end=T_END, n_snapshots=N_FRAMES,
    L=case["L"], x_diaphragm=case["x_diaphragm"],
    p_left=case["p_left"], T_left=case["T_left"],
    p_right=case["p_right"], T_right=case["T_right"],
)
print(f"    {fv.n_steps} steps, {fv.wall_time:.1f}s; "
      f"{len(fv.snap_times)} snapshots", flush=True)

x_exact = np.linspace(0.0, case["L"], 2001)

print("[2] rendering animation", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
specs = [("p", "Pressure [kPa]", 1e-3), ("u", "Velocity [m/s]", 1.0),
         ("rho", "Density [kg/m$^3$]", 1.0), ("T", "Temperature [K]", 1.0)]

lines_ex, lines_fv = {}, {}
ref0 = exact_reference(x_exact, T_END, gas, case)   # final time -> axis limits
for ax, (key, lab, sc) in zip(axes.flat, specs):
    (le,) = ax.plot([], [], "k-", lw=1.8, label="Exact Riemann solution")
    (lf,) = ax.plot([], [], color="tab:red", lw=1.0, alpha=0.9,
                    label=f"FV MUSCL-Hancock + HLLC ({n_cells} cells)")
    ax.set_xlim(0, case["L"])
    full = np.concatenate([ref0[key], getattr(fv, f"snap_{key}")[0]]) * sc
    pad = 0.06 * (full.max() - full.min())
    ax.set_ylim(full.min() - pad, full.max() + pad)
    ax.set_xlabel("x [m]")
    ax.set_ylabel(lab)
    ax.grid(alpha=0.3)
    lines_ex[key] = (le, sc)
    lines_fv[key] = (lf, sc)
axes[0, 0].legend(fontsize=8, loc="lower left")
ttl = axes[0, 0].set_title("t = 0.000 ms", loc="left", fontsize=10)
fig.suptitle("Single-phase shock tube (air, 500 kPa/400 K | 100 kPa/300 K)")
fig.tight_layout()


def update(i):
    t = fv.snap_times[i]
    ref = exact_reference(x_exact, t, gas, case)
    for key, (ln, sc) in lines_ex.items():
        ln.set_data(x_exact, ref[key] * sc)
    for key, (ln, sc) in lines_fv.items():
        ln.set_data(fv.x, getattr(fv, f"snap_{key}")[i] * sc)
    ttl.set_text(f"t = {t*1e3:.3f} ms")
    return []


anim = FuncAnimation(fig, update, frames=len(fv.snap_times), blit=False)
anim.save(RES / f"shocktube_exact_vs_fv{suffix}.mp4", fps=30, dpi=110,
          writer="ffmpeg")
anim.save(RES / f"shocktube_exact_vs_fv{suffix}.gif", fps=15, dpi=70,
          writer="pillow")
plt.close(fig)
print("done", flush=True)
