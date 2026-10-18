"""Supplementary computations for the technical report.

1. Closed-wellhead stationarity: a hydrostatically initialized column
   with the valve closed must remain numerically stationary. Quantifies
   residual velocities / pressure drift for N2 (ideal) and CO2 (real
   EOS), uniform-T and geothermal initializations, two grids.
2. Discharge-coefficient sensitivity: blowdown mass-flow histories for
   Cd in {0.61, 0.72, 0.84, 1.0}; since demand = Cd*A_t*G(p0,T0), the
   quasi-static discharge should scale in time as 1/Cd (verified).
3. Publication velocity figures (early acoustic + late quasi-static)
   for N2 and CO2 from the stage-1/2 wave + long checkpoints.

Outputs -> results/paper_support.json, paper/figures/velocity_{n2,co2}.png
"""

import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

from hyperbolic1d.eos import NITROGEN
from hyperbolic1d.postprocess import plot_velocity_publication
from hyperbolic1d.realfluid import SpanWagnerCO2
from hyperbolic1d.thermal import Geothermal, lsu_co2_well_stack
from hyperbolic1d.wellbore import ChokeValve, WellboreModel, WellGeometry

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

r1, *_ = lsu_co2_well_stack()
GEOM = WellGeometry(depth=609.6, diameter=2 * r1)
GEO = Geothermal(T_surface=305.4, gradient=0.0255)
CLOSED = ChokeValve(p_back=1.0e9)   # back pressure above any p0: valve shut
out = {}

# ---------------------------------------------------------------- 1. static
print("[static] closed-well stationarity", flush=True)
out["stationarity"] = {}
for label, eos, geo in (("n2_uniform", NITROGEN, None),
                        ("n2_geothermal", NITROGEN, GEO),
                        ("co2_uniform", SpanWagnerCO2(), None),
                        ("co2_geothermal", SpanWagnerCO2(), GEO)):
    for n in (100, 300):
        m = WellboreModel(geometry=GEOM, choke=CLOSED, eos=eos,
                          gravity=True, friction=True, n_cells=n)
        r = m.solve(5e6, 300.0, 600.0, init="hydrostatic", n_snapshots=3,
                    hist_stride=200, geothermal=geo)
        umax = float(np.max(np.abs(r.u[-1])))
        dp = float(np.max(np.abs(r.p[-1] - r.p[0])))
        out["stationarity"][f"{label}_n{n}"] = {
            "max_abs_u_m_per_s": umax,
            "max_abs_dp_Pa": dp,
            "rel_dp": dp / 5e6,
            "mass_balance_error": r.mass_balance_error,
        }
        print(f"    {label} n={n}: max|u|={umax:.2e} m/s, "
              f"max|dp|={dp:.1f} Pa", flush=True)

# ------------------------------------------------------------------- 2. Cd
print("[cd] discharge-coefficient sensitivity (N2, ideal, 600 s)",
      flush=True)
out["cd_sensitivity"] = {}
t_half = {}
for cd in (0.61, 0.72, 0.84, 1.0):
    m = WellboreModel(geometry=GEOM, choke=ChokeValve(cd=cd), eos=NITROGEN,
                      gravity=True, friction=True, n_cells=120)
    r = m.solve(5e6, 300.0, 1200.0, init="hydrostatic", n_snapshots=3,
                hist_stride=20)
    # time at which inventory falls to half
    th = float(np.interp(0.5, r.mass[::-1] / r.mass[0], r.t_hist[::-1]))
    t_half[cd] = th
    out["cd_sensitivity"][str(cd)] = {
        "mdot_initial_kg_s": float(r.mdot[1]),
        "t_half_mass_s": th,
    }
    print(f"    Cd={cd}: mdot0={r.mdot[1]:.3f} kg/s, t_half={th:.1f} s",
          flush=True)
# 1/Cd scaling check against Cd=0.84
sc = {str(cd): t_half[cd] * cd / (t_half[0.84] * 0.84) for cd in t_half}
out["cd_sensitivity"]["t_half_times_cd_normalized"] = sc
print("    Cd*t_half (normalized to 0.84):", sc, flush=True)

# ------------------------------------------------------------- 3. velocity
print("[fig] velocity figures", flush=True)
with open(RES / "co2" / "ckpt_wave_n2.pkl", "rb") as f:
    wn = pickle.load(f)
with open(RES / "co2" / "ckpt_n2_conjugate.pkl", "rb") as f:
    rn = pickle.load(f)
with open(RES / "co2_stage2" / "ckpt_wave_co2_geo.pkl", "rb") as f:
    wc = pickle.load(f)
with open(RES / "co2_stage2" / "ckpt_co2_conjugate_geo.pkl", "rb") as f:
    rc = pickle.load(f)
plot_velocity_publication(wn, rn, FIG / "velocity_n2.png",
                          early_times=(0.5, 1.2, 2.2, 3.0),
                          late_times=(120, 600, 1800, 3000))
plot_velocity_publication(wc, rc, FIG / "velocity_co2.png",
                          early_times=(0.8, 1.8, 3.2, 4.5),
                          late_times=(120, 600, 1800, 3000))

with open(RES / "paper_support.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print("done", flush=True)
