"""CO2 geothermal-case animations in report conventions: depth zeta
(ft, increasing downward) and US-customary primary units with SI
secondary axes, matching scripts/make_report_figures_us.py.

Three outputs (results/co2_stage2/):

1. temperature_sensors_co2_geo_us.mp4 -- fiber-optic sensor animation,
   co2_conjugate_geo extended to 4200 s (new checkpointed solve,
   n = 300, snapshot every 10 s);
2. blowdown_fields_co2_geo_us.mp4 -- p, T, rho, u profiles vs zeta for
   the same 4200 s run, with the wellhead-mass-flow cursor inset;
3. waves_co2_geo_us.mp4 -- early-time (0-20 s) wave animation showing
   the expansion wave in p, u, T descending and reflecting off the
   bottom ~4 times (round-trip time 2L/a ~ 5 s for CO2 at 5 MPa);
   fields are shown as deviations from the initial hydrostatic +
   geothermal profiles so the traveling wave is visible.

Solves are checkpointed; re-running skips completed cases.
"""

import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from hyperbolic1d.postprocess import FT, lsu_fiber_sensors
from hyperbolic1d.realfluid import SpanWagnerCO2
from hyperbolic1d.thermal import (FormationCoupledWall, Geothermal,
                                  lsu_co2_well_stack)
from hyperbolic1d.wellbore import ChokeValve, WellboreModel, WellGeometry

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
S2 = RES / "co2_stage2"
S2.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- case setup
r1, *_ = lsu_co2_well_stack()
GEOM = WellGeometry(depth=609.6, diameter=2 * r1)
CHOKE = ChokeValve()
P0, T0 = 5.0e6, 300.0
GEO_SUMMER = Geothermal(T_surface=305.4, gradient=0.0255)
DEPTHS_FT = (50, 500, 1000, 1500, 1950)

# ------------------------------------------------------------------ units
PSI = 6894.757293
LBM = 0.45359237
DEN = LBM / FT**3


def K2F(K):
    return (np.asarray(K) - 273.15) * 1.8 + 32.0


def F2K(F):
    return (np.asarray(F) - 32.0) / 1.8 + 273.15


SI = {
    "p_MPa": (lambda x: x * PSI / 1e6, lambda x: x * 1e6 / PSI),
    "p_kPa": (lambda x: x * PSI / 1e3, lambda x: x * 1e3 / PSI),
    "T_K": (F2K, K2F),
    "dT_K": (lambda x: np.asarray(x) / 1.8, lambda x: np.asarray(x) * 1.8),
    "u_mps": (lambda x: x * FT, lambda x: x / FT),
    "md_kgps": (lambda x: x * LBM, lambda x: x / LBM),
    "L_m": (lambda x: x * FT, lambda x: x / FT),
    "rho_kgm3": (lambda x: x * DEN, lambda x: x / DEN),
}


def sx(ax, kind, label):
    """SI scale on the upper abscissa."""
    ax.tick_params(top=False, which="both")
    s = ax.secondary_xaxis("top", functions=SI[kind])
    s.set_xlabel(label, fontsize=8)
    s.tick_params(labelsize=7)
    return s


def sy(ax, kind, label):
    """SI scale on the right ordinate."""
    ax.tick_params(right=False, which="both")
    s = ax.secondary_yaxis("right", functions=SI[kind])
    s.set_ylabel(label, fontsize=8)
    s.tick_params(labelsize=7)
    return s


def zeta_ft(r):
    """Depth coordinate zeta = L - z [ft], increasing downward."""
    return (GEOM.depth - r.z) / FT


# ------------------------------------------------------------------ solves
def solve_ckpt(name, **kw):
    ckpt = S2 / f"ckpt_{name}.pkl"
    if ckpt.exists():
        print(f"[run] {name}: checkpoint found, skipping", flush=True)
        with open(ckpt, "rb") as f:
            return pickle.load(f)
    print(f"[run] {name}", flush=True)
    m = WellboreModel(geometry=GEOM, choke=CHOKE, eos=SpanWagnerCO2(),
                      gravity=True, friction=True,
                      n_cells=kw.pop("n_cells"),
                      wall_model=kw.pop("wall_model", None))
    r = m.solve(P0, T0, kw.pop("t_end"), init="hydrostatic",
                geothermal=GEO_SUMMER, **kw)
    print(f"    {r.n_steps} steps, {r.wall_time:.0f}s wall; budgets "
          f"m={r.mass_balance_error:.0e} E={r.energy_balance_error:.0e}; "
          f"{'TWO-PHASE ENTRY' if r.two_phase_entry else 'single-phase'}",
          flush=True)
    with open(ckpt, "wb") as f:
        pickle.dump(r, f)
    return r


print("[1] solves", flush=True)
cg = solve_ckpt(
    "co2_conjugate_geo_4200", n_cells=300, t_end=4200.0,
    wall_model=FormationCoupledWall(geothermal=GEO_SUMMER),
    n_snapshots=421, hist_stride=50, sensors=lsu_fiber_sensors(DEPTHS_FT))
wv = solve_ckpt(
    "wave_co2_geo_20s", n_cells=600, t_end=20.0,
    wall_model=FormationCoupledWall(geothermal=GEO_SUMMER),
    n_snapshots=500, hist_stride=5)


# -------------------------------------------- [2] temperature sensors anim
print("[2] temperature-sensor animation", flush=True)
meta = cg.sensor_meta
names = list(cg.sensor_data)
fluid_names = sorted((n for n in names if meta[n]["location"] == "fluid"),
                     key=lambda n: meta[n]["depth"])
solid_names = sorted((n for n in names if meta[n]["location"] != "fluid"),
                     key=lambda n: meta[n]["depth"])
zf = zeta_ft(cg)
TF = K2F(cg.T)
ts_min = cg.sensor_t / 60.0

fig, (axp, axt) = plt.subplots(1, 2, figsize=(12, 5.4),
                               gridspec_kw={"width_ratios": [0.85, 1.2]})
(lnT,) = axp.plot(TF[0], zf, "k-", lw=1.4, label=r"fluid $T(\zeta)$")
fl_pts, = axp.plot([], [], "o", color="tab:red", ms=5,
                   label="fluid fiber samples")
so_pts, = axp.plot([], [], "s", color="tab:blue", ms=5,
                   label='13-3/8" external fiber')
allF = np.concatenate([TF.ravel()] +
                      [K2F(v) for v in cg.sensor_data.values()])
pad = 0.05 * (allF.max() - allF.min() + 1e-9)
axp.set_xlim(allF.min() - pad, allF.max() + pad)
axp.set_ylim(GEOM.depth / FT, 0.0)          # zeta increases downward
axp.set_xlabel(r"Temperature [$^\circ$F]")
axp.set_ylabel(r"Depth $\zeta$ [ft]")
axp.grid(alpha=0.3)
axp.legend(fontsize=8, loc="lower left")
sx(axp, "T_K", "[K]")
sy(axp, "L_m", "[m]")

cmap_f, cmap_s = plt.cm.autumn, plt.cm.winter
for i, n in enumerate(fluid_names):
    axt.plot(ts_min, K2F(cg.sensor_data[n]),
             color=cmap_f(i / max(len(fluid_names) - 1, 1)), lw=1.0,
             label=f"fluid {meta[n]['depth']/FT:.0f} ft")
for i, n in enumerate(solid_names):
    axt.plot(ts_min, K2F(cg.sensor_data[n]),
             color=cmap_s(i / max(len(solid_names) - 1, 1)), lw=1.0,
             ls="--", label=f"fiber {meta[n]['depth']/FT:.0f} ft")
cursor = axt.axvline(0.0, color="k", lw=0.8)
axt.set_xlabel("$t$ [min]")
axt.set_ylabel(r"Sensor temperature [$^\circ$F]")
axt.grid(alpha=0.3)
axt.legend(fontsize=6.5, ncol=2)
sy(axt, "T_K", "[K]")
ttl = axp.set_title("t = 0 s", loc="left", fontsize=10)
fig.suptitle(r"CO$_2$ blowdown, geothermal summer IC, HEM choke -- "
             "fiber-optic sensors, coupled thermal model")
fig.tight_layout()


def upd_sensors(i):
    t = cg.snap_times[i]
    lnT.set_xdata(TF[i])
    j = min(np.searchsorted(cg.sensor_t, t), cg.sensor_t.size - 1)
    fl_pts.set_data([K2F(cg.sensor_data[n][j]) for n in fluid_names],
                    [meta[n]["depth"] / FT for n in fluid_names])
    so_pts.set_data([K2F(cg.sensor_data[n][j]) for n in solid_names],
                    [meta[n]["depth"] / FT for n in solid_names])
    cursor.set_xdata([ts_min[j], ts_min[j]])
    ttl.set_text(f"t = {t:.0f} s = {t/60.0:.1f} min")
    return []


anim = FuncAnimation(fig, upd_sensors, frames=len(cg.snap_times), blit=False)
anim.save(S2 / "temperature_sensors_co2_geo_us.mp4", fps=25, dpi=110,
          writer="ffmpeg")
plt.close(fig)

# ------------------------------------------------ [3] blowdown fields anim
print("[3] blowdown-fields animation", flush=True)
fields = (("p", r"$p$ [psi]", 1.0 / PSI, "p_MPa", "[MPa]"),
          ("T", r"$T$ [$^\circ$F]", None, "T_K", "[K]"),
          ("rho", r"$\rho$ [lbm/ft$^3$]", 1.0 / DEN, "rho_kgm3",
           r"[kg/m$^3$]"),
          ("u", r"$u$ [ft/s]", 1.0 / FT, "u_mps", "[m/s]"))
fig, axes = plt.subplots(1, 4, figsize=(13, 5.6), sharey=True)
lines = {}
for k, (ax, (key, lab, sc, sik, silab)) in enumerate(zip(axes, fields)):
    data = K2F(getattr(cg, key)) if sc is None else getattr(cg, key) * sc
    (ln,) = ax.plot(data[0], zf, lw=1.4)
    lo, hi = data.min(), data.max()
    pad = 0.05 * (hi - lo + 1e-12)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(lab)
    ax.grid(alpha=0.3)
    for sd in DEPTHS_FT:
        ax.axhline(sd, color="crimson", lw=0.6, ls=(0, (2, 2)), alpha=0.7)
    sx(ax, sik, silab)
    if k == len(fields) - 1:
        sy(ax, "L_m", "[m]")
    lines[key] = (ln, data)
axes[0].set_ylabel(r"Depth $\zeta$ [ft]")
axes[0].set_ylim(GEOM.depth / FT, 0.0)

axm = axes[3].inset_axes([0.35, 0.08, 0.58, 0.30])
axm.plot(cg.t_hist / 60.0, cg.mdot / LBM, lw=0.8, color="tab:gray")
(mdot_pt,) = axm.plot([cg.t_hist[0] / 60.0], [cg.mdot[0] / LBM], "ro", ms=4)
axm.set_xlabel("$t$ [min]", fontsize=7)
axm.set_ylabel(r"$\dot m$ [lbm/s]", fontsize=7)
axm.tick_params(labelsize=6)

ttl = axes[0].set_title("t = 0 s", loc="left", fontsize=10)
fig.suptitle(r"CO$_2$ blowdown, geothermal summer IC, HEM choke "
             "(dashed: fiber sensor depths)")
fig.tight_layout()


def upd_fields(i):
    for key, (ln, data) in lines.items():
        ln.set_xdata(data[i])
    t = cg.snap_times[i]
    j = min(np.searchsorted(cg.t_hist, t), cg.t_hist.size - 1)
    mdot_pt.set_data([cg.t_hist[j] / 60.0], [cg.mdot[j] / LBM])
    ttl.set_text(f"t = {t:.0f} s")
    return []


anim = FuncAnimation(fig, upd_fields, frames=len(cg.snap_times), blit=False)
anim.save(S2 / "blowdown_fields_co2_geo_us.mp4", fps=25, dpi=110,
          writer="ffmpeg")
plt.close(fig)

# ----------------------------------------------------- [4] early-wave anim
print("[4] early-time wave animation", flush=True)
zw = zeta_ft(wv)
dp = (wv.p - wv.p[0][None, :]) / PSI                 # psi
du = (wv.u - wv.u[0][None, :]) / FT                  # ft/s
dT = (wv.T - wv.T[0][None, :]) * 1.8                 # delta degF
wfields = ((dp, r"$p - p_{init}(\zeta)$ [psi]", "p_kPa", "[kPa]"),
           (du, r"$u$ [ft/s]", "u_mps", "[m/s]"),
           (dT, r"$T - T_{init}(\zeta)$ [$^\circ$F]", "dT_K", "[K]"))
fig, axes = plt.subplots(1, 3, figsize=(11, 5.6), sharey=True)
wlines = []
for k, (ax, (data, lab, sik, silab)) in enumerate(zip(axes, wfields)):
    (ln,) = ax.plot(data[0], zw, lw=1.2)
    lo, hi = data.min(), data.max()
    pad = 0.05 * (hi - lo + 1e-12)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(lab)
    ax.grid(alpha=0.3)
    sx(ax, sik, silab)
    if k == 2:
        sy(ax, "L_m", "[m]")
    wlines.append((ln, data))
axes[0].set_ylabel(r"Depth $\zeta$ [ft]")
axes[0].set_ylim(GEOM.depth / FT, 0.0)
ttl = axes[0].set_title("t = 0.00 s", loc="left", fontsize=10)
fig.suptitle(r"Valve-opening expansion wave -- CO$_2$, geothermal IC "
             "(deviations from initial hydrostatic/geothermal profiles)")
fig.tight_layout()


def upd_wave(i):
    for ln, data in wlines:
        ln.set_xdata(data[i])
    ttl.set_text(f"t = {wv.snap_times[i]:.2f} s")
    return []


anim = FuncAnimation(fig, upd_wave, frames=len(wv.snap_times), blit=False)
anim.save(S2 / "waves_co2_geo_us.mp4", fps=25, dpi=110, writer="ffmpeg")
plt.close(fig)
print("done", flush=True)
