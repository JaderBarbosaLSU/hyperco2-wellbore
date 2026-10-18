"""Stage 2: HEM choke + geothermal initial temperature profile.

Evaluates the two approaches requested for keeping the CO2 wellbore
single-phase (docs/co2_eos.md SS3b-3c):

  1. HEM two-phase choke (flashing allowed across the valve only; the
     wellbore conservation equations remain single-phase, and the dome
     monitor still halts the run on any in-wellbore dome entry);
  2. Louisiana-summer geothermal initial profile
     T(z,0) = T_surface + G z with T_surface = 305.4 K (90 F) and
     G = 25.5 K/km (1.4 F/100 ft), formation initialized consistently.

Cases (all 5 MPa wellhead, hydrostatic init, LSU geometry, 12.7 mm
choke, n = 300, t_end = 3600 s; HEM choke active for all CO2 cases):

  co2_{conjugate,adiabatic}_hem   uniform 300 K  (approach 1 alone)
  co2_{conjugate,adiabatic}_geo   geothermal     (approaches 1 + 2)
  n2_{conjugate,adiabatic}_geo    N2 baseline at the same geothermal IC

The stage-1 uniform-300 K N2 runs and the single-phase-bound CO2 runs
(results/co2/ckpt_*.pkl) are reused for cross-stage comparison where
present. Checkpointed per case; re-running skips completed cases.

Outputs -> results/co2_stage2/...  +  results/co2_campaign2.json
"""

import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperbolic1d.animate import animate_blowdown, animate_temperature
from hyperbolic1d.eos import NITROGEN
from hyperbolic1d.postprocess import (FT, export_csv, lsu_fiber_sensors,
                                      plot_profiles_at_times,
                                      plot_sensor_histories,
                                      plot_wellhead_histories, pub_style)
from hyperbolic1d.realfluid import SpanWagnerCO2
from hyperbolic1d.thermal import (FormationCoupledWall, Geothermal,
                                  lsu_co2_well_stack)
from hyperbolic1d.wellbore import (AdiabaticWall, ChokeValve, WellboreModel,
                                   WellGeometry)

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
S2 = RES / "co2_stage2"
S2.mkdir(parents=True, exist_ok=True)
S1 = RES / "co2"

r1, *_ = lsu_co2_well_stack()
GEOM = WellGeometry(depth=609.6, diameter=2 * r1)
CHOKE = ChokeValve()
P0, T0, T_END = 5.0e6, 300.0, 3600.0
N = 300
DEPTHS_FT = (50, 500, 1000, 1500, 1950)
GEO_SUMMER = Geothermal(T_surface=305.4, gradient=0.0255)  # 90 F, 1.4 F/100ft

report = {"parameters": {
    "T_surface_K": GEO_SUMMER.T_surface,
    "geothermal_gradient_K_per_m": GEO_SUMMER.gradient,
    "p_wellhead_Pa": P0, "T_uniform_K": T0,
    "initial_T_at_depth_K": {f"{d} ft": round(float(
        GEO_SUMMER(d * FT)), 1) for d in (0, 500, 1000, 1500, 2000)},
}}


def run_case(key, eos, thermal, geothermal):
    ckpt = S2 / f"ckpt_{key}.pkl"
    if ckpt.exists():
        print(f"[run] {key} (checkpoint found, skipping)", flush=True)
        with open(ckpt, "rb") as f:
            return pickle.load(f)
    print(f"[run] {key}", flush=True)
    if thermal == "conjugate":
        wall = FormationCoupledWall(geothermal=geothermal or
                                    Geothermal())  # stage-1 default profile
        sensors = lsu_fiber_sensors(DEPTHS_FT)
    else:
        wall = AdiabaticWall()
        sensors = lsu_fiber_sensors(DEPTHS_FT, casing_external=False)
    m = WellboreModel(geometry=GEOM, choke=CHOKE, eos=eos, gravity=True,
                      friction=True, n_cells=N, wall_model=wall)
    r = m.solve(P0, T0, T_END, init="hydrostatic", n_snapshots=301,
                hist_stride=50, sensors=sensors, geothermal=geothermal)
    with open(ckpt, "wb") as f:
        pickle.dump(r, f)
    return r


def describe(key, r):
    entry = r.two_phase_entry
    fl = getattr(r, "choke_flashing", None)
    ql = getattr(r, "choke_quality", None)
    d = {
        "n_steps": r.n_steps, "wall_time_s": round(r.wall_time, 1),
        "mass_balance_error": r.mass_balance_error,
        "energy_balance_error": r.energy_balance_error,
        "T_wellhead_min_K": float(r.T_wellhead.min()),
        "mass_end_kg": float(r.mass[-1]),
        "heat_input_MJ": round(r.heat_input_J / 1e6, 3),
        "t_end_reached_s": float(r.t_hist[-1]),
        "two_phase_entry": entry,
    }
    if r.sat_margin_min is not None and r.sat_margin_min.size:
        j = int(np.argmin(r.sat_margin_min))
        d["min_sat_margin_K"] = float(r.sat_margin_min[j])
        smd = getattr(r, "sat_margin_depth", None)
        d["min_sat_margin_at"] = {
            "t_s": float(r.t_hist[j]),
            "depth_m": (float(smd[j]) if smd is not None else None)}
    if fl is not None and fl.size:
        d["choke_flashing_fraction"] = float(np.mean(fl))
        if np.any(fl):
            qv = ql[np.asarray(fl, bool)]
            qv = qv[np.isfinite(qv)]
            d["throat_quality_range"] = ([float(qv.min()), float(qv.max())]
                                         if qv.size else None)
    msg = (f"TWO-PHASE ENTRY t={entry['t']:.1f}s depth="
           f"{entry['depth']:.0f}m p={entry['p']/1e6:.3f}MPa "
           f"T={entry['T']:.1f}K (Tsat={entry['T_sat']:.1f}K)"
           if entry else "single-phase throughout")
    print(f"    {r.n_steps} steps {r.wall_time:.0f}s; budgets "
          f"m={r.mass_balance_error:.0e} E={r.energy_balance_error:.0e}; "
          f"{msg}", flush=True)
    return d


runs = {}
CASES = [
    ("co2_conjugate_hem", lambda: SpanWagnerCO2(), "conjugate", None),
    ("co2_adiabatic_hem", lambda: SpanWagnerCO2(), "adiabatic", None),
    ("co2_conjugate_geo", lambda: SpanWagnerCO2(), "conjugate", GEO_SUMMER),
    ("co2_adiabatic_geo", lambda: SpanWagnerCO2(), "adiabatic", GEO_SUMMER),
    ("n2_conjugate_geo", lambda: NITROGEN, "conjugate", GEO_SUMMER),
    ("n2_adiabatic_geo", lambda: NITROGEN, "adiabatic", GEO_SUMMER),
]
for key, eos_f, thermal, geo in CASES:
    r = run_case(key, eos_f(), thermal, geo)
    runs[key] = r
    report[key] = describe(key, r)

# geothermal early-wave runs (600 cells, 10 s)
print("[waves] geothermal early-time runs", flush=True)
waves = {}
for fluid, eos_f in (("co2", lambda: SpanWagnerCO2()),
                     ("n2", lambda: NITROGEN)):
    ckpt = S2 / f"ckpt_wave_{fluid}_geo.pkl"
    if ckpt.exists():
        with open(ckpt, "rb") as f:
            waves[fluid] = pickle.load(f)
        continue
    m = WellboreModel(geometry=GEOM, choke=CHOKE, eos=eos_f(), gravity=True,
                      friction=True, n_cells=600)
    waves[fluid] = m.solve(P0, T0, 10.0, init="hydrostatic", n_snapshots=200,
                           hist_stride=5, geothermal=GEO_SUMMER)
    with open(ckpt, "wb") as f:
        pickle.dump(waves[fluid], f)

# stage-1 runs for cross-stage comparison (if available)
stage1 = {}
for k in ("co2_conjugate", "co2_adiabatic", "n2_conjugate", "n2_adiabatic"):
    p = S1 / f"ckpt_{k}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            stage1[k] = pickle.load(f)

eos_sat = SpanWagnerCO2()
eos_sat._ensure_sat_tables()


def margin_field(r):
    """m(z, t) = T - T_sat(p) on the snapshot grid."""
    return np.array([eos_sat.saturation_margin(r.p[k], r.T[k])
                     for k in range(r.p.shape[0])])


# ----------------------------------------------------------------- figures
print("[post] stage-2 figures", flush=True)
cg, ag = runs["co2_conjugate_geo"], runs["co2_adiabatic_geo"]
ch, ah = runs["co2_conjugate_hem"], runs["co2_adiabatic_hem"]
ng, na = runs["n2_conjugate_geo"], runs["n2_adiabatic_geo"]
d_ft = (GEOM.depth - cg.z) / FT

with pub_style("double"):
    # saturation-margin field m(z, t): uniform vs geothermal IC
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True,
                             sharey=True)
    panels = ((ch, "uniform 300 K, coupled"), (cg, "geothermal, coupled"),
              (ah, "uniform 300 K, adiabatic"), (ag, "geothermal, adiabatic"))
    vmax = 40.0
    for ax, (r, lab) in zip(axes.flat, panels):
        mzt = margin_field(r)
        pc = ax.pcolormesh(r.snap_times / 60.0, d_ft, mzt.T, cmap="RdYlBu",
                           vmin=0.0, vmax=vmax, shading="nearest")
        c0 = ax.contour(r.snap_times / 60.0, d_ft, mzt.T, levels=[5.0],
                        colors="k", linewidths=0.7, linestyles="--")
        ax.clabel(c0, fmt="%g K", fontsize=6)
        ax.set_title(lab, fontsize=8)
        if r.two_phase_entry:
            ax.plot(r.two_phase_entry["t"] / 60.0,
                    r.two_phase_entry["depth"] / FT, "kx", ms=7)
    axes[0, 0].invert_yaxis()
    for ax in axes[1]:
        ax.set_xlabel("$t$ [min]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Depth [ft]")
    fig.colorbar(pc, ax=axes, label=r"$T - T_{sat}(p)$ [K]", shrink=0.85)
    fig.suptitle("CO$_2$ saturation margin $m(z,t)$ "
                 "(x: flagged dome entry; dashed: 5 K)", fontsize=9)
    fig.savefig(S2 / "fig2_margin_zt.png")
    plt.close(fig)

    # min margin + its depth vs time
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for r, lab, c, ls in ((ch, "uniform, coupled", "tab:blue", "-"),
                          (ah, "uniform, adiabatic", "tab:blue", ":"),
                          (cg, "geothermal, coupled", "tab:red", "-"),
                          (ag, "geothermal, adiabatic", "tab:red", ":")):
        axes[0].plot(r.t_hist / 60.0, r.sat_margin_min, c=c, ls=ls,
                     label=lab)
        smd = getattr(r, "sat_margin_depth", None)
        if smd is not None:
            axes[1].plot(r.t_hist / 60.0, smd / FT, c=c, ls=ls)
    axes[0].axhline(0.0, c="k", lw=0.6)
    axes[0].set_ylabel(r"$\min_z\,(T - T_{sat})$ [K]")
    axes[1].set_ylabel("Depth of minimum [ft]")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.set_xlabel("$t$ [min]")
        ax.grid(True)
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Minimum saturation margin and its location", fontsize=9)
    fig.tight_layout()
    fig.savefig(S2 / "fig2_min_margin.png")
    plt.close(fig)

    # HEM choke: mass flow + flashing / throat quality
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for r, lab, c in ((ch, "uniform 300 K", "tab:blue"),
                      (cg, "geothermal", "tab:red")):
        axes[0].plot(r.t_hist / 60.0, r.mdot, c=c, label=lab)
        ql = getattr(r, "choke_quality", None)
        if ql is not None:
            axes[1].plot(r.t_hist / 60.0, ql, c=c)
    axes[0].set_ylabel(r"$\dot m$ [kg s$^{-1}$]")
    axes[1].set_ylabel("HEM throat quality $x_t$ [-]")
    axes[1].set_ylim(0.0, 1.05)
    for ax in axes:
        ax.set_xlabel("$t$ [min]")
        ax.grid(True)
    axes[0].legend(fontsize=7)
    fig.suptitle("HEM choke: discharge and throat quality (gaps: "
                 "single-phase throat)", fontsize=9)
    fig.tight_layout()
    fig.savefig(S2 / "fig2_choke_hem.png")
    plt.close(fig)

    # N2 vs CO2 at geothermal conditions (histories)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    for r, lab, c, ls in ((ng, "N$_2$ coupled", "tab:blue", "-"),
                          (cg, "CO$_2$ coupled", "tab:red", "-"),
                          (na, "N$_2$ adiabatic", "tab:blue", ":"),
                          (ag, "CO$_2$ adiabatic", "tab:red", ":")):
        tm = r.t_hist / 60.0
        axes[0, 0].plot(tm, r.p_wellhead / 1e6, c=c, ls=ls, label=lab)
        axes[0, 1].plot(tm, r.T_wellhead, c=c, ls=ls)
        axes[1, 0].plot(tm, r.mdot, c=c, ls=ls)
        axes[1, 1].plot(tm, r.mass / r.mass[0], c=c, ls=ls)
    axes[0, 0].set_ylabel("$p_{wh}$ [MPa]")
    axes[0, 1].set_ylabel("$T_{wh}$ [K]")
    axes[1, 0].set_ylabel(r"$\dot m$ [kg s$^{-1}$]")
    axes[1, 1].set_ylabel("$M/M_0$ [-]")
    for ax in axes[1]:
        ax.set_xlabel("$t$ [min]")
    for ax in axes.flat:
        ax.grid(True)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("N$_2$ vs CO$_2$, geothermal summer IC "
                 "(305.4 K + 25.5 K/km), HEM choke", fontsize=9)
    fig.tight_layout()
    fig.savefig(S2 / "fig2_n2_co2_histories_geo.png")
    plt.close(fig)

    # expansion-wave comparison at geothermal IC
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for fluid, lab, c in (("n2", "N$_2$", "tab:blue"),
                          ("co2", "CO$_2$", "tab:red")):
        w = waves[fluid]
        ax.plot(w.snap_times, (w.p[:, 0] - w.p[0, 0]) / 1e3, c=c, label=lab)
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel(r"$p_{bottom} - p_{bottom,0}$ [kPa]")
    ax.grid(True)
    ax.legend()
    ax.set_title("Bottom-hole expansion-wave arrival, geothermal IC",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(S2 / "fig2_wave_geo.png")
    plt.close(fig)

    # p-T trajectories vs saturation curve, all CO2 cases
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(eos_sat._T_sat, np.exp(eos_sat._lnp_sat) / 1e6, "k-", lw=1.2,
            label="saturation")
    ax.plot(eos_sat.T_crit, eos_sat.p_crit / 1e6, "k*", ms=8)
    for r, lab, c in ((ch, "uniform, coupled", "tab:blue"),
                      (ah, "uniform, adiab.", "cornflowerblue"),
                      (cg, "geothermal, coupled", "tab:red"),
                      (ag, "geothermal, adiab.", "salmon")):
        ax.plot(r.T[:, -1], r.p[:, -1] / 1e6, c=c, lw=0.9, label=lab)
        ax.plot(r.T[:, 0], r.p[:, 0] / 1e6, c=c, lw=0.9, ls="--")
        if r.two_phase_entry:
            ax.plot(r.two_phase_entry["T"], r.two_phase_entry["p"] / 1e6,
                    "x", c=c, ms=7)
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$p$ [MPa]")
    ax.grid(True)
    ax.legend(fontsize=6)
    ax.set_title("CO$_2$ trajectories (solid: wellhead, dashed: bottom; "
                 "x: dome entry)", fontsize=8)
    fig.tight_layout()
    fig.savefig(S2 / "fig2_saturation_paths.png")
    plt.close(fig)

# headline CO2 case: full output set (sensors, profiles, wellhead, CSV)
print("[post] headline case output set", flush=True)
export_csv(cg, S2 / "export", profile_times=(0, 120, 600, 1800, 3600))
plot_sensor_histories(cg, S2 / "fig2_sensors_T_co2_geo.png", t_unit="min")
plot_profiles_at_times(cg, times=(0, 120, 600, 1800, 3600),
                       fields=("T", "p"),
                       filename=S2 / "fig2_profiles_co2_geo.png",
                       sensor_depths_ft=DEPTHS_FT)
plot_wellhead_histories(cg, S2 / "fig2_wellhead_co2_geo.png")

print("[anim] stage-2 animations", flush=True)
animate_temperature(
    cg, str(S2 / "temperature_sensors_co2_geo.mp4"), fps=25,
    title="CO$_2$ blowdown, geothermal summer IC, HEM choke — "
          "fiber-optic sensors, coupled thermal model")
animate_blowdown(
    cg, str(S2 / "blowdown_fields_co2_geo.mp4"), fps=25,
    title="CO$_2$ blowdown, geothermal summer IC, HEM choke "
          "(dashed: fiber sensor depths)",
    time_label="t = {t:.0f} s",
    sensor_depths=[d * FT for d in DEPTHS_FT])

with open(RES / "co2_campaign2.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print("done", flush=True)
