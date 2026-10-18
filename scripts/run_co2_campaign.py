"""CO2 real-fluid blowdown campaign + matched-N2 comparison.

Matched conditions for BOTH fluids (deviation from the original N2
baseline, which is retained untouched: at 13.79 MPa / 300 K CO2 is not a
single-phase gas, so the fluid-comparison pair runs at a wellhead
pressure of 5 MPa / 300 K):

  LSU geometry (ID 170.3 mm, 2,000 ft), 12.7 mm choke (Cd = 0.84),
  gravity + friction, hydrostatic initialization (the physically correct
  shut-in state; also avoids the artificial settling transient of a
  uniform column), n = 300 cells, hist_stride = 50, t_end = 3600 s,
  identical sensors, tolerances, output times and post-processing.

Cases: {CO2, N2} x {coupled formation model, adiabatic}; plus 8-s
early-wave runs (600 cells) for the expansion-wave comparison.

Outputs -> results/co2/..., results/cmp_*.png, results/co2/export/*.csv
"""

import json
import pathlib
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
from hyperbolic1d.thermal import FormationCoupledWall, lsu_co2_well_stack
from hyperbolic1d.wellbore import (AdiabaticWall, ChokeValve, WellboreModel,
                                   WellGeometry)

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
CO2DIR = RES / "co2"
CO2DIR.mkdir(parents=True, exist_ok=True)

r1, *_ = lsu_co2_well_stack()
GEOM = WellGeometry(depth=609.6, diameter=2 * r1)
CHOKE = ChokeValve()
P0, T0, T_END = 5.0e6, 300.0, 3600.0
N = 300
DEPTHS_FT = (50, 500, 1000, 1500, 1950)

report = {}
runs = {}

for fluid, eos_f in (("co2", lambda: SpanWagnerCO2()),
                     ("n2", lambda: NITROGEN)):
    for thermal in ("conjugate", "adiabatic"):
        key = f"{fluid}_{thermal}"
        print(f"[run] {key}", flush=True)
        wall = FormationCoupledWall() if thermal == "conjugate" \
            else AdiabaticWall()
        sensors = lsu_fiber_sensors(DEPTHS_FT) if thermal == "conjugate" \
            else lsu_fiber_sensors(DEPTHS_FT, casing_external=False)
        m = WellboreModel(geometry=GEOM, choke=CHOKE, eos=eos_f(),
                          gravity=True, friction=True, n_cells=N,
                          wall_model=wall)
        r = m.solve(P0, T0, T_END, init="hydrostatic", n_snapshots=301,
                    hist_stride=50, sensors=sensors)
        runs[key] = r
        entry = r.two_phase_entry
        report[key] = {
            "n_steps": r.n_steps, "wall_time_s": round(r.wall_time, 1),
            "mass_balance_error": r.mass_balance_error,
            "energy_balance_error": r.energy_balance_error,
            "T_wellhead_min_K": float(r.T_wellhead.min()),
            "mass_end_kg": float(r.mass[-1]),
            "heat_input_MJ": round(r.heat_input_J / 1e6, 3),
            "min_sat_margin_K": (float(np.min(r.sat_margin_min))
                                 if r.sat_margin_min is not None
                                 and r.sat_margin_min.size else None),
            "two_phase_entry": entry,
            "t_end_reached_s": float(r.t_hist[-1]),
        }
        msg = (f"TWO-PHASE ENTRY t={entry['t']:.1f}s depth="
               f"{entry['depth']:.0f}m p={entry['p']/1e6:.3f}MPa "
               f"T={entry['T']:.1f}K (Tsat={entry['T_sat']:.1f}K)"
               if entry else "single-phase throughout")
        print(f"    {r.n_steps} steps {r.wall_time:.0f}s; budgets "
              f"m={r.mass_balance_error:.0e} E={r.energy_balance_error:.0e}; "
              f"{msg}", flush=True)

print("[waves] early-time runs", flush=True)
waves = {}
for fluid, eos_f in (("co2", lambda: SpanWagnerCO2()),
                     ("n2", lambda: NITROGEN)):
    m = WellboreModel(geometry=GEOM, choke=CHOKE, eos=eos_f(), gravity=True,
                      friction=True, n_cells=600)
    waves[fluid] = m.solve(P0, T0, 10.0, init="hydrostatic",
                           n_snapshots=200, hist_stride=5)

# ---------------------------------------------------------------- CO2 set
print("[post] CO2 figure set + CSV", flush=True)
rc = runs["co2_conjugate"]
export_csv(rc, CO2DIR / "export", profile_times=(0, 120, 600, 1800, 3600))
plot_sensor_histories(rc, CO2DIR / "fig_sensors_T_co2.png", t_unit="min")
plot_profiles_at_times(rc, times=(0, 120, 600, 1800, 3600),
                       fields=("T", "p"),
                       filename=CO2DIR / "fig_profiles_co2.png",
                       sensor_depths_ft=DEPTHS_FT)
plot_wellhead_histories(rc, CO2DIR / "fig_wellhead_co2.png")

# ------------------------------------------------------------- comparisons
print("[post] comparison figures", flush=True)
cc, cn = runs["co2_conjugate"], runs["n2_conjugate"]
ca, na = runs["co2_adiabatic"], runs["n2_adiabatic"]

with pub_style("double"):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    for r, lab, c, ls in ((cn, "N$_2$ coupled", "tab:blue", "-"),
                          (cc, "CO$_2$ coupled", "tab:red", "-"),
                          (na, "N$_2$ adiabatic", "tab:blue", ":"),
                          (ca, "CO$_2$ adiabatic", "tab:red", ":")):
        tm = r.t_hist / 60.0
        axes[0, 0].plot(tm, r.p_wellhead / 1e6, c=c, ls=ls, label=lab)
        axes[0, 1].plot(tm, r.T_wellhead, c=c, ls=ls)
        axes[1, 0].plot(tm, r.mdot, c=c, ls=ls)
        axes[1, 1].plot(tm, r.mass / r.mass[0], c=c, ls=ls)
    if ca.two_phase_entry:
        axes[0, 1].plot(ca.two_phase_entry["t"] / 60.0,
                        ca.two_phase_entry["T"], "x", c="tab:red", ms=6)
        axes[0, 1].annotate("dome entry", fontsize=7, color="tab:red",
                            xy=(ca.two_phase_entry["t"] / 60.0,
                                ca.two_phase_entry["T"]),
                            xytext=(5, -10), textcoords="offset points")
    axes[0, 0].set_ylabel("$p_{wh}$ [MPa]")
    axes[0, 1].set_ylabel("$T_{wh}$ [K]")
    axes[1, 0].set_ylabel(r"$\dot m$ [kg s$^{-1}$]")
    axes[1, 1].set_ylabel("$M/M_0$ [-]")
    for ax in axes[1]:
        ax.set_xlabel("$t$ [min]")
    for ax in axes.flat:
        ax.grid(True)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("N$_2$ vs CO$_2$ blowdown — matched conditions "
                 "(5 MPa wellhead, 300 K, 12.7 mm choke)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "cmp_n2_co2_histories.png")
    plt.close(fig)

    # sensor temperature comparison (fluid DTS at two depths)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    for ax, d in zip(axes, (500, 1950)):
        for r, lab, c in ((cn, "N$_2$", "tab:blue"), (cc, "CO$_2$", "tab:red")):
            ax.plot(r.sensor_t / 60.0, r.sensor_data[f"fluid_{d}ft"],
                    c=c, label=lab)
        ax.set_xlabel("$t$ [min]")
        ax.set_ylabel("$T$ [K]")
        ax.set_title(f"bore DTS, {d} ft", fontsize=8)
        ax.grid(True)
    axes[0].legend()
    fig.suptitle("Fluid-fiber temperature, coupled thermal model", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "cmp_n2_co2_sensors.png")
    plt.close(fig)

    # density and temperature profiles at matched times
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    d_ft = (GEOM.depth - cc.z) / FT
    for i, tt in enumerate((120, 1800)):
        for r, lab, c in ((cn, "N$_2$", "tab:blue"), (cc, "CO$_2$", "tab:red")):
            k = int(np.argmin(np.abs(r.snap_times - tt)))
            ls = "-" if i == 0 else "--"
            axes[0].plot(r.rho[k], d_ft, c=c, ls=ls,
                         label=f"{lab}, {tt} s" if True else None)
            axes[1].plot(r.T[k], d_ft, c=c, ls=ls)
    axes[0].set_xlabel(r"$\rho$ [kg m$^{-3}$]")
    axes[1].set_xlabel("$T$ [K]")
    axes[0].set_ylabel("Depth [ft]")
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(True)
    axes[0].legend(fontsize=7)
    fig.suptitle("Profiles at 120 s (solid) and 1800 s (dashed), coupled",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "cmp_n2_co2_profiles.png")
    plt.close(fig)

    # expansion-wave comparison (early runs, pressure perturbation at the
    # bottom cell vs time: arrival and reverberation timing)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for fluid, lab, c in (("n2", "N$_2$", "tab:blue"),
                          ("co2", "CO$_2$", "tab:red")):
        w = waves[fluid]
        dpb = (w.p[:, 0] - w.p[0, 0]) / 1e3
        ax.plot(w.snap_times, dpb, c=c, label=lab)
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel(r"$p_{bottom} - p_{bottom,0}$ [kPa]")
    ax.grid(True)
    ax.legend()
    ax.set_title("Bottom-hole arrival of the valve-opening expansion wave",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "cmp_n2_co2_wave.png")
    plt.close(fig)

    # single-phase validity: p-T trajectories vs the saturation curve
    eos_sat = SpanWagnerCO2()
    eos_sat._ensure_sat_tables()
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.plot(eos_sat._T_sat, np.exp(eos_sat._lnp_sat) / 1e6, "k-", lw=1.2,
            label="CO$_2$ saturation")
    ax.plot(eos_sat.T_crit, eos_sat.p_crit / 1e6, "k*", ms=8)
    for r, lab, c in ((cc, "coupled", "tab:red"), (ca, "adiabatic", "gray")):
        for j, m_ in ((-1, "wellhead"), (0, "bottom")):
            ls = "-" if j == -1 else "--"
            ax.plot(r.T[:, j], r.p[:, j] / 1e6, c=c, ls=ls, lw=0.9,
                    label=f"{lab}, {m_}")
    if ca.two_phase_entry:
        ax.plot(ca.two_phase_entry["T"], ca.two_phase_entry["p"] / 1e6,
                "x", c="k", ms=7, label="flagged dome entry")
    ax.set_xlabel("$T$ [K]")
    ax.set_ylabel("$p$ [MPa]")
    ax.grid(True)
    ax.legend(fontsize=6.5)
    ax.set_title("CO$_2$ state trajectories vs saturation", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "cmp_co2_saturation_path.png")
    plt.close(fig)

# ---------------------------------------------------------------- animations
print("[anim] CO2 animations", flush=True)
animate_temperature(
    rc, str(CO2DIR / "temperature_sensors_co2.mp4"), fps=25,
    title="CO$_2$ blowdown (Span-Wagner EOS) — fluid temperature and "
          "fiber-optic sensors, coupled thermal model")
animate_blowdown(
    rc, str(CO2DIR / "blowdown_fields_co2.mp4"), fps=25,
    title="CO$_2$ blowdown, LSU well — coupled thermal model "
          "(dashed: fiber sensor depths)",
    time_label="t = {t:.0f} s",
    sensor_depths=[d * FT for d in DEPTHS_FT])

with open(RES / "co2_campaign.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print("done", flush=True)
