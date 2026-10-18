"""Regenerate ALL report figures with US-customary primary axes and SI
secondary scales (upper abscissa / right ordinate), and with every
spatial result reported as depth zeta = L - z (ft, increasing downward).

Units (primary | secondary): psi | MPa (or kPa for perturbations),
degF | K (differences: degF | K), ft/s | m/s, lbm/s | kg/s, ft | m,
lbm/ft^3 | kg/m^3, lbm | kg.

Legacy N2-baseline runs are recomputed (checkpointed pickles in
results/paper_us/); campaign results are loaded from the stage-1/2
checkpoints. Output: paper/figures/*.png (same file names as before,
so main.tex includes are unchanged).
"""

import dataclasses
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperbolic1d.eos import NITROGEN
from hyperbolic1d.postprocess import FT, pub_style
from hyperbolic1d.realfluid import SpanWagnerCO2
from hyperbolic1d.thermal import (GEOTHERMAL_DEFAULT, FormationCoupledWall,
                                  lsu_co2_well_stack)
from hyperbolic1d.wellbore import (AdiabaticWall, ChokeValve,
                                   PrescribedWallTemperature, WellboreModel,
                                   WellGeometry, vessel_blowdown_ode)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
CKPT = RES / "paper_us"
CKPT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- units
PSI = 6894.757293       # Pa
LBM = 0.45359237        # kg
DEN = LBM / FT**3       # lbm/ft^3 in kg/m^3  (16.0185)


def K2F(K):
    return (np.asarray(K) - 273.15) * 1.8 + 32.0


def F2K(F):
    return (np.asarray(F) - 32.0) / 1.8 + 273.15


# secondary-axis function pairs: (us -> si, si -> us)
SI = {
    "p_MPa": (lambda x: x * PSI / 1e6, lambda x: x * 1e6 / PSI),
    "p_kPa": (lambda x: x * PSI / 1e3, lambda x: x * 1e3 / PSI),
    "T_K": (F2K, K2F),
    "dT_K": (lambda x: np.asarray(x) / 1.8, lambda x: np.asarray(x) * 1.8),
    "u_mps": (lambda x: x * FT, lambda x: x / FT),
    "md_kgps": (lambda x: x * LBM, lambda x: x / LBM),
    "L_m": (lambda x: x * FT, lambda x: x / FT),
    "rho_kgm3": (lambda x: x * DEN, lambda x: x / DEN),
    "m_kg": (lambda x: x * LBM, lambda x: x / LBM),
}


def sy(ax, kind, label):
    """SI scale on the right ordinate."""
    ax.tick_params(right=False, which="both")
    s = ax.secondary_yaxis("right", functions=SI[kind])
    s.set_ylabel(label)
    return s


def sx(ax, kind, label):
    """SI scale on the upper abscissa."""
    ax.tick_params(top=False, which="both")
    s = ax.secondary_xaxis("top", functions=SI[kind])
    s.set_xlabel(label)
    return s


def run(key, fn):
    p = CKPT / f"{key}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    print(f"[run] {key}", flush=True)
    r = fn()
    with open(p, "wb") as f:
        pickle.dump(r, f)
    return r


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


P0B, T0 = 1.379e7, 300.0          # original N2 baseline
GEOM_B = WellGeometry()           # D = 0.100 m
CHOKE = ChokeValve()
r1, *_ = lsu_co2_well_stack()
GEOM_L = WellGeometry(depth=609.6, diameter=2 * r1)
LB = GEOM_B.depth                 # 609.6 m (both geometries)

# ========================================================== legacy reruns
r_ves = run("vessel", lambda: WellboreModel(
    geometry=GEOM_B, choke=CHOKE, eos=NITROGEN, gravity=False,
    n_cells=200).solve(P0B, T0, 600.0, n_snapshots=2))
ves_ref = vessel_blowdown_ode(NITROGEN, GEOM_B.volume, CHOKE, P0B, T0, 600.0)

sweep = {}
for d, label in [(0.00635, "1/4 in"), (0.0127, "1/2 in"), (0.0254, "1 in"),
                 (0.0508, "2 in"), (GEOM_B.diameter, "open pipe")]:
    cd = 1.0 if label == "open pipe" else 0.84
    sweep[label] = run(f"sweep_{label.replace(' ', '').replace('/', '')}",
                       lambda d=d, cd=cd: WellboreModel(
                           geometry=GEOM_B,
                           choke=ChokeValve(diameter=d, cd=cd),
                           gravity=True, n_cells=200).solve(
                           P0B, T0, 20.0, n_snapshots=2))

r_wave = run("wave8s", lambda: WellboreModel(
    geometry=GEOM_B, choke=CHOKE, gravity=True, n_cells=600).solve(
    P0B, T0, 8.0, init="hydrostatic", n_snapshots=400, hist_stride=5))

r_demo = run("demo600s", lambda: WellboreModel(
    geometry=GEOM_B, choke=CHOKE, gravity=True, n_cells=400).solve(
    P0B, T0, 600.0, n_snapshots=7))

N_H = 300
zc_h = (np.arange(N_H) + 0.5) * (GEOM_L.depth / N_H)
T_geo = GEOTHERMAL_DEFAULT.T(GEOM_L.depth - zc_h)
hier = {}
hier_walls = {}
for name, wall_f in (
        ("adiabatic", AdiabaticWall),
        ("prescribed $T_w$ (geothermal)",
         lambda: PrescribedWallTemperature(T_wall=T_geo)),
        ("coupled formation model", FormationCoupledWall)):
    key = "hier_" + name.split()[0].strip("$")
    p = CKPT / f"{key}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            hier[name], hier_walls[name] = pickle.load(f)
    else:
        print(f"[run] {key}", flush=True)
        wall = wall_f()
        m = WellboreModel(geometry=GEOM_L, choke=CHOKE, gravity=True,
                          n_cells=N_H, wall_model=wall)
        r = m.solve(P0B, T0, 1800.0, n_snapshots=7)
        hier[name] = r
        hier_walls[name] = (wall.radial_temperatures.copy(),
                            np.asarray(wall.radial_positions).copy()) \
            if hasattr(wall, "radial_temperatures") else None
        with open(p, "wb") as f:
            pickle.dump((r, hier_walls[name]), f)

# ------------------------------------------------------- campaign loads
cc = load(RES / "co2" / "ckpt_n2_conjugate.pkl")          # n2 uniform conj
na_u = load(RES / "co2" / "ckpt_n2_adiabatic.pkl")
ch = load(RES / "co2_stage2" / "ckpt_co2_conjugate_hem.pkl")
ah = load(RES / "co2_stage2" / "ckpt_co2_adiabatic_hem.pkl")
cg = load(RES / "co2_stage2" / "ckpt_co2_conjugate_geo.pkl")
ag = load(RES / "co2_stage2" / "ckpt_co2_adiabatic_geo.pkl")
ng = load(RES / "co2_stage2" / "ckpt_n2_conjugate_geo.pkl")
na_g = load(RES / "co2_stage2" / "ckpt_n2_adiabatic_geo.pkl")
w_n2_g = load(RES / "co2_stage2" / "ckpt_wave_n2_geo.pkl")
w_co2_g = load(RES / "co2_stage2" / "ckpt_wave_co2_geo.pkl")
wave_n2_u = load(RES / "co2" / "ckpt_wave_n2.pkl")
rn_u = load(RES / "co2" / "ckpt_n2_conjugate.pkl")

eos_sat = SpanWagnerCO2()
eos_sat._ensure_sat_tables()


def margin_field(r):
    return np.array([eos_sat.saturation_margin(r.p[k], r.T[k])
                     for k in range(r.p.shape[0])])


def depth_ft(r, L=609.6):
    return (L - r.z) / FT


print("[fig] rendering", flush=True)

# ================================================================ figures
with pub_style("double"):
    # ---------------------------------------------- blowdown_vs_vessel_ode
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    axes[0].plot(ves_ref["t"], ves_ref["p"] / PSI, "k-", lw=1.6,
                 label="0-D vessel ODE")
    axes[0].plot(r_ves.t_hist, r_ves.p_wellhead / PSI, "r--",
                 label="1-D FV (wellhead)")
    axes[0].set_ylabel("$p_{wh}$ [psi]")
    sy(axes[0], "p_MPa", "[MPa]")
    axes[1].plot(ves_ref["t"], ves_ref["mdot"] / LBM, "k-", lw=1.6)
    axes[1].plot(r_ves.t_hist, r_ves.mdot / LBM, "r--")
    axes[1].set_ylabel(r"$\dot m$ [lbm/s]")
    sy(axes[1], "md_kgps", "[kg/s]")
    axes[2].plot(ves_ref["t"], ves_ref["mass"] / LBM, "k-", lw=1.6)
    axes[2].plot(r_ves.t_hist, r_ves.mass / LBM, "r--")
    axes[2].set_ylabel("$M$ [lbm]")
    sy(axes[2], "m_kg", "[kg]")
    for ax in axes:
        ax.set_xlabel("$t$ [s]")
        ax.grid(True)
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "blowdown_vs_vessel_ode.png")
    plt.close(fig)

    # -------------------------------------------------- blowdown_choke_sweep
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for label, r in sweep.items():
        ax.plot(r.t_hist, r.mdot / LBM, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel(r"$\dot m$ [lbm/s]")
    s = ax.secondary_yaxis("right", functions=SI["md_kgps"])
    s.set_ylabel("[kg/s]")
    ax.tick_params(right=False, which="both")
    ax.grid(True)
    ax.legend(title="choke bore", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "blowdown_choke_sweep.png")
    plt.close(fig)

    # ------------------------------------------------------- blowdown_xt
    dp = (r_wave.p - r_wave.p[0][None, :]) / PSI
    dft = depth_ft(r_wave)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    pc = ax.pcolormesh(r_wave.snap_times, dft, dp.T, shading="auto",
                       cmap="viridis", rasterized=True)
    cb = fig.colorbar(pc, ax=ax, pad=0.13)
    cb.set_label(r"$p - p_{init}(\zeta)$ [psi]"
                 "\n(1 psi = 6.895 kPa)")
    ax.invert_yaxis()
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel(r"Depth $\zeta$ [ft]")
    sy(ax, "L_m", "[m]")
    fig.tight_layout()
    fig.savefig(FIG / "blowdown_xt.png")
    plt.close(fig)

    # -------------------------------------------------- blowdown_profiles
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.6), sharey=True)
    cmap = plt.cm.viridis
    dft = depth_ft(r_demo)
    for i, ts in enumerate(r_demo.snap_times):
        c = cmap(i / max(len(r_demo.snap_times) - 1, 1))
        kw = dict(color=c, lw=1.0, label=f"{ts:.0f} s")
        axes[0].plot(r_demo.p[i] / PSI, dft, **kw)
        axes[1].plot(K2F(r_demo.T[i]), dft, **kw)
        axes[2].plot(r_demo.rho[i] / DEN, dft, **kw)
        axes[3].plot(r_demo.u[i] / FT, dft, **kw)
    axes[0].set_xlabel("$p$ [psi]")
    sx(axes[0], "p_MPa", "[MPa]")
    axes[1].set_xlabel("$T$ [$^\\circ$F]")
    sx(axes[1], "T_K", "[K]")
    axes[2].set_xlabel(r"$\rho$ [lbm/ft$^3$]")
    sx(axes[2], "rho_kgm3", "[kg/m$^3$]")
    axes[3].set_xlabel("$u$ [ft/s]")
    sx(axes[3], "u_mps", "[m/s]")
    axes[0].set_ylabel(r"Depth $\zeta$ [ft]")
    axes[0].invert_yaxis()
    sy(axes[3], "L_m", r"$\zeta$ [m]")
    for ax in axes:
        ax.grid(True)
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "blowdown_profiles.png")
    plt.close(fig)

    # ------------------------------------------------- blowdown_histories
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    axes[0].plot(r_demo.t_hist, r_demo.mdot / LBM, "tab:blue")
    axes[0].set_ylabel(r"$\dot m$ [lbm/s]")
    sy(axes[0], "md_kgps", "[kg/s]")
    axes[1].plot(r_demo.t_hist, r_demo.mass / LBM, "tab:blue")
    axes[1].set_ylabel("$M$ [lbm]")
    sy(axes[1], "m_kg", "[kg]")
    axes[2].plot(r_demo.t_hist, K2F(r_demo.T_wellhead), "tab:blue")
    axes[2].set_ylabel("$T_{wh}$ [$^\\circ$F]")
    sy(axes[2], "T_K", "[K]")
    for ax in axes:
        ax.set_xlabel("$t$ [s]")
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIG / "blowdown_histories.png")
    plt.close(fig)

    # ------------------------------------------------- thermal_hierarchy
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    colors = {"adiabatic": "tab:red",
              "prescribed $T_w$ (geothermal)": "tab:green",
              "coupled formation model": "tab:blue"}
    for name, r in hier.items():
        c = colors[name]
        axes[0].plot(r.t_hist / 60.0, K2F(r.T_wellhead), color=c,
                     label=name)
        axes[1].plot(r.t_hist / 60.0, r.mdot / LBM, color=c)
        axes[2].plot(r.t_hist / 60.0, r.mass / LBM, color=c)
    axes[0].set_ylabel("$T_{wh}$ [$^\\circ$F]")
    sy(axes[0], "T_K", "[K]")
    axes[1].set_ylabel(r"$\dot m$ [lbm/s]")
    sy(axes[1], "md_kgps", "[kg/s]")
    axes[2].set_ylabel("$M$ [lbm]")
    sy(axes[2], "m_kg", "[kg]")
    for ax in axes:
        ax.set_xlabel("$t$ [min]")
        ax.grid(True)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "thermal_hierarchy.png")
    plt.close(fig)

    # ---------------------------------------------------- thermal_radial
    Tr, rr = hier_walls["coupled formation model"]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for frac, lab in ((0.95, "near wellhead"), (0.5, "mid depth"),
                      (0.05, "near bottom")):
        j = int(frac * (N_H - 1))
        ax.semilogx(rr / FT, K2F(Tr[j]), ".-", ms=3,
                    label=f"{lab} ($\\zeta$ = "
                          f"{(GEOM_L.depth - zc_h[j]) / FT:.0f} ft)")
    for x, lab in ((r1, "bore"), (0.0968, "7-5/8 OD"),
                   (0.1568, "13-3/8 ID"), (0.3302, "borehole")):
        ax.axvline(x / FT, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("$r$ [ft]")
    ax.set_ylabel("$T$ [$^\\circ$F]")
    sy(ax, "T_K", "[K]")
    ax.grid(True, which="both")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "thermal_radial.png")
    plt.close(fig)

    # ------------------------------------- cmp_histories (uniform + geo)
    def cmp_histories(runs4, fname, entry_run=None):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
        for r, lab, c, ls in runs4:
            tm = r.t_hist / 60.0
            axes[0, 0].plot(tm, r.p_wellhead / PSI, c=c, ls=ls, label=lab)
            axes[0, 1].plot(tm, K2F(r.T_wellhead), c=c, ls=ls)
            axes[1, 0].plot(tm, r.mdot / LBM, c=c, ls=ls)
            axes[1, 1].plot(tm, r.mass / r.mass[0], c=c, ls=ls)
        if entry_run is not None and entry_run.two_phase_entry:
            e = entry_run.two_phase_entry
            axes[0, 1].plot(e["t"] / 60.0, K2F(e["T"]), "x", c="tab:red",
                            ms=6)
        axes[0, 0].set_ylabel("$p_{wh}$ [psi]")
        sy(axes[0, 0], "p_MPa", "[MPa]")
        axes[0, 1].set_ylabel("$T_{wh}$ [$^\\circ$F]")
        sy(axes[0, 1], "T_K", "[K]")
        axes[1, 0].set_ylabel(r"$\dot m$ [lbm/s]")
        sy(axes[1, 0], "md_kgps", "[kg/s]")
        axes[1, 1].set_ylabel("$M/M_0$ [-]")
        for ax in axes[1]:
            ax.set_xlabel("$t$ [min]")
        for ax in axes.flat:
            ax.grid(True)
        axes[0, 0].legend(fontsize=6.5)
        fig.tight_layout()
        fig.savefig(FIG / fname)
        plt.close(fig)

    cmp_histories(((cc, "N$_2$ coupled", "tab:blue", "-"),
                   (ch, "CO$_2$ coupled", "tab:red", "-"),
                   (na_u, "N$_2$ adiabatic", "tab:blue", ":"),
                   (ah, "CO$_2$ adiabatic", "tab:red", ":")),
                  "cmp_histories_uniform.png", entry_run=ah)
    cmp_histories(((ng, "N$_2$ coupled", "tab:blue", "-"),
                   (cg, "CO$_2$ coupled", "tab:red", "-"),
                   (na_g, "N$_2$ adiabatic", "tab:blue", ":"),
                   (ag, "CO$_2$ adiabatic", "tab:red", ":")),
                  "cmp_histories_geo.png", entry_run=ag)

    # ---------------------------------------------------- cmp_sensors
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, d in zip(axes, (500, 1950)):
        for r, lab, c in ((cc, "N$_2$", "tab:blue"),
                          (ch, "CO$_2$", "tab:red")):
            ax.plot(r.sensor_t / 60.0, K2F(r.sensor_data[f"fluid_{d}ft"]),
                    c=c, label=lab)
        ax.set_xlabel("$t$ [min]")
        ax.set_ylabel("$T$ [$^\\circ$F]")
        sy(ax, "T_K", "[K]")
        ax.set_title(f"bore DTS, $\\zeta$ = {d} ft", fontsize=8)
        ax.grid(True)
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "cmp_sensors.png")
    plt.close(fig)

    # ------------------------------------------------------- margin_zt
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True,
                             sharey=True)
    panels = ((ch, "uniform 300 K, coupled"), (cg, "geothermal, coupled"),
              (ah, "uniform 300 K, adiabatic"),
              (ag, "geothermal, adiabatic"))
    for ax, (r, lab) in zip(axes.flat, panels):
        mzt = margin_field(r) * 1.8          # deltaF
        dft = depth_ft(r)
        pc = ax.pcolormesh(r.snap_times / 60.0, dft, mzt.T, cmap="RdYlBu",
                           vmin=0.0, vmax=72.0, shading="nearest")
        if mzt.shape[0] >= 2:
            c0 = ax.contour(r.snap_times / 60.0, dft, mzt.T, levels=[9.0],
                            colors="k", linewidths=0.7, linestyles="--")
            if c0.levels is not None and len(c0.allsegs[0]):
                ax.clabel(c0, fmt="%g $^\\circ$F", fontsize=6)
        ax.set_title(lab, fontsize=8)
        if r.two_phase_entry:
            ax.plot(r.two_phase_entry["t"] / 60.0,
                    r.two_phase_entry["depth"] / FT, "kx", ms=7)
    axes[0, 0].invert_yaxis()
    for ax in axes[1]:
        ax.set_xlabel("$t$ [min]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Depth $\zeta$ [ft]")
    fig.colorbar(pc, ax=axes, shrink=0.85,
                 label="$T - T_{sat}(p)$ [$^\\circ$F]"
                       "\n(1 K = 1.8 $^\\circ$F)")
    fig.suptitle("CO$_2$ saturation margin $m(\\zeta,t)$ "
                 "($\\times$: flagged dome entry; dashed: 9 $^\\circ$F "
                 "= 5 K)", fontsize=9)
    fig.savefig(FIG / "margin_zt.png")
    plt.close(fig)

    # ------------------------------------------------------ min_margin
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for r, lab, c, ls in ((ch, "uniform, coupled", "tab:blue", "-"),
                          (ah, "uniform, adiabatic", "tab:blue", ":"),
                          (cg, "geothermal, coupled", "tab:red", "-"),
                          (ag, "geothermal, adiabatic", "tab:red", ":")):
        axes[0].plot(r.t_hist / 60.0, r.sat_margin_min * 1.8, c=c, ls=ls,
                     label=lab)
        smd = getattr(r, "sat_margin_depth", None)
        if smd is not None:
            axes[1].plot(r.t_hist / 60.0, smd / FT, c=c, ls=ls)
    axes[0].axhline(0.0, c="k", lw=0.6)
    axes[0].set_ylabel(r"$\min_\zeta\,(T - T_{sat})$ [$^\circ$F]")
    sy(axes[0], "dT_K", "[K]")
    axes[1].set_ylabel(r"Depth of minimum $\zeta$ [ft]")
    sy(axes[1], "L_m", "[m]")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.set_xlabel("$t$ [min]")
        ax.grid(True)
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Minimum saturation margin and its location", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "min_margin.png")
    plt.close(fig)

    # ------------------------------------------------- saturation_paths
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.plot(K2F(eos_sat._T_sat), np.exp(eos_sat._lnp_sat) / PSI, "k-",
            lw=1.2, label="saturation")
    ax.plot(K2F(eos_sat.T_crit), eos_sat.p_crit / PSI, "k*", ms=8)
    for r, lab, c in ((ch, "uniform, coupled", "tab:blue"),
                      (ah, "uniform, adiab.", "cornflowerblue"),
                      (cg, "geothermal, coupled", "tab:red"),
                      (ag, "geothermal, adiab.", "salmon")):
        ax.plot(K2F(r.T[:, -1]), r.p[:, -1] / PSI, c=c, lw=0.9, label=lab)
        ax.plot(K2F(r.T[:, 0]), r.p[:, 0] / PSI, c=c, lw=0.9, ls="--")
        if r.two_phase_entry:
            ax.plot(K2F(r.two_phase_entry["T"]),
                    r.two_phase_entry["p"] / PSI, "x", c=c, ms=7)
    ax.set_xlabel("$T$ [$^\\circ$F]")
    sx(ax, "T_K", "$T$ [K]")
    ax.set_ylabel("$p$ [psi]")
    sy(ax, "p_MPa", "$p$ [MPa]")
    ax.grid(True)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "saturation_paths.png")
    plt.close(fig)

    # --------------------------------------------------------- choke_hem
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for r, lab, c in ((ch, "uniform 300 K", "tab:blue"),
                      (cg, "geothermal", "tab:red")):
        axes[0].plot(r.t_hist / 60.0, r.mdot / LBM, c=c, label=lab)
        ql = getattr(r, "choke_quality", None)
        if ql is not None:
            axes[1].plot(r.t_hist / 60.0, ql, c=c)
    axes[0].set_ylabel(r"$\dot m$ [lbm/s]")
    sy(axes[0], "md_kgps", "[kg/s]")
    axes[1].set_ylabel("HEM throat quality $x_t$ [-]")
    axes[1].set_ylim(0.0, 1.05)
    for ax in axes:
        ax.set_xlabel("$t$ [min]")
        ax.grid(True)
    axes[0].legend(fontsize=7)
    fig.suptitle("HEM choke: discharge and throat quality (gaps: "
                 "single-phase throat)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "choke_hem.png")
    plt.close(fig)

    # ---------------------------------------------------------- wave_geo
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for w, lab, c in ((w_n2_g, "N$_2$", "tab:blue"),
                      (w_co2_g, "CO$_2$", "tab:red")):
        ax.plot(w.snap_times, (w.p[:, 0] - w.p[0, 0]) / PSI, c=c,
                label=lab)
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel(r"$p_{bh} - p_{bh,0}$ [psi]")
    sy(ax, "p_kPa", "[kPa]")
    ax.grid(True)
    ax.legend()
    ax.set_title("Bottomhole arrival of the valve-opening expansion wave "
                 "($\\zeta$ = 2,000 ft)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "wave_geo.png")
    plt.close(fig)

    # ------------------------------------------------- sensors_T_co2_geo
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    cmap = plt.cm.viridis
    depths = (50, 500, 1000, 1500, 1950)
    for i, d in enumerate(depths):
        c = cmap(i / (len(depths) - 1))
        axes[0].plot(cg.sensor_t / 60.0, K2F(cg.sensor_data[f"fluid_{d}ft"]),
                     c=c, label=f"{d} ft")
        k = f"fiber13_{d}ft"
        if cg.sensor_data and k in cg.sensor_data:
            axes[1].plot(cg.sensor_t / 60.0, K2F(cg.sensor_data[k]), c=c)
    axes[0].set_title("bore DTS (fluid)", fontsize=8)
    axes[1].set_title("external fiber, 13-3/8 in casing OD", fontsize=8)
    for ax in axes:
        ax.set_xlabel("$t$ [min]")
        ax.set_ylabel("$T$ [$^\\circ$F]")
        sy(ax, "T_K", "[K]")
        ax.grid(True)
    axes[0].legend(fontsize=6, title="depth $\\zeta$", title_fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "sensors_T_co2_geo.png")
    plt.close(fig)

    # ------------------------------------------------ profiles_co2_geo
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)
    dft = depth_ft(cg)
    cmap = plt.cm.viridis
    times = (0, 120, 600, 1800, 3600)
    for i, tt in enumerate(times):
        k = int(np.argmin(np.abs(cg.snap_times - tt)))
        c = cmap(i / (len(times) - 1))
        kw = dict(color=c, lw=1.0, label=f"{tt} s")
        axes[0].plot(K2F(cg.T[k]), dft, **kw)
        axes[1].plot(cg.p[k] / PSI, dft, **kw)
    for d in depths:
        for ax in axes:
            ax.axhline(d, color="gray", lw=0.5, ls=":")
    axes[0].set_xlabel("$T$ [$^\\circ$F]")
    sx(axes[0], "T_K", "[K]")
    axes[1].set_xlabel("$p$ [psi]")
    sx(axes[1], "p_MPa", "[MPa]")
    axes[0].set_ylabel(r"Depth $\zeta$ [ft]")
    axes[0].invert_yaxis()
    sy(axes[1], "L_m", r"$\zeta$ [m]")
    for ax in axes:
        ax.grid(True)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "profiles_co2_geo.png")
    plt.close(fig)

    # -------------------------------------------- velocity_{n2,co2}
    def velocity_fig(w, r, fname, early_times, late_times, L=609.6):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
        cmap = plt.cm.viridis
        for ax, rr, tt, tfmt in ((axes[0], w, early_times, "{:.1f} s"),
                                 (axes[1], r, late_times, "{:.0f} s")):
            dft = depth_ft(rr, L)
            for i, t in enumerate(tt):
                k = int(np.argmin(np.abs(rr.snap_times - t)))
                c = cmap(i / (len(tt) - 1))
                ax.plot(rr.u[k] / FT, dft, color=c, lw=1.0,
                        label=tfmt.format(rr.snap_times[k]))
            ax.set_xlabel("$u$ [ft/s]")
            sx(ax, "u_mps", "[m/s]")
            ax.grid(True)
            ax.legend(fontsize=6)
        axes[0].set_ylabel(r"Depth $\zeta$ [ft]")
        axes[0].invert_yaxis()
        sy(axes[1], "L_m", r"$\zeta$ [m]")
        axes[0].set_title("early acoustic phase", fontsize=8)
        axes[1].set_title("quasi-static discharge", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / fname)
        plt.close(fig)

    velocity_fig(wave_n2_u, rn_u, "velocity_n2.png",
                 (0.5, 1.2, 2.2, 3.0), (120, 600, 1800, 3000))
    velocity_fig(w_co2_g, cg, "velocity_co2.png",
                 (0.8, 1.8, 3.2, 4.5), (120, 600, 1800, 3000))

print("done", flush=True)
