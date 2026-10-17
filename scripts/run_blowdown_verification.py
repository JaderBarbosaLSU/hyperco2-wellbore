"""Verification battery + demonstration run for the wellbore blowdown model.

Case: N2 (ideal gas), 2,000-ft (609.6 m) vertical well, ID = 0.100 m,
initially uniform 13.79 MPa (2,000 psi) / 300 K, closed bottom, wellhead
choke d = 12.7 mm (Cd = 0.84) discharging to 101.325 kPa.

Outputs -> results/blowdown_*.png, results/blowdown_verification.json
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperbolic1d.eos import NITROGEN
from hyperbolic1d.wellbore import (ChokeValve, WellboreModel, WellGeometry,
                                   vessel_blowdown_ode)

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
RES.mkdir(exist_ok=True)

P0, T0 = 1.379e7, 300.0
T_END = 600.0
geom = WellGeometry()
choke = ChokeValve()
report = {}

# ---------------------------------------------------------------------------
# 1. Quasi-static verification vs 0-D adiabatic-vessel ODE (gravity off)
# ---------------------------------------------------------------------------
print("[1] 0-D vessel comparison (gravity off)", flush=True)
m = WellboreModel(geometry=geom, choke=choke, eos=NITROGEN,
                  gravity=False, n_cells=200)
r0 = m.solve(P0, T0, T_END, n_snapshots=2)
ref = vessel_blowdown_ode(NITROGEN, geom.volume, choke, P0, T0, T_END)

p_num = np.interp(ref["t"][1:], r0.t_hist, r0.p_wellhead)
md_num = np.interp(ref["t"][1:], r0.t_hist, r0.mdot)
mask = ref["mdot"][1:] > 0.01 * ref["mdot"][1]
err_p = float(np.max(np.abs(p_num - ref["p"][1:]) / ref["p"][1:]))
err_md = float(np.max(np.abs(md_num[mask] - ref["mdot"][1:][mask])
                      / ref["mdot"][1:][mask]))
report["vessel_ode"] = {
    "max_rel_err_p_wellhead": err_p,
    "max_rel_err_mdot": err_md,
    "mass_balance_error": r0.mass_balance_error,
    "energy_balance_error": r0.energy_balance_error,
}
print(f"    max rel err: p={err_p:.3e}, mdot={err_md:.3e}; "
      f"budgets: m={r0.mass_balance_error:.1e}, E={r0.energy_balance_error:.1e}",
      flush=True)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].plot(ref["t"], ref["p"] / 1e6, "k-", lw=2, label="0-D vessel ODE")
axes[0].plot(r0.t_hist, r0.p_wellhead / 1e6, "r--", label="1-D FV (wellhead)")
axes[0].set_ylabel("Pressure [MPa]")
axes[1].plot(ref["t"], ref["mdot"], "k-", lw=2, label="0-D vessel ODE")
axes[1].plot(r0.t_hist, r0.mdot, "r--", label="1-D FV")
axes[1].set_ylabel("Wellhead mass flow rate [kg/s]")
axes[2].plot(ref["t"], ref["mass"], "k-", lw=2, label="0-D vessel ODE")
axes[2].plot(r0.t_hist, r0.mass, "r--", label="1-D FV")
axes[2].set_ylabel("Mass in well [kg]")
for ax in axes:
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Verification: 1-D FV (no gravity) vs lumped adiabatic-vessel ODE")
fig.tight_layout()
fig.savefig(RES / "blowdown_vs_vessel_ode.png", dpi=150)

# ---------------------------------------------------------------------------
# 2. Grid independence (60-s window, gravity on)
# ---------------------------------------------------------------------------
print("[2] grid independence", flush=True)
grids = [100, 200, 400, 800]
md_curves = {}
t_probe = np.linspace(1.0, 60.0, 200)
for n in grids:
    m = WellboreModel(geometry=geom, choke=choke, gravity=True, n_cells=n)
    r = m.solve(P0, T0, 60.0, n_snapshots=2)
    md_curves[n] = np.interp(t_probe, r.t_hist, r.mdot)
    print(f"    n={n}: {r.n_steps} steps, {r.wall_time:.1f}s", flush=True)
grid_err = {
    n: float(np.max(np.abs(md_curves[n] - md_curves[800]) / md_curves[800]))
    for n in grids[:-1]
}
report["grid_independence_mdot_vs_n800"] = grid_err
print(f"    max rel dev vs n=800: {grid_err}", flush=True)

# ---------------------------------------------------------------------------
# 3. CFL sensitivity
# ---------------------------------------------------------------------------
print("[3] CFL sensitivity", flush=True)
cfl_err = {}
base = None
for cfl in (0.9, 0.5, 0.2):
    m = WellboreModel(geometry=geom, choke=choke, gravity=True,
                      n_cells=200, cfl=cfl)
    r = m.solve(P0, T0, 60.0, n_snapshots=2)
    cur = np.interp(t_probe, r.t_hist, r.mdot)
    if base is None:
        base = cur
    else:
        cfl_err[cfl] = float(np.max(np.abs(cur - base) / base))
report["cfl_sensitivity_mdot_vs_cfl0.9"] = cfl_err
print(f"    max rel dev vs CFL=0.9: {cfl_err}", flush=True)

# ---------------------------------------------------------------------------
# 4. Choke sensitivity, incl. the open-pipe limit
# ---------------------------------------------------------------------------
print("[4] choke sweep", flush=True)
sweep = {}
for d, label in [(0.00635, "1/4 in"), (0.0127, "1/2 in"),
                 (0.0254, "1 in"), (0.0508, "2 in"),
                 (geom.diameter, "open pipe")]:
    cd = 1.0 if label == "open pipe" else 0.84
    m = WellboreModel(geometry=geom,
                      choke=ChokeValve(diameter=d, cd=cd),
                      gravity=True, n_cells=200)
    r = m.solve(P0, T0, 20.0, n_snapshots=2)
    sweep[label] = (r.t_hist, r.mdot, r.mass_balance_error)
    print(f"    {label}: mdot(0+)={r.mdot[0]:.2f} kg/s, "
          f"mass err={r.mass_balance_error:.1e}", flush=True)
report["choke_sweep_mdot0"] = {k: float(v[1][0]) for k, v in sweep.items()}
report["choke_sweep_mass_err"] = {k: float(v[2]) for k, v in sweep.items()}

fig, ax = plt.subplots(figsize=(7, 4.5))
for label, (t, md, _) in sweep.items():
    ax.plot(t, md, label=label)
ax.set_xlabel("t [s]")
ax.set_ylabel("Wellhead mass flow rate [kg/s]")
ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.legend(title="choke bore", fontsize=8)
ax.set_title("Choke sensitivity (N$_2$, 13.79 MPa / 300 K, first 20 s)")
fig.tight_layout()
fig.savefig(RES / "blowdown_choke_sweep.png", dpi=150)

# ---------------------------------------------------------------------------
# 5. Demonstration run: full blowdown with gravity, profiles + histories
# ---------------------------------------------------------------------------
print("[5] demonstration run (gravity on, full blowdown)", flush=True)
m = WellboreModel(geometry=geom, choke=choke, gravity=True, n_cells=400)
r = m.solve(P0, T0, T_END, n_snapshots=7)
report["demo"] = {
    "mass_balance_error": r.mass_balance_error,
    "energy_balance_error": r.energy_balance_error,
    "n_steps": r.n_steps,
    "wall_time_s": round(r.wall_time, 1),
    "T_wellhead_min_K": float(r.T_wellhead.min()),
    "choked_until_s": float(r.t_hist[r.choked][-1]) if r.choked.any() else 0.0,
}
print(f"    {r.n_steps} steps, {r.wall_time:.0f}s wall; "
      f"T_wh min={r.T_wellhead.min():.1f} K; budgets m={r.mass_balance_error:.1e} "
      f"E={r.energy_balance_error:.1e}", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
cmap = plt.cm.viridis
for i, ts in enumerate(r.snap_times):
    c = cmap(i / max(len(r.snap_times) - 1, 1))
    kw = dict(color=c, lw=1.2, label=f"t = {ts:.0f} s")
    axes[0, 0].plot(r.z, r.p[i] / 1e6, **kw)
    axes[0, 1].plot(r.z, r.T[i], **kw)
    axes[1, 0].plot(r.z, r.rho[i], **kw)
    axes[1, 1].plot(r.z, r.u[i], **kw)
for ax, lab in zip(axes.flat, ["Pressure [MPa]", "Temperature [K]",
                               "Density [kg/m$^3$]", "Velocity [m/s]"]):
    ax.set_xlabel("z [m]  (0 = bottom, 609.6 = wellhead)")
    ax.set_ylabel(lab)
    ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=7)
fig.suptitle("N$_2$ blowdown of a 2,000-ft well — 12.7 mm choke, adiabatic")
fig.tight_layout()
fig.savefig(RES / "blowdown_profiles.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].plot(r.t_hist, r.mdot)
axes[0].set_ylabel("Wellhead mass flow rate [kg/s]")
ch = r.choked
if ch.any() and (~ch).any():
    tch = r.t_hist[ch][-1]
    axes[0].axvline(tch, color="r", ls=":", lw=1)
    axes[0].annotate("choked | unchoked", (tch, 0.5 * r.mdot.max()),
                     rotation=90, fontsize=8, color="r")
axes[1].plot(r.t_hist, r.mass)
axes[1].set_ylabel("Mass in well [kg]")
axes[2].plot(r.t_hist, r.T_wellhead)
axes[2].set_ylabel("Wellhead fluid temperature [K]")
for ax in axes:
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3)
fig.suptitle("Blowdown histories (adiabatic, gravity on)")
fig.tight_layout()
fig.savefig(RES / "blowdown_histories.png", dpi=150)

with open(RES / "blowdown_verification.json", "w") as f:
    json.dump(report, f, indent=2)
print("done", flush=True)
