"""Validation of the HEM wellbore solver against the SINTEF ECCSEL
depressurization experiments (Munkejord et al. 2020, Energy 211:118560;
raw data Zenodo 10.5281/zenodo.3928227, CC BY 4.0).

Rig: horizontal 61.668 m SS316 tube, ID 40.8 mm, Ra 0.2-0.3 um, one end
closed, full-bore rupture-disc opening at the other, discharge to
atmosphere. Mapping onto the wellbore solver: gravity off (horizontal),
"depth" = pipe length with the closed end at z = 0 and the open end at
z = L; the full-bore opening is the choke boundary with the pipe's own
diameter and cd = 1 (the quasi-steady isentropic exit then chokes at
the pipe area, i.e. the standard full-bore HEM outflow). Walls
adiabatic -- valid for the decompression-wave window (< 1 s) and an
increasingly poor approximation at long times (noted on the figures).

Usage: python run_eccsel_validation.py [Exp03|Exp08|...] [--n 600]
Data files are read from DATA_DIR (xlsx as published on Zenodo) and
cached as npz. Outputs -> results/eccsel/.
"""

import pathlib
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperbolic1d.postprocess import FiberSensor, SensorArray
from hyperbolic1d.realfluid import SpanWagnerCO2
from hyperbolic1d.eos import NITROGEN
from hyperbolic1d.wellbore import ChokeValve, WellboreModel, WellGeometry

DATA_DIR = pathlib.Path("/mnt/user-data/uploads/CO2 - HyperCO2/SINTEF DATA")
RES = pathlib.Path(__file__).resolve().parents[1] / "results" / "eccsel"
RES.mkdir(parents=True, exist_ok=True)

L_PIPE = 61.668
D_PIPE = 0.0408

TESTS = {
    "Exp03": dict(file="Exp03_CO2_40.4bar_10.2C.xlsx", fluid="co2",
                  p0=4.04e6, T0=283.35),
    "Exp04": dict(file="Exp04_CO2_125.4bar_21.1C.xlsx", fluid="co2",
                  p0=12.54e6, T0=294.25),
    "Exp06": dict(file="Exp06_CO2_104.0bar_40.0C.xlsx", fluid="co2",
                  p0=10.40e6, T0=313.15),
    "Exp08": dict(file="Exp08_CO2_122.2bar_24.6C.xlsx", fluid="co2",
                  p0=12.22e6, T0=297.75),
    "Exp11": dict(file="Exp11_N2_51.3bar_10.0C.xlsx", fluid="n2",
                  p0=5.13e6, T0=283.15),
}


# ---------------------------------------------------------------- data
def load_test(key):
    """Extract (tPT, {PTname: (x_m, p_Pa)}, tTT, {TTname: (x_m, T_K)})
    from the published xlsx; cached as npz."""
    cache = RES / f"{key}_data.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return (d["tPT"], d["pt"].item(), d["tTT"], d["tt"].item())
    import openpyxl
    wb = openpyxl.load_workbook(DATA_DIR / TESTS[key]["file"],
                                read_only=True)
    ws = wb["Data"]
    rows = ws.iter_rows(values_only=True)
    names = next(rows)
    pos_row = next(rows)
    next(rows)                                   # units row
    cols = list(zip(*[r for r in rows]))
    wb.close()

    def col(name):
        return np.array(
            [v for v in cols[names.index(name)] if v is not None],
            dtype=float)

    tPT = col("tPT")
    pt = {}
    tt = {}
    tTT = col("tTT") if "tTT" in names else None
    for k, nm in enumerate(names):
        if nm is None:
            continue
        m = re.match(r"(PT|TT)(\d+)$", str(nm))
        if not m or str(nm) in ("PT200B", "TT200"):
            continue
        pos = re.search(r"([\d.]+)\s*m", str(pos_row[k]) or "")
        if not pos:
            continue
        x = float(pos.group(1))
        v = col(nm)
        if m.group(1) == "PT":
            pt[nm] = (x, v[:tPT.size] * 1e5)          # bar -> Pa
        else:
            # keep only TT channels on the shared tTT time base (the
            # TT219+ group runs on its own per-sensor time columns)
            if tTT is not None and abs(v.size - tTT.size) <= 1:
                tt[nm] = (x, v[:tTT.size] + 273.15)   # C -> K
    np.savez(cache, tPT=tPT, pt=np.array(pt, dtype=object),
             tTT=tTT, tt=np.array(tt, dtype=object))
    return tPT, pt, tTT, tt


# --------------------------------------------------------------- model
def run_model(key, n_cells=600, t_end=10.0):
    ckpt = RES / f"{key}_model_n{n_cells}_t{t_end:g}.pkl"
    if ckpt.exists():
        with open(ckpt, "rb") as f:
            return pickle.load(f)
    cfg = TESTS[key]
    if cfg["fluid"] == "co2":
        eos = SpanWagnerCO2(two_phase=True)
        hem = True
    else:
        eos = NITROGEN
        hem = False
    _, pt, _, _ = load_test(key)
    sensors = SensorArray([FiberSensor(nm, depth=x, location="fluid")
                           for nm, (x, _) in sorted(pt.items())])
    m = WellboreModel(
        geometry=WellGeometry(depth=L_PIPE, diameter=D_PIPE,
                              roughness=0.25e-6),
        choke=ChokeValve(diameter=D_PIPE, cd=1.0, p_back=101325.0),
        eos=eos, gravity=False, friction=True, n_cells=n_cells,
        allow_two_phase=hem)
    r = m.solve(cfg["p0"], cfg["T0"], t_end, init="uniform",
                n_snapshots=201, hist_stride=4, sensors=sensors)
    print(f"[{key}] {r.n_steps} steps, {r.wall_time:.0f} s wall; "
          f"budgets m={r.mass_balance_error:.0e} "
          f"E={r.energy_balance_error:.0e}; "
          f"entry={r.two_phase_entry}; table_exit={r.table_exit}",
          flush=True)
    with open(ckpt, "wb") as f:
        pickle.dump(r, f)
    return r


# -------------------------------------------------------------- figures
def compare(key, r_wave, r_long,
            stations=("PT204", "PT208", "PT212", "PT214"),
            t_wave=0.6, t_long=10.0):
    tPT, pt, tTT, tt = load_test(key)

    for r, tmax, tag in ((r_wave, t_wave, "wave"),
                         (r_long, t_long, "long")):
        ts = r.sensor_t
        ex = r.sensor_fluid_extra
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True)
        for ax, nm in zip(axes.flat, stations):
            if nm not in pt:
                continue
            x, pdat = pt[nm]
            msk = (tPT >= -0.01) & (tPT <= tmax)
            ax.plot(tPT[msk], pdat[msk] / 1e6, "k-", lw=0.9,
                    label="experiment")
            if nm in ex:
                ax.plot(ts, ex[nm]["p"] / 1e6, "r--", lw=1.1,
                        label="HEM model")
            ax.set_xlim(-0.005 if tmax < 1 else -0.1, tmax)
            ax.set_title(f"{nm}  (x = {x:.2f} m from open end)",
                         fontsize=9)
            ax.grid(alpha=0.3)
        for ax in axes[1]:
            ax.set_xlabel("$t$ [s]")
        for ax in axes[:, 0]:
            ax.set_ylabel("$p$ [MPa]")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"ECCSEL {key}: p(t), model (n={r.z.size}, adiabatic, "
            "full-bore HEM outflow) vs experiment", fontsize=10)
        fig.tight_layout()
        fig.savefig(RES / f"{key}_p_{tag}.png", dpi=150)
        plt.close(fig)

    # temperature at two stations (long window; adiabatic caveat)
    r = r_long
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, nm in zip(axes, ("TT206", "TT212")):
        if nm in tt and tTT is not None:
            x, Td = tt[nm]
            msk = (tTT >= -0.1) & (tTT <= t_long)
            ax.plot(tTT[msk], Td[msk], "k-", lw=0.9, label="experiment")
            k = np.argmin(np.abs((L_PIPE - r.z) - x))
            ax.plot(r.snap_times, r.T[:, k], "r--", lw=1.1,
                    label="HEM model (adiabatic)")
            ax.set_title(f"{nm} (x = {x:.2f} m)", fontsize=9)
        ax.set_xlabel("$t$ [s]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("$T$ [K]")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"ECCSEL {key}: fluid temperature (model is adiabatic; "
                 "late-time deviation expected)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / f"{key}_T.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    keys = [a for a in sys.argv[1:] if a in TESTS] or ["Exp03", "Exp08"]
    n = 600
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    for key in keys:
        r_wave = run_model(key, n_cells=n, t_end=1.2)      # wave window
        r_long = run_model(key, n_cells=n // 2, t_end=10.0)  # bulk window
        compare(key, r_wave, r_long)
        print(f"[{key}] figures written", flush=True)
    print("done", flush=True)
