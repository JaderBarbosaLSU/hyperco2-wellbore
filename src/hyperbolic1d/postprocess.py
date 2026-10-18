"""Post-processing: fiber-optic sensors, publication figures, CSV export.

Two levels of output are supported, per the project requirements:

* interactive/animated diagnostics (see :mod:`hyperbolic1d.animate`), and
* clean, publication-quality static figures plus tabular (CSV) export of
  every quantity needed for later comparison with the fiber-optic
  measurements of the LSU CO2 research well.

Fiber-optic sensors
-------------------
The well instrumentation includes distributed fibers in the bore/tubing
region and an external fiber on the 13-3/8" intermediate casing, plus
discrete P&T gauges. These are represented by :class:`FiberSensor`
objects carrying a depth (below the wellhead) and a radial station:

* ``location="fluid"``      — samples the flowing-fluid cell (T, p, rho, u);
* ``location="casing_in"``  — innermost solid node (7-5/8" CRA casing);
* ``location="casing_ext"`` — node nearest the 13-3/8" casing (external
  fiber position);
* ``location="cement"``     — node nearest the mid-cement radius;
* ``location="formation"``  — node nearest ``r`` (required) in the rock;
* explicit ``r=...``        — node nearest that radius.

Solid stations require the conjugate wall model
(:class:`hyperbolic1d.thermal.FormationCoupledWall`); with a stateless
wall model they are skipped with a warning. A :class:`SensorArray` is
passed to ``WellboreModel.solve(sensors=...)``; sampled histories are
attached to the result as ``result.sensor_t`` / ``result.sensor_data``.
"""

from __future__ import annotations

import csv
import pathlib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "FiberSensor",
    "SensorArray",
    "lsu_fiber_sensors",
    "pub_style",
    "plot_sensor_histories",
    "plot_profiles_at_times",
    "plot_velocity_publication",
    "plot_wellhead_histories",
    "export_csv",
]

FT = 0.3048


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FiberSensor:
    """One sampling station: depth below the wellhead + radial location."""

    name: str
    depth: float                   # m below the wellhead
    location: str = "fluid"        # fluid|casing_in|casing_ext|cement|formation
    r: float | None = None         # explicit radius [m] (optional)


@dataclass
class SensorArray:
    """Set of sensors; bound to a model grid inside the solver.

    The wellbore solver calls :meth:`bind` once and :meth:`sample` at the
    history-recording cadence. Fluid sensors record (T, p, rho, u); solid
    sensors record the temperature of the nearest radial node.
    """

    sensors: list = field(default_factory=list)
    _bound: bool = field(default=False, repr=False)

    def bind(self, zc, L, wall_model):
        self._L = L
        self._zc = zc
        self._wall = wall_model
        # NOTE: keyed on 'advance' (like the solver). 'radial_temperatures'
        # is a property that raises before the network is built, which
        # would make hasattr() return False here.
        self._conjugate = hasattr(wall_model, "advance")
        self._fluid, self._solid = [], []
        for s in self.sensors:
            z = L - s.depth                     # cell coordinate (z up)
            j = int(np.clip(np.argmin(np.abs(zc - z)), 0, zc.size - 1))
            if s.location == "fluid":
                self._fluid.append((s, j))
            else:
                if not self._conjugate:
                    warnings.warn(
                        f"sensor {s.name!r} at a solid station skipped: "
                        "wall model has no radial temperature field")
                    continue
                self._solid.append((s, j, None))   # node idx resolved lazily
        self._rows = {s.name: [] for s, *_ in self._fluid}
        for s, *_ in self._solid:
            self._rows[s.name] = []
        self._fluid_extra = {s.name: {"p": [], "rho": [], "u": []}
                             for s, _ in self._fluid}
        self._bound = True
        return self

    def _resolve_solid_nodes(self):
        rr = self._wall.radial_positions
        resolved = []
        for s, j, _ in self._solid:
            if s.location == "casing_in":
                k = 0
            elif s.location == "casing_ext":
                # node nearest the 13-3/8" casing wall (external fiber)
                k = int(np.argmin(np.abs(rr - 0.1634)))
            elif s.location == "cement":
                k = int(np.argmin(np.abs(rr - 0.20)))
            elif s.location == "formation":
                if s.r is None:
                    raise ValueError(f"sensor {s.name!r}: formation "
                                     "station requires r=")
                k = int(np.argmin(np.abs(rr - s.r)))
            elif s.r is not None:
                k = int(np.argmin(np.abs(rr - s.r)))
            else:
                raise ValueError(f"sensor {s.name!r}: unknown location "
                                 f"{s.location!r}")
            resolved.append((s, j, k))
        self._solid = resolved

    def sample(self, T, rho, u, p):
        for s, j in self._fluid:
            self._rows[s.name].append(T[j])
            self._fluid_extra[s.name]["p"].append(p[j])
            self._fluid_extra[s.name]["rho"].append(rho[j])
            self._fluid_extra[s.name]["u"].append(u[j])
        if self._solid:
            if self._solid[0][2] is None:
                # the conjugate wall builds its network on the first flow
                # step, before the first history record; resolve nodes now
                self._resolve_solid_nodes()
            Tw = self._wall.radial_temperatures
            for s, j, k in self._solid:
                self._rows[s.name].append(Tw[j, k])

    def finalize(self):
        data = {k: np.array(v) for k, v in self._rows.items()}
        extra = {k: {q: np.array(a) for q, a in d.items()}
                 for k, d in self._fluid_extra.items()}
        meta = {}
        for s, j in self._fluid:
            meta[s.name] = dict(depth=s.depth, location="fluid",
                                z=float(self._zc[j]))
        for s, j, k in self._solid:
            meta[s.name] = dict(
                depth=s.depth, location=s.location, z=float(self._zc[j]),
                r=float(self._wall.radial_positions[k]) if k is not None else None)
        return data, extra, meta


def lsu_fiber_sensors(depths_ft=(50, 500, 1000, 1500, 1950),
                      fluid=True, casing_external=True):
    """Default sensor set for the LSU well: distributed-fiber sampling
    points in the bore and on the external fiber of the 13-3/8" casing,
    at the given depths [ft below wellhead]."""
    sensors = []
    for d in depths_ft:
        if fluid:
            sensors.append(FiberSensor(f"fluid_{d}ft", d * FT, "fluid"))
        if casing_external:
            sensors.append(FiberSensor(f"fiber13_{d}ft", d * FT, "casing_ext"))
    return SensorArray(sensors)


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
@contextmanager
def pub_style(column: str = "double"):
    """Matplotlib context for publication figures.

    ``column='single'`` -> 3.5 in wide; ``'double'`` -> 7.0 in.
    """
    import matplotlib as mpl

    w = 3.5 if column == "single" else 7.0
    rc = {
        "figure.figsize": (w, w * 0.62),
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8.5,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.1,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
    with mpl.rc_context(rc):
        yield


def _depth(result):
    L = result.z[-1] + (result.z[1] - result.z[0]) / 2
    return L - result.z          # depth below wellhead [m]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_sensor_histories(result, filename=None, locations=None,
                          t_unit="s", title=None):
    """T(t) at the sensor stations, grouped by radial location."""
    import matplotlib.pyplot as plt

    meta = result.sensor_meta
    names = [n for n in result.sensor_data
             if locations is None or meta[n]["location"] in locations]
    locs = sorted({meta[n]["location"] for n in names})
    ts = result.sensor_t / (60.0 if t_unit == "min" else 1.0)

    with pub_style("double"):
        fig, axes = plt.subplots(1, len(locs), figsize=(3.5 * len(locs), 2.6),
                                 sharex=True)
        if len(locs) == 1:
            axes = [axes]
        cmap = plt.cm.viridis
        for ax, loc in zip(axes, locs):
            group = [n for n in names if meta[n]["location"] == loc]
            group.sort(key=lambda n: meta[n]["depth"])
            for i, n in enumerate(group):
                c = cmap(i / max(len(group) - 1, 1))
                ax.plot(ts, result.sensor_data[n], color=c,
                        label=f"{meta[n]['depth']/FT:.0f} ft")
            ax.set_xlabel(f"$t$ [{t_unit}]")
            ax.set_ylabel("$T$ [K]")
            ax.set_title({"fluid": "Fluid (bore DTS)",
                          "casing_ext": 'External fiber (13-3/8" casing)',
                          "casing_in": '7-5/8" casing',
                          "cement": "Cement",
                          "formation": "Formation"}.get(loc, loc))
            ax.grid(True)
            ax.legend(title="depth", ncol=1)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        if filename:
            fig.savefig(filename)
            plt.close(fig)
            return filename
        return fig


def plot_profiles_at_times(result, times, fields=("T", "p"),
                           filename=None, sensor_depths_ft=None):
    """Vertical-depth-oriented profiles at selected times (snapshots are
    interpolated to the nearest stored instant). Depth increases downward
    (petroleum convention)."""
    import matplotlib.pyplot as plt

    d = _depth(result) / FT      # ft
    labels = {"T": "$T$ [K]", "p": "$p$ [MPa]",
              "rho": r"$\rho$ [kg m$^{-3}$]", "u": "$u$ [m s$^{-1}$]"}
    scale = {"T": 1.0, "p": 1e-6, "rho": 1.0, "u": 1.0}

    with pub_style("double"):
        fig, axes = plt.subplots(1, len(fields),
                                 figsize=(2.6 * len(fields), 3.6),
                                 sharey=True)
        if len(fields) == 1:
            axes = [axes]
        cmap = plt.cm.viridis
        for ax, f in zip(axes, fields):
            data = getattr(result, f)
            for i, t in enumerate(times):
                k = int(np.argmin(np.abs(result.snap_times - t)))
                c = cmap(i / max(len(times) - 1, 1))
                ax.plot(data[k] * scale[f], d, color=c,
                        label=f"{result.snap_times[k]:.0f} s")
            ax.set_xlabel(labels[f])
            ax.grid(True)
            if sensor_depths_ft is not None:
                for sd in sensor_depths_ft:
                    ax.axhline(sd, color="crimson", lw=0.5, ls=(0, (2, 2)))
        axes[0].set_ylabel("Depth [ft]")
        axes[0].invert_yaxis()
        axes[0].legend(title="$t$")
        if sensor_depths_ft is not None:
            axes[-1].plot([], [], color="crimson", lw=0.5, ls=(0, (2, 2)),
                          label="fiber sensors")
            axes[-1].legend()
        fig.tight_layout()
        if filename:
            fig.savefig(filename)
            plt.close(fig)
            return filename
        return fig


def plot_velocity_publication(result_early, result_late, filename=None,
                              early_times=(0.5, 1.2, 2.2, 3.0),
                              late_times=(100, 300, 600, 1200)):
    """Two-panel velocity figure separating the early acoustic phase from
    the quasi-static discharge, with a wellhead-zoom inset on the late
    panel. Numerical content is shown as computed (no smoothing)."""
    import matplotlib.pyplot as plt

    with pub_style("double"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
        cmap = plt.cm.viridis
        for i, t in enumerate(early_times):
            k = int(np.argmin(np.abs(result_early.snap_times - t)))
            ax1.plot(result_early.z, result_early.u[k],
                     color=cmap(i / max(len(early_times) - 1, 1)),
                     label=f"{result_early.snap_times[k]:.1f} s")
        ax1.set_xlabel("$z$ [m] (0 = bottom)")
        ax1.set_ylabel("$u$ [m s$^{-1}$]")
        ax1.set_title("(a) acoustic phase", loc="left")
        ax1.grid(True)
        ax1.legend(title="$t$")

        for i, t in enumerate(late_times):
            k = int(np.argmin(np.abs(result_late.snap_times - t)))
            ax2.plot(result_late.z, result_late.u[k],
                     color=cmap(i / max(len(late_times) - 1, 1)),
                     label=f"{result_late.snap_times[k]:.0f} s")
        ax2.set_xlabel("$z$ [m]")
        ax2.set_ylabel("$u$ [m s$^{-1}$]")
        ax2.set_title("(b) quasi-static discharge", loc="left")
        ax2.grid(True)
        ax2.legend(title="$t$")
        fig.tight_layout()
        if filename:
            fig.savefig(filename)
            plt.close(fig)
            return filename
        return fig


def plot_wellhead_histories(result, filename=None, t_unit="min"):
    """Wellhead p, T, u_face, and mdot vs time (publication layout)."""
    import matplotlib.pyplot as plt

    ts = result.t_hist / (60.0 if t_unit == "min" else 1.0)
    with pub_style("double"):
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
        axes[0, 0].plot(ts, result.p_wellhead / 1e6)
        axes[0, 0].set_ylabel("$p_{wh}$ [MPa]")
        axes[0, 1].plot(ts, result.T_wellhead)
        axes[0, 1].set_ylabel("$T_{wh}$ [K]")
        axes[1, 0].plot(ts, result.u_wellhead_face)
        axes[1, 0].set_ylabel("$u_{f}$ [m s$^{-1}$]")
        axes[1, 1].plot(ts, result.mdot)
        axes[1, 1].set_ylabel(r"$\dot m$ [kg s$^{-1}$]")
        for ax in axes[1]:
            ax.set_xlabel(f"$t$ [{t_unit}]")
        for ax in axes.flat:
            ax.grid(True)
        fig.tight_layout()
        if filename:
            fig.savefig(filename)
            plt.close(fig)
            return filename
        return fig


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def export_csv(result, outdir, profile_times=None):
    """Write the tabular outputs needed for experimental comparison.

    Files written to ``outdir``:

    * ``wellhead_history.csv`` — t, p_wh, T_wh, u_face, mdot, mass, choked
    * ``sensors_T.csv``        — t + one column per sensor (K), with a
      metadata header row (location, depth, z, r)
    * ``sensors_fluid_pru.csv`` — t + p/rho/u columns per fluid sensor
    * ``profiles_t<+>.csv``    — depth, z, p, T, rho, u at selected times
    """
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # wellhead history
    with open(out / "wellhead_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "p_wellhead_Pa", "T_wellhead_K",
                    "u_face_m_s", "mdot_kg_s", "mass_kg", "choked"])
        uface = getattr(result, "u_wellhead_face",
                        np.full_like(result.t_hist, np.nan))
        for i in range(result.t_hist.size):
            w.writerow([f"{result.t_hist[i]:.6g}",
                        f"{result.p_wellhead[i]:.8g}",
                        f"{result.T_wellhead[i]:.8g}",
                        f"{uface[i]:.8g}",
                        f"{result.mdot[i]:.8g}",
                        f"{result.mass[i]:.8g}",
                        int(result.choked[i])])

    # sensors
    if getattr(result, "sensor_data", None):
        meta = result.sensor_meta
        names = list(result.sensor_data)
        with open(out / "sensors_T.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# location"] + [meta[n]["location"] for n in names])
            w.writerow(["# depth_m"] + [f"{meta[n]['depth']:.4g}" for n in names])
            w.writerow(["# r_m"] + [f"{meta[n].get('r', '')}" for n in names])
            w.writerow(["t_s"] + names)
            for i in range(result.sensor_t.size):
                w.writerow([f"{result.sensor_t[i]:.6g}"] +
                           [f"{result.sensor_data[n][i]:.7g}" for n in names])
        fe = getattr(result, "sensor_fluid_extra", {})
        if fe:
            with open(out / "sensors_fluid_pru.csv", "w", newline="") as f:
                w = csv.writer(f)
                cols = []
                for n in fe:
                    cols += [f"{n}:p_Pa", f"{n}:rho_kg_m3", f"{n}:u_m_s"]
                w.writerow(["t_s"] + cols)
                npts = result.sensor_t.size
                for i in range(npts):
                    row = [f"{result.sensor_t[i]:.6g}"]
                    for n in fe:
                        row += [f"{fe[n]['p'][i]:.8g}",
                                f"{fe[n]['rho'][i]:.7g}",
                                f"{fe[n]['u'][i]:.7g}"]
                    w.writerow(row)

    # profiles at selected times
    if profile_times is None:
        profile_times = result.snap_times
    L = result.z[-1] + (result.z[1] - result.z[0]) / 2
    for t in profile_times:
        k = int(np.argmin(np.abs(result.snap_times - t)))
        tt = result.snap_times[k]
        with open(out / f"profiles_t{tt:07.1f}s.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["depth_m", "z_m", "p_Pa", "T_K",
                        "rho_kg_m3", "u_m_s"])
            for j in range(result.z.size):
                w.writerow([f"{L - result.z[j]:.5g}", f"{result.z[j]:.5g}",
                            f"{result.p[k][j]:.8g}", f"{result.T[k][j]:.7g}",
                            f"{result.rho[k][j]:.7g}", f"{result.u[k][j]:.7g}"])
    return out
