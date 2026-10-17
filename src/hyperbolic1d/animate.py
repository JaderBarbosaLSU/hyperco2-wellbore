"""Animation utilities for wellbore blowdown results.

Produces MP4 (ffmpeg) or GIF (pillow) animations of the p, T, rho, u
profiles along the well, with a synchronized wellhead mass-flow-rate
trace, from densely sampled snapshots. Also provides an x-t (distance-
time) pressure diagram for wave-propagation analysis.

Snapshot density is the controlling choice: resolving the acoustic wave
requires sampling at a fraction of the transit time L/a (~1.7 s for the
N2 well), i.e. hundreds of snapshots over the first seconds, whereas the
bulk depressurization is well represented by a frame every few seconds.
Use two runs (early-time and full-blowdown) rather than one oversampled
run.
"""

from __future__ import annotations

import numpy as np

__all__ = ["animate_blowdown", "xt_diagram"]


def animate_blowdown(result, filename: str, fps: int = 25,
                     title: str = "", mdot_inset: bool = True,
                     dpi: int = 110, time_label: str = "t = {t:.2f} s",
                     p_ref=None, p_label: str = "Pressure [MPa]",
                     p_scale: float = 1e-6):
    """Render an animation from a BlowdownResult with dense snapshots.

    ``result`` must have been produced with a large ``n_snapshots`` (one
    per frame). Writes MP4 if ``filename`` ends in .mp4 (ffmpeg), GIF if
    .gif (pillow). Returns the matplotlib FuncAnimation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    z = result.z
    ns = len(result.snap_times)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    (axp, axT), (axr, axu) = axes

    lines = {}
    for ax, key, lab, scale in ((axp, "p", p_label, p_scale),
                                (axT, "T", "Temperature [K]", 1.0),
                                (axr, "rho", "Density [kg/m$^3$]", 1.0),
                                (axu, "u", "Velocity [m/s]", 1.0)):
        data = getattr(result, key)
        if key == "p" and p_ref is not None:
            data = data - p_ref[None, :]
        data = data * scale
        (ln,) = ax.plot(z, data[0], lw=1.4)
        ax.set_xlim(z[0], z[-1])
        lo, hi = data.min(), data.max()
        pad = 0.05 * (hi - lo + 1e-12)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("z [m]  (0 = bottom)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        lines[key] = (ln, data)

    ttxt = axp.set_title(time_label.format(t=0.0), loc="left", fontsize=10)
    if title:
        fig.suptitle(title)

    if mdot_inset and result.t_hist.size:
        axm = axu.inset_axes([0.55, 0.55, 0.42, 0.4])
        axm.plot(result.t_hist, result.mdot, lw=0.8, color="tab:gray")
        (mdot_pt,) = axm.plot([result.t_hist[0]], [result.mdot[0]], "ro", ms=4)
        axm.set_xlabel("t [s]", fontsize=7)
        axm.set_ylabel("$\\dot m$ [kg/s]", fontsize=7)
        axm.tick_params(labelsize=6)
    else:
        mdot_pt = None

    fig.tight_layout()

    def update(i):
        for key, (ln, data) in lines.items():
            ln.set_ydata(data[i])
        t = result.snap_times[i]
        ttxt.set_text(time_label.format(t=t))
        if mdot_pt is not None:
            j = np.searchsorted(result.t_hist, t)
            j = min(j, result.t_hist.size - 1)
            mdot_pt.set_data([result.t_hist[j]], [result.mdot[j]])
        return []

    anim = FuncAnimation(fig, update, frames=ns, blit=False)
    writer = "ffmpeg" if filename.endswith(".mp4") else "pillow"
    anim.save(filename, fps=fps, dpi=dpi, writer=writer)
    plt.close(fig)
    return filename


def xt_diagram(result, filename: str | None = None, field: str = "p",
               scale: float = 1e-6, label: str = "Pressure [MPa]",
               title: str = ""):
    """Distance-time contour diagram of a field (wave-propagation view)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = getattr(result, field) * scale
    fig, ax = plt.subplots(figsize=(7.5, 5))
    pc = ax.pcolormesh(result.z, result.snap_times, data,
                       shading="auto", cmap="viridis", rasterized=True)
    fig.colorbar(pc, ax=ax, label=label)
    ax.set_xlabel("z [m]  (0 = bottom, wellhead right)")
    ax.set_ylabel("t [s]")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        return filename
    return fig, ax
