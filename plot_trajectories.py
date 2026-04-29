"""Render one trajectory-on-target plot per method.

Reads canonical trajectory.npz files (TOES + OT-CFM) and the Milano grid,
producing one PNG per method with the smoothed density as a contour and
the trajectory overlaid (start point in black).
"""
from __future__ import annotations

import argparse, os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from fair_comparison import TrajectoryRecord


class Target:
    """Bare-minimum Target stand-in: xlim/ylim + pdf_np(X, Y)."""
    def __init__(self, grid, xlim, ylim):
        self.xlim, self.ylim = xlim, ylim
        h, w = grid.shape
        xs = np.linspace(xlim[0], xlim[1], w)
        ys = np.linspace(ylim[0], ylim[1], h)
        cell_area = ((xlim[1] - xlim[0]) / max(w - 1, 1)
                     * (ylim[1] - ylim[0]) / max(h - 1, 1))
        mass = grid / grid.sum()
        self._interp = RegularGridInterpolator(
            (ys, xs), mass / max(cell_area, 1e-16),
            bounds_error=False, fill_value=0.0, method="linear")

    def pdf_np(self, X, Y):
        pts = np.column_stack([np.asarray(Y).ravel(), np.asarray(X).ravel()])
        return self._interp(pts).reshape(np.broadcast(X, Y).shape)


def _as_batched(traj):
    a = np.asarray(traj)
    return a[None, ...] if a.ndim == 2 else a


def plot_trajectory(target, traj, x0, out_path, title="", max_points=8000,
                    nofly=None):
    nx, ny = 160, 160
    X, Y = np.meshgrid(np.linspace(*target.xlim, nx),
                       np.linspace(*target.ylim, ny))
    P = np.asarray(target.pdf_np(X, Y))

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    levels = np.linspace(P.min(), P.max(), 11)
    if levels[0] == levels[-1]:
        levels = None
    ax.contourf(X, Y, P, levels=levels, cmap="Purples")

    traj_b = _as_batched(traj)
    x0_b = np.asarray(x0)
    if x0_b.ndim == 1:
        x0_b = x0_b[None, :]
    n_agents = traj_b.shape[0]
    cmap = plt.get_cmap("tab10" if n_agents <= 10 else "hsv")
    T_total = traj_b.shape[1]

    # subsample for the line, density-style for markers
    if T_total > max_points:
        stride = max(1, T_total // max_points)
        idx = np.arange(0, T_total, stride)
        T_shown = len(idx)
    else:
        idx = np.arange(T_total)
        T_shown = T_total

    # density-aware styling
    if n_agents == 1:
        lw = 2.0 if T_shown <= 1000 else (1.0 if T_shown <= 5000 else 0.5)
        ms = 3.0 if T_shown <= 1000 else (1.5 if T_shown <= 5000 else 0.8)
        line_alpha = 0.9 if T_shown <= 1000 else (0.45 if T_shown <= 5000 else 0.2)
        marker_alpha = 0.5 if T_shown <= 1000 else (0.25 if T_shown <= 5000 else 0.12)
    else:
        lw = max(0.5, 1.2 - 0.05 * n_agents)
        ms = max(1.0, 2.0 - 0.05 * n_agents)
        line_alpha = 0.9
        marker_alpha = 0.5

    for i in range(n_agents):
        c = cmap(i % cmap.N) if n_agents > 1 else "C1"
        ax.plot(traj_b[i, idx, 0], traj_b[i, idx, 1], "-", color=c, lw=lw,
                alpha=line_alpha)
        ax.plot(traj_b[i, idx, 0], traj_b[i, idx, 1], "o", color=c, ms=ms,
                alpha=marker_alpha, markeredgecolor="none")
    ax.plot(x0_b[:, 0], x0_b[:, 1], "o", color="black", ms=8)
    if nofly is not None:
        cx, cy, rr = nofly
        circ = plt.Circle((cx, cy), rr, edgecolor="red", facecolor="red",
                          alpha=0.18, lw=1.5, zorder=5)
        ax.add_patch(circ)
    ax.set_xlim(target.xlim); ax.set_ylim(target.ylim)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        full = f"{title}  (showing {T_shown}/{T_total})" if T_shown < T_total else title
        ax.set_title(full, fontsize=10)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", action="append", required=True,
                  help="label:path/to/trajectory.npz (repeatable)")
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--max-points", type=int, default=8000,
                  help="uniform-stride subsample cap for plotting")
    p.add_argument("--nofly", nargs=3, type=float, default=None,
                  metavar=("CX", "CY", "R"),
                  help="Overlay a no-fly disc (centre x, centre y, radius)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    grid = np.load(args.milano_grid).astype(np.float64)
    if args.smooth_sigma > 0:
        grid = gaussian_filter(grid, sigma=float(args.smooth_sigma))
    grid = np.maximum(grid, 1e-12)

    for spec in args.traj:
        if ":" not in spec:
            raise SystemExit(f"--traj must be label:path (got {spec})")
        label, path = spec.split(":", 1)
        rec = TrajectoryRecord.from_npz(path)
        xlim = (rec.domain[0], rec.domain[1])
        ylim = (rec.domain[2], rec.domain[3])
        target = Target(grid, xlim, ylim)
        out_path = os.path.join(args.out_dir, f"trajectory_{label}.png")
        plot_trajectory(target, rec.pos, rec.pos[0], out_path,
                       title=f"{label}  (N={len(rec.pos)})",
                       max_points=args.max_points,
                       nofly=tuple(args.nofly) if args.nofly else None)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
