"""Side-by-side plot: vanilla OT-CFM vs NFZ-constrained OT-CFM on Milano.

Shows the smoothed Milano density as a contour, the trajectory subsampled
and time-coloured, and the no-fly disc as a red ring + soft fill. Annotates
each panel with the violation stats from eval_nofly.
"""
from __future__ import annotations

import argparse, os, sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from ergodic_cfm.fair_comparison import TrajectoryRecord
from evaluation.eval_nofly import violations


def _stride(n, cap):
    return max(1, n // cap)


def _panel(ax, grid, xlim, ylim, pos, nofly, label, vstats, max_points=8000):
    h, w = grid.shape
    X, Y = np.meshgrid(np.linspace(*xlim, w), np.linspace(*ylim, h))

    # density background — purple cmap, log-stretched for legibility
    Pn = grid / grid.max()
    Pn = np.log1p(20.0 * Pn) / np.log1p(20.0)
    ax.imshow(Pn, extent=(*xlim, *ylim), origin="lower",
              cmap="Purples", alpha=0.85, aspect="equal", zorder=1)

    # subsampled trajectory, time-coloured
    n = len(pos)
    s = _stride(n, max_points)
    idx = np.arange(0, n, s)
    t_norm = np.linspace(0, 1, len(idx))

    cmap_traj = LinearSegmentedColormap.from_list(
        "ocfm", ["#ffd166", "#ef476f", "#073b4c"], N=256)
    ax.scatter(pos[idx, 0], pos[idx, 1], c=t_norm, cmap=cmap_traj,
               s=1.5, alpha=0.5, linewidths=0, zorder=2,
               rasterized=True)

    # no-fly disc — soft fill + dashed boundary
    if nofly is not None:
        cx, cy, rr = nofly
        ax.add_patch(plt.Circle((cx, cy), rr, facecolor="#ff3b3b",
                               alpha=0.18, edgecolor="none", zorder=3))
        ax.add_patch(plt.Circle((cx, cy), rr, facecolor="none",
                               edgecolor="#c40000", lw=2.0, ls="--",
                               zorder=4))

    # start point
    ax.plot(pos[0, 0], pos[0, 1], "o", color="black", ms=7,
            markeredgecolor="white", markeredgewidth=1.2, zorder=6)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#333"); spine.set_linewidth(1.0)

    title = f"{label}    N = {n:,}    showing {len(idx):,}"
    sub = (f"violations: {vstats['n_inside']} pts  "
           f"({vstats['frac_inside']*100:.2f}%)   "
           f"dwell {vstats['dwell_time_s']:.2f}s   "
           f"incursions {vstats['n_incursions']}")
    ax.set_title(title + "\n" + sub, fontsize=10, pad=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla", required=True,
                  help="path to baseline OT-CFM trajectory.npz")
    p.add_argument("--nfz", required=True,
                  help="path to NFZ-trained trajectory.npz")
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--center", nargs=2, type=float, default=[0.19, 0.03])
    p.add_argument("--radius", type=float, default=0.1)
    p.add_argument("--max-points", type=int, default=8000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    grid = np.load(args.milano_grid).astype(np.float64)
    if args.smooth_sigma > 0:
        grid = gaussian_filter(grid, sigma=float(args.smooth_sigma))
    grid = np.maximum(grid, 1e-12)

    rec_v = TrajectoryRecord.from_npz(args.vanilla)
    rec_n = TrajectoryRecord.from_npz(args.nfz)
    xlim = (rec_v.domain[0], rec_v.domain[1])
    ylim = (rec_v.domain[2], rec_v.domain[3])

    nofly = (args.center[0], args.center[1], args.radius)
    dt_v = float(rec_v.meta.get("dt_s", rec_v.times[1] - rec_v.times[0]))
    dt_n = float(rec_n.meta.get("dt_s", rec_n.times[1] - rec_n.times[0]))
    vstats_v = violations(rec_v.pos, dt_v, args.center, args.radius)
    vstats_n = violations(rec_n.pos, dt_n, args.center, args.radius)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), dpi=160)
    fig.suptitle(
        f"OT-CFM trajectory on Milano cellular density   "
        f"(NFZ disc: c=({nofly[0]:.2f}, {nofly[1]:.2f}), r={nofly[2]:.2f})",
        fontsize=12, y=0.99)

    _panel(axes[0], grid, xlim, ylim, rec_v.pos, nofly,
           "OT-CFM v2 (baseline)", vstats_v, args.max_points)
    _panel(axes[1], grid, xlim, ylim, rec_n.pos, nofly,
           "OT-CFM v2 + NFZ penalty", vstats_n, args.max_points)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
