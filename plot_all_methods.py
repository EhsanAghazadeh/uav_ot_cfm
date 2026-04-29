"""Multi-panel trajectory plot: one panel per method, on the Milano density.

Optional no-fly disc overlay + per-panel violation stats (n_inside, dwell,
incursions) when --nofly is passed. Renders with a log-stretched purple
density contour and time-coloured trajectory points; subsamples for high-N.

Usage:
  python plot_all_methods.py \
      --traj "TOES:results/toes_milano_long_4x/trajectory.npz" \
      --traj "NonUniformErg:results/nonuniform_erg_milano_4x/trajectory.npz" \
      ... \
      [--nofly 0.19 0.03 0.1] \
      --out results/.../trajectories_all.png
"""
from __future__ import annotations

import argparse, math, os, sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fair_comparison import TrajectoryRecord
from eval_nofly import violations


def _stride(n, cap):
    return max(1, n // cap)


def _panel(ax, grid, xlim, ylim, pos, label, nofly, vstats, max_points):
    h, w = grid.shape
    Pn = grid / grid.max()
    Pn = np.log1p(20.0 * Pn) / np.log1p(20.0)
    ax.imshow(Pn, extent=(*xlim, *ylim), origin="lower",
              cmap="Purples", alpha=0.85, aspect="equal", zorder=1)

    n = len(pos)
    s = _stride(n, max_points)
    idx = np.arange(0, n, s)
    t_norm = np.linspace(0, 1, len(idx))

    cmap_traj = LinearSegmentedColormap.from_list(
        "ocfm", ["#ffd166", "#ef476f", "#073b4c"], N=256)
    ax.scatter(pos[idx, 0], pos[idx, 1], c=t_norm, cmap=cmap_traj,
               s=1.5, alpha=0.5, linewidths=0, zorder=2,
               rasterized=True)

    if nofly is not None:
        cx, cy, rr = nofly
        ax.add_patch(plt.Circle((cx, cy), rr, facecolor="#ff3b3b",
                               alpha=0.18, edgecolor="none", zorder=3))
        ax.add_patch(plt.Circle((cx, cy), rr, facecolor="none",
                               edgecolor="#c40000", lw=2.0, ls="--",
                               zorder=4))

    ax.plot(pos[0, 0], pos[0, 1], "o", color="black", ms=6,
            markeredgecolor="white", markeredgewidth=1.0, zorder=6)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#333"); spine.set_linewidth(0.8)

    title = f"{label}    N={n:,}  (showing {len(idx):,})"
    if vstats is not None:
        sub = (f"NFZ: {vstats['n_inside']} pts "
               f"({vstats['frac_inside']*100:.2f}%)  "
               f"dwell {vstats['dwell_time_s']:.1f}s  "
               f"incurs {vstats['n_incursions']}")
        title = title + "\n" + sub
    ax.set_title(title, fontsize=9, pad=6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", action="append", required=True,
                  help="label:path/to/trajectory.npz (repeatable)")
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--nofly", nargs=3, type=float, default=None,
                  metavar=("CX", "CY", "R"))
    p.add_argument("--max-points", type=int, default=8000)
    p.add_argument("--ncols", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--suptitle", default=None)
    args = p.parse_args()

    grid = np.load(args.milano_grid).astype(np.float64)
    if args.smooth_sigma > 0:
        grid = gaussian_filter(grid, sigma=float(args.smooth_sigma))
    grid = np.maximum(grid, 1e-12)

    items = []
    for spec in args.traj:
        if ":" not in spec:
            raise SystemExit(f"--traj must be label:path (got {spec})")
        label, path = spec.split(":", 1)
        rec = TrajectoryRecord.from_npz(path)
        dt = float(rec.meta.get("dt_s", rec.times[1] - rec.times[0]))
        vs = (violations(rec.pos, dt, args.nofly[:2], args.nofly[2])
              if args.nofly else None)
        items.append((label, rec, vs))

    n = len(items)
    ncols = min(args.ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.6 * nrows),
                            dpi=160, squeeze=False)
    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=13, y=0.995)

    for i, (label, rec, vs) in enumerate(items):
        ax = axes[i // ncols][i % ncols]
        xlim = (rec.domain[0], rec.domain[1])
        ylim = (rec.domain[2], rec.domain[3])
        _panel(ax, grid, xlim, ylim, rec.pos, label,
               tuple(args.nofly) if args.nofly else None,
               vs, args.max_points)

    # blank unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97 if args.suptitle else 1.0))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
