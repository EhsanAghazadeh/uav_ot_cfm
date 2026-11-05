"""Probe Milano density at candidate NFZ disc placements.

Loads the smoothed Milano density (the same pipeline used by training
and eval) and reports, for each candidate (cx, cy, r), the fraction of
total target mass that falls inside the disc. A low-density placement
is one where this fraction is small (e.g., < 1 %); the current paper
setup at (0.19, 0.03), r=0.10 covers roughly 20-30 % and is the
"non-trivial" condition.

Usage:
    python -m evaluation.probe_nfz_density \
        --milano-grid data/milano_grid_corrected.npy \
        --candidate -0.6 -0.6 0.10 \
        --candidate  0.7 -0.5 0.10 \
        --candidate -0.7  0.5 0.10 \
        --candidate  0.19 0.03 0.10
"""
from __future__ import annotations

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ergodic_cfm.target_density import load_milano_target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--candidate", action="append", nargs=3, type=float,
                   metavar=("CX", "CY", "R"),
                   help="Candidate disc as (cx, cy, r) in normalised "
                        "[-1, 1]^2 coords. Repeat for multiple candidates.")
    p.add_argument("--out", default="results/nfz_density_probe.png",
                   help="Output figure path.")
    args = p.parse_args()

    if not args.candidate:
        # Default: a sensible first sweep
        args.candidate = [
            [0.19,  0.03, 0.10],   # current (high-density)
            [-0.6, -0.6,  0.10],
            [ 0.7, -0.5,  0.10],
            [-0.7,  0.5,  0.10],
            [ 0.6,  0.6,  0.10],
            [-0.5, -0.5,  0.10],
        ]

    t = load_milano_target(args.milano_grid, smooth_sigma=args.smooth_sigma)
    pmf = t["pmf"]
    xs, ys = t["xs"], t["ys"]
    XX, YY = np.meshgrid(xs, ys)

    print(f"\n{'cx':>8s}{'cy':>8s}{'r':>8s}  {'mass inside (%)':>16s}")
    print("-" * 48)
    rows = []
    for cx, cy, r in args.candidate:
        mask = (XX - cx) ** 2 + (YY - cy) ** 2 < r * r
        frac = float(pmf[mask].sum() * 100.0)
        rows.append((cx, cy, r, frac))
        marker = " * lowest" if frac == min(rr[3] for rr in rows) else ""
        print(f"{cx:>8.2f}{cy:>8.2f}{r:>8.2f}  {frac:>16.3f}{marker}")

    rows_sorted = sorted(rows, key=lambda x: x[3])
    print(f"\nSuggested low-density placement: cx={rows_sorted[0][0]}, "
          f"cy={rows_sorted[0][1]}, r={rows_sorted[0][2]} "
          f"(only {rows_sorted[0][3]:.2f}% of target mass).")

    # Visual: density heatmap + candidate discs annotated by mass
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.imshow(pmf, extent=(xs[0], xs[-1], ys[0], ys[-1]),
              origin="lower", cmap="viridis", aspect="equal")
    cmap = plt.cm.tab10
    for i, (cx, cy, r, frac) in enumerate(rows):
        col = cmap(i % 10)
        circ = plt.Circle((cx, cy), r, facecolor="none",
                          edgecolor=col, linewidth=2.0,
                          linestyle="--", zorder=3)
        ax.add_patch(circ)
        ax.text(cx, cy, f"{frac:.1f}%", color="white", fontsize=8,
                ha="center", va="center",
                bbox=dict(facecolor=col, alpha=0.85, edgecolor="none",
                          boxstyle="round,pad=0.2"))
    ax.set_title("Milano target density with candidate NFZ discs\n"
                 "(label = % of target mass inside the disc)")
    ax.set_xlabel("x (normalised)"); ax.set_ylabel("y (normalised)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nfigure: {args.out}")


if __name__ == "__main__":
    main()
