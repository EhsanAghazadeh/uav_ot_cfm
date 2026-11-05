"""Coverage--energy Pareto scatter from eval_fair.py per_run.json.

Produces:
  - pareto.png            mean Zeng power vs Fourier ergodic (log-log)
  - pareto_cov_l1.png     mean Zeng power vs coverage L1 (alt metric)
  - pareto_per_metre.png  J/m vs Fourier ergodic (efficiency view)

Each marker = one (method, seed) pair. Convex hull per method drawn as a
shaded polygon to make method clusters visually obvious.

Usage:
  python -m plotting.plot_pareto \
      --per-run results/fair_metrics/per_run.json \
      --out-dir results/fair_metrics
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from ergodic_cfm.method_style import color as _color, label as _label


def _convex_hull(points):
    """Andrew's monotone chain; returns hull pts in CCW order or None if <3 pts."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        return None
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _pareto_scatter(per_run, x_key, y_key, x_label, y_label, title, out_path,
                    log_x=True, log_y=True):
    by_method = {}
    for r in per_run:
        if x_key not in r or y_key not in r:
            continue
        x = r[x_key]; y = r[y_key]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        by_method.setdefault(r["method"], []).append((x, y))

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    for m, pts in by_method.items():
        col = _color(m)
        arr = np.asarray(pts)
        # convex hull (transparent fill) when >=3 distinct points
        if len(set(map(tuple, pts))) >= 3:
            hull = _convex_hull(pts)
            if hull is not None:
                ax.add_patch(Polygon(hull, alpha=0.12, facecolor=col,
                                     edgecolor=col, linewidth=0.7))
        # method centroid (bigger marker)
        cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
        ax.scatter(cx, cy, s=120, marker="*", color=col, edgecolor="black",
                   linewidth=0.6, zorder=5,
                   label=f"{_label(m)}  (n={len(pts)})")
        # individual seeds
        ax.scatter(arr[:, 0], arr[:, 1], s=42, marker="o", color=col,
                   edgecolor="black", linewidth=0.4, alpha=0.85, zorder=4)

    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-run", required=True,
                  help="per_run.json from eval_fair.py")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.per_run) as f:
        per_run = json.load(f)
    print(f"loaded {len(per_run)} (method, seed) records")

    _pareto_scatter(
        per_run, "P_mean_W", "fourier_erg",
        x_label=r"mean Zeng power  $\bar P$  (W)",
        y_label=r"Fourier ergodic  $\mathcal{E}$  (lower better)",
        title="Coverage--energy Pareto (Milano)",
        out_path=os.path.join(args.out_dir, "pareto.png"))

    _pareto_scatter(
        per_run, "P_mean_W", "l1",
        x_label=r"mean Zeng power  $\bar P$  (W)",
        y_label=r"coverage  $\ell_1$  (lower better)",
        title="Coverage ($\\ell_1$) vs energy",
        out_path=os.path.join(args.out_dir, "pareto_cov_l1.png"),
        log_y=False)

    _pareto_scatter(
        per_run, "E_per_metre_J", "fourier_erg",
        x_label=r"energy per metre  $E/L$  (J/m)",
        y_label=r"Fourier ergodic  $\mathcal{E}$  (lower better)",
        title="Per-metre flight efficiency vs coverage",
        out_path=os.path.join(args.out_dir, "pareto_per_metre.png"))

    # NFZ panels — only produced when nfz_frac_inside is present in records
    nfz_keys = [r for r in per_run if "nfz_frac_inside" in r]
    if nfz_keys:
        _pareto_scatter(
            per_run, "fourier_erg", "nfz_frac_inside",
            x_label=r"Fourier ergodic  $\mathcal{E}$  (lower better)",
            y_label="frac. of trajectory inside NFZ  (lower better)",
            title="Coverage vs NFZ fraction",
            out_path=os.path.join(args.out_dir, "pareto_nfz_frac.png"),
            log_x=True, log_y=False)

        _pareto_scatter(
            per_run, "fourier_erg", "nfz_n_incursions",
            x_label=r"Fourier ergodic  $\mathcal{E}$  (lower better)",
            y_label="number of NFZ incursions  (lower better)",
            title="Coverage vs NFZ incursions",
            out_path=os.path.join(args.out_dir, "pareto_nfz_incursions.png"),
            log_x=True, log_y=False)

        _pareto_scatter(
            per_run, "P_mean_W", "nfz_frac_inside",
            x_label=r"mean Zeng power  $\bar P$  (W)",
            y_label="frac. of trajectory inside NFZ  (lower better)",
            title="Energy vs NFZ fraction",
            out_path=os.path.join(args.out_dir, "pareto_energy_nfz_frac.png"),
            log_x=True, log_y=False)

        _pareto_scatter(
            per_run, "P_mean_W", "nfz_n_incursions",
            x_label=r"mean Zeng power  $\bar P$  (W)",
            y_label="number of NFZ incursions  (lower better)",
            title="Energy vs NFZ incursions",
            out_path=os.path.join(args.out_dir, "pareto_energy_nfz_incursions.png"),
            log_x=True, log_y=False)


if __name__ == "__main__":
    main()
