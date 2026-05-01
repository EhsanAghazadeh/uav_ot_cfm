"""Render the three experiment plots from one or more prefix_curve.npz files.

Each plot has one curve per method (here, just TOES). The harness is built
so additional methods can be appended via --curve method:path; the plotter
expects each curve already pre-aggregated (mean ± band would come from
multiple seed runs of a stochastic method).
"""
from __future__ import annotations

import argparse, os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from method_style import color as _color, label as _label


METRIC_LABELS = {
    "fourier_erg": "Fourier ergodic metric  (lower is better)",
    "density_corr": "density Pearson corr  (higher is better)",
    "l1": "L1(visitation, target)  (lower is better)",
    "l2": "L2(visitation, target)  (lower is better)",
}
METRIC_BETTER_HIGH = {"fourier_erg": False, "density_corr": True,
                     "l1": False, "l2": False}


def _interp_at(x_q, x, y):
    """1D monotone interp: returns y at x_q. NaN outside x range."""
    order = np.argsort(x)
    x = x[order]; y = y[order]
    out = np.full_like(x_q, np.nan, dtype=np.float64)
    mask = (x_q >= x[0]) & (x_q <= x[-1])
    out[mask] = np.interp(x_q[mask], x, y)
    return out


def plot_exp1_fixed_energy(curves, out_path, metric):
    """X = energy budget, Y = coverage (one of the four metrics)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    e_lo = max(c["energy"][0] for c in curves)
    e_hi = min(c["energy"][-1] for c in curves)
    e_grid = np.geomspace(max(e_lo, 1e-6), e_hi, 80)

    for c in curves:
        y = _interp_at(e_grid, c["energy"], c[metric])
        ax.plot(e_grid, y, lw=1.6, color=_color(c["method"]),
                label=_label(c["method"]))
    ax.set_xscale("log")
    ax.set_xlabel("energy budget  Σ ‖v‖² · dt")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"Exp 1 — fixed energy budget → coverage ({metric})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_exp2_fixed_coverage(curves, out_path, metric):
    """X = coverage level, Y = energy needed to reach it."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sign = 1.0 if METRIC_BETTER_HIGH[metric] else -1.0
    cov_lo = max((sign * c[metric]).min() for c in curves)
    cov_hi = min((sign * c[metric]).max() for c in curves)
    cov_grid = np.linspace(cov_lo, cov_hi, 80)

    for c in curves:
        # walk monotone in "better-direction" coverage; y(coverage)=energy
        cov_signed = sign * c[metric]
        y = _interp_at(cov_grid, cov_signed, c["energy"])
        ax.plot(sign * cov_grid, y, lw=1.6, color=_color(c["method"]),
                label=_label(c["method"]))
    ax.set_yscale("log")
    ax.set_xlabel(METRIC_LABELS[metric])
    ax.set_ylabel("energy needed  Σ ‖v‖² · dt")
    ax.set_title(f"Exp 2 — fixed coverage → energy ({metric})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_exp3_fixed_n(curves, out_path):
    """X = number of samples (=prefix length), three stacked panels:
    Σ‖v‖²·dt, Zeng E_total (kJ), and Fourier ergodic metric."""
    has_kj = all("E_total_kJ" in c for c in curves)
    n_panels = 3 if has_kj else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(6.5, 3.5 * n_panels),
                             sharex=True)
    for c in curves:
        axes[0].plot(c["n_points"], c["energy"], lw=1.6,
                     color=_color(c["method"]), label=_label(c["method"]))
    axes[0].set_ylabel("energy  Σ ‖v‖² · dt")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3); axes[0].legend(); axes[0].set_title(
        "Exp 3 — fixed N samples → energy & coverage")
    next_ax = 1
    if has_kj:
        for c in curves:
            axes[next_ax].plot(c["n_points"], c["E_total_kJ"], lw=1.6,
                               color=_color(c["method"]),
                               label=_label(c["method"]))
        axes[next_ax].set_ylabel("E_total  (kJ)  [Zeng 2019]")
        axes[next_ax].set_xscale("log"); axes[next_ax].set_yscale("log")
        axes[next_ax].grid(alpha=0.3); axes[next_ax].legend()
        next_ax += 1
    for c in curves:
        axes[next_ax].plot(c["n_points"], c["fourier_erg"], lw=1.6,
                           color=_color(c["method"]),
                           label=_label(c["method"]))
    axes[next_ax].set_ylabel("Fourier ergodic metric")
    axes[next_ax].set_yscale("log"); axes[next_ax].set_xscale("log")
    axes[next_ax].set_xlabel("number of samples")
    axes[next_ax].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_exp4_fixed_kj(curves, out_path, metric):
    """X = E_total (kJ) [Zeng], Y = coverage. Skipped if any curve lacks E_total_kJ."""
    if not all("E_total_kJ" in c for c in curves):
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    e_lo = max(c["E_total_kJ"][0] for c in curves)
    e_hi = min(c["E_total_kJ"][-1] for c in curves)
    e_grid = np.geomspace(max(e_lo, 1e-6), e_hi, 80)
    for c in curves:
        y = _interp_at(e_grid, c["E_total_kJ"], c[metric])
        ax.plot(e_grid, y, lw=1.6, color=_color(c["method"]),
                label=_label(c["method"]))
    ax.set_xscale("log")
    ax.set_xlabel("E_total  (kJ)  [Zeng 2019 rotary-wing]")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"Exp 4 — fixed Zeng E_total → coverage ({metric})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curve", action="append", required=True,
                  help="method:path/to/prefix_curve.npz (repeatable)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--metric-exp1", default="fourier_erg",
                  choices=list(METRIC_LABELS))
    p.add_argument("--metric-exp2", default="fourier_erg",
                  choices=list(METRIC_LABELS))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    curves = []
    for spec in args.curve:
        if ":" not in spec:
            raise SystemExit(f"--curve must be method:path (got {spec})")
        method, path = spec.split(":", 1)
        d = dict(np.load(path))
        d["method"] = method
        curves.append(d)
    print(f"loaded {len(curves)} curves: {[c['method'] for c in curves]}")

    plot_exp1_fixed_energy(curves, os.path.join(args.out_dir, "exp1_fixed_energy.png"),
                          args.metric_exp1)
    plot_exp2_fixed_coverage(curves, os.path.join(args.out_dir, "exp2_fixed_coverage.png"),
                            args.metric_exp2)
    plot_exp3_fixed_n(curves, os.path.join(args.out_dir, "exp3_fixed_n.png"))
    if plot_exp4_fixed_kj(curves,
                         os.path.join(args.out_dir, "exp4_fixed_kj.png"),
                         args.metric_exp1):
        print("  + exp4_fixed_kj.png (Zeng E_total)")

    # also a 2x2 grid of all 4 coverage metrics for Exp 1 (handy overview)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, metric in zip(axes.flat, METRIC_LABELS):
        e_lo = max(c["energy"][0] for c in curves)
        e_hi = min(c["energy"][-1] for c in curves)
        e_grid = np.geomspace(max(e_lo, 1e-6), e_hi, 80)
        for c in curves:
            y = _interp_at(e_grid, c["energy"], c[metric])
            ax.plot(e_grid, y, lw=1.5, color=_color(c["method"]),
                    label=_label(c["method"]))
        ax.set_xscale("log")
        ax.set_xlabel("energy budget"); ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Exp 1 — coverage vs energy (all 4 metrics)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "exp1_all_metrics.png"), dpi=140)
    plt.close(fig)
    print(f"plots in {args.out_dir}")


if __name__ == "__main__":
    main()
