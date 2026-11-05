"""Render experiment plots from one or more prefix_curve.npz files.

Multiple seeds of the same method are passed as repeated --curve method:path
specs with the same method name. Each method gets:
  - one solid mean line (full opacity)
  - per-seed lines in light (alpha 0.25)
  - ±1 std shaded band
"""
from __future__ import annotations

import argparse, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ergodic_cfm.method_style import color as _color, label as _label


METRIC_LABELS = {
    "fourier_erg":  "Fourier ergodic  $\\mathcal{E}$  (lower better)",
    "density_corr": "density Pearson corr  (higher better)",
    "l1":           "$\\ell_1$(visitation, target)  (lower better)",
    "l2":           "$\\ell_2$(visitation, target)  (lower better)",
}
METRIC_BETTER_HIGH = {"fourier_erg": False, "density_corr": True,
                      "l1": False, "l2": False}
NFZ_LABELS = {
    "nfz_frac_inside":   "frac. of trajectory inside NFZ  (lower better)",
    "nfz_n_incursions":  "number of NFZ incursions  (lower better)",
}


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------
def _interp_at(x_q, x, y):
    """1D monotone interp; NaN outside range."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    out = np.full_like(x_q, np.nan, dtype=np.float64)
    mask = (x_q >= xs[0]) & (x_q <= xs[-1])
    out[mask] = np.interp(x_q[mask], xs, ys)
    return out


# ---------------------------------------------------------------------------
# Per-method aggregation plotter
# ---------------------------------------------------------------------------
def _plot_group(ax, method, curves, x_key, y_key,
                log_x=False, sign_x=1.0, sign_y=1.0):
    """Plot per-seed (light) + mean (solid) + std band for one method.

    sign_x / sign_y flip axes before aggregation (used when
    metric direction requires negation for monotone interp).
    """
    col = _color(method)
    lbl = _label(method)
    xs = [sign_x * c[x_key] for c in curves]
    ys = [sign_y * c[y_key] for c in curves]

    if len(curves) == 1:
        ax.plot(sign_x * xs[0], sign_y * ys[0], lw=1.8, color=col, label=lbl)
        return

    finite_xs = [x[np.isfinite(x)] for x in xs]
    if any(len(fx) == 0 for fx in finite_xs):
        ax.plot(sign_x * xs[0], sign_y * ys[0], lw=1.8, color=col, label=lbl)
        return
    x_lo = max(float(fx.min()) for fx in finite_xs)
    x_hi = min(float(fx.max()) for fx in finite_xs)
    if x_lo >= x_hi:
        ax.plot(sign_x * xs[0], sign_y * ys[0], lw=1.8, color=col, label=lbl)
        return

    x_grid = (np.geomspace(max(x_lo, 1e-12), x_hi, 100)
              if log_x else np.linspace(x_lo, x_hi, 100))
    y_mat = np.array([_interp_at(x_grid, xv, yv) for xv, yv in zip(xs, ys)])

    # individual seeds (light)
    for xv, yv in zip(xs, ys):
        ax.plot(sign_x * xv, sign_y * yv, lw=0.7, color=col, alpha=0.25)

    y_mean = np.nanmean(y_mat, axis=0)
    y_std  = np.nanstd(y_mat, axis=0)
    ax.plot(sign_x * x_grid, sign_y * y_mean, lw=2.0, color=col, label=lbl)
    ax.fill_between(sign_x * x_grid,
                    sign_y * (y_mean - y_std),
                    sign_y * (y_mean + y_std),
                    alpha=0.15, color=col)


# ---------------------------------------------------------------------------
# Exp 1 — fixed energy → coverage
# ---------------------------------------------------------------------------
def plot_exp1_fixed_energy(by_method, out_path, metric):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, curves in by_method.items():
        _plot_group(ax, method, curves, "energy", metric, log_x=True)
    ax.set_xscale("log")
    ax.set_xlabel("energy budget  $\\sum\\|v\\|^2\\cdot dt$")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 2 — fixed coverage → energy
# ---------------------------------------------------------------------------
def plot_exp2_fixed_coverage(by_method, out_path, metric):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sign = 1.0 if METRIC_BETTER_HIGH[metric] else -1.0
    for method, curves in by_method.items():
        _plot_group(ax, method, curves, metric, "energy",
                    sign_x=sign, sign_y=1.0)
    ax.set_yscale("log")
    ax.set_xlabel(METRIC_LABELS[metric])
    ax.set_ylabel("energy needed  $\\sum\\|v\\|^2\\cdot dt$")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 3 — fixed N → energy + coverage
# ---------------------------------------------------------------------------
def plot_exp3_fixed_n(by_method, out_path):
    has_kj = all("E_total_kJ" in c
                 for g in by_method.values() for c in g)
    n_panels = 3 if has_kj else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(6.5, 3.5 * n_panels),
                             sharex=True)
    for method, curves in by_method.items():
        _plot_group(axes[0], method, curves, "n_points", "energy", log_x=True)
    axes[0].set_ylabel("energy  $\\sum\\|v\\|^2\\cdot dt$")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    nxt = 1
    if has_kj:
        for method, curves in by_method.items():
            _plot_group(axes[nxt], method, curves, "n_points", "E_total_kJ",
                        log_x=True)
        axes[nxt].set_ylabel("$E_\\mathrm{total}$ (kJ)  [Zeng 2019]")
        axes[nxt].set_xscale("log"); axes[nxt].set_yscale("log")
        axes[nxt].grid(alpha=0.3); axes[nxt].legend(fontsize=8)
        nxt += 1
    for method, curves in by_method.items():
        _plot_group(axes[nxt], method, curves, "n_points", "fourier_erg",
                    log_x=True)
    axes[nxt].set_ylabel("Fourier ergodic $\\mathcal{E}$")
    axes[nxt].set_xscale("log"); axes[nxt].set_yscale("log")
    axes[nxt].set_xlabel("number of samples")
    axes[nxt].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 4 — fixed Zeng E_total → coverage
# ---------------------------------------------------------------------------
def plot_exp4_fixed_kj(by_method, out_path, metric):
    if not all("E_total_kJ" in c
               for g in by_method.values() for c in g):
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, curves in by_method.items():
        _plot_group(ax, method, curves, "E_total_kJ", metric, log_x=True)
    ax.set_xscale("log")
    ax.set_xlabel("$E_\\mathrm{total}$ (kJ)  [Zeng 2019]")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# NFZ plots — fixed coverage/energy → NFZ metric
# ---------------------------------------------------------------------------
def _has_nfz(by_method, nfz_key):
    return all(nfz_key in c
               for g in by_method.values() for c in g)


def plot_nfz_fixed_coverage(by_method, out_path, coverage_metric, nfz_key):
    """X = coverage level, Y = NFZ metric at that coverage."""
    if not _has_nfz(by_method, nfz_key):
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sign = 1.0 if METRIC_BETTER_HIGH[coverage_metric] else -1.0
    for method, curves in by_method.items():
        _plot_group(ax, method, curves, coverage_metric, nfz_key, sign_x=sign)
    ax.set_xlabel(METRIC_LABELS[coverage_metric])
    ax.set_ylabel(NFZ_LABELS[nfz_key])
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    return True


def plot_nfz_fixed_energy(by_method, out_path, nfz_key):
    """X = energy budget, Y = NFZ metric at that energy."""
    if not _has_nfz(by_method, nfz_key):
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, curves in by_method.items():
        _plot_group(ax, method, curves, "energy", nfz_key, log_x=True)
    ax.set_xscale("log")
    ax.set_xlabel("energy budget  $\\sum\\|v\\|^2\\cdot dt$")
    ax.set_ylabel(NFZ_LABELS[nfz_key])
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curve", action="append", required=True,
                  help="method:path/to/prefix_curve.npz (repeatable; "
                       "same method name for multiple seeds)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--metric-exp1", default="fourier_erg",
                  choices=list(METRIC_LABELS))
    p.add_argument("--metric-exp2", default="fourier_erg",
                  choices=list(METRIC_LABELS))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Group curves by method name (multiple seeds → same key)
    by_method: dict[str, list] = defaultdict(list)
    for spec in args.curve:
        if ":" not in spec:
            raise SystemExit(f"--curve must be method:path (got {spec})")
        method, path = spec.split(":", 1)
        d = dict(np.load(path, allow_pickle=False))
        d["method"] = method
        by_method[method].append(d)

    print(f"loaded {sum(len(g) for g in by_method.values())} curves "
          f"across {len(by_method)} methods: {list(by_method)}")

    od = args.out_dir
    m1, m2 = args.metric_exp1, args.metric_exp2

    plot_exp1_fixed_energy(by_method,
                           os.path.join(od, "exp1_fixed_energy.png"), m1)
    plot_exp2_fixed_coverage(by_method,
                             os.path.join(od, "exp2_fixed_coverage.png"), m2)
    plot_exp3_fixed_n(by_method, os.path.join(od, "exp3_fixed_n.png"))
    if plot_exp4_fixed_kj(by_method,
                          os.path.join(od, "exp4_fixed_kj.png"), m1):
        print("  + exp4_fixed_kj.png")

    # NFZ plots (silently skipped if NFZ metrics not in curves):
    # One figure per coverage metric (pearson, l1) with frac + incursions side by side.
    for cov_metric, cov_tag in [("density_corr", "pearson"), ("l1", "l1")]:
        nfz_keys = ("nfz_frac_inside", "nfz_n_incursions")
        if _has_nfz(by_method, nfz_keys[0]) and _has_nfz(by_method, nfz_keys[1]):
            sign = 1.0 if METRIC_BETTER_HIGH[cov_metric] else -1.0
            fig, (ax_f, ax_i) = plt.subplots(1, 2, figsize=(12, 4.5))
            for method, curves in by_method.items():
                _plot_group(ax_f, method, curves, cov_metric, "nfz_frac_inside",
                            sign_x=sign)
                _plot_group(ax_i, method, curves, cov_metric, "nfz_n_incursions",
                            sign_x=sign)
            for ax, nfz_key in [(ax_f, "nfz_frac_inside"),
                                 (ax_i, "nfz_n_incursions")]:
                ax.set_xlabel(METRIC_LABELS[cov_metric])
                ax.set_ylabel(NFZ_LABELS[nfz_key])
                ax.grid(alpha=0.3); ax.legend(fontsize=8)
            fig.tight_layout()
            fname = f"nfz_fixed_cov_{cov_tag}.png"
            fig.savefig(os.path.join(od, fname), dpi=140)
            plt.close(fig)
            print(f"  + {fname}")

        for nfz_key in nfz_keys:
            tag = "frac" if "frac" in nfz_key else "incursions"
            plot_nfz_fixed_coverage(by_method,
                                    os.path.join(od, f"nfz_fixed_cov_{cov_tag}_{tag}.png"),
                                    cov_metric, nfz_key)
            plot_nfz_fixed_energy(by_method,
                                  os.path.join(od, f"nfz_fixed_energy_{cov_tag}_{tag}.png"),
                                  nfz_key)

    # Individual plots per metric (fixed energy → coverage)
    for metric in METRIC_LABELS:
        plot_exp1_fixed_energy(by_method,
                               os.path.join(od, f"exp1_metric_{metric}.png"),
                               metric)

    # Paired: Pearson (left) + L1 (right)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, curves in by_method.items():
        _plot_group(ax_l, method, curves, "energy", "density_corr", log_x=True)
        _plot_group(ax_r, method, curves, "energy", "l1", log_x=True)
    for ax, metric in [(ax_l, "density_corr"), (ax_r, "l1")]:
        ax.set_xscale("log")
        ax.set_xlabel("energy budget  $\\sum\\|v\\|^2\\cdot dt$")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "exp1_pearson_l1.png"), dpi=140)
    plt.close(fig)

    # Paired: L2 (left) + Fourier ergodic (right)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, curves in by_method.items():
        _plot_group(ax_l, method, curves, "energy", "l2", log_x=True)
        _plot_group(ax_r, method, curves, "energy", "fourier_erg", log_x=True)
    for ax, metric in [(ax_l, "l2"), (ax_r, "fourier_erg")]:
        ax.set_xscale("log")
        ax.set_xlabel("energy budget  $\\sum\\|v\\|^2\\cdot dt$")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "exp1_l2_ergodic.png"), dpi=140)
    plt.close(fig)

    print(f"plots in {od}")


if __name__ == "__main__":
    main()
