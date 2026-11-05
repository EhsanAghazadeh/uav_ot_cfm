"""Per-method velocity distribution.

Loads a set of trajectory.npz files, computes per-step instantaneous
speed v_mps in physical units (no clipping), and renders three
diagnostic views:

  1. Overlapping histograms (log-x) of v_mps per method.
  2. ECDF per method (log-x), with vertical reference lines at
     hover (V=0), V_mp ~ 4 m/s, and the realistic UAV cap (30 m/s).
  3. A small bar chart of "fraction of samples above the realistic
     cap" per method.

Also prints a summary table of per-method statistics: median speed,
99th-percentile speed, max speed, fraction over the realistic cap,
and Zeng power both at the headline cap (100 m/s) and the realistic
cap (default 30 m/s). The gap between the two power columns is the
teleport-pathology indicator.

Usage:
    python -m plotting.plot_velocity_distribution \\
        --traj toes:results/toes_seed42/trajectory.npz \\
        --traj otcfm:results/otcfm_seed42/trajectory.npz \\
        --traj opt_transport:results/opt_transport_seed42/trajectory.npz \\
        --traj sac:results/sac_seed42/trajectory.npz \\
        --domain-size 200.0 --time-scale 10.0 \\
        --v-cap-realistic 30.0 \\
        --out-dir results/velocity_dist
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ergodic_cfm.baselines.milano_baselines import RotaryWingEnergyModel
from ergodic_cfm.fair_comparison import TrajectoryRecord
from ergodic_cfm.method_style import color as _color, label as _label


def per_step_speeds(rec, domain_size: float, time_scale: float):
    """Return v_mps (length T-1) in physical units."""
    pos = rec.pos
    times = rec.times * float(time_scale)
    dx = np.diff(pos, axis=0)
    dt = np.diff(times)
    dt = np.maximum(dt, 1e-12)
    step_m = np.linalg.norm(dx, axis=1) * float(domain_size)
    return step_m / dt


def power_with_cap(v_mps, cap, em):
    v = np.clip(v_mps, 0.01, cap)
    P = np.array([em.power(float(vi)) for vi in v])
    return P


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", action="append", required=True,
                   help="method:path/to/trajectory.npz (repeatable). "
                        "Multiple seeds for the same method are concatenated.")
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy",
                   help="(unused; kept for CLI compat with eval_fair).")
    p.add_argument("--domain-size", type=float, default=200.0)
    p.add_argument("--time-scale", type=float, default=10.0)
    p.add_argument("--v-cap-headline", type=float, default=100.0,
                   help="Numerical-guard cap (matches eval_fair --v-clip-max).")
    p.add_argument("--v-cap-realistic", type=float, default=30.0,
                   help="Physical UAV speed cap (typical rotorcraft max).")
    p.add_argument("--xmin", type=float, default=1e-2,
                   help="Lower bound of the log-x axis (m/s).")
    p.add_argument("--xmax", type=float, default=2e3,
                   help="Upper bound of the log-x axis (m/s).")
    p.add_argument("--n-bins", type=int, default=80)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Group --traj entries by method
    by_method = {}
    for spec in args.traj:
        if ":" not in spec:
            raise SystemExit(f"--traj must be method:path (got {spec})")
        m, path = spec.split(":", 1)
        rec = TrajectoryRecord.from_npz(path)
        v = per_step_speeds(rec, args.domain_size, args.time_scale)
        by_method.setdefault(m, []).append(v)

    methods = list(by_method.keys())
    em = RotaryWingEnergyModel()

    # Concatenate per-method speeds
    speeds = {m: np.concatenate(by_method[m]) for m in methods}

    # ---- Summary table ----
    # We need per-step dt to compute realizable-execution metrics, so
    # rebuild from records.
    cap_h = args.v_cap_headline
    cap_r = args.v_cap_realistic
    rows = []

    print(f"\n{'method':<32s} {'median':>8s} {'p99':>9s} {'max':>9s}  "
          f"{'frac>'+str(int(cap_r)):>9s}  "
          f"{'P(c='+str(int(cap_h))+')':>11s}  "
          f"{'P(c='+str(int(cap_r))+')':>11s}  "
          f"{'P_realiz':>11s}  {'T-factor':>9s}")
    print("-" * 124)

    # Recompute dt per method (concatenate seeds in the same order as speeds[m])
    dts = {m: [] for m in methods}
    for spec in args.traj:
        m, path = spec.split(":", 1)
        rec = TrajectoryRecord.from_npz(path)
        ts = rec.times * float(args.time_scale)
        dts[m].append(np.maximum(np.diff(ts), 1e-12))
    dts = {m: np.concatenate(dts[m]) for m in methods}

    for m in methods:
        v = speeds[m]
        dt = dts[m]
        # Per-step distance in metres (V * dt)
        step_m = v * dt
        # Headline-cap power
        Ph_steps = power_with_cap(v, cap_h, em)
        Eh = float((Ph_steps * dt).sum())
        Tt = float(dt.sum())
        Ph = Eh / max(Tt, 1e-12)
        # Realistic-cap power (V capped, dt unchanged)
        Pr_steps = power_with_cap(v, cap_r, em)
        Er = float((Pr_steps * dt).sum())
        Pr = Er / max(Tt, 1e-12)
        # Realizable-execution power (dt extended where V > cap)
        dt_real = np.maximum(dt, step_m / max(cap_r, 1e-9))
        v_real = step_m / np.maximum(dt_real, 1e-12)
        Pre_steps = power_with_cap(v_real, cap_r, em)
        Ere = float((Pre_steps * dt_real).sum())
        Tre = float(dt_real.sum())
        Pre = Ere / max(Tre, 1e-12)
        T_factor = Tre / max(Tt, 1e-12)
        frac = float((v > cap_r).mean())
        row = {
            "method": m,
            "median_v_mps": float(np.median(v)),
            "p99_v_mps": float(np.percentile(v, 99)),
            "max_v_mps": float(v.max()),
            "frac_v_over_realistic_cap": frac,
            f"P_mean_W_cap{int(cap_h)}": float(Ph),
            f"P_mean_W_cap{int(cap_r)}": float(Pr),
            "P_mean_W_realizable": float(Pre),
            "T_realizable_s": float(Tre),
            "realizable_time_factor": float(T_factor),
            "teleport_factor": float(Pre / max(Ph, 1e-12)),
        }
        rows.append(row)
        print(f"{m:<32s} {row['median_v_mps']:>8.2f} "
              f"{row['p99_v_mps']:>9.2f} {row['max_v_mps']:>9.2f}  "
              f"{frac:>9.3f}  {Ph:>11.1f}  {Pr:>11.1f}  "
              f"{Pre:>11.1f}  {T_factor:>9.2f}")

    with open(os.path.join(args.out_dir, "velocity_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out_dir}/velocity_summary.json")

    # ---- Plots ----
    log_bins = np.geomspace(max(args.xmin, 1e-3), args.xmax, args.n_bins)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=160,
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1]})

    # Panel 1: overlapping histograms (semi-transparent fill, line outline)
    ax = axes[0]
    for m in methods:
        v = np.clip(speeds[m], log_bins[0], log_bins[-1])
        ax.hist(v, bins=log_bins, color=_color(m), alpha=0.35,
                density=True, label=_label(m), histtype="stepfilled")
        ax.hist(v, bins=log_bins, color=_color(m), alpha=0.95,
                density=True, histtype="step", linewidth=1.4)
    ax.set_xscale("log")
    ax.axvline(cap_r, color="#222", ls="--", lw=1.0,
               label=f"realistic cap {cap_r:g} m/s")
    ax.axvline(4.03, color="#888", ls=":", lw=1.0, label=r"$V_{mp}\approx 4$")
    ax.set_xlabel("instantaneous speed  $V$ (m/s)")
    ax.set_ylabel("density (over steps)")
    ax.set_title("Per-step velocity distribution")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)

    # Panel 2: ECDF
    ax = axes[1]
    for m in methods:
        v = np.sort(speeds[m])
        ecdf = np.arange(1, len(v) + 1) / len(v)
        ax.plot(np.maximum(v, args.xmin), ecdf,
                color=_color(m), lw=1.6, label=_label(m))
    ax.set_xscale("log")
    ax.axvline(cap_r, color="#222", ls="--", lw=1.0)
    ax.axvline(4.03, color="#888", ls=":", lw=1.0)
    ax.set_xlabel("instantaneous speed  $V$ (m/s)")
    ax.set_ylabel("ECDF")
    ax.set_title("Cumulative speed distribution")
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.95)

    # Panel 3: bar chart of *realizable* time-extension factor
    # (T_realizable / T_reported). =1 for continuous methods; >> 1 for
    # teleporting baselines, since their reported mission time is too
    # short to physically execute the trajectory.
    ax = axes[2]
    factors = [r["realizable_time_factor"] for r in rows]
    colors  = [_color(m) for m in methods]
    labels  = [_label(m) for m in methods]
    bars = ax.barh(range(len(methods)), factors,
                   color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1.0, color="#222", ls="--", lw=1.0,
               label="continuous (no teleport)")
    ax.set_xscale("log")
    ax.set_xlabel(rf"$T_\mathrm{{realizable}} / T_\mathrm{{reported}}$ at "
                  rf"$V_\max{{=}}{cap_r:g}$ m/s")
    ax.set_title("Time-extension factor")
    ax.grid(True, which="both", alpha=0.3, axis="x")
    for i, f_ in enumerate(factors):
        ax.text(f_, i, f"  {f_:.2f}", va="center", ha="left", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "velocity_distribution.png")
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
