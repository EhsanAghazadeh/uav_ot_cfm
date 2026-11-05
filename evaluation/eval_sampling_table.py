"""Sampling-protocol table for the appendix.

Walks <root>/<method>_seed*/ trees, parses plan_meta.json + the per_run
fair-metrics output, and emits a LaTeX table giving, per method:

    N             sample count
    Delta t       per-step sampling rate (s)
    T_reported    N * Delta t  (s)
    T_realizable  realisable-execution time at V_max = 30 m/s (s)
    L             arc length (m, rescaled)
    mean V        mean physical speed (m/s)

The point of this table is to make the sampling-rate asymmetry
explicit: T = N * Delta t is reported per method but Delta t is a
sampling convention, not an algorithm property. The realisable column
is the method-intrinsic time floor for a UAV-realisable execution.

Usage:
    python -m evaluation.eval_sampling_table \
        --per-run results/fair_metrics_rescaled/per_run.json \
        --root results \
        --out results/fair_metrics_rescaled/sampling_protocol.tex
"""
from __future__ import annotations

import argparse, json, os, re
from collections import defaultdict
import numpy as np


def find_meta(root, method, seed):
    d = os.path.join(root, f"{method}_seed{seed}")
    p = os.path.join(d, "plan_meta.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return None


def extract_dt_seconds(meta):
    """Best-effort dt-per-step extraction from a plan_meta.json dict."""
    for k in ("dt_s", "dt", "delta_t"):
        if k in meta:
            try:
                return float(meta[k])
            except (TypeError, ValueError):
                continue
    return None


def fmt_time(t):
    if t is None or not np.isfinite(t):
        return "--"
    if t < 1.0:
        return f"{t*1000:.0f} ms"
    if t < 60:
        return f"{t:.1f} s"
    if t < 3600:
        return f"{t/60:.1f} min"
    return f"{t/3600:.2f} h"


def fmt_int(x):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{int(round(x)):,}".replace(",", r"{,}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-run", required=True,
                   help="per_run.json from eval_fair.py (rescaled).")
    p.add_argument("--root", default="results",
                   help="Root containing <method>_seed<S>/plan_meta.json.")
    p.add_argument("--out", required=True,
                   help="Output .tex path for the LaTeX appendix table.")
    p.add_argument("--label", default="tab:sampling")
    args = p.parse_args()

    with open(args.per_run) as f:
        per_run = json.load(f)

    # Group per-run records by method
    by_method = defaultdict(list)
    for r in per_run:
        by_method[r["method"]].append(r)

    # For dt extraction we need plan_meta.json; the per_run record has a
    # "path" field pointing at trajectory.npz next to plan_meta.json.
    rows = []
    for method, recs in by_method.items():
        Ns = []; dts = []; Ts = []; Trs = []; Ls = []; Vs = []
        for r in recs:
            Ns.append(r.get("N_samples"))
            Ts.append(r.get("T_total_s"))
            Trs.append(r.get("T_realizable_s"))
            Ls.append(r.get("arc_length_m"))
            Vs.append(r.get("mean_speed_mps"))
            # dt: prefer the trajectory's intrinsic value (from meta)
            traj_path = r.get("path", "")
            run_dir = os.path.dirname(traj_path)
            meta_path = os.path.join(run_dir, "plan_meta.json")
            dt_v = None
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    dt_v = extract_dt_seconds(meta)
                except Exception:
                    dt_v = None
            # Fallback: derive from T / (N - 1)
            if dt_v is None and r.get("T_total_s") is not None and r.get("N_samples"):
                N = float(r["N_samples"])
                dt_v = float(r["T_total_s"]) / max(N - 1, 1)
            dts.append(dt_v)

        def _agg(arr):
            arr = [a for a in arr if a is not None and np.isfinite(a)]
            if not arr:
                return None, None
            return float(np.mean(arr)), float(np.std(arr))

        N_m, _    = _agg(Ns)
        dt_m, _   = _agg(dts)
        T_m, _    = _agg(Ts)
        Tr_m, _   = _agg(Trs)
        L_m, _    = _agg(Ls)
        V_m, _    = _agg(Vs)
        rows.append({
            "method": method,
            "N": N_m, "dt": dt_m, "T_reported": T_m,
            "T_realizable": Tr_m, "L": L_m, "V_mean": V_m,
        })

    # Stable ordering by method name
    rows.sort(key=lambda r: r["method"])

    # Render LaTeX
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w") as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Sampling protocol per method. $T_\text{reported}=N\cdot\Delta t$ "
                r"with $\Delta t$ a per-method sampling-rate convention (algorithm "
                r"discretisation, environment-step convention, or latent-cycle pts-per-cycle), "
                r"not a property of the planned trajectory. $T_\text{realizable}$ is the "
                r"mission time required to physically execute the trajectory at "
                r"$V_\max{=}30$ m/s; for continuous methods $T_\text{realizable}\!\approx\!T_\text{reported}$, "
                r"and the gap measures teleport pathology. $\bar V$ is the mean kinematic "
                r"speed at the rescaled domain.}" + "\n")
        f.write(r"\label{" + args.label + "}" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\setlength{\tabcolsep}{4pt}" + "\n")
        f.write(r"\begin{tabular}{lrrrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Method & $N$ & $\Delta t$ (s) & $T_\text{reported}$ "
                r"& $T_\text{realizable}$ & $L$ (m) & $\bar V$ (m/s) \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in rows:
            method = r["method"].replace("_", r"\_")
            N = fmt_int(r["N"])
            dt = (f"{r['dt']:.4g}" if r['dt'] is not None else "--")
            T = fmt_time(r["T_reported"])
            Tr = fmt_time(r["T_realizable"])
            L = (f"{r['L']:.3g}" if r['L'] is not None else "--")
            V = (f"{r['V_mean']:.2f}" if r['V_mean'] is not None else "--")
            f.write(f"{method} & {N} & {dt} & {T} & {Tr} & {L} & {V} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"wrote {args.out}")

    # Plain-text echo for stdout sanity
    print(f"\n{'method':<32s} {'N':>10s} {'dt':>10s} "
          f"{'T_reported':>14s} {'T_realizable':>14s} "
          f"{'L (m)':>10s} {'V (m/s)':>10s}")
    print("-" * 110)
    for r in rows:
        dt_s = "--" if r["dt"] is None else f"{r['dt']:.4g}"
        L_s  = "--" if r["L"] is None else f"{r['L']:.3g}"
        V_s  = "--" if r["V_mean"] is None else f"{r['V_mean']:.2f}"
        print(f"{r['method']:<32s} {fmt_int(r['N']):>10s} {dt_s:>10s} "
              f"{fmt_time(r['T_reported']):>14s} {fmt_time(r['T_realizable']):>14s} "
              f"{L_s:>10s} {V_s:>10s}")


if __name__ == "__main__":
    main()
