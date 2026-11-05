"""Pick best lambda settings per loss-combo from the sweep results.

Discovers runs under results/sweep_*/, computes coverage + Zeng-energy
+ NFZ metrics via eval_fair.py, applies a combo-specific rank-based
composite score, and emits:

    sweep_winners/
        sweep_metrics.json        full per-run metrics (every lambda)
        winners.json              best lambdas per combo + their metrics
        winners_table.tex         LaTeX summary
        winners_summary.txt       human-readable copy
        grid_<combo>.png          5-panel trajectory grid for that combo
        curves_<combo>.png        prefix-stop curves overlaid (5 lambdas)
        winners_grid.png          5-panel grid of the chosen winners
        winners_curves.png        prefix-stop curves of just the winners

Composite score (lower = better) is rank-based per combo: each metric
contributes its rank-within-combo (1 = best) weighted by the combo
spec. Rank-based means we don't need a separate baseline run for
normalisation, and outliers can't blow up the ordering.

Usage
-----
    python -m experiments.milano.pick_sweep_winners \
        --sweep-root results \
        --milano-grid data/milano_grid_corrected.npy \
        --nfz-center 0.19 0.03 --nfz-radius 0.10 \
        --out-dir results/sweep_winners
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np


# --- Combo specifications ----------------------------------------------------

# Each combo: (name, sweep dir, lambda field names parsed from run dirname,
# scoring spec [(metric, weight)], dir-pattern for run discovery).
# Score: lower rank = better; final score = sum_i w_i * rank_i across runs
# in the combo. For "higher better" metrics (e.g. density_corr) the rank is
# inverted before scoring, controlled by a third tuple entry "min" / "max".

COMBOS = [
    {
        "name": "nfz",
        "dir": "sweep_nfz",
        "lambda_fields": ["lnfz"],
        "score": [
            ("nfz_frac_inside", 5.0, "min"),
            ("fourier_erg",     1.0, "min"),
        ],
        "label_fmt": "λ_nfz={lnfz}",
    },
    {
        "name": "energy",
        "dir": "sweep_energy",
        "lambda_fields": ["le"],
        "score": [
            ("P_mean_W",    2.0, "min"),
            ("fourier_erg", 1.0, "min"),
        ],
        "label_fmt": "λ_E={le}",
    },
    {
        "name": "acc",
        "dir": "sweep_acc",
        "lambda_fields": ["la"],
        "score": [
            ("fourier_erg", 1.0, "min"),
            ("l1",          0.5, "min"),
        ],
        "label_fmt": "λ_A={la}",
    },
    {
        "name": "nfz_energy",
        "dir": "sweep_nfz_energy",
        "lambda_fields": ["lnfz", "le"],
        "score": [
            ("nfz_frac_inside", 5.0, "min"),
            ("P_mean_W",        2.0, "min"),
            ("fourier_erg",     1.0, "min"),
        ],
        "label_fmt": "λ_nfz={lnfz},λ_E={le}",
    },
    {
        "name": "nfz_energy_acc",
        "dir": "sweep_nfz_energy_acc",
        "lambda_fields": ["lnfz", "le", "la"],
        "score": [
            ("nfz_frac_inside", 5.0, "min"),
            ("P_mean_W",        2.0, "min"),
            ("fourier_erg",     1.0, "min"),
        ],
        "label_fmt": "λ_nfz={lnfz},λ_E={le},λ_A={la}",
    },
]


# --- Discovery + tagging -----------------------------------------------------


_LAMBDA_RE = re.compile(r"(lnfz|le|la)([0-9eE+\-.]+)")
_SEED_RE   = re.compile(r"seed(\d+)")


def parse_run_name(run_dirname: str) -> dict:
    """Extract {lnfz, le, la, seed} from a sweep run directory name.

    Names look like: lnfz5_seed42, le1e-3_seed42, lnfz5_le1e-4_seed42, ...
    """
    out = {}
    for m in _LAMBDA_RE.finditer(run_dirname):
        out[m.group(1)] = m.group(2)
    sm = _SEED_RE.search(run_dirname)
    if sm:
        out["seed"] = int(sm.group(1))
    return out


def discover_runs(sweep_root: str):
    """Yield (combo_dict, run_dir, parsed_fields) for every sweep run."""
    for combo in COMBOS:
        combo_root = os.path.join(sweep_root, combo["dir"])
        if not os.path.isdir(combo_root):
            print(f"[discover] skip missing {combo_root}")
            continue
        for entry in sorted(os.listdir(combo_root)):
            run_dir = os.path.join(combo_root, entry)
            traj = os.path.join(run_dir, "trajectory.npz")
            if not os.path.isfile(traj):
                continue
            parsed = parse_run_name(entry)
            yield combo, run_dir, parsed


def tag_for(combo, parsed) -> str:
    """Unique tag combining combo name + lambdas, used as the eval_fair method id."""
    parts = [combo["name"]] + [f"{k}{parsed.get(k, 'NA')}"
                               for k in combo["lambda_fields"]]
    return "__".join(parts)


# --- Metric collection via eval_fair.py --------------------------------------


def run_eval_fair(traj_args, milano_grid, nfz_center, nfz_radius,
                  out_dir) -> dict:
    """Drive eval_fair.py and return the loaded per_run.json (list of dicts)."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "evaluation.eval_fair", "--milano-grid", milano_grid,
           "--out-dir", out_dir]
    for tag, path in traj_args:
        cmd += ["--traj", f"{tag}:{path}"]
    if nfz_center is not None and nfz_radius is not None:
        cmd += ["--nfz-center", f"{nfz_center[0]}", f"{nfz_center[1]}",
                "--nfz-radius", f"{nfz_radius}"]
    print("[eval_fair] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    with open(os.path.join(out_dir, "per_run.json")) as f:
        return json.load(f)


# --- Composite scoring -------------------------------------------------------


def rank(values, mode):
    """Return 1-based fractional ranks of `values`.

    mode='min': smaller = rank 1.  Ties share the average of their slot
    indices (e.g., three values all = 0 occupying slots 1..3 each get
    rank 2.0).  Non-finite values are pushed to the worst slot.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    work = arr.copy()
    if mode == "max":
        work = -work
    work = np.where(np.isfinite(work), work, np.inf)  # bad → worst
    order = np.argsort(work, kind="stable")
    sorted_vals = work[order]
    sorted_ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average
        sorted_ranks[i : j + 1] = avg
        i = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def composite_scores(records, score_spec):
    """records: list of dicts. Returns list of float scores aligned to records."""
    if not records:
        return []
    n = len(records)
    total = np.zeros(n)
    for metric, weight, mode in score_spec:
        vals = [r.get(metric, np.nan) for r in records]
        if all((not np.isfinite(v)) for v in vals):
            print(f"  warning: metric {metric} all-NaN, dropping from score")
            continue
        ranks = rank(vals, mode)
        total += weight * ranks
    return total.tolist()


# --- LaTeX rendering ---------------------------------------------------------


def fmt_metric(name: str, val):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "--"
    if name in ("P_mean_W", "E_total_kJ", "T_total_s", "arc_length_m",
                "E_per_metre_J"):
        return f"{val:.1f}"
    if name == "nfz_frac_inside":
        return f"{val:.3f}"
    if name in ("fourier_erg", "l1"):
        return f"{val:.4f}"
    if name == "density_corr":
        return f"{val:.3f}"
    return f"{val:.4g}"


def render_winners_latex(winners, path):
    """Compact LaTeX table — one row per combo with its winning lambdas + metrics."""
    cols = [("Combo",           lambda w: w["combo"].replace("_", r"\_")),
            ("Lambdas",         lambda w: w["lambdas_str"]),
            ("Fourier $\\mathcal{E}$", lambda w: fmt_metric("fourier_erg", w["metrics"].get("fourier_erg"))),
            ("$\\ell_1$",       lambda w: fmt_metric("l1", w["metrics"].get("l1"))),
            ("$\\bar P$ (W)",   lambda w: fmt_metric("P_mean_W", w["metrics"].get("P_mean_W"))),
            ("NFZ frac",        lambda w: fmt_metric("nfz_frac_inside", w["metrics"].get("nfz_frac_inside"))),
            ("Score",           lambda w: f"{w['score']:.2f}")]
    spec = "l" + "c" * (len(cols) - 1)
    lines = [r"\begin{tabular}{" + spec + r"}",
             r"\toprule",
             " & ".join(c[0] for c in cols) + r" \\",
             r"\midrule"]
    for w in winners:
        row = " & ".join(c[1](w) for c in cols) + r" \\"
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def render_full_grid_latex(per_combo_records, path):
    """Wide table — every (combo, lambda) row with metrics + score, rank within combo."""
    lines = []
    for combo_name, recs in per_combo_records.items():
        lines.append(r"\paragraph{" + combo_name.replace("_", r"\_") + "}")
        lines.append(r"\begin{tabular}{lcccccc}")
        lines.append(r"\toprule")
        lines.append(r"Lambdas & Fourier $\mathcal{E}$ & $\ell_1$ & "
                     r"$\bar P$ (W) & NFZ frac & Score & Rank \\")
        lines.append(r"\midrule")
        recs_sorted = sorted(recs, key=lambda r: r["score"])
        for i, r in enumerate(recs_sorted):
            row = " & ".join([
                r["lambdas_str"],
                fmt_metric("fourier_erg", r["metrics"].get("fourier_erg")),
                fmt_metric("l1", r["metrics"].get("l1")),
                fmt_metric("P_mean_W", r["metrics"].get("P_mean_W")),
                fmt_metric("nfz_frac_inside", r["metrics"].get("nfz_frac_inside")),
                f"{r['score']:.2f}",
                f"{i+1}",
            ])
            lines.append(row + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# --- Plot drivers ------------------------------------------------------------


def safe_subprocess(cmd):
    print("[run] " + " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"  warning: command failed (rc={res.returncode})")


def plot_combo_grid(combo_records, milano_grid, nfz_center, nfz_radius,
                     out_path, suptitle):
    """5-panel trajectory grid via plot_all_methods.py."""
    if not combo_records:
        return
    cmd = [sys.executable, "-m", "plotting.plot_all_methods",
           "--milano-grid", milano_grid, "--out", out_path,
           "--ncols", "5", "--suptitle", suptitle]
    if nfz_center is not None and nfz_radius is not None:
        cmd += ["--nofly", f"{nfz_center[0]}", f"{nfz_center[1]}",
                f"{nfz_radius}"]
    for r in combo_records:
        cmd += ["--traj", f"{r['short_label']}:{r['traj_path']}"]
    safe_subprocess(cmd)


def plot_combo_curves(combo_records, out_dir):
    """Prefix-stop curves overlaid via plot_curves.py.

    Requires that eval_prefix_stop.py has already produced
    <run_dir>/curves/prefix_curve.npz for each run.
    """
    if not combo_records:
        return
    # Filter to runs that have a prefix curve
    have_curves = []
    for r in combo_records:
        cp = os.path.join(r["run_dir"], "curves", "prefix_curve.npz")
        if os.path.isfile(cp):
            have_curves.append((r, cp))
    if not have_curves:
        print(f"  no prefix curves found for combo, skipping curve plot")
        return
    cmd = [sys.executable, "-m", "plotting.plot_curves", "--out-dir", out_dir]
    for r, cp in have_curves:
        cmd += ["--curve", f"{r['short_label']}:{cp}"]
    safe_subprocess(cmd)


# --- Main --------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", default="results",
                   help="Directory containing sweep_<combo>/ subdirs.")
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--nfz-center", nargs=2, type=float, default=[0.19, 0.03])
    p.add_argument("--nfz-radius", type=float, default=0.10)
    p.add_argument("--out-dir", default="results/sweep_winners")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Discover runs across all combos
    runs = list(discover_runs(args.sweep_root))
    print(f"[discover] {len(runs)} runs across {len(COMBOS)} combos")
    if not runs:
        print("no runs found, exiting"); return

    # 2. Drive eval_fair.py once for all runs
    traj_args = []
    tag_to_run = {}  # tag -> dict
    for combo, run_dir, parsed in runs:
        tag = tag_for(combo, parsed)
        traj_path = os.path.join(run_dir, "trajectory.npz")
        traj_args.append((tag, traj_path))
        tag_to_run[tag] = {
            "combo": combo, "run_dir": run_dir, "parsed": parsed,
            "traj_path": traj_path}
    fair_dir = os.path.join(args.out_dir, "fair")
    per_run = run_eval_fair(traj_args, args.milano_grid,
                            args.nfz_center, args.nfz_radius, fair_dir)
    metric_by_tag = {r["method"]: r for r in per_run}

    # 3. Group by combo, compute composite scores
    per_combo_records = defaultdict(list)
    for tag, info in tag_to_run.items():
        m = metric_by_tag.get(tag, {})
        info["metrics"] = m
        # human-readable labels
        info["lambdas_str"] = ", ".join(
            f"{k}={info['parsed'].get(k, 'NA')}"
            for k in info["combo"]["lambda_fields"])
        info["short_label"] = info["combo"]["label_fmt"].format(**{
            k: info["parsed"].get(k, "NA")
            for k in info["combo"]["lambda_fields"]})
        per_combo_records[info["combo"]["name"]].append(info)

    winners = []
    for combo in COMBOS:
        recs = per_combo_records.get(combo["name"], [])
        if not recs:
            print(f"[score] skip {combo['name']} (no runs)")
            continue
        scores = composite_scores([r["metrics"] for r in recs], combo["score"])
        for r, s in zip(recs, scores):
            r["score"] = float(s)
        recs_sorted = sorted(recs, key=lambda r: r["score"])
        per_combo_records[combo["name"]] = recs_sorted
        best = recs_sorted[0]
        print(f"[score] {combo['name']}: best = {best['lambdas_str']}  "
              f"score={best['score']:.2f}")
        winners.append({
            "combo": combo["name"],
            "lambdas": {k: best["parsed"].get(k) for k in combo["lambda_fields"]},
            "lambdas_str": best["lambdas_str"],
            "short_label": best["short_label"],
            "metrics": {k: best["metrics"].get(k) for k in (
                "fourier_erg", "l1", "density_corr", "P_mean_W",
                "E_total_kJ", "T_total_s", "arc_length_m",
                "E_per_metre_J", "nfz_frac_inside")},
            "score": best["score"],
            "run_dir": best["run_dir"],
            "traj_path": best["traj_path"],
        })

    # 4. Emit JSON / TXT / TeX
    sweep_path = os.path.join(args.out_dir, "sweep_metrics.json")
    with open(sweep_path, "w") as f:
        # Strip nested combo dict before serialising
        out = {name: [{
            "lambdas_str": r["lambdas_str"],
            "metrics": r["metrics"],
            "score": r["score"],
            "run_dir": r["run_dir"]} for r in recs]
            for name, recs in per_combo_records.items()}
        json.dump(out, f, indent=2, default=float)
    print(f"  wrote {sweep_path}")

    win_path = os.path.join(args.out_dir, "winners.json")
    with open(win_path, "w") as f:
        json.dump(winners, f, indent=2, default=float)
    print(f"  wrote {win_path}")

    render_winners_latex(winners, os.path.join(args.out_dir, "winners_table.tex"))
    render_full_grid_latex(per_combo_records,
                           os.path.join(args.out_dir, "sweep_grid_table.tex"))

    # plain-text summary
    with open(os.path.join(args.out_dir, "winners_summary.txt"), "w") as f:
        for w in winners:
            f.write(f"== {w['combo']} ==\n")
            f.write(f"  lambdas    : {w['lambdas_str']}\n")
            f.write(f"  score      : {w['score']:.3f}\n")
            for k, v in w["metrics"].items():
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    continue
                f.write(f"  {k:18s}: {fmt_metric(k, v)}\n")
            f.write("\n")
    print(f"  wrote {os.path.join(args.out_dir, 'winners_summary.txt')}")

    # 5. Per-combo trajectory grid + curves
    for combo in COMBOS:
        recs = per_combo_records.get(combo["name"], [])
        if not recs:
            continue
        plot_combo_grid(
            recs, args.milano_grid, args.nfz_center, args.nfz_radius,
            os.path.join(args.out_dir, f"grid_{combo['name']}.png"),
            suptitle=f"{combo['name']}: 5 lambda settings")
        plot_combo_curves(recs, os.path.join(args.out_dir, f"curves_{combo['name']}"))

    # 6. Winners-only grid + curves
    if winners:
        winners_records = [{
            "short_label": w["short_label"] + f" [{w['combo']}]",
            "traj_path": w["traj_path"],
            "run_dir": w["run_dir"],
        } for w in winners]
        plot_combo_grid(
            winners_records, args.milano_grid,
            args.nfz_center, args.nfz_radius,
            os.path.join(args.out_dir, "winners_grid.png"),
            suptitle="Sweep winners (one per loss combo)")
        plot_combo_curves(winners_records,
                          os.path.join(args.out_dir, "winners_curves"))

    print("done.")


if __name__ == "__main__":
    main()
