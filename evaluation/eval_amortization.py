"""Wall-clock / amortization comparison table.

Walks results/<method>_seed*/ and results_nfz_masked/<method>_seed*/,
parses plan_meta.json for per-method wall-clock, and emits a LaTeX
table summarising:

    Per-scenario wall-clock     -- single-seed mean (and std)
    Multi-agent (N=10) cost     -- qualitative: 10x retrain or shared
    Per-new-target cost         -- qualitative: full re-derive vs amortized

OT-CFM is the only row whose multi-agent and per-new-target columns
are not "full re-...". That's the structural amortization claim from
the abstract; this table is the empirical instantiation.

Usage:
    python -m evaluation.eval_amortization \
        --root results --out results/fair_metrics_rescaled/amortization.tex
"""
from __future__ import annotations

import argparse, json, os, re
import numpy as np


# Each entry: (display name, dirname prefix matched as <prefix>_seed<S>,
# qualitative costs).
METHODS = [
    {"name": "TOES",
     "prefix": "toes",
     "multi_n10": r"$10\times$ SQP solve",
     "per_target": "full re-solve"},
    {"name": "NonUniformErg",
     "prefix": "nonuniform_erg",
     "multi_n10": r"$10\times$ re-derive",
     "per_target": "full re-derive"},
    {"name": "OT",
     "prefix": "opt_transport",
     "multi_n10": r"$10\times$ OT plan",
     "per_target": "full re-plan"},
    {"name": "SAC",
     "prefix": "sac",
     "multi_n10": r"$10\times$ retrain (or shared at quality cost)",
     "per_target": "full retrain"},
    {"name": "FMEC-Stein",
     "prefix": "fmec_stein",
     "multi_n10": r"$10\times$ re-optimize",
     "per_target": "full re-optimize"},
    {"name": "FMEC-Sinkhorn",
     "prefix": "fmec_sinkhorn",
     "multi_n10": r"$10\times$ re-optimize",
     "per_target": "full re-optimize"},
    # OT-CFM family: training amortizes; inference is cheap
    {"name": r"\textbf{OT-CFM} (ours)",
     "prefix": "otcfm",
     "multi_n10": r"$10\times$ inference ($\sim$ms each)",
     "per_target": r"$0$ (same target/constraints)"},
]


def find_runs(root, prefix):
    """Return list of (entry_dirname, plan_meta_dict) under root that match
    exact `<prefix>_seed<S>`."""
    out = []
    if not os.path.isdir(root):
        return out
    pattern = re.compile(rf"^{re.escape(prefix)}_seed\d+$")
    for entry in sorted(os.listdir(root)):
        if not pattern.match(entry):
            continue
        meta_path = os.path.join(root, entry, "plan_meta.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError:
                continue
        out.append((entry, meta))
    return out


def wall_clock_seconds(meta):
    """Total per-scenario wall-clock (training + inference) in seconds."""
    if "plan_wall_clock_s" in meta:
        return float(meta["plan_wall_clock_s"])
    if "train_wall_clock_s" in meta:
        # OT-CFM: train + latent gen + transport
        t = float(meta["train_wall_clock_s"])
        t += float(meta.get("transport_wall_clock_s", 0.0))
        t += float(meta.get("latent_gen_wall_clock_s", 0.0))
        return t
    s = sum(float(meta.get(k, 0.0)) for k in (
        "train_wall_clock_s", "rollout_wall_clock_s",
        "transport_wall_clock_s", "latent_gen_wall_clock_s"))
    return s if s > 0 else None


def inference_seconds(meta):
    """Per-trajectory inference cost once the model is trained.

    For OT-CFM this is latent generation + ODE transport (milliseconds).
    For baselines there is no training/inference split, so returns None
    (caller renders it as '= per-scenario').
    """
    lat = meta.get("latent_gen_wall_clock_s")
    trp = meta.get("transport_wall_clock_s")
    if lat is not None or trp is not None:
        return float(lat or 0.0) + float(trp or 0.0)
    return None


def fmt_time(t):
    if t is None:
        return "--"
    if t < 1.0:
        return f"{t*1000:.0f} ms"
    if t < 60:
        return f"{t:.1f} s"
    if t < 3600:
        return f"{t/60:.1f} min"
    return f"{t/3600:.2f} h"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", action="append", default=None,
                   help="Directory containing <method>_seed<S>/ subdirs. "
                        "Repeatable — seeds from all roots are pooled per method.")
    p.add_argument("--out", required=True,
                   help="Output .tex path for the LaTeX table.")
    args = p.parse_args()
    if args.root is None:
        args.root = ["results"]

    # Special case: OT-CFM has many sub-variants. We aggregate over the
    # "vanilla" otcfm_seed<S> dirs and ignore otcfm_nfz_l5, otcfm_energy_*
    # etc., to give a representative training time.
    roots_str = " + ".join(args.root)
    rows = []
    print(f"\nAmortization wall-clocks ({roots_str}):")
    print(f"  {'Method':<25s}  {'Per-scenario':<18s}  {'Inference':<18s}  n")
    print("-" * 75)
    for spec in METHODS:
        runs = []
        for root in args.root:
            runs.extend(find_runs(root, spec["prefix"]))
        if not runs:
            mean_wc = None; std_wc = None
            mean_inf = None; n = 0
            mean_str = "--"; inf_str = "--"
        else:
            wcs = [w for w in (wall_clock_seconds(m) for _, m in runs)
                   if w is not None]
            infs = [w for w in (inference_seconds(m) for _, m in runs)
                    if w is not None]
            if wcs:
                mean_wc = float(np.mean(wcs)); std_wc = float(np.std(wcs))
                mean_str = fmt_time(mean_wc)
                if len(wcs) > 1 and std_wc > 0:
                    mean_str += rf" $\pm$ {fmt_time(std_wc)}"
            else:
                mean_wc = None; std_wc = None; mean_str = "--"
            if infs:
                mean_inf = float(np.mean(infs))
                std_inf  = float(np.std(infs))
                inf_str  = fmt_time(mean_inf)
                if len(infs) > 1 and std_inf > 0:
                    inf_str += rf" $\pm$ {fmt_time(std_inf)}"
            else:
                mean_inf = None
                inf_str = r"$=$ per-scenario"
            n = len(wcs)
        rows.append({
            "name": spec["name"],
            "wall_clock": mean_str,
            "wall_clock_s": mean_wc,
            "wall_clock_std_s": std_wc,
            "inference": inf_str,
            "inference_s": mean_inf,
            "multi_n10": spec["multi_n10"],
            "per_target": spec["per_target"],
            "n_seeds": n,
        })
        print(f"  {spec['name']:<25s}  {mean_str:<18s}  {inf_str:<18s}  {n}")

    # Render LaTeX
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w") as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Amortization. Per-scenario wall-clock is the "
                r"per-seed mean on the headline mission. Multi-agent and "
                r"per-new-target columns describe the cost of additional "
                r"queries to a deployed system. Only OT-CFM amortizes the "
                r"training cost across both new agents and (if the target "
                r"and constraint set are unchanged) new flights -- this is "
                r"the structural advantage no metric in Table~\ref{tab:main} "
                r"can show.}" + "\n")
        f.write(r"\label{tab:amortization}" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\setlength{\tabcolsep}{4pt}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Method & Per-scenario & Inference & "
                r"Multi-agent ($N{=}10$) & Per-new-target \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in rows:
            f.write(f"{r['name']} & {r['wall_clock']} & {r['inference']} & "
                    f"{r['multi_n10']} & {r['per_target']} \\\\" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"\nwrote {args.out}")

    json_path = args.out.replace(".tex", ".json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
