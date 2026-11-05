"""Appendix: fleet separation diagnostics.

N independent latent trajectories share one trained OT-CFM map
(same fleet construction as eval_multi_agent.py, same seed base).
We report pairwise separation statistics in physical metres under
two dispatch conventions:

  sync   -- all agents start their cycle at the same instant
            (worst case: agents co-locate near the base station).
  phase  -- each agent gets an independent uniform phase offset,
            matching the paper's Phi_0 ~ Uniform[0,2) convention
            (Proposition 1). Implemented as a random circular roll.

For each convention and fleet size N we report min / p1 / median
pairwise separation and the fraction of time any pair is closer
than 5 m and 10 m. A base-gated variant excludes samples where
either agent's latent radius is < 0.15 (the base neighbourhood,
which the paper excludes from mission time).

Usage:
    python eval_fleet_separation.py \
        --model-dir results/otcfm_seed42 \
        --fleet-sizes 2 5 10 20 --domain-size 200.0 \
        --out-dir results/appendix/fleet_separation
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latent_trajectory import radial_trajectory
from ot_cfm import build_model, transport


def pair_stats(pos_phys, lat_r, half_domain_m, thresholds, base_gate=None,
               rng=None, phase_roll=False):
    """pos_phys: list of (T,2) arrays in [-1,1]; lat_r: list of (T,) latent radii."""
    N = len(pos_phys)
    T = min(p.shape[0] for p in pos_phys)
    P = np.stack([p[:T] for p in pos_phys])          # (N, T, 2)
    R = np.stack([r[:T] for r in lat_r])             # (N, T)
    if phase_roll:
        assert rng is not None
        for k in range(N):
            off = int(rng.integers(0, T))
            P[k] = np.roll(P[k], off, axis=0)
            R[k] = np.roll(R[k], off, axis=0)
    dmins = []
    below = {th: 0.0 for th in thresholds}
    n_gated = 0
    all_d = []
    for a in range(N):
        for b in range(a + 1, N):
            d = np.linalg.norm(P[a] - P[b], axis=1) * half_domain_m  # metres
            if base_gate is not None:
                keep = (R[a] >= base_gate) & (R[b] >= base_gate)
                d = d[keep]
                n_gated += int((~keep).sum())
                if d.size == 0:
                    continue
            dmins.append(d.min())
            all_d.append(d)
            for th in thresholds:
                below[th] += float((d < th).mean())
    n_pairs = max(len(dmins), 1)
    cat = np.concatenate(all_d) if all_d else np.array([np.nan])
    return {
        "min_sep_m": float(np.min(dmins)) if dmins else float("nan"),
        "p1_sep_m": float(np.percentile(cat, 1)),
        "median_sep_m": float(np.median(cat)),
        **{f"frac_time_below_{th}m": below[th] / n_pairs for th in thresholds},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="results/otcfm_seed42")
    p.add_argument("--fleet-sizes", nargs="+", type=int, default=[2, 5, 10, 20])
    p.add_argument("--n-cycles", type=int, default=128)
    p.add_argument("--pts-per-cycle", type=int, default=200)
    p.add_argument("--transport-steps", type=int, default=30)
    p.add_argument("--domain-size", type=float, default=200.0)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--base-gate", type=float, default=0.15,
                   help="latent-radius gate for the base neighbourhood")
    p.add_argument("--phase-seeds", type=int, default=5,
                   help="number of random phase draws for the phase variant")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.model_dir, "plan_meta.json")) as f:
        meta = json.load(f)
    delta = float(meta.get("delta", 0.05))
    model = build_model(arch=meta.get("arch", "v1"),
                        hidden_dim=int(meta.get("hidden_dim", 256)),
                        n_layers=int(meta.get("n_layers", 12)),
                        n_time_freqs=int(meta.get("n_time_freqs", 16)))
    sd = torch.load(os.path.join(args.model_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded {args.model_dir}/model.pt  delta={delta}")

    max_N = max(args.fleet_sizes)
    pos_phys, lat_r = [], []
    for k in range(max_N):
        pos_lat, t_lat, _ = radial_trajectory(
            n_cycles=args.n_cycles, points_per_cycle=args.pts_per_cycle,
            delta=delta, seed=args.seed_base + 1000 * k)
        mask = t_lat > 2.0  # same burn-in as eval_multi_agent.py
        pos_lat = pos_lat[mask]
        pp = transport(model, pos_lat, n_steps=args.transport_steps,
                       device="cpu")
        pos_phys.append(pp)
        lat_r.append(np.linalg.norm(pos_lat, axis=1))
        print(f"  agent {k:2d} transported ({pp.shape[0]} samples)")

    half_domain = args.domain_size / 2.0  # 1 unit in [-1,1] = D/2 metres
    thresholds = (5.0, 10.0)
    rows = []
    for N in args.fleet_sizes:
        row = {"N": N}
        row["sync"] = pair_stats(pos_phys[:N], lat_r[:N], half_domain,
                                 thresholds)
        row["sync_base_gated"] = pair_stats(pos_phys[:N], lat_r[:N],
                                            half_domain, thresholds,
                                            base_gate=args.base_gate)
        # Phase convention: average over several random phase draws
        ph = []
        for ps in range(args.phase_seeds):
            rng = np.random.default_rng(1234 + ps)
            ph.append(pair_stats(pos_phys[:N], lat_r[:N], half_domain,
                                 thresholds, rng=rng, phase_roll=True))
        row["phase"] = {k: float(np.mean([x[k] for x in ph]))
                        for k in ph[0]}
        row["phase_min_over_draws"] = float(min(x["min_sep_m"] for x in ph))
        rows.append(row)
        print(f"N={N:2d}  sync min={row['sync']['min_sep_m']:.2f} m  "
              f"gated min={row['sync_base_gated']['min_sep_m']:.2f} m  "
              f"phase min(mean)={row['phase']['min_sep_m']:.2f} m")

    out = {"args": vars(args), "delta": delta, "rows": rows}
    path = os.path.join(args.out_dir, "fleet_separation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
