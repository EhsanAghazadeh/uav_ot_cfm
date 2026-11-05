"""Multi-agent reuse ablation.

Loads a trained OT-CFM checkpoint and tests Theorem 4 (collective
$O(1/\\sqrt{NK})$ convergence): N independent latent trajectories
with phase-staggered angle seeds are pushed through the *same*
$G_\\theta$, and the aggregate Fourier ergodic metric is reported as
N grows.

Two outputs:
    multi_agent.json  -- per-N: aggregate eps, median per-agent P_bar
    multi_agent.png   -- 2-panel figure (eps vs N with O(1/sqrt(N))
                         reference, median P vs N showing fleet-invariance)

Usage:
    python -m evaluation.eval_multi_agent \\
        --model-dir results/otcfm_seed42 \\
        --milano-grid data/milano_grid_corrected.npy \\
        --fleet-sizes 1 2 5 10 20 \\
        --domain-size 200.0 --time-scale 10.0 \\
        --out-dir results/multi_agent
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ergodic_cfm.latent_trajectory import radial_trajectory
from ergodic_cfm.ot_cfm import build_model, transport
from ergodic_cfm.target_density import load_milano_target
from ergodic_cfm.baselines.milano_baselines import RotaryWingEnergyModel


# --- Fourier-coefficient helpers (mirror eval_fair internals) ----------------


def _build_basis(n_basis):
    K = []
    for i in range(n_basis):
        for j in range(n_basis):
            K.append([i, j])
    K = np.asarray(K, dtype=np.float64)
    hk = np.where((K[:, 0] == 0) & (K[:, 1] == 0), 1.0, np.sqrt(0.5))
    norm = np.linalg.norm(K, axis=1)
    lam = (1.0 + norm) ** -1.5
    return K, hk, lam


def _phik_from_target(grid, K, hk):
    h, w = grid.shape
    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)
    p = grid / grid.sum()
    phik = np.zeros(len(K))
    for ki, k in enumerate(K):
        cosx = np.cos(np.pi * k[0] * xs)
        cosy = np.cos(np.pi * k[1] * ys)
        basis = np.outer(cosy, cosx) / hk[ki]
        phik[ki] = (basis * p).sum()
    return phik


def fourier_ergodic(pos_unit, phik, K, hk, lam):
    ck = np.zeros(len(K))
    for ki, k in enumerate(K):
        cosx = np.cos(np.pi * k[0] * pos_unit[:, 0])
        cosy = np.cos(np.pi * k[1] * pos_unit[:, 1])
        ck[ki] = ((cosx * cosy) / hk[ki]).mean()
    return float((lam * (ck - phik) ** 2).sum())


# --- Main --------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True,
                   help="Trained OT-CFM checkpoint directory.")
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--fleet-sizes", nargs="+", type=int,
                   default=[1, 2, 5, 10, 20],
                   help="Fleet sizes N to evaluate. Max determines how many "
                        "agents we transport.")
    p.add_argument("--n-cycles", type=int, default=128)
    p.add_argument("--pts-per-cycle", type=int, default=200)
    p.add_argument("--transport-steps", type=int, default=30)
    p.add_argument("--domain-size", type=float, default=200.0)
    p.add_argument("--time-scale", type=float, default=10.0)
    p.add_argument("--n-basis", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--v-clip-max", type=float, default=100.0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model from plan_meta.json
    with open(os.path.join(args.model_dir, "plan_meta.json")) as f:
        meta = json.load(f)
    arch = meta.get("arch", "v1")
    hidden_dim = int(meta.get("hidden_dim", 256))
    n_layers = int(meta.get("n_layers", 12))
    n_time_freqs = int(meta.get("n_time_freqs", 16))
    delta = float(meta.get("delta", 0.05))
    extra = {}
    if "n_spatial_freqs" in meta:
        extra["n_spatial_freqs"] = int(meta["n_spatial_freqs"])
    if "sigma_x" in meta:
        extra["sigma_x"] = float(meta["sigma_x"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch=arch, hidden_dim=hidden_dim, n_layers=n_layers,
                        n_time_freqs=n_time_freqs, **extra).to(device)
    sd = torch.load(os.path.join(args.model_dir, "model.pt"),
                    map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded {args.model_dir}/model.pt  arch={arch}  device={device}")

    # Target (for Fourier basis)
    t_target = load_milano_target(args.milano_grid,
                                  smooth_sigma=args.smooth_sigma)
    target_grid = t_target["grid"]
    K, hk, lam = _build_basis(args.n_basis)
    phik = _phik_from_target(target_grid, K, hk)

    # Generate max_N agents, each with a different angular phase seed
    max_N = max(args.fleet_sizes)
    print(f"transporting {max_N} agents (each {args.n_cycles} cycles, "
          f"{args.pts_per_cycle} pts/cycle, RK4 steps={args.transport_steps})")
    agent_pos_unit = []
    agent_v_phys = []
    em = RotaryWingEnergyModel()
    for k in range(max_N):
        pos_lat, t_lat, _ = radial_trajectory(
            n_cycles=args.n_cycles, points_per_cycle=args.pts_per_cycle,
            delta=delta, seed=args.seed_base + 1000 * k)
        if args.n_cycles > 1:
            mask = t_lat > 2.0
            pos_lat = pos_lat[mask]
            t_lat = t_lat[mask]
        pos_phys = transport(model, pos_lat, n_steps=args.transport_steps,
                             device=device)
        # Map [-1,1] -> [0,1] for Fourier basis (matches eval_fair)
        pos_unit_01 = (pos_phys + 1.0) / 2.0
        # Times scaled to UAV-mission protocol
        times = (t_lat - t_lat[0]) * float(args.time_scale)
        dt = float(times[1] - times[0])
        # Speeds in physical units
        step_m = (np.linalg.norm(np.diff(pos_phys, axis=0), axis=1)
                  * args.domain_size)
        v_mps = step_m / max(dt, 1e-12)
        agent_pos_unit.append(pos_unit_01)
        agent_v_phys.append(v_mps)
        print(f"  agent {k:2d}: T={times[-1]:.1f}s  mean v={v_mps.mean():.2f} m/s")

    # Per-fleet-size aggregate metrics
    rows = []
    for N in args.fleet_sizes:
        if N > max_N:
            continue
        all_pos = np.vstack(agent_pos_unit[:N])
        eps_N = fourier_ergodic(all_pos, phik, K, hk, lam)
        per_agent_p = []
        for v_mps in agent_v_phys[:N]:
            v_clipped = np.clip(v_mps, 0.01, float(args.v_clip_max))
            P = np.array([em.power(float(v)) for v in v_clipped])
            per_agent_p.append(float(P.mean()))
        per_agent_p = np.asarray(per_agent_p)
        row = {
            "N": int(N),
            "fourier_erg": float(eps_N),
            "median_P_W": float(np.median(per_agent_p)),
            "mean_P_W": float(per_agent_p.mean()),
            "std_P_W": float(per_agent_p.std()),
        }
        rows.append(row)
        print(f"  N={N:2d}  eps={eps_N:.5e}  median P={row['median_P_W']:.1f} W")

    json_path = os.path.join(args.out_dir, "multi_agent.json")
    with open(json_path, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    print(f"wrote {json_path}")

    # 2-panel figure
    Ns = np.array([r["N"] for r in rows], dtype=float)
    epsN = np.array([r["fourier_erg"] for r in rows])
    pN = np.array([r["median_P_W"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)

    ax = axes[0]
    ax.loglog(Ns, epsN, "o-", lw=1.7, ms=8, color="#1f77b4",
              label="Aggregate $\\mathcal{E}_N$")
    ref = epsN[0] * np.sqrt(Ns[0] / Ns)
    ax.loglog(Ns, ref, "--", color="#888", lw=1.0,
              label=r"$\mathcal{E}_1\,\sqrt{1/N}$ ref")
    ax.set_xlabel("Fleet size $N$")
    ax.set_ylabel(r"Aggregate Fourier ergodic $\mathcal{E}$")
    ax.set_title("Multi-agent collective convergence")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")

    ax = axes[1]
    ax.semilogx(Ns, pN, "s-", lw=1.7, ms=8, color="#d62728")
    ax.set_xlabel("Fleet size $N$")
    ax.set_ylabel(r"Median per-agent $\bar P$ (W)")
    ax.set_title("Per-agent power vs fleet size (expect flat)")
    ax.grid(True, alpha=0.3, which="both")
    # y-range padding so the curve isn't a flat line at the top
    ymid = float(np.median(pN))
    ax.set_ylim(ymid * 0.5, ymid * 1.5)

    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "multi_agent.png")
    fig.savefig(fig_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
