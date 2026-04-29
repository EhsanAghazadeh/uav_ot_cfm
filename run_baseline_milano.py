"""Unified baseline dispatcher for the Milano comparison.

Runs one of {toes, nonuniform_erg, opt_transport, sac} on the smoothed
Milano density and saves a canonical trajectory.npz at <out-dir>/trajectory.npz
so eval_prefix_stop.py + plot_curves.py work unchanged across all methods.

Examples:
  python run_baseline_milano.py --method toes \
      --t-total 320 --n-knots 6400 --out-dir results/toes_milano

  python run_baseline_milano.py --method nonuniform_erg \
      --num-intervals 32 --points-per-interval 200 \
      --out-dir results/nonuniform_erg_milano

  python run_baseline_milano.py --method opt_transport \
      --num-points 12800 --out-dir results/opt_transport_milano

  python run_baseline_milano.py --method sac \
      --num-points 12800 --sac-train-steps 200000 \
      --out-dir results/sac_milano
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fair_comparison import save_trajectory_npz
from milano_baselines import (
    MilanoRateMap,
    NonUniformErgodic,
    OptimalTransportCoverage,
    SACTrajectoryGenerator,
)


# ---------------------------------------------------------------------------
# TOES leg (mirrors run_toes_milano.main but as a callable)
# ---------------------------------------------------------------------------
def _remap_to_physical(pos_unit, xlim, ylim):
    x = pos_unit[..., 0] * (xlim[1] - xlim[0]) + xlim[0]
    y = pos_unit[..., 1] * (ylim[1] - ylim[0]) + ylim[0]
    return np.stack([x, y], axis=-1)


def plan_toes(args):
    from scipy.ndimage import gaussian_filter
    from toes_baseline import TOESConfig, solve_toes

    grid = np.load(args.milano_grid).astype(np.float64)
    if args.smooth_sigma > 0:
        grid = gaussian_filter(grid, sigma=float(args.smooth_sigma))
    grid = np.maximum(grid, args.eps_floor)
    xlim, ylim = (-1.0, 1.0), (-1.0, 1.0)

    cfg = TOESConfig(
        mode="fixed_time",
        N=args.n_knots,
        n_basis=args.n_basis,
        u_max=args.u_max,
        workspace_bnds=((0., 1.), (0., 1.)),
        step_size=args.step_size,
        c_init=args.c_init,
        max_iter=args.max_iter,
        convergence_eps=args.convergence_eps,
        T_total=args.t_total,
    )
    print(f"TOES horizon : T={args.t_total}s, N={args.n_knots}")
    t0 = time.perf_counter()
    sol = solve_toes(phi_vals=grid, xlim=xlim, ylim=ylim, cfg=cfg,
                    x0_phys=np.asarray(args.x0), xf_phys=np.asarray(args.xf))
    plan_s = time.perf_counter() - t0
    pos = _remap_to_physical(sol.pos_unit, xlim, ylim)
    times = sol.dt * np.arange(sol.pos_unit.shape[0])
    meta = {
        "method_kind": "toes",
        "T_total_s": float(sol.tf),
        "dt_s": float(sol.dt),
        "N": int(pos.shape[0]),
        "mode": "fixed_time",
        "n_basis": cfg.n_basis,
        "u_max": cfg.u_max,
        "step_size": cfg.step_size,
        "max_iter": cfg.max_iter,
        "n_iter_run": int(sol.n_iter),
        "erg_metric_fixed_horizon": float(sol.erg_metric),
        "milano_smooth_sigma": float(args.smooth_sigma),
        "x0_phys": [float(x) for x in args.x0],
        "xf_phys": [float(x) for x in args.xf],
        "plan_wall_clock_s": float(plan_s),
    }
    return pos, times, meta, "toes_fixed_horizon"


# ---------------------------------------------------------------------------
# Polar baselines (ergodic / opt-transport / SAC)
# ---------------------------------------------------------------------------
def plan_nonuniform_erg(args):
    rate_map = MilanoRateMap(args.milano_grid, smooth_sigma=args.smooth_sigma)
    t0 = time.perf_counter()
    gen = NonUniformErgodic(rate_map, phi_max=args.phi_max,
                            alpha_res=args.alpha_res, s_res=args.s_res)
    pos = gen.generate(num_intervals=args.num_intervals,
                       points_per_interval=args.points_per_interval,
                       seed=args.seed)
    plan_s = time.perf_counter() - t0
    times = args.dt * np.arange(pos.shape[0])
    meta = {
        "method_kind": "nonuniform_erg",
        "num_intervals": int(args.num_intervals),
        "points_per_interval": int(args.points_per_interval),
        "phi_max": float(args.phi_max),
        "alpha_res": int(args.alpha_res),
        "s_res": int(args.s_res),
        "dt_s": float(args.dt),
        "T_total_s": float(times[-1]),
        "N": int(pos.shape[0]),
        "milano_smooth_sigma": float(args.smooth_sigma),
        "seed": int(args.seed),
        "plan_wall_clock_s": float(plan_s),
    }
    return pos, times, meta, "nonuniform_erg_milano"


def plan_opt_transport(args):
    rate_map = MilanoRateMap(args.milano_grid, smooth_sigma=args.smooth_sigma)
    t0 = time.perf_counter()
    gen = OptimalTransportCoverage(rate_map, grid_resolution=args.ot_grid_res)
    pos = gen.generate(num_points=args.num_points, seed=args.seed)
    plan_s = time.perf_counter() - t0
    times = args.dt * np.arange(pos.shape[0])
    meta = {
        "method_kind": "optimal_transport",
        "num_points": int(args.num_points),
        "ot_grid_res": int(args.ot_grid_res),
        "dt_s": float(args.dt),
        "T_total_s": float(times[-1]),
        "N": int(pos.shape[0]),
        "milano_smooth_sigma": float(args.smooth_sigma),
        "seed": int(args.seed),
        "plan_wall_clock_s": float(plan_s),
    }
    return pos, times, meta, "opt_transport_milano"


def plan_sac(args):
    rate_map = MilanoRateMap(args.milano_grid, smooth_sigma=args.smooth_sigma)
    gen = SACTrajectoryGenerator(rate_map, grid_size=args.sac_grid_size,
                                 comm_range=args.sac_comm_range,
                                 train_steps=args.sac_train_steps)
    print(f"training SAC  : {args.sac_train_steps} steps "
          f"(grid={args.sac_grid_size}, comm_range={args.sac_comm_range})")
    t0 = time.perf_counter()
    gen.train(seed=args.seed)
    train_s = time.perf_counter() - t0
    print(f"  train wall t : {train_s:.1f}s")
    print(f"rolling out   : up to {args.num_points} points")
    t0 = time.perf_counter()
    pos = gen.generate(num_points=args.num_points, seed=args.seed)
    rollout_s = time.perf_counter() - t0
    plan_s = float(train_s + rollout_s)
    times = args.dt * np.arange(pos.shape[0])
    meta = {
        "method_kind": "sac",
        "num_points": int(args.num_points),
        "sac_grid_size": int(args.sac_grid_size),
        "sac_comm_range": float(args.sac_comm_range),
        "sac_train_steps": int(args.sac_train_steps),
        "train_wall_clock_s": float(train_s),
        "rollout_wall_clock_s": float(rollout_s),
        "dt_s": float(args.dt),
        "T_total_s": float(times[-1]),
        "N": int(pos.shape[0]),
        "milano_smooth_sigma": float(args.smooth_sigma),
        "seed": int(args.seed),
        "plan_wall_clock_s": plan_s,
    }
    return pos, times, meta, "sac_milano"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                  choices=["toes", "nonuniform_erg", "opt_transport", "sac"])
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--eps-floor", type=float, default=1e-12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dt", type=float, default=0.1,
                  help="time step (s) for polar baselines (Zeng convention)")
    p.add_argument("--out-dir", required=True)

    # TOES
    p.add_argument("--t-total", type=float, default=320.0)
    p.add_argument("--n-knots", type=int, default=6400)
    p.add_argument("--n-basis", type=int, default=8)
    p.add_argument("--u-max", type=float, default=6.0)
    p.add_argument("--step-size", type=float, default=5e-4)
    p.add_argument("--c-init", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=15000)
    p.add_argument("--convergence-eps", type=float, default=1e-7)
    p.add_argument("--x0", type=float, nargs=4, default=[-0.8, -0.8, 0.0, 0.0])
    p.add_argument("--xf", type=float, nargs=4, default=[0.8, 0.8, 0.0, 0.0])

    # nonuniform_erg
    p.add_argument("--num-intervals", type=int, default=32)
    p.add_argument("--points-per-interval", type=int, default=200)
    p.add_argument("--phi-max", type=float, default=np.pi / 5)
    p.add_argument("--alpha-res", type=int, default=200)
    p.add_argument("--s-res", type=int, default=100)

    # opt_transport
    p.add_argument("--num-points", type=int, default=12800,
                  help="rollout length (opt_transport, sac)")
    p.add_argument("--ot-grid-res", type=int, default=15)

    # sac
    p.add_argument("--sac-grid-size", type=int, default=50)
    p.add_argument("--sac-comm-range", type=float, default=0.18)
    p.add_argument("--sac-train-steps", type=int, default=200_000)

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"method       : {args.method}")
    print(f"target       : {args.milano_grid} (smoothed σ={args.smooth_sigma})")
    if args.method == "toes":
        pos, times, meta, label = plan_toes(args)
    elif args.method == "nonuniform_erg":
        pos, times, meta, label = plan_nonuniform_erg(args)
    elif args.method == "opt_transport":
        pos, times, meta, label = plan_opt_transport(args)
    elif args.method == "sac":
        pos, times, meta, label = plan_sac(args)
    else:
        raise SystemExit(f"unknown method {args.method!r}")

    out = os.path.join(args.out_dir, "trajectory.npz")
    save_trajectory_npz(
        path=out, pos=pos.astype(np.float64), times=times.astype(np.float64),
        domain=(-1.0, 1.0, -1.0, 1.0), method=label, meta=meta)
    with open(os.path.join(args.out_dir, "plan_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"plan wall t  : {meta['plan_wall_clock_s']:.2f}s")
    print(f"trajectory   : N={meta['N']}, dt={meta['dt_s']:.5g}s, "
          f"T={meta['T_total_s']:.2f}s")
    print(f"saved        : {out}")


if __name__ == "__main__":
    main()
