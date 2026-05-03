"""Unified baseline dispatcher v1: original methods + Flow Matching Ergodic Coverage.

Extends ``run_baseline_milano.py`` with ``fmec_stein`` and ``fmec_sinkhorn`` without
modifying the original module. Writes the same ``trajectory.npz`` format.

Examples:
  python run_baseline_milano_v1.py --method fmec_stein \\
      --fmec-tsteps 200 --fmec-num-iters 50 --out-dir results/fmec_stein_test

  python run_baseline_milano_v1.py --method fmec_sinkhorn \\
      --fmec-tsteps 200 --out-dir results/fmec_sink_test
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


def _remap_to_physical(pos_unit, xlim, ylim):
    x = pos_unit[..., 0] * (xlim[1] - xlim[0]) + xlim[0]
    y = pos_unit[..., 1] * (ylim[1] - ylim[0]) + ylim[0]
    return np.stack([x, y], axis=-1)


def plan_toes(args):
    from toes_baseline import TOESConfig, solve_toes
    from target_density import load_milano_target

    nfz_c = (
        tuple(args.nfz_center)
        if args.mask_nfz_target and args.nfz_center is not None
        else None
    )
    nfz_r = (
        float(args.nfz_radius)
        if args.mask_nfz_target and args.nfz_radius is not None
        else None
    )
    t = load_milano_target(
        args.milano_grid,
        smooth_sigma=args.smooth_sigma,
        eps_floor=args.eps_floor,
        nfz_center=nfz_c,
        nfz_radius=nfz_r,
    )
    grid = t["grid"]
    xlim, ylim = (-1.0, 1.0), (-1.0, 1.0)

    cfg = TOESConfig(
        mode="fixed_time",
        N=args.n_knots,
        n_basis=args.n_basis,
        u_max=args.u_max,
        workspace_bnds=((0.0, 1.0), (0.0, 1.0)),
        step_size=args.step_size,
        c_init=args.c_init,
        max_iter=args.max_iter,
        convergence_eps=args.convergence_eps,
        T_total=args.t_total,
    )
    print(f"TOES horizon : T={args.t_total}s, N={args.n_knots}")
    t0 = time.perf_counter()
    sol = solve_toes(
        phi_vals=grid,
        xlim=xlim,
        ylim=ylim,
        cfg=cfg,
        x0_phys=np.asarray(args.x0),
        xf_phys=np.asarray(args.xf),
    )
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
        "mask_nfz_target": bool(args.mask_nfz_target),
        "nfz_center": (
            list(args.nfz_center)
            if args.mask_nfz_target and args.nfz_center is not None
            else None
        ),
        "nfz_radius": (
            float(args.nfz_radius)
            if args.mask_nfz_target and args.nfz_radius is not None
            else None
        ),
        "x0_phys": [float(x) for x in args.x0],
        "xf_phys": [float(x) for x in args.xf],
        "plan_wall_clock_s": float(plan_s),
    }
    return pos, times, meta, "toes_fixed_horizon"


def plan_nonuniform_erg(args):
    _nfz_c = (
        tuple(args.nfz_center)
        if args.mask_nfz_target and args.nfz_center is not None
        else None
    )
    _nfz_r = (
        float(args.nfz_radius)
        if args.mask_nfz_target and args.nfz_radius is not None
        else None
    )
    rate_map = MilanoRateMap(
        args.milano_grid,
        smooth_sigma=args.smooth_sigma,
        nfz_center=_nfz_c,
        nfz_radius=_nfz_r,
    )
    t0 = time.perf_counter()
    gen = NonUniformErgodic(
        rate_map,
        phi_max=args.phi_max,
        alpha_res=args.alpha_res,
        s_res=args.s_res,
    )
    pos = gen.generate(
        num_intervals=args.num_intervals,
        points_per_interval=args.points_per_interval,
        seed=args.seed,
    )
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
        "mask_nfz_target": bool(args.mask_nfz_target),
        "nfz_center": (
            list(args.nfz_center)
            if args.mask_nfz_target and args.nfz_center is not None
            else None
        ),
        "nfz_radius": (
            float(args.nfz_radius)
            if args.mask_nfz_target and args.nfz_radius is not None
            else None
        ),
        "seed": int(args.seed),
        "plan_wall_clock_s": float(plan_s),
    }
    return pos, times, meta, "nonuniform_erg_milano"


def plan_opt_transport(args):
    _nfz_c = (
        tuple(args.nfz_center)
        if args.mask_nfz_target and args.nfz_center is not None
        else None
    )
    _nfz_r = (
        float(args.nfz_radius)
        if args.mask_nfz_target and args.nfz_radius is not None
        else None
    )
    rate_map = MilanoRateMap(
        args.milano_grid,
        smooth_sigma=args.smooth_sigma,
        nfz_center=_nfz_c,
        nfz_radius=_nfz_r,
    )
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
        "mask_nfz_target": bool(args.mask_nfz_target),
        "nfz_center": (
            list(args.nfz_center)
            if args.mask_nfz_target and args.nfz_center is not None
            else None
        ),
        "nfz_radius": (
            float(args.nfz_radius)
            if args.mask_nfz_target and args.nfz_radius is not None
            else None
        ),
        "seed": int(args.seed),
        "plan_wall_clock_s": float(plan_s),
    }
    return pos, times, meta, "opt_transport_milano"


def plan_sac(args):
    _nfz_c = (
        tuple(args.nfz_center)
        if args.mask_nfz_target and args.nfz_center is not None
        else None
    )
    _nfz_r = (
        float(args.nfz_radius)
        if args.mask_nfz_target and args.nfz_radius is not None
        else None
    )
    rate_map = MilanoRateMap(
        args.milano_grid,
        smooth_sigma=args.smooth_sigma,
        nfz_center=_nfz_c,
        nfz_radius=_nfz_r,
    )
    gen = SACTrajectoryGenerator(
        rate_map,
        grid_size=args.sac_grid_size,
        comm_range=args.sac_comm_range,
        train_steps=args.sac_train_steps,
    )
    print(
        f"training SAC  : {args.sac_train_steps} steps "
        f"(grid={args.sac_grid_size}, comm_range={args.sac_comm_range})"
    )
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
        "mask_nfz_target": bool(args.mask_nfz_target),
        "nfz_center": (
            list(args.nfz_center)
            if args.mask_nfz_target and args.nfz_center is not None
            else None
        ),
        "nfz_radius": (
            float(args.nfz_radius)
            if args.mask_nfz_target and args.nfz_radius is not None
            else None
        ),
        "seed": int(args.seed),
        "plan_wall_clock_s": plan_s,
    }
    return pos, times, meta, "sac_milano"


def plan_fmec(args, variant: str):
    from flow_matching_coverage import plan_milano_fmec_for_baseline

    return plan_milano_fmec_for_baseline(args, variant)


def _apply_fmec_defaults(args):
    if args.method == "fmec_stein":
        if args.fmec_num_iters is None:
            args.fmec_num_iters = 500
        if args.fmec_step_size is None:
            args.fmec_step_size = 0.01
    elif args.method == "fmec_sinkhorn":
        if args.fmec_num_iters is None:
            args.fmec_num_iters = 200
        if args.fmec_step_size is None:
            args.fmec_step_size = 0.005


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        required=True,
        choices=[
            "toes",
            "nonuniform_erg",
            "opt_transport",
            "sac",
            "fmec_stein",
            "fmec_sinkhorn",
        ],
    )
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--eps-floor", type=float, default=1e-12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="time step (s) for polar baselines (Zeng convention)",
    )
    p.add_argument("--out-dir", required=True)

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

    p.add_argument("--num-intervals", type=int, default=32)
    p.add_argument("--points-per-interval", type=int, default=200)
    p.add_argument("--phi-max", type=float, default=np.pi / 5)
    p.add_argument("--alpha-res", type=int, default=200)
    p.add_argument("--s-res", type=int, default=100)

    p.add_argument(
        "--num-points",
        type=int,
        default=12800,
        help="rollout length (opt_transport, sac)",
    )
    p.add_argument("--ot-grid-res", type=int, default=15)

    p.add_argument("--sac-grid-size", type=int, default=50)
    p.add_argument("--sac-comm-range", type=float, default=0.18)
    p.add_argument("--sac-train-steps", type=int, default=200_000)

    p.add_argument(
        "--nfz-center",
        nargs=2,
        type=float,
        default=None,
        metavar=("CX", "CY"),
    )
    p.add_argument("--nfz-radius", type=float, default=None)
    p.add_argument(
        "--mask-nfz-target",
        action="store_true",
        help="Zero out target density inside the NFZ disc.",
    )

    p.add_argument(
        "--fmec-dt",
        type=float,
        default=0.05,
        help="FMEC LQR time step (independent of --dt for polar baselines).",
    )
    p.add_argument("--fmec-tsteps", type=int, default=200)
    p.add_argument(
        "--fmec-num-iters",
        type=int,
        default=None,
        help="Outer FM iterations; default 500 (stein) / 200 (sinkhorn).",
    )
    p.add_argument(
        "--fmec-step-size",
        type=float,
        default=None,
        help="LQR gradient step; default 0.01 (stein) / 0.005 (sinkhorn).",
    )
    p.add_argument("--fmec-num-samples", type=int, default=1000)
    p.add_argument("--fmec-kernel-h", type=float, default=None)
    p.add_argument("--fmec-x0", type=float, nargs=2, default=None)
    p.add_argument("--fmec-n-agents", type=int, default=1)
    p.add_argument("--fmec-init-radius-frac", type=float, default=0.35)
    p.add_argument(
        "--fmec-grad-subsample",
        type=int,
        default=2000,
        help="0 disables subsampling (may be slow/OOM for large N*T).",
    )
    p.add_argument("--fmec-metric-subsample", type=int, default=2000)
    p.add_argument("--fmec-lambda-nfz", type=float, default=0.0)
    p.add_argument("--fmec-nfz-margin", type=float, default=0.05)

    args = p.parse_args()
    if args.mask_nfz_target and (
        args.nfz_center is None or args.nfz_radius is None
    ):
        raise SystemExit(
            "--mask-nfz-target requires both --nfz-center and --nfz-radius."
        )
    if args.method.startswith("fmec_"):
        _apply_fmec_defaults(args)

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
    elif args.method == "fmec_stein":
        pos, times, meta, label = plan_fmec(args, "stein")
    elif args.method == "fmec_sinkhorn":
        pos, times, meta, label = plan_fmec(args, "sinkhorn")
    else:
        raise SystemExit(f"unknown method {args.method!r}")

    out = os.path.join(args.out_dir, "trajectory.npz")
    save_trajectory_npz(
        path=out,
        pos=pos.astype(np.float64),
        times=times.astype(np.float64),
        domain=(-1.0, 1.0, -1.0, 1.0),
        method=label,
        meta=meta,
    )
    with open(os.path.join(args.out_dir, "plan_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"plan wall t  : {meta['plan_wall_clock_s']:.2f}s")
    print(
        f"trajectory   : N={meta['N']}, dt={meta['dt_s']:.5g}s, "
        f"T={meta['T_total_s']:.2f}s"
    )
    print(f"saved        : {out}")


if __name__ == "__main__":
    main()
