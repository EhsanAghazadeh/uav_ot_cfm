"""Flow Matching Ergodic Coverage — paper replication + Milano baseline.

Sun, M. M. & Murphey, T. "Flow Matching Ergodic Coverage." RSS 2025.
https://arxiv.org/abs/2504.17872

Milano uses ``target_density``-consistent loading via ``milano_target_fmec``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fmec_dynamics import PointMassLQR, make_sinkhorn_grad, make_stein_grad
from fmec_targets import (
    Target,
    exp3_gmm_target,
    gmm_target,
    milano_target,
    pointcloud_target,
)
from fmec_viz_metrics import (
    evaluate_final,
    plot_metric,
    plot_trajectory,
    save_animation,
)


def _resolve_device():
    cpu = jax.devices("cpu")[0]
    try:
        gpu = jax.devices("cuda")[0]
    except Exception:
        gpu = cpu
    return cpu, gpu


def run_fmec_optimization(
    target: Target,
    method: str,
    *,
    dt: float,
    tsteps: int,
    num_iters: int,
    step_size: float,
    num_samples: int,
    kernel_h: Optional[float],
    seed: int,
    n_agents: int,
    init_radius_frac: float,
    x0_xy: Optional[Tuple[float, float]],
    grad_subsample: int,
    metric_subsample: int,
    nfz_center: Optional[Tuple[float, float]],
    nfz_radius: float,
    nfz_margin: float,
    lambda_nfz: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[float], float, list[np.ndarray]
]:
    """Returns (final_x_traj, u_traj, x0_batch, metric_history, wall_seconds,
    x_traj_snapshots per iteration)."""
    assert method in ("stein", "sinkhorn")
    x_lo, x_hi = target.xlim
    y_lo, y_hi = target.ylim
    domain_w = max(x_hi - x_lo, y_hi - y_lo)
    h = kernel_h if kernel_h is not None else (0.01 * (domain_w**2))

    N = max(int(n_agents), 1)
    cx, cy = target.center
    T_horizon = dt * tsteps
    dx_dom = x_hi - x_lo
    dy_dom = y_hi - y_lo
    dom_cx = 0.5 * (x_lo + x_hi)
    dom_cy = 0.5 * (y_lo + y_hi)

    if N == 1:
        if x0_xy is not None:
            p0x, p0y = x0_xy
        elif target.raw_points is not None:
            p0x = float(target.raw_points[0, 0])
            p0y = float(target.raw_points[0, 1])
        else:
            p0x = x_lo + 0.1 * dx_dom
            p0y = y_lo + 0.2 * dy_dom
        starts = np.array([[p0x, p0y]], dtype=np.float32)
    else:
        r = init_radius_frac * max(dx_dom, dy_dom)
        thetas = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
        starts = np.stack(
            [
                dom_cx + r * np.cos(thetas),
                dom_cy + r * np.sin(thetas),
            ],
            axis=1,
        ).astype(np.float32)
    vels = np.stack(
        [
            2.0 * (cx - starts[:, 0]) / T_horizon,
            2.0 * (cy - starts[:, 1]) / T_horizon,
        ],
        axis=1,
    ).astype(np.float32)
    x0_batch = jnp.asarray(np.concatenate([starts, vels], axis=1))

    cpu, gpu = _resolve_device()

    Q = jnp.diag(jnp.array([1.0, 1.0, 0.001, 0.001]))
    R = jnp.diag(jnp.array([0.01, 0.01]))
    model = PointMassLQR(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
    linearize_dyn_batched = jit(vmap(model.linearize_dyn, in_axes=(0, 0)), device=cpu)
    solve_lqr_batched = jit(vmap(model.solve, in_axes=(0, 0, 0, 0)), device=cpu)
    traj_sim_batched = vmap(model.traj_sim, in_axes=(0, 0))

    _NUM_SAMPLES_DEFAULT = 1000
    if (
        target.raw_points is not None
        and num_samples == _NUM_SAMPLES_DEFAULT
    ):
        num_samples_eff = int(len(target.raw_points))
        tgt_samples_np = np.asarray(target.raw_points, dtype=np.float32)
    else:
        num_samples_eff = num_samples
        tgt_samples_np = target.sample(num_samples_eff, seed=seed)
    tgt_samples = jnp.array(tgt_samples_np)

    pool_size = N * tsteps
    grad_sub = int(grad_subsample)
    use_grad_sub = grad_sub > 0 and pool_size > grad_sub
    grad_rng = np.random.default_rng(int(seed) + 2)

    if method == "stein":
        stein_full, stein_pair = make_stein_grad(target.log_pdf, h=h)
        if use_grad_sub:
            stein_pair_jit = jit(stein_pair, device=gpu)

            def grad_fn_single(flat):
                flat_g = jax.device_put(flat, gpu)
                idx = grad_rng.choice(int(flat_g.shape[0]), size=grad_sub, replace=False)
                idx_j = jax.device_put(jnp.asarray(idx), gpu)
                emp = flat_g[idx_j]
                return stein_pair_jit(flat_g, emp)
        else:
            grad_fn_single = jit(stein_full, device=gpu)
    else:
        sinkhorn_grad_fn, _ = make_sinkhorn_grad(tgt_samples)
        if use_grad_sub:
            sinkhorn_grad_jit = jit(sinkhorn_grad_fn, device=gpu)

            def grad_fn_single(flat):
                flat_g = jax.device_put(flat, gpu)
                idx = grad_rng.choice(int(flat_g.shape[0]), size=grad_sub, replace=False)
                idx_j = jax.device_put(jnp.asarray(idx), gpu)
                sub = flat_g[idx_j]
                g_sub = sinkhorn_grad_jit(sub)
                out = jnp.zeros_like(flat_g)
                return out.at[idx_j].set(g_sub)
        else:
            grad_fn_single = jit(sinkhorn_grad_fn, device=gpu)

    _, sinkhorn_div_fn = make_sinkhorn_grad(tgt_samples)
    sinkhorn_div_jit = jit(sinkhorn_div_fn, device=gpu)

    def _base_grad_fn_pooled(x_NT4: jnp.ndarray) -> jnp.ndarray:
        flat = x_NT4.reshape(-1, 4)
        g_flat = grad_fn_single(flat)
        return g_flat.reshape(x_NT4.shape)

    nfz_on = (
        lambda_nfz > 0.0
        and nfz_center is not None
        and nfz_radius > 0.0
    )
    if nfz_on:
        _nfz_c = jax.device_put(
            jnp.asarray(np.array(nfz_center, dtype=np.float32)), gpu
        )
        _nfz_r = float(nfz_radius)
        _nfz_m = float(nfz_margin)
        _nfz_l = float(lambda_nfz)

        @jit
        def _nfz_grad_NT4(x_NT4_g: jnp.ndarray) -> jnp.ndarray:
            pos = x_NT4_g[..., :2]
            diff = pos - _nfz_c
            d = jnp.linalg.norm(diff, axis=-1, keepdims=True) + 1e-8
            viol = jnp.maximum(0.0, _nfz_r + _nfz_m - d)
            dpos = _nfz_l * viol * diff / d
            zeros_vel = jnp.zeros_like(dpos)
            return jnp.concatenate([dpos, zeros_vel], axis=-1)

        def grad_fn_pooled(x_NT4: jnp.ndarray) -> jnp.ndarray:
            x_gpu = jax.device_put(x_NT4, gpu)
            g = _base_grad_fn_pooled(x_gpu)
            return g + _nfz_grad_NT4(x_gpu)
    else:
        grad_fn_pooled = _base_grad_fn_pooled

    metric_sub = int(metric_subsample)
    metric_rng = np.random.default_rng(int(seed) + 1)
    if metric_sub > 0 and pool_size > metric_sub:
        metric_idx_np = metric_rng.choice(pool_size, size=metric_sub, replace=False)
        metric_idx = jnp.asarray(metric_idx_np)

        def metric_pooled(x_NT4: jnp.ndarray) -> jnp.ndarray:
            flat = x_NT4.reshape(-1, 4)
            return sinkhorn_div_jit(flat[metric_idx])
    else:

        def metric_pooled(x_NT4: jnp.ndarray) -> jnp.ndarray:
            return sinkhorn_div_jit(x_NT4.reshape(-1, 4))

    u_traj = jnp.zeros((N, tsteps, 2))
    z0 = jnp.zeros((N, 4))
    x_snapshots: list[np.ndarray] = []
    metric_history = []
    t0 = time.time()
    for _ in tqdm(range(num_iters), desc=f"{method} flow (N={N})"):
        x_traj, A_traj, B_traj = linearize_dyn_batched(x0_batch, u_traj)
        dx_traj = grad_fn_pooled(x_traj)
        v_traj, z_traj = solve_lqr_batched(z0, A_traj, B_traj, dx_traj)
        u_traj = u_traj + step_size * v_traj
        x_snapshots.append(np.asarray(x_traj))
        metric_history.append(float(metric_pooled(x_traj)))

    final_x_traj = np.asarray(traj_sim_batched(x0_batch, u_traj))
    wall = time.time() - t0
    return (
        final_x_traj,
        np.asarray(u_traj),
        np.asarray(x0_batch),
        metric_history,
        wall,
        x_snapshots,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Flow Matching Ergodic Coverage (arxiv 2504.17872)"
    )
    p.add_argument("--method", choices=["stein", "sinkhorn"], default="stein")
    p.add_argument(
        "--target",
        choices=["gmm", "exp3_gmm", "pointcloud", "milano"],
        default="gmm",
    )
    p.add_argument("--milano-grid", type=str, default=None)
    p.add_argument("--pointcloud-file", type=str, default="heart")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--tsteps", type=int, default=200)
    p.add_argument("--num-iters", type=int, default=None)
    p.add_argument("--step-size", type=float, default=None)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--kernel-h", type=float, default=None)
    p.add_argument("--x0", type=float, nargs=2, default=None)
    p.add_argument("--n-agents", type=int, default=1)
    p.add_argument("--init-radius-frac", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--gif-stride", type=int, default=10)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--eval-grid-res", type=int, default=60)
    p.add_argument("--ergodic-k-max", type=int, default=10)
    p.add_argument("--ergodic-weight-power", type=float, default=1.5)
    p.add_argument("--w2-n", type=int, default=500)
    p.add_argument("--w2-method", choices=["sinkhorn", "emd"], default="sinkhorn")
    p.add_argument("--sinkhorn-reg", type=float, default=0.05)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--metric-subsample", type=int, default=2000)
    p.add_argument("--nfz-center", type=float, nargs=2, default=None)
    p.add_argument("--nfz-radius", type=float, default=0.0)
    p.add_argument("--nfz-margin", type=float, default=0.05)
    p.add_argument("--lambda-nfz", type=float, default=0.0)
    p.add_argument("--grad-subsample", type=int, default=2000)
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--eps-floor", type=float, default=1e-12)
    return p.parse_args()


def build_target(args) -> Target:
    if args.target == "gmm":
        return gmm_target()
    if args.target == "exp3_gmm":
        return exp3_gmm_target()
    if args.target == "pointcloud":
        return pointcloud_target(args.pointcloud_file)
    if args.target == "milano":
        if args.milano_grid is None:
            raise SystemExit("--milano-grid is required when --target milano")
        nfz_c = tuple(args.nfz_center) if args.nfz_center is not None else None
        nfz_r = (
            float(args.nfz_radius)
            if (args.nfz_radius is not None and float(args.nfz_radius) > 0.0)
            else None
        )
        if nfz_c is None:
            nfz_r = None
        return milano_target(
            args.milano_grid,
            smooth_sigma=args.smooth_sigma,
            eps_floor=args.eps_floor,
            seed=args.seed,
            nfz_center=nfz_c,
            nfz_radius=nfz_r,
        )
    raise SystemExit(f"unknown target {args.target}")


def plan_milano_fmec_for_baseline(args: Any, fmec_method: str):
    """Integrate with ``run_baseline_milano_v1``: canonical trajectory + meta.

    ``args`` must expose milano_grid, smooth_sigma, eps_floor, mask_nfz_target,
    nfz_center, nfz_radius, seed, and fmec_* fields (see v1 parser).
    """
    assert fmec_method in ("stein", "sinkhorn")
    nfz_discs = getattr(args, "nfz_discs", None) or []
    mask_discs = nfz_discs if args.mask_nfz_target else []
    target = milano_target(
        args.milano_grid,
        smooth_sigma=args.smooth_sigma,
        eps_floor=args.eps_floor,
        nfz_discs=mask_discs if mask_discs else None,
        seed=args.seed,
    )

    num_iters = args.fmec_num_iters
    step_size = args.fmec_step_size
    x0_xy = (
        (float(args.fmec_x0[0]), float(args.fmec_x0[1]))
        if getattr(args, "fmec_x0", None) is not None
        else None
    )
    # Soft NFZ penalty: fmec only supports single disc; use first disc if present.
    soft_nfz_c = None
    soft_nfz_r = 0.0
    if getattr(args, "fmec_lambda_nfz", 0.0) > 0.0 and nfz_discs:
        soft_nfz_c = nfz_discs[0][0]
        soft_nfz_r = float(nfz_discs[0][1])

    t0 = time.perf_counter()
    final_x_traj, _u, _x0b, metric_history, wall_loop, _snaps = (
        run_fmec_optimization(
            target,
            fmec_method,
            dt=float(args.fmec_dt),
            tsteps=int(args.fmec_tsteps),
            num_iters=int(num_iters),
            step_size=float(step_size),
            num_samples=int(args.fmec_num_samples),
            kernel_h=args.fmec_kernel_h,
            seed=int(args.seed),
            n_agents=int(args.fmec_n_agents),
            init_radius_frac=float(args.fmec_init_radius_frac),
            x0_xy=x0_xy,
            grad_subsample=int(args.fmec_grad_subsample),
            metric_subsample=int(args.fmec_metric_subsample),
            nfz_center=soft_nfz_c,
            nfz_radius=soft_nfz_r,
            nfz_margin=float(args.fmec_nfz_margin),
            lambda_nfz=float(args.fmec_lambda_nfz),
        )
    )
    plan_s = time.perf_counter() - t0

    fx = np.asarray(final_x_traj, dtype=np.float64)
    if fx.ndim == 2:
        fx = fx[None, ...]
    pos = fx[0, :, :2] if fx.shape[0] == 1 else fx[:, :, :2]
    dt = float(args.fmec_dt)
    times = np.arange(fx.shape[1], dtype=np.float64) * dt
    meta = {
        "method_kind": f"fmec_{fmec_method}",
        "fmec_variant": fmec_method,
        "dt_s": dt,
        "fmec_tsteps": int(args.fmec_tsteps),
        "fmec_num_iters": int(num_iters),
        "fmec_step_size": float(step_size),
        "fmec_num_samples": int(args.fmec_num_samples),
        "fmec_n_agents": int(args.fmec_n_agents),
        "fmec_final_sinkhorn_div_ott": float(metric_history[-1])
        if metric_history
        else float("nan"),
        "milano_smooth_sigma": float(args.smooth_sigma),
        "mask_nfz_target": bool(args.mask_nfz_target),
        "nfz_discs": [[[float(c[0]), float(c[1])], float(r)] for c, r in mask_discs],
        "fmec_lambda_nfz": float(args.fmec_lambda_nfz),
        "fmec_grad_subsample": int(args.fmec_grad_subsample),
        "fmec_metric_subsample": int(args.fmec_metric_subsample),
        "seed": int(args.seed),
        "plan_wall_clock_s": float(plan_s),
        "fmec_inner_wall_s": float(wall_loop),
    }
    if pos.ndim == 3:
        pos_out = pos.reshape(-1, 2)
        times_out = np.repeat(times[None, :], pos.shape[0], axis=0).reshape(-1)
        meta["fmec_multiagent_flattened"] = True
    else:
        pos_out = pos
        times_out = times
    meta["N"] = int(pos_out.shape[0])
    meta["T_total_s"] = float(times_out[-1])
    label = f"fmec_{fmec_method}_milano"
    return pos_out, times_out, meta, label


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    num_iters = (
        args.num_iters
        if args.num_iters is not None
        else (500 if args.method == "stein" else 200)
    )
    step_size = (
        args.step_size
        if args.step_size is not None
        else (0.01 if args.method == "stein" else 0.005)
    )

    target = build_target(args)
    N = max(int(args.n_agents), 1)
    x0_xy = tuple(args.x0) if args.x0 is not None else None

    nfz_c = tuple(args.nfz_center) if args.nfz_center is not None else None
    nfz_r = float(args.nfz_radius) if args.nfz_radius else 0.0

    final_x_traj, u_traj, x0_batch, metric_history, wall, x_snapshots = (
        run_fmec_optimization(
        target,
        args.method,
        dt=args.dt,
        tsteps=args.tsteps,
        num_iters=num_iters,
        step_size=step_size,
        num_samples=args.num_samples,
        kernel_h=args.kernel_h,
        seed=args.seed,
        n_agents=N,
        init_radius_frac=args.init_radius_frac,
        x0_xy=x0_xy,
        grad_subsample=args.grad_subsample,
        metric_subsample=args.metric_subsample,
        nfz_center=nfz_c,
        nfz_radius=nfz_r,
        nfz_margin=args.nfz_margin,
        lambda_nfz=args.lambda_nfz,
        )
    )
    print(f"done in {wall:.1f}s; final metric = {metric_history[-1]:.4f}")

    tag = f"{args.method}_{args.target}"
    if N > 1:
        tag = f"{tag}_N{N}"

    plot_trajectory(
        target,
        final_x_traj,
        np.asarray(x0_batch),
        os.path.join(args.out_dir, f"{tag}_trajectory.png"),
        title=f"{args.method} / {args.target} (N={N}, iter {num_iters})",
    )
    plot_metric(
        metric_history, os.path.join(args.out_dir, f"{tag}_metric.png")
    )
    eval_metrics: dict = {}
    if not args.no_eval:
        eval_metrics = evaluate_final(
            target,
            final_x_traj,
            args.dt,
            args.out_dir,
            tag,
            grid_res=args.eval_grid_res,
            ergodic_k_max=args.ergodic_k_max,
            ergodic_weight_power=args.ergodic_weight_power,
            w2_n=args.w2_n,
            w2_method=args.w2_method,
            sinkhorn_reg=args.sinkhorn_reg,
            seed=args.seed,
        )
        print("\n" + "=" * 60)
        print("FINAL METRICS")
        for k, v in eval_metrics.items():
            print(f"  {k}: {v}")
        print("=" * 60)

    if not args.no_gif:
        x_traj_arr = np.stack(x_snapshots + [final_x_traj], axis=0)
        save_animation(
            target,
            x_traj_arr,
            np.asarray(x0_batch),
            os.path.join(args.out_dir, f"{tag}_trajectory.gif"),
            stride=args.gif_stride,
        )

    np.savez(
        os.path.join(args.out_dir, f"{tag}.npz"),
        u_traj=u_traj,
        x0=np.asarray(x0_batch),
        metric_history=np.asarray(metric_history),
        dt=args.dt,
        tsteps=args.tsteps,
        num_iters=num_iters,
        step_size=step_size,
        n_agents=N,
        method=args.method,
        target=args.target,
        **{f"eval_{k}": v for k, v in eval_metrics.items()},
    )

    fx = np.asarray(final_x_traj, dtype=np.float64)
    if fx.ndim == 2:
        fx = fx[None, ...]
    pos_NT2 = fx[:, :, :2]
    times_T = np.arange(pos_NT2.shape[1], dtype=np.float64) * args.dt
    traj_npz = os.path.join(args.out_dir, f"{tag}_trajectory.npz")
    meta_json = json.dumps(
        {
            "n_agents": int(N),
            "dt": float(args.dt),
            "tsteps": int(args.tsteps),
            "num_iters": int(num_iters),
            "step_size": float(step_size),
            "target": str(args.target),
        }
    )
    np.savez(
        traj_npz,
        pos=pos_NT2 if N > 1 else pos_NT2[0],
        times=times_T,
        domain=np.array(
            [target.xlim[0], target.xlim[1], target.ylim[0], target.ylim[1]],
            dtype=np.float64,
        ),
        method=f"lq_{args.method}",
        meta=meta_json,
    )
    print(f"[info] canonical trajectory saved to {traj_npz}")
    print(f"outputs written to {args.out_dir}/{tag}_*")


if __name__ == "__main__":
    main()
