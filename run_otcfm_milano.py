"""Train OT-CFM on Milano + generate one transported trajectory.

Mirrors run_toes_milano.py: produces a single canonical-format
trajectory.npz so eval_prefix_stop.py + plot_curves.py work unchanged.

  python run_otcfm_milano.py \
      --milano-grid data/milano_grid_corrected.npy \
      --epochs 1000 --batch-size 1024 --hidden-dim 512 --n-layers 6 \
      --n-cycles 300 --pts-per-cycle 200 --transport-steps 20 \
      --out-dir results/otcfm_milano

Then:
  python eval_prefix_stop.py --traj results/otcfm_milano/trajectory.npz \
      --milano-grid data/milano_grid_corrected.npy --smooth-sigma 1.5 \
      --out-dir results/otcfm_milano/curves
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fair_comparison import save_trajectory_npz
from latent_trajectory import radial_trajectory, sample_uniform_annulus
from ot_cfm import (VelocityNet, build_model, cfm_loss, transport,
                    transport_train, nfz_penalty, energy_penalty,
                    acceleration_penalty, ModelEMA)


def make_grid_sampler(grid_path, smooth_sigma, eps_floor,
                      xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), seed=0,
                      nfz_center=None, nfz_radius=None):
    """Build a Milano-density sampler.

    When `nfz_center` and `nfz_radius` are provided, the underlying
    target density is zeroed inside the disc and renormalised, so
    sampler output never lands in the NFZ. This is the principled
    encoding of the constraint at the *distribution* level (cf. §2.3
    of the paper); pair with the soft penalty for residual gradient
    signal at the boundary.
    """
    from target_density import load_milano_target
    t = load_milano_target(grid_path, smooth_sigma=smooth_sigma,
                           eps_floor=eps_floor, xlim=xlim, ylim=ylim,
                           nfz_center=nfz_center, nfz_radius=nfz_radius)
    g = t["grid"]
    xs, ys = t["xs"], t["ys"]
    h, w = g.shape
    p = t["pmf"].reshape(-1)
    cdf = np.cumsum(p); cdf[-1] = 1.0
    dx = (xlim[1] - xlim[0]) / max(w - 1, 1)
    dy = (ylim[1] - ylim[0]) / max(h - 1, 1)
    rng = np.random.default_rng(seed)

    def sampler(n, seed=None):
        rrng = rng if seed is None else np.random.default_rng(seed)
        u = rrng.uniform(0, 1, n)
        idx = np.clip(np.searchsorted(cdf, u, side="right"), 0, h * w - 1)
        ii, jj = idx // w, idx % w
        x = xs[jj] + rrng.uniform(-0.5, 0.5, n) * dx
        y = ys[ii] + rrng.uniform(-0.5, 0.5, n) * dy
        x = np.clip(x, xlim[0], xlim[1])
        y = np.clip(y, ylim[0], ylim[1])
        return np.column_stack([x, y]).astype(np.float32)
    return sampler


def train_cfm(sample_target, n_epochs, batch_size, lr, hidden_dim,
              n_layers, delta, seed, arch="v1", n_time_freqs=16,
              n_spatial_freqs=16, sigma_x=5.0,
              ema_decay=0.0,
              nfz_center=None, nfz_radius=None, nfz_margin=0.05,
              nfz_lambda=0.0, nfz_steps=8, nfz_reject_targets=False,
              energy_lambda=0.0, energy_chunk_size=64, energy_steps=8,
              energy_n_cycles=50, energy_pts_per_cycle=200,
              acc_lambda=0.0,
              domain_size=8.0, device="cpu"):
    torch.manual_seed(seed)
    model = build_model(arch=arch, hidden_dim=hidden_dim, n_layers=n_layers,
                        n_time_freqs=n_time_freqs,
                        n_spatial_freqs=n_spatial_freqs,
                        sigma_x=sigma_x).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0.0 else None
    if ema is not None:
        print(f"  EMA on: decay={ema_decay:.4g}")

    nfz_on = (nfz_center is not None and nfz_radius is not None
              and nfz_lambda > 0.0)
    if nfz_on:
        nfz_c = np.asarray(nfz_center, dtype=np.float32)
        print(f"  NFZ on: c=({nfz_c[0]:.3f},{nfz_c[1]:.3f})  r={nfz_radius:.3f}"
              f"  margin={nfz_margin:.3f}  lambda={nfz_lambda:.3f}  "
              f"steps={nfz_steps}  reject_targets={nfz_reject_targets}")

    energy_on = energy_lambda > 0.0
    acc_on = acc_lambda > 0.0
    kin_on = energy_on or acc_on
    if kin_on:
        pos_lat_full, t_lat_full, _ = radial_trajectory(
            n_cycles=energy_n_cycles, points_per_cycle=energy_pts_per_cycle,
            delta=delta, seed=seed + 7)
        pos_lat_full_t = torch.as_tensor(
            pos_lat_full, dtype=torch.float32, device=device)
        n_lat_full = pos_lat_full_t.shape[0]
        dt_s = float(t_lat_full[1] - t_lat_full[0])
        kin_rng = np.random.default_rng(seed + 8)
        max_start = n_lat_full - energy_chunk_size - 1
        if max_start <= 0:
            raise ValueError(
                f"energy_chunk_size={energy_chunk_size} too large for"
                f" radial trajectory of length {n_lat_full}")
        if energy_on:
            print(f"  Energy on: lambda={energy_lambda:.4g}  chunk={energy_chunk_size}"
                  f"  steps={energy_steps}  dt_s={dt_s:.5g}  "
                  f"domain_size={domain_size}")
        if acc_on:
            print(f"  Acc on: lambda={acc_lambda:.4g}  chunk={energy_chunk_size}"
                  f"  steps={energy_steps}  dt_s={dt_s:.5g}  "
                  f"domain_size={domain_size}")

    def _resample_outside_nfz(x, c, r, sampler, base_seed):
        """Replace any x_i inside disc B(c, r) with fresh samples."""
        c_a = np.asarray(c, dtype=x.dtype)
        retries = 0
        while True:
            d = np.linalg.norm(x - c_a, axis=1)
            mask = d < r
            n_bad = int(mask.sum())
            if n_bad == 0 or retries > 20:
                return x
            fresh = sampler(n_bad, seed=base_seed + 17 * retries + 1)
            x[mask] = fresh
            retries += 1

    losses, cfm_log, nfz_log, energy_log, acc_log = [], [], [], [], []
    for ep in range(n_epochs):
        model.train()
        z0 = sample_uniform_annulus(batch_size, delta=delta,
                                    seed=ep * 1000 + seed)
        x1 = sample_target(batch_size, seed=ep * 1000 + seed + 1)

        if nfz_on and nfz_reject_targets:
            x1 = _resample_outside_nfz(
                x1, nfz_c, float(nfz_radius), sample_target,
                base_seed=ep * 1000 + seed + 1)

        loss_cfm = cfm_loss(model, z0, x1, device=device)
        loss = loss_cfm
        pen_val = 0.0
        eng_val = 0.0
        acc_val = 0.0
        if nfz_on:
            z0_t = torch.as_tensor(z0, dtype=torch.float32, device=device)
            yT = transport_train(model, z0_t, n_steps=nfz_steps)
            pen = nfz_penalty(yT, nfz_c, float(nfz_radius), float(nfz_margin))
            loss = loss + float(nfz_lambda) * pen
            pen_val = float(pen.item())

        if kin_on:
            start = int(kin_rng.integers(0, max_start))
            chunk = pos_lat_full_t[start : start + energy_chunk_size + 1]
            yK = transport_train(model, chunk, n_steps=energy_steps)
            if energy_on:
                pen_E = energy_penalty(yK, dt_s, domain_size=domain_size)
                loss = loss + float(energy_lambda) * pen_E
                eng_val = float(pen_E.item())
            if acc_on:
                pen_A = acceleration_penalty(yK, dt_s, domain_size=domain_size)
                loss = loss + float(acc_lambda) * pen_A
                acc_val = float(pen_A.item())

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ema is not None:
            ema.update(model)
        losses.append(loss.item())
        cfm_log.append(float(loss_cfm.item()))
        nfz_log.append(pen_val)
        energy_log.append(eng_val)
        acc_log.append(acc_val)
        if (ep + 1) % 100 == 0:
            extra = ""
            if nfz_on:
                extra += f"  cfm={cfm_log[-1]:.5f}  nfz_pen={nfz_log[-1]:.5f}"
            if energy_on:
                extra += f"  energy_W={energy_log[-1]:.2f}"
            if acc_on:
                extra += f"  acc={acc_log[-1]:.3g}"
            print(f"  epoch {ep+1}/{n_epochs}  loss={loss.item():.6f}{extra}")
    if ema is not None:
        # Use EMA weights for inference (standard CFM/diffusion recipe).
        ema.copy_to(model)
    return model, losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--eps-floor", type=float, default=1e-12)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--arch", choices=["v1", "v2", "v3"], default="v1",
                  help="v1 = original MLP, v2 = Fourier time + residual + LN, "
                       "v3 = spatial RFF + FiLM-conditioned ResMLP")
    p.add_argument("--n-time-freqs", type=int, default=16,
                  help="Fourier time-embedding frequencies (v2/v3).")
    p.add_argument("--n-spatial-freqs", type=int, default=16,
                  help="Random Fourier features for spatial input (v3 only).")
    p.add_argument("--sigma-x", type=float, default=5.0,
                  help="RFF bandwidth for spatial input (v3 only). Higher "
                       "= sharper spatial detail; tune on {1, 2, 5, 10, 20}.")
    p.add_argument("--ema-decay", type=float, default=0.0,
                  help="EMA decay for inference weights. 0 disables; "
                       "0.999 is the standard CFM/diffusion choice.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--n-cycles", type=int, default=300)
    p.add_argument("--pts-per-cycle", type=int, default=200)
    p.add_argument("--transport-steps", type=int, default=20)
    p.add_argument("--nfz-center", nargs=2, type=float, default=None,
                  metavar=("CX", "CY"),
                  help="No-fly disc centre (normalised). If unset, NFZ is off.")
    p.add_argument("--nfz-radius", type=float, default=None,
                  help="No-fly disc radius (normalised).")
    p.add_argument("--nfz-margin", type=float, default=0.05,
                  help="Safety margin beyond radius (default 0.05).")
    p.add_argument("--nfz-lambda", type=float, default=0.0,
                  help="Weight of NFZ penalty in total loss (0 = off).")
    p.add_argument("--nfz-steps", type=int, default=8,
                  help="RK4 steps for in-training transport (default 8).")
    p.add_argument("--nfz-reject-targets", action="store_true",
                  help="Reject target samples that fall inside NFZ during training.")
    p.add_argument("--energy-lambda", type=float, default=0.0,
                  help="Weight of mean Zeng-power penalty (0 = off).")
    p.add_argument("--energy-chunk-size", type=int, default=64,
                  help="Consecutive latent points used per step for the "
                       "energy chunk (longer = lower-variance speed estimate).")
    p.add_argument("--energy-steps", type=int, default=8,
                  help="RK4 steps for the differentiable energy transport.")
    p.add_argument("--energy-n-cycles", type=int, default=50,
                  help="Cycles in the radial trajectory pre-built for "
                       "energy chunks.")
    p.add_argument("--energy-pts-per-cycle", type=int, default=200,
                  help="Points per cycle in the energy radial trajectory.")
    p.add_argument("--acc-lambda", type=float, default=0.0,
                  help="Weight of acceleration / Hessian-Frobenius penalty "
                       "(0 = off). Discrete proxy for E[||H_{G_θ}||_F^2] "
                       "via second finite differences along the transported "
                       "chunk. Reuses --energy-chunk-size and --energy-steps.")
    p.add_argument("--domain-size", type=float, default=8.0,
                  help="Physical extent of the unit interval in metres "
                       "(matches RotaryWingEnergyModel default).")
    p.add_argument("--mask-nfz-target", action="store_true",
                  help="Zero out the target density inside the NFZ disc "
                       "before training (requires --nfz-center / "
                       "--nfz-radius). The sampler then never produces "
                       "targets inside the disc, in addition to the soft "
                       "penalty controlled by --nfz-lambda.")
    p.add_argument("--reuse-model-from", default=None,
                  help="Load checkpoint if it exists; else train and save")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--out-dir", default="results/otcfm_milano")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mask_nfz_c = (tuple(args.nfz_center)
                  if args.mask_nfz_target and args.nfz_center is not None
                  else None)
    mask_nfz_r = (float(args.nfz_radius)
                  if args.mask_nfz_target and args.nfz_radius is not None
                  else None)
    if args.mask_nfz_target and (mask_nfz_c is None or mask_nfz_r is None):
        raise SystemExit("--mask-nfz-target requires --nfz-center and "
                         "--nfz-radius.")
    if mask_nfz_c is not None:
        print(f"  Target density: NFZ-masked at c={mask_nfz_c}, r={mask_nfz_r}")
    sample_target = make_grid_sampler(args.milano_grid, args.smooth_sigma,
                                      args.eps_floor,
                                      nfz_center=mask_nfz_c,
                                      nfz_radius=mask_nfz_r)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    cache = args.reuse_model_from
    cache_hit = (cache is not None and os.path.isfile(cache)
                 and not args.force_retrain)

    train_s = 0.0
    if cache_hit:
        print(f"loading cached model from {cache}  (arch={args.arch})")
        model = build_model(arch=args.arch, hidden_dim=args.hidden_dim,
                            n_layers=args.n_layers,
                            n_time_freqs=args.n_time_freqs,
                            n_spatial_freqs=args.n_spatial_freqs,
                            sigma_x=args.sigma_x).to(device)
        model.load_state_dict(torch.load(cache, map_location=device))
        model.eval()
    else:
        print(f"training OT-CFM: arch={args.arch} epochs={args.epochs} "
              f"batch={args.batch_size} hidden={args.hidden_dim} "
              f"layers={args.n_layers}")
        t0 = time.perf_counter()
        model, losses = train_cfm(
            sample_target, args.epochs, args.batch_size, args.lr,
            args.hidden_dim, args.n_layers, args.delta, args.seed,
            arch=args.arch, n_time_freqs=args.n_time_freqs,
            n_spatial_freqs=args.n_spatial_freqs, sigma_x=args.sigma_x,
            ema_decay=args.ema_decay,
            nfz_center=args.nfz_center, nfz_radius=args.nfz_radius,
            nfz_margin=args.nfz_margin, nfz_lambda=args.nfz_lambda,
            nfz_steps=args.nfz_steps,
            nfz_reject_targets=args.nfz_reject_targets,
            energy_lambda=args.energy_lambda,
            energy_chunk_size=args.energy_chunk_size,
            energy_steps=args.energy_steps,
            energy_n_cycles=args.energy_n_cycles,
            energy_pts_per_cycle=args.energy_pts_per_cycle,
            acc_lambda=args.acc_lambda,
            domain_size=args.domain_size,
            device=device)
        train_s = time.perf_counter() - t0
        print(f"train wall t : {train_s:.1f}s")
        if cache is not None:
            os.makedirs(os.path.dirname(os.path.abspath(cache)) or ".",
                        exist_ok=True)
            torch.save(model.state_dict(), cache)
            print(f"saved checkpoint -> {cache}")
    torch.save(model.state_dict(),
               os.path.join(args.out_dir, "model.pt"))

    # latent radial trajectory + 1-cycle burn-in (paper convention)
    print("generating latent trajectory ...")
    t0 = time.perf_counter()
    pos_lat, t_lat, _ = radial_trajectory(
        n_cycles=args.n_cycles, points_per_cycle=args.pts_per_cycle,
        delta=args.delta, seed=42)
    if args.n_cycles > 1:
        mask = t_lat > 2.0
        pos_lat = pos_lat[mask]; t_lat = t_lat[mask]
    latent_s = time.perf_counter() - t0

    print(f"transporting {len(pos_lat)} points (RK4 steps={args.transport_steps}) ...")
    t0 = time.perf_counter()
    pos_phys = transport(model, pos_lat, n_steps=args.transport_steps,
                         device=device)
    transport_s = time.perf_counter() - t0
    print(f"  transport wall t : {transport_s:.2f}s")

    # canonical export — same shape as TOES output so downstream tools
    # (eval_prefix_stop.py + plot_curves.py) work unchanged.
    times = t_lat - t_lat[0]
    dt = float(times[1] - times[0])
    plan_s = float(train_s + latent_s + transport_s)
    meta = {
        "method_kind": "ot_cfm",
        "arch": str(args.arch),
        "n_time_freqs": int(args.n_time_freqs),
        "n_spatial_freqs": int(args.n_spatial_freqs),
        "sigma_x": float(args.sigma_x),
        "ema_decay": float(args.ema_decay),
        "delta": float(args.delta),
        "n_cycles": int(args.n_cycles),
        "pts_per_cycle": int(args.pts_per_cycle),
        "transport_steps": int(args.transport_steps),
        "epochs": int(args.epochs),
        "hidden_dim": int(args.hidden_dim),
        "n_layers": int(args.n_layers),
        "seed": int(args.seed),
        "T_total_s": float(times[-1]),
        "dt_s": dt,
        "N": int(len(pos_phys)),
        "milano_smooth_sigma": float(args.smooth_sigma),
        "nfz_center": (list(args.nfz_center) if args.nfz_center else None),
        "nfz_radius": (float(args.nfz_radius)
                       if args.nfz_radius is not None else None),
        "nfz_margin": float(args.nfz_margin),
        "nfz_lambda": float(args.nfz_lambda),
        "nfz_steps": int(args.nfz_steps),
        "nfz_reject_targets": bool(args.nfz_reject_targets),
        "energy_lambda": float(args.energy_lambda),
        "energy_chunk_size": int(args.energy_chunk_size),
        "energy_steps": int(args.energy_steps),
        "energy_n_cycles": int(args.energy_n_cycles),
        "energy_pts_per_cycle": int(args.energy_pts_per_cycle),
        "acc_lambda": float(args.acc_lambda),
        "mask_nfz_target": bool(args.mask_nfz_target),
        "domain_size": float(args.domain_size),
        "train_wall_clock_s": float(train_s),
        "latent_gen_wall_clock_s": float(latent_s),
        "transport_wall_clock_s": float(transport_s),
        "plan_wall_clock_s": plan_s,  # eval_prefix_stop.py reads this key
    }
    out = os.path.join(args.out_dir, "trajectory.npz")
    save_trajectory_npz(
        path=out, pos=pos_phys.astype(np.float64), times=times,
        domain=(-1.0, 1.0, -1.0, 1.0),
        method="otcfm_milano", meta=meta)
    with open(os.path.join(args.out_dir, "plan_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved        : {out}")
    print(f"plan_wall_clock_s = {plan_s:.1f}s   (train+latent+transport)")
    print(f"trajectory   : N={len(pos_phys)}, dt={dt:.5g}s, T={times[-1]:.2f}s")


if __name__ == "__main__":
    main()
