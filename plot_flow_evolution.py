"""Flow-matching evolution plot: latent (annulus) → target (Milano).

Renders the OT-CFM transport as a row of 2D-histogram panels at evenly
spaced integration times s ∈ [0, 1], in the style of the canonical flow-
matching figure (e.g. Tong et al. 2023, Fig. 2). Reads a trained model
checkpoint from --model-dir and integrates with RK4, saving snapshots at
each requested frame.

Usage:
  python plot_flow_evolution.py \
      --model-dir results/otcfm_milano_4x_v2 \
      --milano-grid data/milano_grid_corrected.npy \
      --n-cycles 50 --pts-per-cycle 200 \
      --n-frames 8 \
      --out results/otcfm_milano_4x_v2/flow_evolution.png
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latent_trajectory import radial_trajectory, sample_uniform_annulus
from ot_cfm import build_model


@torch.no_grad()
def transport_with_snapshots(model, z0, n_steps, n_frames, device="cpu"):
    """RK4 transport saving (n_frames) snapshots at evenly-spaced s ∈ [0, 1].

    Always includes s=0 and s=1. n_steps must be ≥ n_frames - 1.
    Returns (snapshots, s_values): snapshots is shape (n_frames, N, 2).
    """
    if n_steps < n_frames - 1:
        n_steps = n_frames - 1

    y = torch.as_tensor(z0, dtype=torch.float32, device=device)
    dt = 1.0 / n_steps
    N = y.shape[0]

    snap_idx = np.unique(np.round(np.linspace(0, n_steps, n_frames))
                         .astype(int))
    snapshots = [y.cpu().numpy().copy()]
    s_values = [0.0]

    for i in range(n_steps):
        s_i = i * dt
        s1 = torch.full((N,), s_i, device=device)
        s2 = torch.full((N,), s_i + 0.5 * dt, device=device)
        s3 = torch.full((N,), s_i + dt, device=device)
        k1 = model(s1, y)
        k2 = model(s2, y + 0.5 * dt * k1)
        k3 = model(s2, y + 0.5 * dt * k2)
        k4 = model(s3, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if (i + 1) in snap_idx:
            snapshots.append(y.cpu().numpy().copy())
            s_values.append((i + 1) * dt)

    return np.stack(snapshots), np.array(s_values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True,
                  help="Directory containing model.pt + plan_meta.json")
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--n-cycles", type=int, default=50,
                  help="Cycles in the latent radial trajectory (source=radial)")
    p.add_argument("--pts-per-cycle", type=int, default=200)
    p.add_argument("--source", choices=["radial", "uniform"], default="uniform",
                  help="Source distribution: 'uniform' samples annulus i.i.d.;"
                       " 'radial' uses the structured planning trajectory.")
    p.add_argument("--n-particles", type=int, default=30000,
                  help="If source=uniform, number of latent particles.")
    p.add_argument("--n-frames", type=int, default=8,
                  help="Number of panels (including s=0 and s=1).")
    p.add_argument("--n-steps", type=int, default=40,
                  help="Total RK4 steps for the integration grid.")
    p.add_argument("--bins", type=int, default=160,
                  help="2D histogram resolution per axis.")
    p.add_argument("--smooth-hist", type=float, default=1.5,
                  help="Gaussian smoothing sigma applied to the histogram.")
    p.add_argument("--cmap", default="viridis",
                  help="Colormap for the particle density panels.")
    p.add_argument("--show-target", action="store_true", default=True,
                  help="Overlay Milano density contour on each panel.")
    p.add_argument("--contour-color", default="#ff6a00",
                  help="Color of the Milano density contour overlay "
                       "(default warm-orange, visible on viridis).")
    p.add_argument("--contour-alpha", type=float, default=0.85,
                  help="Opacity of the Milano contour lines (default 0.85).")
    p.add_argument("--contour-lw", type=float, default=0.9,
                  help="Linewidth of the Milano contour lines.")
    p.add_argument("--nfz-center", nargs=2, type=float, default=None,
                  metavar=("CX", "CY"),
                  help="No-fly disc centre (normalised). Draws a red circle "
                       "on every panel if set.")
    p.add_argument("--nfz-radius", type=float, default=None,
                  help="No-fly disc radius (normalised).")
    p.add_argument("--nfz-color", default="#ff2a2a",
                  help="No-fly-zone overlay color (default bright red).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # ---- Load model ----
    meta_path = os.path.join(args.model_dir, "plan_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    arch = meta.get("arch", "v1")
    hidden_dim = int(meta.get("hidden_dim", 512))
    n_layers = int(meta.get("n_layers", 6))
    n_time_freqs = int(meta.get("n_time_freqs", 16))
    delta = float(meta.get("delta", 0.05))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch=arch, hidden_dim=hidden_dim, n_layers=n_layers,
                       n_time_freqs=n_time_freqs).to(device)
    sd = torch.load(os.path.join(args.model_dir, "model.pt"),
                    map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded {args.model_dir}/model.pt  arch={arch}  hidden={hidden_dim}  layers={n_layers}  device={device}")

    # ---- Latent samples ----
    if args.source == "radial":
        z0, _, _ = radial_trajectory(
            n_cycles=args.n_cycles, points_per_cycle=args.pts_per_cycle,
            delta=delta, seed=args.seed)
    else:
        z0 = sample_uniform_annulus(args.n_particles, delta=delta,
                                   seed=args.seed)
    print(f"latent: N={len(z0)}  source={args.source}")

    # ---- Integrate with snapshots ----
    snaps, s_vals = transport_with_snapshots(
        model, z0, args.n_steps, args.n_frames, device=device)
    print(f"snapshots: {len(snaps)}  s={np.round(s_vals, 3).tolist()}")

    # ---- Target for overlay ----
    target = np.load(args.milano_grid).astype(np.float64)
    if args.smooth_sigma > 0:
        target = gaussian_filter(target, sigma=args.smooth_sigma)
    target = np.maximum(target, 1e-12)

    # ---- Render ----
    xlim, ylim = (-1.0, 1.0), (-1.0, 1.0)
    n = len(snaps)
    fig, axes = plt.subplots(1, n, figsize=(2.05 * n, 2.4), dpi=170)
    if n == 1:
        axes = [axes]

    th, tw = target.shape
    Tx = np.linspace(xlim[0], xlim[1], tw)
    Ty = np.linspace(ylim[0], ylim[1], th)
    TX, TY = np.meshgrid(Tx, Ty)
    target_levels = np.linspace(target.min(), target.max(), 6)[1:-1]

    for i, (snap, s) in enumerate(zip(snaps, s_vals)):
        ax = axes[i]
        H, _, _ = np.histogram2d(
            snap[:, 0], snap[:, 1], bins=args.bins,
            range=[xlim, ylim])
        if args.smooth_hist > 0:
            H = gaussian_filter(H, sigma=args.smooth_hist)

        ax.imshow(H.T, origin="lower", extent=(*xlim, *ylim),
                 cmap=args.cmap, aspect="equal", interpolation="bilinear")

        if args.show_target:
            ax.contour(TX, TY, target, levels=target_levels,
                      colors=args.contour_color, alpha=args.contour_alpha,
                      linewidths=args.contour_lw)

        if args.nfz_center is not None and args.nfz_radius is not None:
            disc = plt.Circle(tuple(args.nfz_center), args.nfz_radius,
                              facecolor=args.nfz_color, alpha=0.18,
                              edgecolor=args.nfz_color, linewidth=1.4,
                              linestyle="--", zorder=5)
            ax.add_patch(disc)

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"s = {s:.2f}", fontsize=9, color="#222", pad=4)

    fig.suptitle(
        f"OT-CFM flow:  annular latent  →  Milano density   "
        f"(N={len(z0):,}, n_steps={args.n_steps})",
        fontsize=11, y=0.99, color="#222")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
