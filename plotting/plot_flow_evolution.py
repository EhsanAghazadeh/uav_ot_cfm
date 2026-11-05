"""Flow-matching evolution plot: latent (annulus) → target (Milano).

Renders the OT-CFM transport as a row of 2D-histogram panels at evenly
spaced integration times s ∈ [0, 1], in the style of the canonical flow-
matching figure (e.g. Tong et al. 2023, Fig. 2). The reference target
distribution is shown as one extra panel after s=1, separated by a
vertical dotted line.

When `--nfz-center` and `--nfz-radius` are given, a hole is punched in
the displayed target distribution (mass zeroed inside the disc and
renormalised) and the disc is overlaid on every evolution panel.

Usage:
  python -m plotting.plot_flow_evolution \
      --model-dir results/otcfm_milano_4x_v2 \
      --milano-grid data/milano_grid_corrected.npy \
      --n-frames 8 \
      --nfz-center 0.19 0.03 --nfz-radius 0.10 \
      --out results/otcfm_milano_4x_v2/flow_evolution.png
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from ergodic_cfm.latent_trajectory import radial_trajectory, sample_uniform_annulus
from ergodic_cfm.ot_cfm import build_model
from ergodic_cfm.target_density import load_milano_target


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
                  help="Number of evolution panels (including s=0 and s=1). "
                       "The target panel is added beyond these.")
    p.add_argument("--n-steps", type=int, default=40,
                  help="Total RK4 steps for the integration grid.")
    p.add_argument("--bins", type=int, default=160,
                  help="2D histogram resolution per axis.")
    p.add_argument("--smooth-hist", type=float, default=1.5,
                  help="Gaussian smoothing sigma applied to the histogram.")
    p.add_argument("--cmap", default="Purples",
                  help="Colormap shared by evolution panels and target panel.")
    p.add_argument("--nfz-center", nargs=2, type=float, default=None,
                  metavar=("CX", "CY"),
                  help="No-fly disc centre (normalised). When set with "
                       "--nfz-radius, punches a hole in the target panel "
                       "(zero density inside the disc, renormalised) and "
                       "overlays the disc on every evolution panel.")
    p.add_argument("--nfz-radius", type=float, default=None,
                  help="No-fly disc radius (normalised).")
    p.add_argument("--nfz-color", default="#c40000",
                  help="No-fly-zone border color.")
    p.add_argument("--target-title", default="Target",
                  help="Title shown above the target panel.")
    p.add_argument("--separator-color", default="#888888",
                  help="Color of the dotted separator between the s=1 panel "
                       "and the target panel.")
    p.add_argument("--separator-lw", type=float, default=1.0,
                  help="Linewidth of the dotted separator.")
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
    n_spatial_freqs = int(meta.get("n_spatial_freqs", 16))
    sigma_x = float(meta.get("sigma_x", 5.0))
    delta = float(meta.get("delta", 0.05))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch=arch, hidden_dim=hidden_dim, n_layers=n_layers,
                       n_time_freqs=n_time_freqs,
                       n_spatial_freqs=n_spatial_freqs,
                       sigma_x=sigma_x).to(device)
    sd = torch.load(os.path.join(args.model_dir, "model.pt"),
                    map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded {args.model_dir}/model.pt  arch={arch}  hidden={hidden_dim}"
          f"  layers={n_layers}  device={device}")

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

    # ---- Target (with optional NFZ hole) ----
    nfz_on = args.nfz_center is not None and args.nfz_radius is not None
    nfz_c = tuple(args.nfz_center) if nfz_on else None
    nfz_r = float(args.nfz_radius) if nfz_on else None
    t = load_milano_target(args.milano_grid,
                           smooth_sigma=args.smooth_sigma,
                           nfz_center=nfz_c, nfz_radius=nfz_r)
    target = t["grid"]
    if nfz_on:
        print(f"  Target panel: NFZ-masked at c={nfz_c}, r={nfz_r}")

    # ---- Render ----
    xlim, ylim = (-1.0, 1.0), (-1.0, 1.0)
    n_evo = len(snaps)
    n_total = n_evo + 1                                 # +1 for target panel
    fig, axes = plt.subplots(1, n_total,
                             figsize=(2.05 * n_total, 2.4), dpi=170)
    if n_total == 1:
        axes = [axes]

    def _style(ax, title):
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#333")
            sp.set_linewidth(0.8)
        ax.set_title(title, fontsize=9, color="#222", pad=6)

    def _add_nfz(ax):
        # 1. Semi-transparent white to lighten the interior
        ax.add_patch(plt.Circle(nfz_c, nfz_r, facecolor="white", alpha=0.6,
                                edgecolor="none", zorder=3))
        # 2. Light grey hatch
        ax.add_patch(plt.Circle(nfz_c, nfz_r, facecolor="none",
                                edgecolor="#cccccc", lw=0, hatch="////", zorder=4))
        # 3. Dashed border 
        ax.add_patch(plt.Circle(nfz_c, nfz_r, facecolor="none",
                                edgecolor=args.nfz_color, lw=2.0, ls="--", zorder=5))

    # Evolution panels (s = 0 ... 1)
    for i, (snap, s) in enumerate(zip(snaps, s_vals)):
        ax = axes[i]
        H, _, _ = np.histogram2d(
            snap[:, 0], snap[:, 1], bins=args.bins,
            range=[xlim, ylim])
        if args.smooth_hist > 0:
            H = gaussian_filter(H, sigma=args.smooth_hist)
        
        # Apply the log density stretch from the reference script
        H_max = H.max() if H.max() > 0 else 1.0
        Pn = H / H_max
        Pn = np.log1p(20.0 * Pn) / np.log1p(20.0)

        ax.imshow(Pn.T, origin="lower", extent=(*xlim, *ylim),
                  cmap=args.cmap, alpha=0.85, aspect="equal", interpolation="bilinear", zorder=1)
        
        if nfz_on:
            _add_nfz(ax)
        _style(ax, f"s = {s:.2f}")

    # Target panel (after s=1, same imshow style as the evolution panels)
    ax_t = axes[-1]
    
    # Apply the same log density stretch to the target panel
    T_max = target.max() if target.max() > 0 else 1.0
    Pn_t = target / T_max
    Pn_t = np.log1p(20.0 * Pn_t) / np.log1p(20.0)

    ax_t.imshow(Pn_t, origin="lower", extent=(*xlim, *ylim),
                cmap=args.cmap, alpha=0.85, aspect="equal", interpolation="bilinear", zorder=1)
    
    if nfz_on:
        _add_nfz(ax_t)
    _style(ax_t, args.target_title)

    fig.tight_layout()

    # Vertical dotted separator between the s=1 panel and the target panel.
    last_evo_pos = axes[-2].get_position()
    target_pos = axes[-1].get_position()
    sep_x = 0.5 * (last_evo_pos.x1 + target_pos.x0)
    y0 = min(last_evo_pos.y0, target_pos.y0) - 0.01
    y1 = max(last_evo_pos.y1, target_pos.y1) + 0.01
    fig.add_artist(Line2D([sep_x, sep_x], [y0, y1],
                          transform=fig.transFigure,
                          color=args.separator_color,
                          linestyle=":", linewidth=args.separator_lw,
                          zorder=10))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()