"""Export the latent -> Milano flow-matching evolution as animation assets.

Integrates the trained OT-CFM velocity field with RK4 from i.i.d. latent
samples on the annulus D_delta (s = 0) to the learned pushforward (s = 1),
and writes, for each requested model directory:

  <out>/<name>_frames.png   grayscale sprite sheet, one density frame per
                            cell (log-stretched, row-major, s ascending)
  <out>/<name>_target.png   grayscale training target (NFZ-masked if any)
  <out>/milano_target.png   grayscale unmasked Milano demand map (page panel)
  <out>/<name>.gif          colour GIF (evolution | target) for READMEs
  <out>/<name>_path.json    first cycles of the latent path z(t) and the
                            mapped UAV path x(t) = G(z(t)) from trajectory.npz
  <out>/<name>_meta.json    s values, grid size, sheet layout, NFZ disc

The PNG sheets are colourised in the browser (project page), which keeps
the assets small and lets the page pick light/dark palettes.

Usage:
  python -m plotting.make_flow_animation \
      --model results/otcfm_seed42:otcfm \
      --model results_nfz_masked/otcfm_nfz_l5_seed42:otcfm_nfz \
      --out docs_assets/flow
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
from ergodic_cfm.latent_trajectory import sample_uniform_annulus
from ergodic_cfm.ot_cfm import build_model
from plotting.plot_flow_evolution import transport_with_snapshots
from ergodic_cfm.target_density import load_milano_target

LIM = (-1.0, 1.0)


def stretch(H):
    m = H.max() if H.max() > 0 else 1.0
    return np.log1p(20.0 * H / m) / np.log1p(20.0)


def to_u8(P):
    return (np.clip(P, 0, 1) * 255).round().astype(np.uint8)


def target_image(grid, bins):
    """Smooth (cubic) upsampling of the target grid, top row = max y."""
    from scipy.ndimage import zoom
    g = zoom(grid, bins / grid.shape[0], order=3)
    return to_u8(stretch(np.clip(g, 0, None))[::-1])


def load_model(model_dir, device):
    with open(os.path.join(model_dir, "plan_meta.json")) as f:
        meta = json.load(f)
    model = build_model(arch=meta.get("arch", "v1"),
                        hidden_dim=int(meta.get("hidden_dim", 512)),
                        n_layers=int(meta.get("n_layers", 6)),
                        n_time_freqs=int(meta.get("n_time_freqs", 16)),
                        n_spatial_freqs=int(meta.get("n_spatial_freqs", 16)),
                        sigma_x=float(meta.get("sigma_x", 5.0))).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"),
                                     map_location=device))
    model.eval()
    return model, meta


def export_path(model_dir, n_cycles, pts_per_cycle):
    """First cycles of latent z(t) and mapped x(t), if stored."""
    path = os.path.join(model_dir, "trajectory.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    pos = d["pos"] if "pos" in keys else None
    lat = next((d[k] for k in ("latent", "z", "latent_pos") if k in keys), None)
    if pos is None:
        return {"keys": keys}
    pos = np.asarray(pos)
    if pos.ndim == 3:            # (agents, N, 2) -> first agent
        pos = pos[0]
    n = min(len(pos), n_cycles * pts_per_cycle)
    out = {"keys": keys, "x": np.round(pos[:n], 3).tolist()}
    if lat is not None:
        lat = np.asarray(lat)
        if lat.ndim == 3:
            lat = lat[0]
        out["z"] = np.round(lat[:n], 3).tolist()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True,
                   help="MODEL_DIR:NAME (repeatable)")
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--n-particles", type=int, default=30000)
    p.add_argument("--n-frames", type=int, default=61)
    p.add_argument("--n-steps", type=int, default=60)
    p.add_argument("--bins", type=int, default=128)
    p.add_argument("--smooth-hist", type=float, default=1.2)
    p.add_argument("--gif-cmap", default="viridis")
    p.add_argument("--path-cycles", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # Unmasked Milano demand map, shared by all models on the project page
    t0 = load_milano_target(args.milano_grid, smooth_sigma=1.5)
    Image.fromarray(target_image(t0["grid"], args.bins)).save(
        os.path.join(args.out, "milano_target.png"), optimize=True)

    for spec in args.model:
        model_dir, name = spec.split(":")
        model, meta = load_model(model_dir, device)
        delta = float(meta.get("delta", 0.05))
        nfz_c = meta.get("nfz_center")
        nfz_r = meta.get("nfz_radius")
        print(f"[{name}] {model_dir}  delta={delta}  nfz={nfz_c},{nfz_r}"
              f"  lambda_nfz={meta.get('nfz_lambda')}", flush=True)

        z0 = sample_uniform_annulus(args.n_particles, delta=delta,
                                    seed=args.seed)
        snaps, s_vals = transport_with_snapshots(model, z0, args.n_steps,
                                                 args.n_frames, device=device)

        frames = []
        for snap in snaps:
            H, _, _ = np.histogram2d(snap[:, 0], snap[:, 1], bins=args.bins,
                                     range=[LIM, LIM])
            if args.smooth_hist > 0:
                H = gaussian_filter(H, sigma=args.smooth_hist)
            # histogram2d is (x, y); image rows are y, top row = max y
            frames.append(to_u8(stretch(H.T[::-1])))

        t = load_milano_target(args.milano_grid, smooth_sigma=1.5,
                               nfz_center=tuple(nfz_c) if nfz_c else None,
                               nfz_radius=nfz_r if nfz_c else None)
        tgt = target_image(t["grid"], args.bins)

        # Sprite sheet
        cols = int(np.ceil(np.sqrt(len(frames))))
        rows = int(np.ceil(len(frames) / cols))
        b = args.bins
        sheet = np.zeros((rows * b, cols * b), np.uint8)
        for i, fr in enumerate(frames):
            r, c = divmod(i, cols)
            sheet[r * b:(r + 1) * b, c * b:(c + 1) * b] = fr
        Image.fromarray(sheet).save(
            os.path.join(args.out, f"{name}_frames.png"), optimize=True)
        Image.fromarray(tgt).save(
            os.path.join(args.out, f"{name}_target.png"), optimize=True)

        # GIF: evolution | target, hold the final frame
        cmap = colormaps[args.gif_cmap]
        gap = np.full((b, 6, 3), 255, np.uint8)
        tgt_rgb = (cmap(tgt / 255.0)[..., :3] * 255).astype(np.uint8)
        gif = []
        for fr in frames:
            rgb = (cmap(fr / 255.0)[..., :3] * 255).astype(np.uint8)
            img = Image.fromarray(np.concatenate([rgb, gap, tgt_rgb], axis=1))
            gif.append(img.resize((img.width * 2, img.height * 2),
                                  Image.BILINEAR))
        seq = gif + [gif[-1]] * 20 + gif[::-1] + [gif[0]] * 10
        seq[0].save(os.path.join(args.out, f"{name}.gif"), save_all=True,
                    append_images=seq[1:], duration=50, loop=0,
                    optimize=True)

        path = export_path(model_dir, args.path_cycles,
                           int(meta.get("pts_per_cycle", 200)))
        if path is not None:
            with open(os.path.join(args.out, f"{name}_path.json"), "w") as f:
                json.dump(path, f, separators=(",", ":"))

        with open(os.path.join(args.out, f"{name}_meta.json"), "w") as f:
            json.dump({"model_dir": model_dir,
                       "s": [round(float(s), 4) for s in s_vals],
                       "bins": b, "cols": cols, "rows": rows,
                       "n_particles": args.n_particles,
                       "lim": LIM, "delta": delta,
                       "nfz_center": nfz_c, "nfz_radius": nfz_r,
                       "nfz_lambda": meta.get("nfz_lambda")}, f, indent=1)
        print(f"[{name}] wrote {len(frames)} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
