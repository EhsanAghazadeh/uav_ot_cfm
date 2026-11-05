"""Appendix experiment: 3D instantiation of the R^d construction.

Instantiates the Appendix "Extension to R^d" construction at d = 3:
    latent domain: spherical shell D_delta in R^3
    radial profile: r(s) = (delta^3 + (1 - delta^3) s)^{1/3}
    heading: uniform on S^2, fresh per cycle
    target: two-mode Gaussian mixture at (+-0.3, 0, 0), std 0.2,
            restricted to the shell (3D analogue of Experiment 1)

Training protocol identical to the 2D synthetic runs (train_synth.py):
Adam 2e-3 cosine, batch 1024, Sinkhorn eps 0.05 / 50 iters, 1000 epochs,
4 hidden layers x 256 SiLU (input is now (s, y) in R^4, output R^3).

Metrics (metrics.json): rho_iid on a 20^3 grid with 100k IID transported
samples, W2_hat (3000 vs 3000, ot.emd2 sqrt sqeuclidean), final CFM loss,
eps_v, empirical L_v_hat (3x3 Jacobian SVD).

Run with --scaling-only to skip training and just verify the predicted
latent acceleration-energy scaling E_acc per half-cycle ~ Theta(delta^-7)
by numerical integration of rdd(s)^2 (no learning involved).
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_synth import cfm_loss_50
from ot_cfm import transport

MODES = np.array([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]])
STD = 0.2


def sample_shell(n, delta, rng):
    """Uniform on the spherical shell delta <= ||z|| <= 1 (inverse CDF)."""
    u = rng.uniform(0, 1, n)
    r = (delta ** 3 + (1 - delta ** 3) * u) ** (1.0 / 3.0)
    d = rng.standard_normal((n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return r[:, None] * d


def sample_target3d(n, delta, rng):
    out = np.empty((0, 3))
    while len(out) < n:
        comp = rng.integers(0, 2, size=2 * n)
        x = MODES[comp] + STD * rng.standard_normal((2 * n, 3))
        r = np.linalg.norm(x, axis=1)
        x = x[(r >= delta) & (r <= 1.0)]
        out = np.vstack([out, x])
    return out[:n]


def target_grid_pmf3d(delta, n_grid=20):
    xs = np.linspace(-1, 1, n_grid)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    P = np.zeros_like(X)
    for m in MODES:
        P += np.exp(-((X - m[0]) ** 2 + (Y - m[1]) ** 2 + (Z - m[2]) ** 2)
                    / (2 * STD ** 2))
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    P[(R < delta) | (R > 1.0)] = 0.0
    return P / P.sum()


class VelocityNet3D(nn.Module):
    def __init__(self, hidden=256, n_hidden=4):
        super().__init__()
        layers = [nn.Linear(4, hidden), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, s, y):
        if s.dim() == 1:
            s = s.unsqueeze(1)
        return self.net(torch.cat([s, y], dim=1))


def empirical_lv3d(model, delta, rng, n_pts=1024, h=1e-4):
    z0 = sample_shell(n_pts, delta, rng)
    x1 = sample_target3d(n_pts, delta, rng)
    s = rng.uniform(0, 1, n_pts)
    y = (1 - s[:, None]) * z0 + s[:, None] * x1
    with torch.no_grad():
        st = torch.tensor(s, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        cols = []
        for d in range(3):
            e = torch.zeros(1, 3)
            e[0, d] = h
            cols.append((model(st, yt + e) - model(st, yt - e)) / (2 * h))
        J = torch.stack(cols, dim=2)
        return float(torch.linalg.matrix_norm(J, ord=2, dim=(1, 2)).max())


def scaling_check():
    """Numeric check of E_acc per half-cycle ~ Theta(delta^-(3d-2)) = delta^-7."""
    print("== 3D latent acceleration-energy scaling check ==")
    print(f"{'delta':>7} {'E_acc(num)':>12} {'pred d^-7':>12}")
    s = np.linspace(1e-6, 1 - 1e-6, 400001)
    vals = {}
    for delta in (0.05, 0.1, 0.2, 0.4):
        d3 = delta ** 3
        u = d3 + (1 - d3) * s
        r = u ** (1.0 / 3.0)
        dr = (1 - d3) / (3 * r ** 2)
        rdd = np.gradient(dr, s)
        E = float(np.trapezoid(rdd ** 2, s))
        vals[delta] = E
        print(f"{delta:>7} {E:>12.4g} {'':>12}")
    # log-log slope
    ds = sorted(vals)
    sl = [(math.log(vals[ds[i]] / vals[ds[i + 1]])
           / math.log(ds[i] / ds[i + 1])) for i in range(len(ds) - 1)]
    print("pairwise log-log slopes:", [f"{x:.2f}" for x in sl],
          " (theory: -7)")
    return {str(k): v for k, v in vals.items()}, sl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--transport-steps", type=int, default=50)
    p.add_argument("--scaling-only", action="store_true")
    p.add_argument("--out-dir", default="results/appendix/synth3d/tmp")
    args = p.parse_args()

    if args.scaling_only:
        vals, sl = scaling_check()
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "scaling.json"), "w") as f:
            json.dump({"E_acc_num": vals, "slopes": sl}, f, indent=2)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(2)

    model = VelocityNet3D()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time()
    losses = []
    for ep in range(args.epochs):
        z0 = sample_shell(args.batch, args.delta, rng)
        x1 = sample_target3d(args.batch, args.delta, rng)
        loss = cfm_loss_50(model, z0, x1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss))
        if (ep + 1) % 200 == 0:
            print(f"  ep {ep+1}/{args.epochs}  loss={np.mean(losses[-50:]):.5f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    model.eval()

    z = sample_shell(100000, args.delta, rng)
    xg = transport(model, z, n_steps=args.transport_steps, device="cpu")
    H, _ = np.histogramdd(xg, bins=20, range=[[-1, 1]] * 3)
    H = H / H.sum()
    tgt = target_grid_pmf3d(args.delta)
    rho_iid = float(np.corrcoef(H.ravel(), tgt.ravel())[0, 1])

    import ot as pot
    zs = sample_shell(3000, args.delta, rng)
    xs = transport(model, zs, n_steps=args.transport_steps, device="cpu")
    ys = sample_target3d(3000, args.delta, rng)
    M = pot.dist(xs, ys, metric="sqeuclidean")
    w2 = float(np.sqrt(pot.emd2(np.ones(3000) / 3000, np.ones(3000) / 3000,
                                M, numItermax=200000)))

    cfm_final = float(np.mean(losses[-50:]))
    metrics = {
        "args": vars(args),
        "rho_iid": rho_iid,
        "w2_hat": w2,
        "cfm_loss_final": cfm_final,
        "eps_v": math.sqrt(cfm_final),
        "L_v_hat": empirical_lv3d(model, args.delta, rng),
        "train_seconds": time.time() - t0,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    print(json.dumps({k: v for k, v in metrics.items() if k != "args"},
                     indent=1))


if __name__ == "__main__":
    main()
