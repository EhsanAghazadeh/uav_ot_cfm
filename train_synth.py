"""Appendix experiments: latent inner-radius (delta) ablation and the
spectral-norm certified end-to-end bound.

Reproduces the paper's Experiment 1 setup: two-mode Gaussian mixture at
(+-0.3, 0), std 0.2, restricted to the annulus D_delta and renormalised.
Source is Uniform(D_delta). Velocity net is the synthetic architecture
(4 hidden layers, hidden dim 256, SiLU => 5 linear layers, 4 activations).
Training: Adam lr 2e-3 with cosine annealing, batch 1024, Sinkhorn OT
coupling reg 0.05, 1000 epochs (one minibatch per epoch), matching
Appendix "Training Protocol".

Options:
    --delta D        inner radius (delta ablation)
    --sn-cap C       if > 0, spectral-normalise every linear layer and scale
                     so the architectural bound L_v_net = 1.0998^4 * prod(c_l)
                     equals C exactly (certified bound). 0 = plain net.

Reported metrics (metrics.json):
    rho_iid          Pearson corr, 50x50 grid, 20000 IID transported samples
    w2_hat           ot.emd2 sqrt(sqeuclidean), 3000 samples per side
    cfm_loss_final   mean CFM loss over last 50 epochs
    eps_v            sqrt(cfm_loss_final)   (paper convention, cf. eps_v ~ 0.039
                     at loss 0.0015 in Appendix Experiment 1)
    L_v_net          1.0998^4 * prod_l sigma_max(W_eff_l)  (certified bound)
    L_v_hat          empirical max sigma_max of the 2x2 spatial Jacobian of
                     v_theta at 1024 sampled (s, y_s) points (lower bound)
    floor_hat        e^{L_v_hat} * eps_v      (diagnostic floor)
    floor_net        e^{L_v_net} * eps_v      (certified floor)
    E_acc_latent_per_cycle   closed form (1-d^2)^4 (1+d^2) / (16 d^4)  (Lemma 1)
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latent_trajectory import sample_uniform_annulus
from ot_cfm import transport

MODES = np.array([[-0.3, 0.0], [0.3, 0.0]])
STD = 0.2
L_SIGMA = 1.0998  # max |silu'|


def sample_target(n, delta, rng):
    """Gaussian mixture restricted to D_delta (rejection sampling)."""
    out = np.empty((0, 2))
    while len(out) < n:
        comp = rng.integers(0, 2, size=2 * n)
        x = MODES[comp] + STD * rng.standard_normal((2 * n, 2))
        r = np.linalg.norm(x, axis=1)
        x = x[(r >= delta) & (r <= 1.0)]
        out = np.vstack([out, x])
    return out[:n]


def target_grid_pmf(delta, n_grid=50):
    xs = np.linspace(-1, 1, n_grid)
    X, Y = np.meshgrid(xs, xs)
    P = np.zeros_like(X)
    for m in MODES:
        P += np.exp(-((X - m[0]) ** 2 + (Y - m[1]) ** 2) / (2 * STD ** 2))
    R = np.sqrt(X ** 2 + Y ** 2)
    P[(R < delta) | (R > 1.0)] = 0.0
    return P / P.sum()


def cfm_loss_50(model, z0, x1, sinkhorn_iters=50, reg=0.05):
    """Same as ot_cfm.cfm_loss but with the Sinkhorn iteration cap of the
    paper's synthetic protocol (Appendix Training Protocol: eps_sink = 0.05,
    50 iterations). ot_cfm.ot_coupling uses the POT default of 1000, which
    is ~20x slower and not what the paper documents for the synthetic runs."""
    import ot as pot
    from scipy.optimize import linear_sum_assignment
    z0 = np.asarray(z0)
    x1 = np.asarray(x1)
    n = len(z0)
    cost = pot.dist(z0, x1, metric="sqeuclidean")
    cost = cost / max(cost.max(), 1e-12)
    a = np.ones(n) / n
    with np.errstate(all="ignore"):
        plan = pot.sinkhorn(a, a, cost, reg=reg, numItermax=sinkhorn_iters)
    if not np.all(np.isfinite(plan)) or plan.sum() <= 0:
        plan = -cost  # fall back to plain assignment on the cost
    row, col = linear_sum_assignment(-plan)
    sigma = np.empty(n, dtype=int)
    sigma[row] = col
    z0_t = torch.as_tensor(z0, dtype=torch.float32)
    x1_t = torch.as_tensor(x1[sigma], dtype=torch.float32)
    s = torch.rand(n)
    s_col = s.unsqueeze(1)
    y_s = (1.0 - s_col) * z0_t + s_col * x1_t
    u_target = x1_t - z0_t
    v_pred = model(s, y_s)
    return ((v_pred - u_target) ** 2).mean()


class SNLinear(nn.Linear):
    """Linear layer with exact spectral normalisation and fixed gain c:
    W_eff = c * W / sigma_max(W). Lipschitz constant is exactly c."""

    def __init__(self, in_f, out_f, gain=1.0):
        super().__init__(in_f, out_f)
        self.gain = float(gain)

    def forward(self, x):
        sigma = torch.linalg.matrix_norm(self.weight, ord=2)
        w = self.gain * self.weight / sigma.clamp_min(1e-12)
        return nn.functional.linear(x, w, self.bias)

    def effective_weight(self):
        with torch.no_grad():
            sigma = torch.linalg.matrix_norm(self.weight, ord=2)
            return self.gain * self.weight / sigma.clamp_min(1e-12)


class SynthVelocityNet(nn.Module):
    """Paper synthetic architecture. n_hidden=4 hidden layers, i.e. 5 linear
    layers and 4 SiLU activations (matches L_v_net = 1.0998^4 * prod of 5
    spectral norms in Appendix eq. for L_v_net)."""

    def __init__(self, hidden=256, n_hidden=4, sn_cap=0.0):
        super().__init__()
        n_linear = n_hidden + 1
        if sn_cap > 0:
            gain = (sn_cap / (L_SIGMA ** n_hidden)) ** (1.0 / n_linear)
            L = lambda i, o: SNLinear(i, o, gain=gain)
        else:
            L = nn.Linear
        layers = [L(3, hidden), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers += [L(hidden, hidden), nn.SiLU()]
        layers.append(L(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, s, y):
        if s.dim() == 1:
            s = s.unsqueeze(1)
        return self.net(torch.cat([s, y], dim=1))

    def linears(self):
        return [m for m in self.net if isinstance(m, (nn.Linear,))]

    def lv_net(self):
        prod = 1.0
        for m in self.linears():
            w = m.effective_weight() if isinstance(m, SNLinear) else m.weight
            prod *= float(torch.linalg.matrix_norm(w.detach(), ord=2))
        n_act = sum(1 for m in self.net if isinstance(m, nn.SiLU))
        return (L_SIGMA ** n_act) * prod


def empirical_lv(model, delta, rng, n_pts=1024, h=1e-4):
    """Max sigma_max of the 2x2 spatial Jacobian of v_theta at sampled
    (s, y_s) points along straight interpolants (exact 2x2 SVD on central
    differences; a lower bound on the true sup, as in Appendix L_v protocol)."""
    z0 = sample_uniform_annulus(n_pts, delta=delta, seed=int(rng.integers(1e9)))
    x1 = sample_target(n_pts, delta, rng)
    s = rng.uniform(0, 1, n_pts)
    y = (1 - s[:, None]) * z0 + s[:, None] * x1
    smax = 0.0
    with torch.no_grad():
        st = torch.tensor(s, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        cols = []
        for d in range(2):
            e = torch.zeros(1, 2)
            e[0, d] = h
            vp = model(st, yt + e)
            vm = model(st, yt - e)
            cols.append((vp - vm) / (2 * h))
        J = torch.stack(cols, dim=2)  # (N, 2, 2), columns d/dy_d
        sv = torch.linalg.matrix_norm(J, ord=2, dim=(1, 2))
        smax = float(sv.max())
    return smax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sn-cap", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-hidden", type=int, default=4)
    p.add_argument("--transport-steps", type=int, default=50)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(2)

    model = SynthVelocityNet(hidden=args.hidden, n_hidden=args.n_hidden,
                             sn_cap=args.sn_cap)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time()
    losses = []
    for ep in range(args.epochs):
        z0 = sample_uniform_annulus(args.batch, delta=args.delta,
                                    seed=int(rng.integers(1e9)))
        x1 = sample_target(args.batch, args.delta, rng)
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

    # rho_iid: 20000 IID transported samples vs target on a 50x50 grid
    z = sample_uniform_annulus(20000, delta=args.delta,
                               seed=int(rng.integers(1e9)))
    xg = transport(model, z, n_steps=args.transport_steps, device="cpu")
    H, _, _ = np.histogram2d(xg[:, 1], xg[:, 0], bins=50,
                             range=[[-1, 1], [-1, 1]])
    H = H / H.sum()
    tgt = target_grid_pmf(args.delta)
    rho_iid = float(np.corrcoef(H.ravel(), tgt.ravel())[0, 1])

    # W2_hat: 3000 vs 3000 samples, ot.emd2 sqeuclidean, sqrt (paper protocol)
    import ot as pot
    zs = sample_uniform_annulus(3000, delta=args.delta,
                                seed=int(rng.integers(1e9)))
    xs = transport(model, zs, n_steps=args.transport_steps, device="cpu")
    ys = sample_target(3000, args.delta, rng)
    M = pot.dist(xs, ys, metric="sqeuclidean")
    w2 = float(np.sqrt(pot.emd2(np.ones(3000) / 3000, np.ones(3000) / 3000, M,
                                numItermax=200000)))

    cfm_final = float(np.mean(losses[-50:]))
    eps_v = math.sqrt(cfm_final)
    lv_net = model.lv_net()
    lv_hat = empirical_lv(model, args.delta, rng)
    d2 = args.delta ** 2
    metrics = {
        "args": vars(args),
        "rho_iid": rho_iid,
        "w2_hat": w2,
        "cfm_loss_final": cfm_final,
        "eps_v": eps_v,
        "L_v_net": lv_net,
        "L_v_hat": lv_hat,
        "floor_hat": math.exp(lv_hat) * eps_v,
        "floor_net": (math.exp(lv_net) * eps_v
                      if lv_net < 700 else float("inf")),
        "E_acc_latent_per_cycle": (1 - d2) ** 4 * (1 + d2) / (16 * args.delta ** 4),
        "train_seconds": time.time() - t0,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    print(json.dumps({k: v for k, v in metrics.items() if k != "args"},
                     indent=1))


if __name__ == "__main__":
    main()
