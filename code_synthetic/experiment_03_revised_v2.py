"""
Experiment 3 (Revised): Constraint Flexibility with Off-Center No-Fly Zone

Revision rationale: The original Experiment 3 placed the NFZ at the origin,
which is the starting point of every radial cycle in the latent trajectory.
This forced G_theta to reroute ALL paths, creating an adversarial worst case.

This revision (v2) places the NFZ at (0.5, -0.5) — off-center, in a
moderate-density region between the two Gaussian modes. This is realistic
(real NFZs are rarely at the coverage center or on top of a mode peak)
and cleanly demonstrates constraint flexibility without adversarial
geometry (origin) or extreme density-constraint conflict (on a mode).

Same three variants as original:
  (a) Unconstrained baseline
  (b) NFZ penalty only (lambda_nfz=50)
  (c) NFZ + acceleration penalty (lambda_nfz=50, lambda_acc=0.1)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from latent_trajectory import radial_trajectory, sample_uniform_disc
from ot_cfm import VelocityNet, ot_coupling, transport


# ===================================================================
# Target density (same as Experiment 1)
# ===================================================================

def sample_gaussian_mixture(n_samples, seed=None):
    rng = np.random.RandomState(seed)
    mu1, sig1, w1 = np.array([0.5, 0.4]), 0.25, 0.6
    mu2, sig2 = np.array([-0.5, -0.3]), 0.20
    n1 = rng.binomial(n_samples, w1)
    n2 = n_samples - n1
    s1 = rng.randn(n1, 2) * sig1 + mu1
    s2 = rng.randn(n2, 2) * sig2 + mu2
    samples = np.vstack([s1, s2]).astype(np.float32)
    rng.shuffle(samples)
    return samples


def gaussian_mixture_density(X, Y):
    from scipy.stats import multivariate_normal
    mu1, sig1, w1 = [0.5, 0.4], 0.25, 0.6
    mu2, sig2, w2 = [-0.5, -0.3], 0.20, 0.4
    pos = np.dstack((X, Y))
    d = w1 * multivariate_normal.pdf(pos, mu1, sig1**2 * np.eye(2)) + \
        w2 * multivariate_normal.pdf(pos, mu2, sig2**2 * np.eye(2))
    return d


# ===================================================================
# No-fly zone definition — OFF-CENTER
# ===================================================================

NFZ_CENTER = np.array([0.5, -0.5])  # Off-center, between modes (moderate density)
NFZ_RADIUS = 0.2   # paper configuration (Experiment 3)


# ===================================================================
# Differentiable transport (for penalty computation during training)
# ===================================================================

def transport_differentiable(model, z0, n_steps=10):
    """
    Transport z0 through the velocity field WITH gradients enabled.
    Uses fewer steps than inference transport for efficiency.
    """
    y = z0.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        s = torch.full((y.shape[0],), i * dt, device=y.device)
        v = model(s, y)
        y = y + v * dt
    return y


# ===================================================================
# Constraint penalties
# ===================================================================

def nfz_penalty(transported_pts, center, radius, margin=0.05):
    """
    No-fly zone penalty: penalize points inside or near the NFZ.
    R_nfz = mean[ max(0, radius + margin - ||G(z) - center||)^2 ]
    """
    center_t = torch.tensor(center, dtype=torch.float32, device=transported_pts.device)
    dist = torch.norm(transported_pts - center_t, dim=1)
    violation = torch.clamp(radius + margin - dist, min=0)
    return (violation ** 2).mean()


def acceleration_penalty(model, z0, n_steps=10, eps=0.01):
    """
    Approximate acceleration penalty via finite-difference Hessian norm.
    """
    G_z = transport_differentiable(model, z0, n_steps)

    e1 = torch.zeros_like(z0)
    e1[:, 0] = eps
    G_z_plus_e1 = transport_differentiable(model, z0 + e1, n_steps)
    G_z_minus_e1 = transport_differentiable(model, z0 - e1, n_steps)

    e2 = torch.zeros_like(z0)
    e2[:, 1] = eps
    G_z_plus_e2 = transport_differentiable(model, z0 + e2, n_steps)
    G_z_minus_e2 = transport_differentiable(model, z0 - e2, n_steps)

    d2G_dz1 = (G_z_plus_e1 + G_z_minus_e1 - 2 * G_z) / (eps ** 2)
    d2G_dz2 = (G_z_plus_e2 + G_z_minus_e2 - 2 * G_z) / (eps ** 2)

    hessian_norm = (d2G_dz1 ** 2).sum(dim=1) + (d2G_dz2 ** 2).sum(dim=1)
    return hessian_norm.mean()


# ===================================================================
# Training with constraints
# ===================================================================

def cfm_loss_with_constraints(model, z0, x1, device='cpu',
                               lambda_nfz=0.0, lambda_acc=0.0,
                               nfz_center=None, nfz_radius=0.0,
                               compute_penalties=True):
    """CFM loss + optional constraint penalties."""
    B = z0.shape[0]

    z0_np = z0.numpy() if isinstance(z0, torch.Tensor) else z0
    x1_np = x1.numpy() if isinstance(x1, torch.Tensor) else x1

    sigma = ot_coupling(z0_np, x1_np)
    x1_paired = x1_np[sigma]

    z0_t = torch.tensor(z0_np, dtype=torch.float32, device=device, requires_grad=False)
    x1_t = torch.tensor(x1_paired, dtype=torch.float32, device=device, requires_grad=False)

    s = torch.rand(B, device=device)
    s_expand = s.unsqueeze(1)
    y_s = (1 - s_expand) * z0_t + s_expand * x1_t
    u_target = x1_t - z0_t

    v_pred = model(s, y_s)
    loss_cfm = ((v_pred - u_target) ** 2).mean()

    loss_nfz = torch.tensor(0.0, device=device)
    loss_acc = torch.tensor(0.0, device=device)

    if compute_penalties and (lambda_nfz > 0 or lambda_acc > 0):
        n_penalty = min(64, B)
        z0_pen = torch.tensor(z0_np[:n_penalty], dtype=torch.float32, device=device)

        if lambda_nfz > 0:
            G_z = transport_differentiable(model, z0_pen, n_steps=5)
            loss_nfz = nfz_penalty(G_z, nfz_center, nfz_radius, margin=0.05)

        if lambda_acc > 0:
            loss_acc = acceleration_penalty(model, z0_pen[:32], n_steps=5, eps=0.01)

    total_loss = loss_cfm + lambda_nfz * loss_nfz + lambda_acc * loss_acc
    return total_loss, loss_cfm.item(), loss_nfz.item(), loss_acc.item()


def train_constrained(n_epochs=1000, batch_size=1024, lr=2e-3,
                      hidden_dim=256, n_layers=4, seed=42,
                      lambda_nfz=0.0, lambda_acc=0.0,
                      nfz_center=None, nfz_radius=0.0,
                      label=""):
    """Train with optional constraints."""
    torch.manual_seed(seed)
    device = 'cpu'

    model = VelocityNet(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = {'cfm': [], 'nfz': [], 'acc': [], 'total': []}

    penalty_freq = 3
    for epoch in range(n_epochs):
        model.train()
        z0 = sample_uniform_disc(batch_size, seed=epoch * 1000 + seed)
        x1 = sample_gaussian_mixture(batch_size, seed=epoch * 1000 + seed + 1)

        do_penalty = (epoch % penalty_freq == 0)
        total_loss, cfm_val, nfz_val, acc_val = cfm_loss_with_constraints(
            model, z0, x1, device=device,
            lambda_nfz=lambda_nfz, lambda_acc=lambda_acc,
            nfz_center=nfz_center, nfz_radius=nfz_radius,
            compute_penalties=do_penalty
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history['total'].append(total_loss.item())
        history['cfm'].append(cfm_val)
        history['nfz'].append(nfz_val)
        history['acc'].append(acc_val)

        if (epoch + 1) % 50 == 0:
            parts = [f"Epoch {epoch+1}/{n_epochs}"]
            parts.append(f"CFM={cfm_val:.5f}")
            if lambda_nfz > 0:
                parts.append(f"NFZ={nfz_val:.5f}")
            if lambda_acc > 0:
                parts.append(f"ACC={acc_val:.4f}")
            print(f"  [{label}] {', '.join(parts)}")

    return model, history


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_model(model, label, results_dir, nfz_center=None, nfz_radius=0.0):
    """Evaluate a trained model and return metrics."""
    device = 'cpu'

    # --- IID transport evaluation ---
    z0_iid = sample_uniform_disc(10000, seed=123)
    x1_iid = transport(model, z0_iid, n_steps=30, device=device)

    # Density histogram
    grid_res = 50
    x_edges = np.linspace(-1.5, 1.5, grid_res + 1)
    y_edges = np.linspace(-1.5, 1.5, grid_res + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
    cell_area = (x_centers[1] - x_centers[0]) * (y_centers[1] - y_centers[0])

    hist_iid, _, _ = np.histogram2d(x1_iid[:, 0], x1_iid[:, 1],
                                     bins=[x_edges, y_edges], density=True)
    target_d = gaussian_mixture_density(X, Y)
    target_d_norm = target_d / (target_d.sum() * cell_area + 1e-12)

    corr_iid = np.corrcoef(hist_iid.flatten(), target_d_norm.flatten())[0, 1]

    # NFZ violations
    if nfz_center is not None:
        dist_to_nfz = np.sqrt((x1_iid[:, 0] - nfz_center[0])**2 +
                               (x1_iid[:, 1] - nfz_center[1])**2)
        nfz_violations = np.sum(dist_to_nfz < nfz_radius)
        nfz_violation_rate = nfz_violations / len(x1_iid)
    else:
        nfz_violations = 0
        nfz_violation_rate = 0.0

    # Trajectory-based acceleration
    pos_latent, times, vels = radial_trajectory(n_cycles=50, points_per_cycle=200, seed=0)
    pts_per_cycle = 200
    start_idx = 2 * pts_per_cycle
    pos_latent = pos_latent[start_idx:]
    times_ss = times[start_idx:]
    pos_target = transport(model, pos_latent, n_steps=30, device=device)

    dt_vals = np.diff(times_ss)
    valid = dt_vals > 1e-8
    dx_t = np.diff(pos_target, axis=0)
    dx_l = np.diff(pos_latent, axis=0)
    vel_t = dx_t[valid] / dt_vals[valid, None]
    vel_l = dx_l[valid] / dt_vals[valid, None]
    dt_valid = dt_vals[valid]
    dv_t = np.diff(vel_t, axis=0)
    dv_l = np.diff(vel_l, axis=0)
    dt_acc = 0.5 * (dt_valid[:-1] + dt_valid[1:])
    acc_t = dv_t / dt_acc[:, None]
    acc_l = dv_l / dt_acc[:, None]
    clip_t = np.percentile(np.sum(acc_t**2, axis=1), 95)
    clip_l = np.percentile(np.sum(acc_l**2, axis=1), 95)
    acc_energy = np.mean(np.clip(np.sum(acc_t**2, axis=1), 0, clip_t))
    acc_energy_lat = np.mean(np.clip(np.sum(acc_l**2, axis=1), 0, clip_l))
    acc_ratio = acc_energy / max(acc_energy_lat, 1e-10)

    metrics = {
        'label': label,
        'corr_iid': corr_iid,
        'nfz_violations': nfz_violations,
        'nfz_violation_rate': nfz_violation_rate,
        'acc_energy': acc_energy,
        'acc_ratio': acc_ratio,
        'transported_iid': x1_iid,
        'pos_target': pos_target[:10*pts_per_cycle],
        'pos_latent': pos_latent[:10*pts_per_cycle],
    }

    print(f"  [{label}] Corr={corr_iid:.4f}, NFZ viol={nfz_violation_rate:.4f} "
          f"({nfz_violations}/{len(x1_iid)}), Accel ratio={acc_ratio:.2f}x")

    return metrics


# ===================================================================
# Plotting
# ===================================================================

def plot_comparison(all_metrics, results_dir):
    """Generate comparison plots across all variants."""

    n_variants = len(all_metrics)
    fig, axes = plt.subplots(2, n_variants, figsize=(6 * n_variants, 10))
    if n_variants == 1:
        axes = axes.reshape(2, 1)

    nfz_circle_kw = dict(radius=NFZ_RADIUS,
                          fill=True, facecolor='gray', edgecolor='red',
                          linewidth=2, alpha=0.4, zorder=5)

    for col, m in enumerate(all_metrics):
        # Top row: IID transported points with NFZ
        ax = axes[0, col]
        pts = m['transported_iid']
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3, alpha=0.2, c='blue')
        nfz = plt.Circle(xy=NFZ_CENTER, **nfz_circle_kw)
        ax.add_patch(nfz)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f"{m['label']}\nCorr={m['corr_iid']:.3f}, "
                     f"NFZ viol={m['nfz_violation_rate']:.1%}",
                     fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('y')

        # Bottom row: trajectory segments
        ax = axes[1, col]
        traj = m['pos_target']
        ax.plot(traj[:, 0], traj[:, 1], linewidth=0.3, alpha=0.5, color='red')
        nfz2 = plt.Circle(xy=NFZ_CENTER, **nfz_circle_kw)
        ax.add_patch(nfz2)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f"Trajectory (20 cycles)\nAccel ratio={m['acc_ratio']:.2f}x",
                     fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('y')

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, 'constraint_comparison_revised.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Summary table figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    headers = ['Variant', 'Density\nCorrelation', 'NFZ\nViolations', 'NFZ\nViol. Rate',
               'Accel\nRatio']
    table_data = []
    for m in all_metrics:
        table_data.append([
            m['label'],
            f"{m['corr_iid']:.3f}",
            f"{m['nfz_violations']}",
            f"{m['nfz_violation_rate']:.1%}",
            f"{m['acc_ratio']:.2f}x"
        ])
    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    for j in range(len(headers)):
        table[0, j].set_text_props(fontweight='bold')
    ax.set_title('Experiment 3 (Revised v2): Off-Center NFZ at (0.5, -0.5)',
                 fontsize=13, fontweight='bold', pad=20)
    fig.savefig(os.path.join(results_dir, 'constraint_summary_table_revised.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\nExperiment 3 summary:")
    for m in all_metrics:
        print(f"  {m['label']:<20} Corr={m['corr_iid']:.2f}, "
              f"NFZ viol={m['nfz_violation_rate']:.1%}, "
              f"Accel={m['acc_ratio']:.2f}x")
    print("=" * 70)


# ===================================================================
# Main
# ===================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("Experiment 3: Off-Center No-Fly Zone")
    print("  Target: 2-Mode Gaussian Mixture (same as Exp 1)")
    print(f"  No-Fly Zone: center={NFZ_CENTER}, radius={NFZ_RADIUS}")
    print("=" * 65)

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiment_03')
    os.makedirs(results_dir, exist_ok=True)

    all_metrics = []
    all_models = {}

    # ---------------------------------------------------------------
    # Variant A: Unconstrained (baseline)
    # ---------------------------------------------------------------
    print("\n--- Variant A: Unconstrained ---")
    model_a, hist_a = train_constrained(
        n_epochs=500, batch_size=1024, seed=42,
        lambda_nfz=0.0, lambda_acc=0.0,
        label="Unconstrained"
    )
    torch.save(model_a.state_dict(), os.path.join(results_dir, 'model_unconstrained.pt'))
    m_a = evaluate_model(model_a, "Unconstrained", results_dir,
                         nfz_center=NFZ_CENTER, nfz_radius=NFZ_RADIUS)
    all_metrics.append(m_a)
    all_models['unconstrained'] = model_a

    # ---------------------------------------------------------------
    # Variant B: No-fly zone only
    # ---------------------------------------------------------------
    print("\n--- Variant B: No-Fly Zone (lambda=50) ---")
    model_b, hist_b = train_constrained(
        n_epochs=500, batch_size=1024, seed=42,
        lambda_nfz=50.0, lambda_acc=0.0,
        nfz_center=NFZ_CENTER, nfz_radius=NFZ_RADIUS,
        label="NFZ Only"
    )
    torch.save(model_b.state_dict(), os.path.join(results_dir, 'model_nfz.pt'))
    m_b = evaluate_model(model_b, "NFZ Only", results_dir,
                         nfz_center=NFZ_CENTER, nfz_radius=NFZ_RADIUS)
    all_metrics.append(m_b)
    all_models['nfz'] = model_b

    # ---------------------------------------------------------------
    # Variant C: No-fly zone + acceleration penalty
    # ---------------------------------------------------------------
    print("\n--- Variant C: NFZ + Acceleration Penalty ---")
    model_c, hist_c = train_constrained(
        n_epochs=500, batch_size=1024, seed=42,
        lambda_nfz=50.0, lambda_acc=0.1,
        nfz_center=NFZ_CENTER, nfz_radius=NFZ_RADIUS,
        label="NFZ + Accel"
    )
    torch.save(model_c.state_dict(), os.path.join(results_dir, 'model_nfz_acc.pt'))
    m_c = evaluate_model(model_c, "NFZ + Accel", results_dir,
                         nfz_center=NFZ_CENTER, nfz_radius=NFZ_RADIUS)
    all_metrics.append(m_c)
    all_models['nfz_acc'] = model_c

    # ---------------------------------------------------------------
    # Plots and summary
    # ---------------------------------------------------------------
    print("\n--- Generating comparison plots ---")
    plot_comparison(all_metrics, results_dir)

    # Print summary
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY (Revised — Off-Center NFZ)")
    print("=" * 65)
    print(f"{'Variant':<20} {'Corr':>8} {'NFZ Viol':>10} {'Rate':>8} {'Accel':>8}")
    print("-" * 58)
    for m in all_metrics:
        print(f"{m['label']:<20} {m['corr_iid']:>8.3f} {m['nfz_violations']:>10d} "
              f"{m['nfz_violation_rate']:>7.1%} {m['acc_ratio']:>7.2f}x")
    print("=" * 65)
    print(f"\nPlots saved to: {os.path.abspath(results_dir)}")
