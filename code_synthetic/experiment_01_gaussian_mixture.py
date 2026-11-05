"""
Experiment 1: Uniform disc -> 2-mode Gaussian mixture.

Validates the full pipeline:
  1. Latent trajectory on unit disc (ergodic w.r.t. uniform)
  2. OT-CFM training to learn G_theta: uniform -> Gaussian mixture
  3. Apply G_theta to latent trajectory -> target trajectory
  4. Evaluate: density matching, ergodic metric, energy, visualization
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from latent_trajectory import radial_trajectory, sample_uniform_disc
from ot_cfm import VelocityNet, cfm_loss, transport


# =============================================================================
# Target distribution: 2-mode Gaussian mixture
# =============================================================================

# Paper-canonical symmetric GM parameters (main.tex L273, L1541; generate_figures.py::target_gaussian_mixture):
#   f_target(x) = 0.5 * N((-0.3, 0), 0.2^2 I) + 0.5 * N((+0.3, 0), 0.2^2 I)
# restricted to the annulus D_delta (delta = 0.01) and renormalized to unit mass,
# so that supp(f_target) is homotopy-equivalent to the latent domain and
# eta_top = 0 in Corollary 1 of the paper.
GM_MU1 = np.array([-0.3, 0.0])
GM_MU2 = np.array([ 0.3, 0.0])
GM_SIG1 = 0.2
GM_SIG2 = 0.2
GM_W1 = 0.5
GM_W2 = 0.5
D_DELTA = 0.01  # annulus inner radius


def sample_gaussian_mixture(n_samples, seed=None, delta=D_DELTA):
    """
    Sample from the paper-canonical symmetric 2-mode Gaussian mixture,
    rejection-truncated to the annulus D_delta = {delta <= ||x|| <= 1}.

    Args:
        n_samples: number of samples to return.
        seed: RNG seed.
        delta: inner radius of the annulus (default 0.01, matches paper).

    Returns:
        (n_samples, 2) array with every row satisfying delta <= ||x|| <= 1.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, 2), dtype=np.float64)
    filled = 0
    # Rejection rate: outer boundary ~2.6%, inner ~1e-5 at delta=0.01
    # (both modes at |mu|=0.3, sigma=0.2 -> outer tails at 3.5 sigma).
    # Oversample by 1.1x to keep the loop typically one-shot.
    oversample = 1.1
    while filled < n_samples:
        need = n_samples - filled
        draw = max(int(np.ceil(need * oversample)), 16)
        n1 = rng.binomial(draw, GM_W1)
        n2 = draw - n1
        s1 = rng.normal(0, GM_SIG1, (n1, 2)) + GM_MU1
        s2 = rng.normal(0, GM_SIG2, (n2, 2)) + GM_MU2
        batch = np.vstack([s1, s2])
        r = np.linalg.norm(batch, axis=1)
        keep = batch[(r >= delta) & (r <= 1.0)]
        take = min(len(keep), need)
        out[filled:filled + take] = keep[:take]
        filled += take
    rng.shuffle(out)
    return out


def gaussian_mixture_density(x, y):
    """Evaluate the (un-truncated, un-renormalized) paper-canonical GM density
    at (x, y). Callers that need the D_delta-truncated-and-renormalized
    density should apply the annular mask and divide by the masked Riemann sum;
    `evaluate()` below does exactly this.
    """
    def gauss2d(xv, yv, mu, sigma):
        r2 = (xv - mu[0])**2 + (yv - mu[1])**2
        return np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)

    return (GM_W1 * gauss2d(x, y, GM_MU1, GM_SIG1)
            + GM_W2 * gauss2d(x, y, GM_MU2, GM_SIG2))


# =============================================================================
# Training
# =============================================================================

def train_cfm(n_epochs=1000, batch_size=1024, lr=2e-3, hidden_dim=256, n_layers=4,
              seed=42):
    """Train the OT-CFM model."""
    torch.manual_seed(seed)
    device = 'cpu'

    model = VelocityNet(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    losses = []
    for epoch in range(n_epochs):
        model.train()
        # Fresh samples each epoch
        z0 = sample_uniform_disc(batch_size, seed=epoch * 1000 + seed)
        x1 = sample_gaussian_mixture(batch_size, seed=epoch * 1000 + seed + 1)

        loss = cfm_loss(model, z0, x1, device=device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")

    return model, losses


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, save_dir):
    """Run full evaluation and generate plots."""
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Generate latent trajectory ---
    print("Generating latent trajectory...")
    n_cycles = 300
    pts_per_cycle = 200
    pos_latent, times, vel_latent = radial_trajectory(
        n_cycles=n_cycles, points_per_cycle=pts_per_cycle, seed=42
    )
    # Discard transient (first 2 time units)
    mask = times > 2.0
    pos_latent = pos_latent[mask]
    vel_latent = vel_latent[mask]
    times_ss = times[mask]
    print(f"  Latent trajectory points: {len(pos_latent)}")

    # --- 2. Apply G_theta to latent trajectory ---
    print("Transporting latent trajectory through G_theta...")
    pos_target = transport(model, pos_latent, n_steps=20)
    print(f"  Target trajectory points: {len(pos_target)}")

    # --- 3. Density matching evaluation ---
    print("Evaluating density matching...")
    # Bin-based comparison
    grid_res = 50
    x_edges = np.linspace(-1.5, 1.5, grid_res + 1)
    y_edges = np.linspace(-1.5, 1.5, grid_res + 1)

    # Empirical density from trajectory
    hist_emp, _, _ = np.histogram2d(pos_target[:, 0], pos_target[:, 1],
                                     bins=[x_edges, y_edges], density=True)

    # Target density on grid: D_delta-masked + renormalized per paper.
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
    target_density = gaussian_mixture_density(X, Y)
    # Zero-out cells outside D_delta = {delta <= ||x|| <= 1}; renormalize so
    # the Riemann integral over the masked cells equals 1.
    R = np.sqrt(X**2 + Y**2)
    mask = (R >= D_DELTA) & (R <= 1.0)
    target_density = target_density * mask
    target_density /= (target_density.sum() * (x_centers[1] - x_centers[0]) *
                        (y_centers[1] - y_centers[0]))

    # L2 error
    cell_area = (x_centers[1] - x_centers[0]) * (y_centers[1] - y_centers[0])
    l2_error = np.sqrt(np.sum((hist_emp - target_density)**2) * cell_area)
    print(f"  L2 density error: {l2_error:.4f}")

    # Correlation
    corr = np.corrcoef(hist_emp.flatten(), target_density.flatten())[0, 1]
    print(f"  Density correlation: {corr:.4f}")

    # --- 4. Compute acceleration energy ---
    print("Computing trajectory energy...")
    # Filter out duplicate time points (occur at cycle boundaries)
    dt_vals = np.diff(times_ss)
    valid = dt_vals > 1e-8
    # Use only valid intervals
    dx_t = np.diff(pos_target, axis=0)
    dx_l = np.diff(pos_latent, axis=0)
    vel_target = dx_t[valid] / dt_vals[valid, None]
    vel_latent_fd = dx_l[valid] / dt_vals[valid, None]

    # Acceleration via finite differences of velocity (use valid-interval dt)
    dt_valid = dt_vals[valid]
    dv_t = np.diff(vel_target, axis=0)
    dv_l = np.diff(vel_latent_fd, axis=0)
    dt_acc = 0.5 * (dt_valid[:-1] + dt_valid[1:])
    acc_target = dv_t / dt_acc[:, None]
    acc_latent = dv_l / dt_acc[:, None]

    # Clip extreme values at cycle boundaries (95th percentile)
    clip_val_t = np.percentile(np.sum(acc_target**2, axis=1), 95)
    clip_val_l = np.percentile(np.sum(acc_latent**2, axis=1), 95)
    acc_sq_t = np.clip(np.sum(acc_target**2, axis=1), 0, clip_val_t)
    acc_sq_l = np.clip(np.sum(acc_latent**2, axis=1), 0, clip_val_l)

    acc_energy = np.mean(acc_sq_t)
    acc_energy_latent = np.mean(acc_sq_l)
    print(f"  Mean acceleration energy (target, 95%-clipped): {acc_energy:.4f}")
    print(f"  Latent acceleration energy (95%-clipped): {acc_energy_latent:.4f}")
    print(f"  Acceleration ratio (target/latent): {acc_energy / max(acc_energy_latent, 1e-10):.2f}x")

    # --- 5. Generate plots ---
    print("Generating plots...")

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # (a) Latent trajectory (first 20 cycles for clarity)
    ax1 = fig.add_subplot(gs[0, 0])
    n_show = 20 * pts_per_cycle
    ax1.plot(pos_latent[:n_show, 0], pos_latent[:n_show, 1],
             linewidth=0.3, alpha=0.5, color='blue')
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=1.5)
    ax1.add_patch(circle)
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_aspect('equal')
    ax1.set_title('(a) Latent Trajectory (20 cycles)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # (b) Transformed trajectory (same 20 cycles)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(pos_target[:n_show, 0], pos_target[:n_show, 1],
             linewidth=0.3, alpha=0.5, color='red')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title('(b) Transformed Trajectory (20 cycles)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # (c) Target density (ground truth)
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.pcolormesh(X, Y, target_density, cmap='Reds', shading='auto')
    plt.colorbar(im, ax=ax3, shrink=0.8)
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.set_title('(c) Target Density (Gaussian Mixture)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    # (d) Empirical density from trajectory
    ax4 = fig.add_subplot(gs[1, 0])
    im2 = ax4.pcolormesh(X, Y, hist_emp, cmap='Reds', shading='auto')
    plt.colorbar(im2, ax=ax4, shrink=0.8)
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_aspect('equal')
    ax4.set_title(f'(d) Empirical Density from Trajectory\nCorr={corr:.3f}, L2={l2_error:.4f}')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')

    # (e) Transport visualization: source -> target point cloud
    ax5 = fig.add_subplot(gs[1, 1])
    n_viz = 500
    z0_viz = sample_uniform_disc(n_viz, seed=99)
    x1_viz = transport(model, z0_viz, n_steps=20)
    ax5.scatter(z0_viz[:, 0], z0_viz[:, 1], s=3, alpha=0.4, c='blue', label='Source (uniform)')
    ax5.scatter(x1_viz[:, 0], x1_viz[:, 1], s=3, alpha=0.4, c='red', label='Transported')
    # Draw some transport lines
    for i in range(0, n_viz, 10):
        ax5.plot([z0_viz[i, 0], x1_viz[i, 0]],
                 [z0_viz[i, 1], x1_viz[i, 1]],
                 'k-', alpha=0.1, linewidth=0.5)
    ax5.set_xlim(-1.5, 1.5)
    ax5.set_ylim(-1.5, 1.5)
    ax5.set_aspect('equal')
    ax5.legend(fontsize=8)
    ax5.set_title('(e) Transport Map: Source -> Target')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')

    # (f) Training loss
    ax6 = fig.add_subplot(gs[1, 2])
    # Load losses from global
    ax6.set_title('(f) Reserved for Convergence Analysis')
    ax6.text(0.5, 0.5, 'See training loss plot\nin separate figure',
             ha='center', va='center', transform=ax6.transAxes, fontsize=12)

    plt.savefig(os.path.join(save_dir, 'experiment_01_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Separate training loss plot
    return {
        'l2_error': l2_error,
        'correlation': corr,
        'acc_energy_target': acc_energy,
        'acc_energy_latent': acc_energy_latent,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiment_01')
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Experiment 1: Uniform Disc -> 2-Mode Gaussian Mixture")
    print("=" * 60)

    # Validate latent trajectory
    print("\n--- Validating latent trajectory ---")
    pos, t, vel = radial_trajectory(n_cycles=200, points_per_cycle=200, seed=42)
    mask = t > 2.0
    r_vals = np.sqrt(pos[mask, 0]**2 + pos[mask, 1]**2)
    test_r = np.linspace(0, 1, 50)
    empirical_cdf = np.array([np.mean(r_vals <= r) for r in test_r])
    theoretical_cdf = test_r**2
    max_cdf_error = np.max(np.abs(empirical_cdf - theoretical_cdf))
    print(f"  Max radial CDF error: {max_cdf_error:.4f} (should be < 0.02)")

    # Train OT-CFM
    print("\n--- Training OT-CFM ---")
    model, losses = train_cfm(
        n_epochs=1000,
        batch_size=1024,
        lr=2e-3,
        hidden_dim=256,
        n_layers=4,
        seed=42
    )

    # Save training loss plot
    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.semilogy(losses)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('CFM Loss')
    ax_loss.set_title('Training Loss (OT-CFM)')
    ax_loss.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Evaluate
    print("\n--- Evaluation ---")
    metrics = evaluate(model, save_dir)

    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Density correlation:     {metrics['correlation']:.4f}")
    print(f"  L2 density error:        {metrics['l2_error']:.4f}")
    print(f"  Acceleration (target):   {metrics['acc_energy_target']:.4f}")
    print(f"  Acceleration (latent):   {metrics['acc_energy_latent']:.4f}")
    print(f"  Accel ratio:             {metrics['acc_energy_target']/max(metrics['acc_energy_latent'],1e-10):.2f}x")
    print(f"\nPlots saved to: {save_dir}")
    print("=" * 60)
