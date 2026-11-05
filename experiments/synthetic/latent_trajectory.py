"""
Latent Trajectory Generator: Radial back-and-forth process on the unit disc.

For all t > 2, the UAV location is uniformly distributed on the unit disc.
This module provides both the continuous trajectory and uniform sampling.
"""

import numpy as np


def radial_trajectory(n_cycles=100, points_per_cycle=200, seed=None):
    """
    Generate a latent trajectory via radial back-and-forth on the unit disc.

    Each cycle:
      - Pick angle theta_k ~ U[0, 2*pi)
      - Outward leg: r(t) = sqrt(t) for t in [0, 1], speed = 1/(2*sqrt(t))
      - Return leg: r(t) = sqrt(2 - t) for t in [1, 2] (mirror)

    With random start delay T ~ U[0, 2], the stationary distribution is
    Uniform(unit disc) for t > 2.

    Args:
        n_cycles: number of out-and-back cycles
        points_per_cycle: number of time points per cycle (half outward, half return)
        seed: random seed

    Returns:
        positions: (N, 2) array of (x, y) positions
        times: (N,) array of time values
        velocities: (N, 2) array of (vx, vy) velocities
    """
    rng = np.random.default_rng(seed)

    half = points_per_cycle // 2
    # Parameterize outward leg: t in (epsilon, 1] to avoid singularity at t=0
    t_out = np.linspace(1e-4, 1.0, half)
    # Return leg mirrors: t in [1, 2-epsilon]
    t_ret = np.linspace(1.0, 2.0 - 1e-4, half)

    all_x, all_y, all_t, all_vx, all_vy = [], [], [], [], []
    current_time = 0.0

    for k in range(n_cycles):
        theta = rng.uniform(0, 2 * np.pi)
        cos_th, sin_th = np.cos(theta), np.sin(theta)

        # Outward leg
        r_out = np.sqrt(t_out)
        speed_out = 1.0 / (2.0 * np.sqrt(t_out))
        x_out = r_out * cos_th
        y_out = r_out * sin_th
        vx_out = speed_out * cos_th
        vy_out = speed_out * sin_th

        # Return leg (mirror: r(t) = sqrt(2 - t))
        r_ret = np.sqrt(2.0 - t_ret)
        speed_ret = 1.0 / (2.0 * np.sqrt(2.0 - t_ret))
        x_ret = r_ret * cos_th
        y_ret = r_ret * sin_th
        vx_ret = -speed_ret * cos_th  # negative: moving inward
        vy_ret = -speed_ret * sin_th

        # Concatenate
        times_cycle = np.concatenate([t_out, t_ret]) + current_time
        x_cycle = np.concatenate([x_out, x_ret])
        y_cycle = np.concatenate([y_out, y_ret])
        vx_cycle = np.concatenate([vx_out, vx_ret])
        vy_cycle = np.concatenate([vy_out, vy_ret])

        all_x.append(x_cycle)
        all_y.append(y_cycle)
        all_t.append(times_cycle)
        all_vx.append(vx_cycle)
        all_vy.append(vy_cycle)
        current_time += 2.0

    positions = np.column_stack([np.concatenate(all_x), np.concatenate(all_y)])
    times = np.concatenate(all_t)
    velocities = np.column_stack([np.concatenate(all_vx), np.concatenate(all_vy)])

    return positions, times, velocities


def sample_uniform_disc(n_samples, seed=None):
    """
    Sample uniformly from the unit disc (for training as source distribution).

    Uses the standard method: r = sqrt(U), theta = 2*pi*V.
    """
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, n_samples))
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


if __name__ == "__main__":
    # Quick validation: check that trajectory positions are ~uniform on disc
    pos, t, vel = radial_trajectory(n_cycles=500, points_per_cycle=200, seed=42)

    # Discard first 2 time units (transient)
    mask = t > 2.0
    pos_ss = pos[mask]

    # Check radial CDF: should be P(R <= r) = r^2
    r_vals = np.sqrt(pos_ss[:, 0]**2 + pos_ss[:, 1]**2)
    test_r = np.linspace(0, 1, 50)
    empirical_cdf = np.array([np.mean(r_vals <= r) for r in test_r])
    theoretical_cdf = test_r**2
    max_error = np.max(np.abs(empirical_cdf - theoretical_cdf))
    print(f"Trajectory points (after transient): {len(pos_ss)}")
    print(f"Max CDF error (radial): {max_error:.4f}")
    print(f"PASS: {'Yes' if max_error < 0.02 else 'No'}")
