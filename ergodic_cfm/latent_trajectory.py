"""
Latent Ergodic Trajectory (paper §2.1).

We fix delta > 0 and work on the ANNULAR latent domain

    D_delta = { z in R^2 : delta <= ||z|| <= 1 }.

The inner disc {||z|| < delta} is excluded (the paper's "base station / airport").
This non-simply-connected shape is what makes the theorems of §2 go through —
see Remark 1 and the topology residual discussion in §2.2.

Each cycle k:
  * draw a fresh angle theta_k ~ Uniform[0, 2*pi)
  * sweep radially OUT along the ray theta_k with
            r(s) = sqrt(delta^2 + (1 - delta^2) * s),   s in [0, 1]
    so that r(0) = delta and r(1) = 1 (paper eq. (2)).
  * sweep back IN along the same ray (mirror), so one full cycle has
    parameter length 2 (s runs 0 -> 1 -> 0).

Proposition 1 (paper): with an independent phase offset Phi_0 ~ Uniform[0, 2),
z(t) is distributed as Uniform(D_delta) with density
    pi_0^delta = 1 / (pi * (1 - delta^2))   for every t >= 0,
and the trajectory is ergodic with respect to pi_0^delta.

In this implementation we use the i.i.d.-cycle ergodicity argument directly and
discard a short "burn-in" at the start (one full cycle) in the caller — this is
asymptotically equivalent to the Phi_0 phase convention (both rely on SLLN over
i.i.d. cycles).
"""

from __future__ import annotations

import numpy as np

__all__ = ["radial_trajectory", "sample_uniform_annulus", "sample_uniform_disc"]


def _radial_profile(s, delta):
    """r(s) = sqrt(delta^2 + (1 - delta^2) s)  — paper eq. (2)."""
    return np.sqrt(delta * delta + (1.0 - delta * delta) * s)


def radial_trajectory(n_cycles=100, points_per_cycle=200, delta=0.05, seed=None):
    """
    Generate the latent trajectory z(t) on D_delta (paper §2.1).

    Args:
        n_cycles: number of out-and-back cycles.
        points_per_cycle: samples per full cycle (half outward, half inward).
        delta: inner radius of the annulus (paper typical range 0.01 - 0.1).
            delta=0.0 degenerates to the full disc (used only for ablation).
        seed: numpy RNG seed.

    Returns:
        positions:  (N, 2)  (x, y) positions on D_delta
        times:      (N,)    parameter time (one cycle has length 2.0)
        velocities: (N, 2)  (dx/dt, dy/dt) analytical velocities
    """
    if not (0.0 <= delta < 1.0):
        raise ValueError(f"delta must be in [0, 1); got {delta}")

    rng = np.random.default_rng(seed)
    half = max(1, points_per_cycle // 2)

    # Parameterize each half-cycle with s in [0,1]. The outgoing leg uses s,
    # the return leg uses 1 - s (so r sweeps 1 -> delta). We use an open grid
    # that avoids the exact turnaround instants {s=0, s=1} — those are a
    # measure-zero set and are excluded in the paper's energy integrals
    # (Theorem 1 and Lemma 1) anyway.
    s_out = np.linspace(0.0, 1.0, half + 1)[:-1] + 0.5 / half
    s_out = np.clip(s_out, 1e-6, 1.0 - 1e-6)
    s_ret = s_out[::-1]  # mirror

    # Radial profile and dr/dt on each half-cycle.
    r_out = _radial_profile(s_out, delta)
    r_ret = _radial_profile(s_ret, delta)

    # dr/ds = (1 - delta^2) / (2 r); time parameterization is ds = dt, so
    # dr/dt = (1 - delta^2) / (2 r(s)). On the return leg the radial velocity
    # has the opposite sign.
    speed_out = (1.0 - delta * delta) / (2.0 * r_out)
    speed_ret = (1.0 - delta * delta) / (2.0 * r_ret)

    all_x, all_y, all_t, all_vx, all_vy = [], [], [], [], []
    current_time = 0.0

    for _ in range(n_cycles):
        theta = rng.uniform(0.0, 2.0 * np.pi)
        cos_th, sin_th = np.cos(theta), np.sin(theta)

        x_out = r_out * cos_th
        y_out = r_out * sin_th
        vx_out = speed_out * cos_th
        vy_out = speed_out * sin_th

        x_ret = r_ret * cos_th
        y_ret = r_ret * sin_th
        vx_ret = -speed_ret * cos_th  # inward
        vy_ret = -speed_ret * sin_th

        times_cycle = np.concatenate([s_out, 1.0 + (1.0 - s_ret)]) + current_time
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


def sample_uniform_annulus(n_samples, delta=0.05, seed=None):
    """
    Sample i.i.d. from Uniform(D_delta) — the stationary density pi_0^delta
    of the latent trajectory (paper Proposition 1).

    Uses inverse-CDF:  R = sqrt(delta^2 + (1 - delta^2) U),  Theta = 2 pi V,
    with U, V ~ Uniform[0,1]. For delta=0 this reduces to the standard
    uniform-disc sampler.

    Returns:
        (n_samples, 2) array of (x, y) points in D_delta.
    """
    if not (0.0 <= delta < 1.0):
        raise ValueError(f"delta must be in [0, 1); got {delta}")
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, n_samples)
    r = np.sqrt(delta * delta + (1.0 - delta * delta) * u)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def sample_uniform_disc(n_samples, seed=None):
    """Backward-compatible alias: uniform on the full disc (delta = 0)."""
    return sample_uniform_annulus(n_samples, delta=0.0, seed=seed)


if __name__ == "__main__":
    # Validate: empirical radial CDF should match (r^2 - delta^2) / (1 - delta^2)
    # on [delta, 1] for the annular stationary density pi_0^delta.
    delta_test = 0.05
    pos, t, vel = radial_trajectory(
        n_cycles=500, points_per_cycle=200, delta=delta_test, seed=42
    )
    mask = t > 2.0  # drop one-cycle burn-in (Phi_0 convention proxy)
    pos_ss = pos[mask]

    r_vals = np.linalg.norm(pos_ss, axis=1)
    test_r = np.linspace(delta_test, 1.0, 50)
    empirical_cdf = np.array([np.mean(r_vals <= r) for r in test_r])
    theoretical_cdf = (test_r**2 - delta_test**2) / (1.0 - delta_test**2)
    max_error = float(np.max(np.abs(empirical_cdf - theoretical_cdf)))

    print(f"delta              : {delta_test}")
    print(f"Trajectory points  : {len(pos_ss)}")
    print(f"Max radial CDF err : {max_error:.4f}")
    print(f"PASS               : {'Yes' if max_error < 0.02 else 'No'}")
