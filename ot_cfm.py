"""
OT-CFM: Conditional Flow Matching with Optimal Transport coupling (paper §2.2).

We learn a velocity field v_theta(s, y), s in [0, 1], so that the ODE

        dy/ds = v_theta(s, y),   y(0) = z_0 ~ pi_0^delta

produces y(1) ~ f_target. The map G_theta(z_0) := y(1) is the pushforward map
of interest. Training minimises the CFM loss (paper eq. (3))

        L_CFM(theta) = E_{s, (z_0, x_1)} || v_theta(s, y_s) - (x_1 - z_0) ||^2,
        y_s = (1 - s) z_0 + s x_1,

with OT-coupled pairs (z_0, x_1) obtained per mini-batch by Sinkhorn +
Hungarian (§2.2: "OT-coupled pairs obtained via the Sinkhorn algorithm
within each mini-batch").
"""

from __future__ import annotations

import numpy as np
import ot  # Python Optimal Transport
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class VelocityNet(nn.Module):
    """
    Time-conditioned velocity field v_theta : [0, 1] x R^2 -> R^2.

    Smooth-activation MLP (Remark 2): Picard-Lindelof gives a diffeomorphism
    on R^2, so G_theta is one-to-one and the only approximation error comes
    from the discretised ODE solver and the finite-capacity network.
    """

    def __init__(self, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (B,) or (B, 1) flow time in [0, 1].
            y: (B, 2) spatial position.
        Returns:
            v: (B, 2) velocity.
        """
        if s.dim() == 1:
            s = s.unsqueeze(1)
        return self.net(torch.cat([s, y], dim=1))


class TimeEmbedding(nn.Module):
    """Sinusoidal time encoding (Fourier features) → linear projection.
    Standard recipe for flow-matching / diffusion velocity nets.
    """
    def __init__(self, dim: int, n_freqs: int = 16):
        super().__init__()
        freqs = 2 * np.pi * torch.exp(
            torch.linspace(0.0, float(np.log(1000.0)), n_freqs))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 * n_freqs, dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B, 1) in [0, 1]
        x = s * self.freqs[None, :]                # (B, n_freqs)
        return self.proj(torch.cat([x.sin(), x.cos()], dim=-1))


class ResBlock(nn.Module):
    """Pre-norm residual MLP block."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.fc2(nn.functional.silu(self.fc1(self.norm(h))))


class VelocityNetV2(nn.Module):
    """Higher-capacity velocity field with Fourier time embedding,
    pre-norm residual blocks, and LayerNorm-then-Linear output head.

    Train interface is identical to VelocityNet (same .forward(s, y)) so
    cfm_loss / transport / save_state_dict all just work.
    """

    def __init__(self, hidden_dim: int = 512, n_layers: int = 8,
                 n_time_freqs: int = 16):
        super().__init__()
        self.t_embed = TimeEmbedding(hidden_dim, n_time_freqs)
        self.in_proj = nn.Linear(2, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(n_layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2))

    def forward(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if s.dim() == 1:
            s = s.unsqueeze(1)
        h = self.in_proj(y) + self.t_embed(s)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


def build_model(arch: str = "v1", hidden_dim: int = 512, n_layers: int = 6,
                n_time_freqs: int = 16) -> nn.Module:
    """Select between VelocityNet (v1) and VelocityNetV2 (v2)."""
    if arch == "v1":
        return VelocityNet(hidden_dim=hidden_dim, n_layers=n_layers)
    if arch == "v2":
        return VelocityNetV2(hidden_dim=hidden_dim, n_layers=n_layers,
                             n_time_freqs=n_time_freqs)
    raise ValueError(f"unknown arch {arch!r}")


def ot_coupling(z0: np.ndarray, x1: np.ndarray, reg: float = 0.05) -> np.ndarray:
    """
    Per-mini-batch OT coupling (paper §2.2). Returns a permutation sigma
    such that (z0_i, x1_{sigma(i)}) is the Sinkhorn-OT pairing.

    Pipeline:
      1. Scale-normalise the squared-Euclidean cost (numerical stability).
      2. Solve the entropic OT plan with Sinkhorn (smooth, stable on small B).
      3. Extract a TRUE permutation by running the Hungarian algorithm on
         the dense plan (maximising transported mass). This avoids the
         many-to-one collapse that an `argmax` of the plan would produce
         and keeps each target point used exactly once.

    Args:
        z0: (N, 2) source points.
        x1: (N, 2) target points.
        reg: Sinkhorn entropic regularisation.

    Returns:
        sigma: (N,) permutation array so that x1[sigma] is paired with z0.
    """
    n = len(z0)
    if n != len(x1):
        raise ValueError("Permutation requires len(z0) == len(x1).")

    cost = ot.dist(z0, x1, metric="sqeuclidean")
    cost = cost / max(cost.max(), 1e-12)

    a = np.ones(n) / n
    b = np.ones(n) / n
    plan = ot.sinkhorn(a, b, cost, reg=reg)

    # Hungarian on -plan <=> maximise transported mass -> true permutation.
    row_ind, col_ind = linear_sum_assignment(-plan)
    sigma = np.empty(n, dtype=int)
    sigma[row_ind] = col_ind
    return sigma


def cfm_loss(
    model: VelocityNet,
    z0: np.ndarray | torch.Tensor,
    x1: np.ndarray | torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute L_CFM on a mini-batch (paper eq. (3)).

        y_s      = (1 - s) z_0 + s x_1,       s ~ U[0, 1]
        u_target = x_1 - z_0                   (straight-line conditional field)
        loss     = mean || v_theta(s, y_s) - u_target ||^2

    Straight-line conditional velocities are what makes this an OT-CFM:
    with OT-paired (z_0, x_1), each conditional trajectory is a geodesic of
    the Benamou-Brenier / Wasserstein-2 geometry (§2.2, dynamic-OT view).
    """
    z0_np = z0.detach().cpu().numpy() if isinstance(z0, torch.Tensor) else np.asarray(z0)
    x1_np = x1.detach().cpu().numpy() if isinstance(x1, torch.Tensor) else np.asarray(x1)

    sigma = ot_coupling(z0_np, x1_np)
    x1_paired = x1_np[sigma]

    z0_t = torch.as_tensor(z0_np, dtype=torch.float32, device=device)
    x1_t = torch.as_tensor(x1_paired, dtype=torch.float32, device=device)

    B = z0_t.shape[0]
    s = torch.rand(B, device=device)
    s_col = s.unsqueeze(1)

    y_s = (1.0 - s_col) * z0_t + s_col * x1_t
    u_target = x1_t - z0_t
    v_pred = model(s, y_s)

    # Mean over (batch, dims). Equivalent up to a factor of 2 to mean of
    # squared 2-norms; keeps a small numeric loss value.
    return ((v_pred - u_target) ** 2).mean()


def nfz_penalty(pts: torch.Tensor, center, radius: float,
                margin: float = 0.05) -> torch.Tensor:
    """Soft no-fly-zone penalty: mean( max(0, r+margin - ||p-c||)^2 ).

    Pushes transported points outside a disc B(center, radius+margin).
    The +margin keeps a buffer so points don't sit exactly on the boundary.
    """
    if not isinstance(center, torch.Tensor):
        center = torch.as_tensor(center, dtype=pts.dtype, device=pts.device)
    dist = torch.linalg.norm(pts - center, dim=1)
    return torch.clamp(radius + margin - dist, min=0).pow(2).mean()


def power_zeng(v: torch.Tensor, P_0: float = 79.86, P_i: float = 88.63,
               U_tip: float = 120.0, v_0: float = 4.03, d_0: float = 0.6,
               rho_air: float = 1.225, s: float = 0.05,
               A: float = 0.503) -> torch.Tensor:
    """Differentiable Zeng-2019 rotary-wing power curve P(V) [Watts].

    Mirror of `RotaryWingEnergyModel.power` in milano_baselines.py, but
    written in torch so autograd can backprop through it.

    Args:
        v: (N,) tensor of speeds in m/s.
    Returns:
        (N,) tensor of instantaneous power in Watts.
    """
    v = torch.clamp(v, min=0.01)
    P_blade = P_0 * (1.0 + 3.0 * v.pow(2) / (U_tip ** 2))
    ratio = v.pow(2) / (2.0 * v_0 ** 2)
    induced = torch.sqrt(
        torch.clamp(1.0 + v.pow(4) / (4.0 * v_0 ** 4) - ratio, min=0.01))
    P_induced = P_i * induced
    P_par = 0.5 * d_0 * rho_air * s * A * v.pow(3)
    return P_blade + P_induced + P_par


def energy_penalty(traj_phys: torch.Tensor, dt_s: float,
                   domain_size: float = 8.0) -> torch.Tensor:
    """Mean Zeng power along a transported chunk (Watts).

    Treats the chunk as time-ordered. Step displacement in normalised
    coordinates is converted to metres via `domain_size`, then divided by
    `dt_s` to get speed in m/s.

    Args:
        traj_phys: (T, 2) transported chunk in normalised [-1, 1].
        dt_s: scalar — physical seconds between consecutive samples.
        domain_size: physical extent of the unit interval (default 8 m).
    Returns:
        Scalar mean power (mean over T-1 segments).
    """
    step_m = torch.linalg.norm(torch.diff(traj_phys, dim=0), dim=1) * domain_size
    v_mps = step_m / dt_s
    return power_zeng(v_mps).mean()


def transport_train(model: nn.Module, z0: torch.Tensor,
                    n_steps: int = 8) -> torch.Tensor:
    """Differentiable RK4 transport used during training for endpoint
    constraint penalties. Same integrator as `transport()` but without
    the @torch.no_grad() decorator so gradients flow through the ODE.
    """
    y = z0
    dt = 1.0 / n_steps
    N = y.shape[0]
    for i in range(n_steps):
        s_i = i * dt
        s1 = torch.full((N,), s_i, device=y.device)
        s2 = torch.full((N,), s_i + 0.5 * dt, device=y.device)
        s3 = torch.full((N,), s_i + dt, device=y.device)
        k1 = model(s1, y)
        k2 = model(s2, y + 0.5 * dt * k1)
        k3 = model(s2, y + 0.5 * dt * k2)
        k4 = model(s3, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y


@torch.no_grad()
def transport(
    model: VelocityNet,
    z0: np.ndarray | torch.Tensor,
    n_steps: int = 50,
    device: str = "cpu",
) -> np.ndarray:
    """
    Integrate dy/ds = v_theta(s, y) from s = 0 to s = 1 using classical
    RK4 on a uniform grid. Returns G_theta(z0) = y(1).

    Args:
        model: trained VelocityNet.
        z0: (N, 2) source points on D_delta.
        n_steps: number of RK4 steps (20-50 is typical).

    Returns:
        (N, 2) numpy array of transported points.
    """
    model.eval()
    if isinstance(z0, np.ndarray):
        y = torch.as_tensor(z0, dtype=torch.float32, device=device)
    else:
        y = z0.clone().to(device)

    dt = 1.0 / n_steps
    N = y.shape[0]
    for i in range(n_steps):
        s_i = i * dt
        s1 = torch.full((N,), s_i, device=device)
        s2 = torch.full((N,), s_i + 0.5 * dt, device=device)
        s3 = torch.full((N,), s_i + dt, device=device)

        k1 = model(s1, y)
        k2 = model(s2, y + 0.5 * dt * k1)
        k3 = model(s2, y + 0.5 * dt * k2)
        k4 = model(s3, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return y.cpu().numpy()
