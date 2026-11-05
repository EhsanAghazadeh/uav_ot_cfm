"""
OT-CFM: Conditional Flow Matching with Optimal Transport coupling.

Trains a velocity field v_theta(s, y) that transports pi_0 (uniform on disc)
to f_target. The map G_theta is obtained by integrating the ODE.
"""

import torch
import torch.nn as nn
import numpy as np
import ot  # Python Optimal Transport library


class VelocityNet(nn.Module):
    """
    Small MLP that predicts the velocity field v(s, y) -> R^2.
    Input: (s, y1, y2) -> 3D
    Output: (v1, v2) -> 2D
    """

    def __init__(self, hidden_dim=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, s, y):
        """
        Args:
            s: (B,) or (B, 1) flow time in [0, 1]
            y: (B, 2) spatial position
        Returns:
            v: (B, 2) velocity
        """
        if s.dim() == 1:
            s = s.unsqueeze(1)
        inp = torch.cat([s, y], dim=1)  # (B, 3)
        return self.net(inp)


def ot_coupling(z0, x1, reg=0.05):
    """
    Compute OT coupling between source z0 and target x1 using Sinkhorn
    (entropic regularization) for smoother, more stable mini-batch coupling.

    Args:
        z0: (N, 2) source points (numpy)
        x1: (N, 2) target points (numpy)
        reg: entropic regularization strength

    Returns:
        sigma: (N,) permutation array
    """
    cost = ot.dist(z0, x1, metric='sqeuclidean')
    a = np.ones(len(z0)) / len(z0)
    b = np.ones(len(x1)) / len(x1)
    # Sinkhorn for smoother coupling (less noisy than exact EMD on small batches)
    plan = ot.sinkhorn(a, b, cost, reg=reg)
    # Extract permutation from transport plan
    sigma = plan.argmax(axis=1)
    return sigma


def cfm_loss(model, z0, x1, device='cpu'):
    """
    Compute the Conditional Flow Matching loss with OT coupling.

    L(theta) = E_{s, i} || v_theta(s, y_s^i) - (x1^sigma(i) - z0^i) ||^2

    Args:
        model: VelocityNet
        z0: (B, 2) source samples (numpy or tensor)
        x1: (B, 2) target samples (numpy or tensor)
        device: torch device

    Returns:
        loss: scalar tensor
    """
    B = z0.shape[0]

    # Convert to numpy for OT coupling
    z0_np = z0.numpy() if isinstance(z0, torch.Tensor) else z0
    x1_np = x1.numpy() if isinstance(x1, torch.Tensor) else x1

    # OT coupling
    sigma = ot_coupling(z0_np, x1_np)
    x1_paired = x1_np[sigma]

    # Convert to tensors
    z0_t = torch.tensor(z0_np, dtype=torch.float32, device=device)
    x1_t = torch.tensor(x1_paired, dtype=torch.float32, device=device)

    # Sample flow time s ~ U[0, 1]
    s = torch.rand(B, device=device)

    # Interpolated points: y_s = (1-s)*z0 + s*x1
    s_expand = s.unsqueeze(1)  # (B, 1)
    y_s = (1 - s_expand) * z0_t + s_expand * x1_t

    # Conditional velocity field (straight line): u = x1 - z0
    u_target = x1_t - z0_t  # (B, 2)

    # Predicted velocity
    v_pred = model(s, y_s)

    # MSE loss
    loss = ((v_pred - u_target) ** 2).mean()
    return loss


@torch.no_grad()
def transport(model, z0, n_steps=50, device='cpu'):
    """
    Transport source points z0 through the learned velocity field to get G(z0).

    Integrates dy/ds = v_theta(s, y) from s=0 to s=1 using RK4.

    Args:
        model: trained VelocityNet
        z0: (N, 2) source points (numpy or tensor)
        n_steps: number of RK4 steps
        device: torch device

    Returns:
        x1: (N, 2) transported points (numpy)
    """
    model.eval()
    if isinstance(z0, np.ndarray):
        y = torch.tensor(z0, dtype=torch.float32, device=device)
    else:
        y = z0.clone().to(device)

    dt = 1.0 / n_steps
    for i in range(n_steps):
        s_i = i * dt
        s1 = torch.full((y.shape[0],), s_i, device=device)
        s2 = torch.full((y.shape[0],), s_i + 0.5 * dt, device=device)
        s3 = torch.full((y.shape[0],), s_i + dt, device=device)

        k1 = model(s1, y)
        k2 = model(s2, y + 0.5 * dt * k1)
        k3 = model(s2, y + 0.5 * dt * k2)
        k4 = model(s3, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return y.cpu().numpy()
