"""LQR point-mass dynamics + Stein / Sinkhorn gradients (RSS 2025 FMEC)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import grad, vmap

from lqrax import LQR
from ott.geometry import pointcloud
from ott.geometry.costs import PNormP
from ott.tools.sinkhorn_divergence import sinkhorn_divergence


class PointMassLQR(LQR):
    def __init__(self, dt, x_dim, u_dim, Q, R):
        super().__init__(dt, x_dim, u_dim, Q, R)

    def dyn(self, xt, ut):
        return jnp.array([xt[2], xt[3], ut[0], ut[1]])


def make_stein_grad(log_pdf, h: float):
    score_pdf = grad(log_pdf)

    def kernel(x1, x2):
        return jnp.exp(-jnp.sum(jnp.square(x1[:2] - x2[:2])) / h)

    d_kernel = grad(kernel, argnums=0)

    def stein_grad_unit(x1, x2):
        return kernel(x2, x1) * score_pdf(x2) + d_kernel(x2, x1)

    def stein_grad_state(x, emp):
        vals = vmap(stein_grad_unit, in_axes=(None, 0))(x, emp)
        return jnp.mean(vals, axis=0)

    def stein_grad_pair(query, emp):
        return vmap(stein_grad_state, in_axes=(0, None))(query, emp)

    def stein_grad(traj):
        return stein_grad_pair(traj, traj)

    return stein_grad, stein_grad_pair


def make_sinkhorn_grad(tgt_samples: jnp.ndarray, eps: float = 0.01,
                       loss_scale: float = 1e3):
    cost_fn = PNormP(p=1)

    def sinkhorn_div(x_samples):
        c = sinkhorn_divergence(
            pointcloud.PointCloud,
            x_samples[:, :2],
            tgt_samples[:, :2],
            cost_fn=cost_fn,
            epsilon=eps,
        )[0]
        return c * loss_scale

    grad_fn = grad(sinkhorn_div)

    def sinkhorn_grad(traj):
        return -grad_fn(traj)

    return sinkhorn_grad, sinkhorn_div
