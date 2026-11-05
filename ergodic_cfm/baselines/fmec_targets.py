"""2-D target distributions for Flow Matching Ergodic Coverage."""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import multivariate_normal as mvn

from ergodic_cfm.baselines.milano_target_fmec import milano_grid_and_sampler


@dataclass
class Target:
    """A 2-D target distribution usable by both Stein and Sinkhorn modes."""

    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    center: Tuple[float, float]
    pdf: Callable[[jnp.ndarray], jnp.ndarray]
    log_pdf: Callable[[jnp.ndarray], jnp.ndarray]
    sample: Callable[[int, Optional[int]], np.ndarray]
    pdf_np: Callable[[np.ndarray, np.ndarray], np.ndarray]
    raw_points: Optional[np.ndarray] = None


def gmm_target() -> Target:
    mean1 = jnp.array([0.3, 0.5])
    cov1 = jnp.array([[0.002, 0.0], [0.0, 0.04]])
    mean2 = jnp.array([0.5, 0.5])
    cov2 = jnp.array([[0.02, -0.018], [-0.018, 0.02]])
    mean3 = jnp.array([0.7, 0.5])
    cov3 = jnp.array([[0.002, 0.0], [0.0, 0.04]])
    w1, w2, w3 = 0.34, 0.34, 0.33

    means = [np.array(m) for m in (mean1, mean2, mean3)]
    covs = [np.array(c) for c in (cov1, cov2, cov3)]
    weights = np.array([w1, w2, w3])
    weights_norm = weights / weights.sum()

    def pdf(x: jnp.ndarray) -> jnp.ndarray:
        return (
            w1 * mvn.pdf(x[:2], mean1, cov1)
            + w2 * mvn.pdf(x[:2], mean2, cov2)
            + w3 * mvn.pdf(x[:2], mean3, cov3)
        )

    def log_pdf(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(pdf(x))

    def sample(n: int, seed: Optional[int] = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        k = rng.choice(3, size=n, p=weights_norm)
        out = np.empty((n, 2), dtype=np.float32)
        for j in range(3):
            m = k == j
            out[m] = rng.multivariate_normal(means[j], covs[j], size=int(m.sum()))
        return out

    def pdf_np(X, Y):
        pts = np.stack([X, Y], axis=-1)
        v = np.zeros_like(X)
        from scipy.stats import multivariate_normal as scipy_mvn
        for wj, mj, cj in zip(weights, means, covs):
            v = v + wj * scipy_mvn.pdf(pts, mean=mj, cov=cj)
        return v

    return Target(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        center=(0.5, 0.5),
        pdf=pdf,
        log_pdf=log_pdf,
        sample=sample,
        pdf_np=pdf_np,
    )


def exp3_gmm_target() -> Target:
    mean1 = jnp.array([0.5, 0.4])
    cov1 = (0.25 ** 2) * jnp.eye(2)
    mean2 = jnp.array([-0.5, -0.3])
    cov2 = (0.20 ** 2) * jnp.eye(2)
    w1, w2 = 0.6, 0.4

    means_np = [np.array([0.5, 0.4]), np.array([-0.5, -0.3])]
    covs_np = [(0.25 ** 2) * np.eye(2), (0.20 ** 2) * np.eye(2)]
    weights = np.array([w1, w2])

    def pdf(x: jnp.ndarray) -> jnp.ndarray:
        return w1 * mvn.pdf(x[:2], mean1, cov1) + w2 * mvn.pdf(
            x[:2], mean2, cov2
        )

    def log_pdf(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(pdf(x) + 1e-30)

    def sample(n: int, seed: Optional[int] = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        k = rng.choice(2, size=n, p=weights)
        out = np.empty((n, 2), dtype=np.float32)
        for j in range(2):
            mask = k == j
            out[mask] = rng.multivariate_normal(
                means_np[j], covs_np[j], size=int(mask.sum())
            )
        return out

    def pdf_np(X, Y):
        pts = np.stack([X, Y], axis=-1)
        from scipy.stats import multivariate_normal as scipy_mvn
        v = np.zeros_like(X)
        for wj, mj, cj in zip(weights, means_np, covs_np):
            v = v + wj * scipy_mvn.pdf(pts, mean=mj, cov=cj)
        return v

    return Target(
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        center=(0.0, 0.0),
        pdf=pdf,
        log_pdf=log_pdf,
        sample=sample,
        pdf_np=pdf_np,
    )


_POINTCLOUD_URL = (
    "https://raw.githubusercontent.com/MurpheyLab/lqr-flow-matching/"
    "refs/heads/main/tutorials/test_objects/2d/{name}.txt"
)


def pointcloud_target(name_or_path: str, kde_sigma: float = 0.02) -> Target:
    if os.path.isfile(name_or_path):
        pts = np.loadtxt(name_or_path)[:, :2]
    else:
        resp = urllib.request.urlopen(_POINTCLOUD_URL.format(name=name_or_path))
        pts = np.loadtxt(resp)[:, :2]
    pts = np.asarray(pts, dtype=np.float32)
    samples_jax = jnp.array(pts)
    center = (0.5, 0.5)

    inv_two_s2 = 1.0 / (2.0 * kde_sigma * kde_sigma)
    norm = 1.0 / (2.0 * np.pi * kde_sigma * kde_sigma)

    def pdf(x: jnp.ndarray) -> jnp.ndarray:
        d2 = jnp.sum((samples_jax - x[:2]) ** 2, axis=-1)
        return norm * jnp.mean(jnp.exp(-inv_two_s2 * d2))

    def log_pdf(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(pdf(x) + 1e-30)

    def sample(n: int, seed: Optional[int] = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if n >= len(pts):
            idx = rng.integers(0, len(pts), size=n)
        else:
            idx = rng.choice(len(pts), size=n, replace=False)
        return pts[idx]

    def pdf_np(X, Y):
        pts2 = np.stack([X, Y], axis=-1)
        d2 = ((pts[None, None, :, :] - pts2[..., None, :]) ** 2).sum(-1)
        return norm * np.mean(np.exp(-inv_two_s2 * d2), axis=-1)

    xmin, ymin = pts.min(axis=0) - 0.05
    xmax, ymax = pts.max(axis=0) + 0.05
    return Target(
        xlim=(float(xmin), float(xmax)),
        ylim=(float(ymin), float(ymax)),
        center=center,
        pdf=pdf,
        log_pdf=log_pdf,
        sample=sample,
        pdf_np=pdf_np,
        raw_points=pts,
    )


def milano_target(
    grid_path: str,
    xlim: Tuple[float, float] = (-1.0, 1.0),
    ylim: Tuple[float, float] = (-1.0, 1.0),
    smooth_sigma: float = 1.5,
    eps_floor: float = 1e-12,
    nfz_discs=None,
    nfz_center: Optional[Tuple[float, float]] = None,
    nfz_radius: Optional[float] = None,
    seed: int = 0,
) -> Target:
    grid, xs, ys, density_np, sampler = milano_grid_and_sampler(
        grid_path,
        smooth_sigma=smooth_sigma,
        eps_floor=eps_floor,
        xlim=xlim,
        ylim=ylim,
        nfz_discs=nfz_discs,
        nfz_center=nfz_center,
        nfz_radius=nfz_radius,
        seed=seed,
    )
    h, w = grid.shape
    mass_np = grid / grid.sum()
    mass_jax = jnp.asarray(mass_np, dtype=jnp.float32)
    x0_grid, x1_grid = float(xs[0]), float(xs[-1])
    y0_grid, y1_grid = float(ys[0]), float(ys[-1])
    dx = (xlim[1] - xlim[0]) / max(w - 1, 1)
    dy = (ylim[1] - ylim[0]) / max(h - 1, 1)
    cell_area = abs(dx * dy)

    cy, cx = np.unravel_index(np.argmax(grid), grid.shape)
    center = (float(xs[cx]), float(ys[cy]))

    def _bilinear(x: jnp.ndarray) -> jnp.ndarray:
        fx = (x[0] - x0_grid) / (x1_grid - x0_grid) * (w - 1)
        fy = (x[1] - y0_grid) / (y1_grid - y0_grid) * (h - 1)
        fx = jnp.clip(fx, 0.0, w - 1.0 - 1e-6)
        fy = jnp.clip(fy, 0.0, h - 1.0 - 1e-6)
        j = jnp.floor(fx).astype(jnp.int32)
        i = jnp.floor(fy).astype(jnp.int32)
        a = fx - j
        b = fy - i
        m00 = mass_jax[i, j]
        m01 = mass_jax[i, j + 1]
        m10 = mass_jax[i + 1, j]
        m11 = mass_jax[i + 1, j + 1]
        return (
            m00 * (1 - a) * (1 - b)
            + m01 * a * (1 - b)
            + m10 * (1 - a) * b
            + m11 * a * b
        )

    def pdf(x: jnp.ndarray) -> jnp.ndarray:
        return _bilinear(x) / cell_area

    def log_pdf(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(pdf(x) + 1e-20)

    def sample(n: int, seed: Optional[int] = None) -> np.ndarray:
        return sampler(n, seed=seed)

    def pdf_np(X, Y):
        return density_np(X, Y)

    return Target(
        xlim=xlim,
        ylim=ylim,
        center=center,
        pdf=pdf,
        log_pdf=log_pdf,
        sample=sample,
        pdf_np=pdf_np,
    )
