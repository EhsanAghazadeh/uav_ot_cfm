"""Milano grid → continuous density + sampler for Flow Matching Ergodic Coverage.

Uses ``target_density.load_milano_target`` so NFZ masking matches TOES / OT-CFM.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from target_density import load_milano_target


def milano_grid_and_sampler(
    grid_path: str,
    *,
    smooth_sigma: float = 1.5,
    eps_floor: float = 1e-12,
    xlim: tuple[float, float] = (-1.0, 1.0),
    ylim: tuple[float, float] = (-1.0, 1.0),
    nfz_discs=None,
    nfz_center: tuple[float, float] | None = None,
    nfz_radius: float | None = None,
    seed: int = 0,
):
    """Load Milano target and build (grid, xs, ys, density_np, sampler)."""
    t = load_milano_target(
        grid_path,
        smooth_sigma=smooth_sigma,
        eps_floor=eps_floor,
        xlim=xlim,
        ylim=ylim,
        nfz_discs=nfz_discs,
        nfz_center=nfz_center,
        nfz_radius=nfz_radius,
    )
    grid = t["grid"]
    xs = t["xs"]
    ys = t["ys"]
    h, w = grid.shape

    mass = grid / np.sum(grid)
    p = mass.reshape(-1)
    cdf = np.cumsum(p)
    cdf[-1] = 1.0

    dx = (xlim[1] - xlim[0]) / max(w - 1, 1)
    dy = (ylim[1] - ylim[0]) / max(h - 1, 1)

    interp = RegularGridInterpolator(
        (ys, xs),
        mass,
        bounds_error=False,
        fill_value=0.0,
        method="linear",
    )
    cell_area = abs(dx * dy)

    def density(x, y):
        pts = np.column_stack([np.asarray(y).ravel(), np.asarray(x).ravel()])
        vals = interp(pts).reshape(np.broadcast(x, y).shape)
        return vals / max(cell_area, 1e-16)

    rng = np.random.default_rng(seed)

    def sampler(n_samples: int, seed=None):
        rrng = rng if seed is None else np.random.default_rng(seed)
        u = rrng.uniform(0.0, 1.0, size=n_samples)
        idx = np.searchsorted(cdf, u, side="right")
        idx = np.clip(idx, 0, h * w - 1)
        ii = idx // w
        jj = idx % w
        x0 = xs[jj]
        y0 = ys[ii]
        x = x0 + rrng.uniform(-0.5, 0.5, size=n_samples) * dx
        y = y0 + rrng.uniform(-0.5, 0.5, size=n_samples) * dy
        x = np.clip(x, xlim[0], xlim[1])
        y = np.clip(y, ylim[0], ylim[1])
        return np.column_stack([x, y]).astype(np.float32)

    return grid, xs, ys, density, sampler
