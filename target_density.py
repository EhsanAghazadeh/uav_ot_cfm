"""Single source of truth for loading the Milano target density.

Optionally zeros out a circular no-fly-zone (NFZ) region and renormalises,
so that training, evaluation, and plotting all see the *same* target
distribution. Mass that originally fell inside the NFZ disc is removed
(not redistributed): the reachable density still integrates to 1.

Usage::

    from target_density import load_milano_target

    # Vanilla (unconstrained) target
    t = load_milano_target("data/milano_grid_corrected.npy", smooth_sigma=1.5)

    # NFZ-masked target (NFZ runs only)
    t = load_milano_target("data/milano_grid_corrected.npy", smooth_sigma=1.5,
                           nfz_center=(0.19, 0.03), nfz_radius=0.10)

    pmf = t["pmf"]                # (h, w) probability mass function on cells
    grid = t["grid"]              # (h, w) masked, floored, smoothed density
    nfz_mask = t["nfz_mask"]      # (h, w) bool, True inside disc — or None

The loader is used by ``run_otcfm_milano.make_grid_sampler``,
``milano_baselines.MilanoRateMap``, ``eval_fair``, ``eval_prefix_stop``
and ``run_baseline_milano`` so that the *same* density object underlies
training, evaluation, and figures.
"""
from __future__ import annotations

import numpy as np


def load_milano_target(grid_path, smooth_sigma=1.5, eps_floor=1e-12,
                       xlim=(-1.0, 1.0), ylim=(-1.0, 1.0),
                       nfz_discs=None,
                       nfz_center=None, nfz_radius=None):
    """Load the smoothed Milano density, optionally with NFZ mass zeroed.

    Parameters
    ----------
    grid_path : str
        Path to the .npy grid (h, w) of activity counts.
    smooth_sigma : float
        Gaussian smoothing in cells (0 disables).
    eps_floor : float
        Lower clip on the density before normalisation, to avoid
        log(0) / division-by-zero downstream.
    xlim, ylim : 2-tuple of float
        Physical extent of the grid in normalised coordinates.
    nfz_center : 2-tuple of float or None
        Disc centre (cx, cy) in `xlim`/`ylim` coordinates. When None
        (and `nfz_radius` is None), no mask is applied — the loader
        is a drop-in replacement for the previous inline loaders.
    nfz_radius : float or None
        Disc radius in normalised coordinates.

    Returns
    -------
    dict with keys:
        ``grid``      (h, w) np.float64 — masked, floored, smoothed density.
        ``pmf``       (h, w) np.float64 — same divided by its sum.
        ``xs``        (w,) cell-centre x coords.
        ``ys``        (h,) cell-centre y coords.
        ``nfz_mask``  (h, w) bool — True inside the disc — or None.
        ``xlim``, ``ylim`` — passed through.
    """
    g = np.load(grid_path).astype(np.float64)
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        g = gaussian_filter(g, sigma=float(smooth_sigma))
    g = np.maximum(g, eps_floor)

    h, w = g.shape
    xs = np.linspace(xlim[0], xlim[1], w)
    ys = np.linspace(ylim[0], ylim[1], h)

    # Build unified disc list (nfz_discs takes priority; old single-disc args kept for compat)
    discs = list(nfz_discs or [])
    if nfz_center is not None and nfz_radius is not None:
        discs = discs + [(nfz_center, nfz_radius)]

    nfz_mask = None
    if discs:
        XX, YY = np.meshgrid(xs, ys)
        combined = np.zeros((h, w), dtype=bool)
        for center, radius in discs:
            cx, cy = float(center[0]), float(center[1])
            r = float(radius)
            combined |= (XX - cx) ** 2 + (YY - cy) ** 2 < r * r
        nfz_mask = combined
        g = g.copy()
        g[nfz_mask] = 0.0
        if g.sum() <= 0:
            raise ValueError(
                f"NFZ discs {discs} zeroed all density. "
                "Check disc placements against the grid extent."
            )

    pmf = g / g.sum()
    return {
        "grid": g, "pmf": pmf,
        "xs": xs, "ys": ys,
        "nfz_mask": nfz_mask,
        "xlim": xlim, "ylim": ylim,
    }
