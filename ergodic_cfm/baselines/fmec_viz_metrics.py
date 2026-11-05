"""Plotting and evaluation helpers for FMEC."""
from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from ergodic_cfm.baselines.fmec_targets import Target

try:
    import ot as pot_ot
    _POT_AVAILABLE = True
except ImportError:
    _POT_AVAILABLE = False


def _as_batched(traj: np.ndarray) -> np.ndarray:
    traj = np.asarray(traj)
    if traj.ndim == 2:
        return traj[None, ...]
    return traj


def plot_trajectory(
    target: Target,
    traj: np.ndarray,
    x0: np.ndarray,
    out_path: str,
    title: str = "",
):
    nx, ny = 160, 160
    X, Y = np.meshgrid(
        np.linspace(*target.xlim, nx), np.linspace(*target.ylim, ny)
    )
    P = np.asarray(target.pdf_np(X, Y))

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    levels = np.linspace(P.min(), P.max(), 11)
    if levels[0] == levels[-1]:
        levels = None
    ax.contourf(X, Y, P, levels=levels, cmap="Purples")

    traj_b = _as_batched(traj)
    x0_b = np.asarray(x0)
    if x0_b.ndim == 1:
        x0_b = x0_b[None, :]
    n_agents = traj_b.shape[0]
    cmap = plt.get_cmap("tab10" if n_agents <= 10 else "hsv")
    lw = 2.0 if n_agents == 1 else max(0.5, 1.2 - 0.05 * n_agents)
    ms = 3.0 if n_agents == 1 else max(1.0, 2.0 - 0.05 * n_agents)
    for i in range(n_agents):
        c = cmap(i % cmap.N) if n_agents > 1 else "C1"
        ax.plot(traj_b[i, :, 0], traj_b[i, :, 1], "-", color=c, lw=lw, alpha=0.9)
        ax.plot(traj_b[i, :, 0], traj_b[i, :, 1], "o", color=c, ms=ms, alpha=0.5)
    ax.plot(x0_b[:, 0], x0_b[:, 1], "o", color="black", ms=8)
    ax.set_xlim(target.xlim)
    ax.set_ylim(target.ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_metric(values, out_path, ylabel="Sinkhorn divergence"):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ax.plot(values, "-o", ms=3, lw=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_animation(
    target: Target,
    x_traj_list: np.ndarray,
    x0: np.ndarray,
    out_path: str,
    stride: int = 10,
    fps: int = 15,
):
    nx, ny = 160, 160
    X, Y = np.meshgrid(
        np.linspace(*target.xlim, nx), np.linspace(*target.ylim, ny)
    )
    P = np.asarray(target.pdf_np(X, Y))

    x_traj_list = np.asarray(x_traj_list)
    if x_traj_list.ndim == 3:
        x_traj_list = x_traj_list[:, None, :, :]
    n_iters, n_agents, T, _ = x_traj_list.shape

    x0_b = np.asarray(x0)
    if x0_b.ndim == 1:
        x0_b = x0_b[None, :]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    levels = np.linspace(P.min(), P.max(), 11)
    if levels[0] == levels[-1]:
        levels = None
    ax.contourf(X, Y, P, levels=levels, cmap="Purples")
    ax.set_xlim(target.xlim)
    ax.set_ylim(target.ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    cmap = plt.get_cmap("tab10" if n_agents <= 10 else "hsv")
    lw = 2.0 if n_agents == 1 else max(0.5, 1.2 - 0.05 * n_agents)
    ms = 3.0 if n_agents == 1 else max(1.0, 2.0 - 0.05 * n_agents)
    lines, dots = [], []
    for i in range(n_agents):
        c = cmap(i % cmap.N) if n_agents > 1 else "C1"
        ln, = ax.plot([], [], "-", color=c, lw=lw, alpha=0.9)
        dt, = ax.plot([], [], "o", color=c, ms=ms, alpha=0.5)
        lines.append(ln)
        dots.append(dt)
    ax.plot(x0_b[:, 0], x0_b[:, 1], "o", color="black", ms=8)
    title = ax.text(0.5, 1.01, "", transform=ax.transAxes, ha="center")

    frames = list(range(0, n_iters, stride))
    if frames[-1] != n_iters - 1:
        frames.append(n_iters - 1)

    def update(k):
        traj = x_traj_list[k]
        for i in range(n_agents):
            lines[i].set_data(traj[i, :, 0], traj[i, :, 1])
            dots[i].set_data(traj[i, :, 0], traj[i, :, 1])
        title.set_text(f"iter {k}")
        return (*lines, *dots, title)

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    try:
        anim.save(out_path, fps=fps, writer="pillow")
    except Exception as e:
        print(f"[warn] could not save gif: {e}")
    plt.close(fig)


def _map_to_unit_box(
    pos: np.ndarray, domain: Tuple[float, float, float, float]
) -> np.ndarray:
    xmin, xmax, ymin, ymax = domain
    u0 = (pos[:, 0] - xmin) / max(xmax - xmin, 1e-12)
    u1 = (pos[:, 1] - ymin) / max(ymax - ymin, 1e-12)
    return np.column_stack([u0, u1])


def fourier_ergodic_metric(
    traj_pos,
    target_grid,
    domain,
    *,
    k_max: int = 10,
    weight_power: float = 1.5,
) -> float:
    u = _map_to_unit_box(np.asarray(traj_pos, dtype=np.float64), domain)
    u = np.clip(u, 0.0, 1.0)

    m0, m1 = target_grid.shape
    if m0 != m1:
        raise ValueError("target_grid must be square (m x m)")
    m = m0
    uu = (np.arange(m) + 0.5) / m
    Ux, Uy = np.meshgrid(uu, uu, indexing="ij")

    xmin, xmax, ymin, ymax = domain
    dA = ((xmax - xmin) / m) * ((ymax - ymin) / m)

    ks = np.arange(0, k_max + 1)
    cos_traj_x = np.cos(np.pi * u[:, 0:1] * ks[None, :])
    cos_traj_y = np.cos(np.pi * u[:, 1:2] * ks[None, :])
    cos_grid_x = np.cos(np.pi * Ux[..., None] * ks[None, None, :])
    cos_grid_y = np.cos(np.pi * Uy[..., None] * ks[None, None, :])

    metric = 0.0
    for kx in ks:
        for ky in ks:
            phi_traj = cos_traj_x[:, kx] * cos_traj_y[:, ky]
            c_k = float(np.mean(phi_traj))
            phi_grid = cos_grid_x[:, :, kx] * cos_grid_y[:, :, ky]
            mu_k = float(np.sum(target_grid * phi_grid) * dA)
            w = (1.0 + float(kx * kx + ky * ky)) ** (-weight_power)
            diff = c_k - mu_k
            metric += w * diff * diff
    return float(metric)


def grid_coverage_error(hist_emp, tgt, cell_area, *, kind: str = "l1") -> float:
    diff = hist_emp - tgt
    if kind == "l1":
        return float(np.sum(np.abs(diff)) * cell_area)
    if kind == "l2":
        return float(np.sqrt(np.sum(diff**2) * cell_area))
    raise ValueError(kind)


def wasserstein_w2_pot(
    x, y, *, method: str = "sinkhorn", reg: float = 0.05
) -> float:
    if not _POT_AVAILABLE:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = np.ones(len(x)) / len(x)
    b = np.ones(len(y)) / len(y)
    M = pot_ot.dist(x, y, metric="sqeuclidean")
    if method == "emd":
        T = pot_ot.emd(a, b, M)
        return float(np.sqrt(np.sum(T * M)))
    return float(np.sqrt(pot_ot.sinkhorn2(a, b, M, reg=reg)))


def sinkhorn_divergence_pot(x, y, *, reg: float = 0.05) -> float:
    if not _POT_AVAILABLE:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = np.ones(len(x)) / len(x)
    b = np.ones(len(y)) / len(y)
    Mxy = pot_ot.dist(x, y, metric="sqeuclidean")
    Mxx = pot_ot.dist(x, x, metric="sqeuclidean")
    Myy = pot_ot.dist(y, y, metric="sqeuclidean")
    return float(
        pot_ot.sinkhorn2(a, b, Mxy, reg=reg)
        - 0.5 * pot_ot.sinkhorn2(a, a, Mxx, reg=reg)
        - 0.5 * pot_ot.sinkhorn2(b, b, Myy, reg=reg)
    )


def evaluate_final(
    target: Target,
    final_x_traj: np.ndarray,
    dt: float,
    out_dir: str,
    tag: str,
    *,
    grid_res: int = 60,
    ergodic_k_max: int = 10,
    ergodic_weight_power: float = 1.5,
    w2_n: int = 500,
    w2_method: str = "sinkhorn",
    sinkhorn_reg: float = 0.05,
    seed: int = 0,
) -> dict:
    xmin, xmax = target.xlim
    ymin, ymax = target.ylim
    domain = (xmin, xmax, ymin, ymax)

    traj = np.asarray(final_x_traj, dtype=np.float64)
    if traj.ndim == 2:
        traj = traj[None, ...]
    n_agents_eval, T_eval, _ = traj.shape
    pos = traj[:, :, :2].reshape(-1, 2)

    x_edges = np.linspace(xmin, xmax, grid_res + 1)
    y_edges = np.linspace(ymin, ymax, grid_res + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    tgt = np.asarray(target.pdf_np(X, Y), dtype=np.float64)
    tgt = np.maximum(tgt, 0.0)
    cell_area = (x_centers[1] - x_centers[0]) * (y_centers[1] - y_centers[0])
    z = float(np.sum(tgt) * cell_area)
    if z > 0:
        tgt = tgt / z

    hist_emp, _, _ = np.histogram2d(
        pos[:, 0], pos[:, 1], bins=[x_edges, y_edges], density=True
    )

    l2_error = float(np.sqrt(np.sum((hist_emp - tgt) ** 2) * cell_area))
    if hist_emp.std() > 0 and tgt.std() > 0:
        corr = float(np.corrcoef(hist_emp.flatten(), tgt.flatten())[0, 1])
    else:
        corr = float("nan")
    coverage_l1 = grid_coverage_error(hist_emp, tgt, cell_area, kind="l1")
    coverage_l2 = grid_coverage_error(hist_emp, tgt, cell_area, kind="l2")

    erg = fourier_ergodic_metric(
        pos,
        tgt,
        domain,
        k_max=ergodic_k_max,
        weight_power=ergodic_weight_power,
    )

    n_ot = int(min(w2_n, len(pos)))
    if len(pos) >= n_ot:
        idx = np.linspace(0, len(pos) - 1, n_ot).astype(int)
        x_hat = pos[idx]
    else:
        x_hat = pos
    x_tgt = np.asarray(target.sample(n_ot, seed=seed + 999), dtype=np.float64)
    w2 = wasserstein_w2_pot(x_hat, x_tgt, method=w2_method, reg=sinkhorn_reg)
    sdiv_pot = sinkhorn_divergence_pot(x_hat, x_tgt, reg=sinkhorn_reg)

    acc_energy = float("nan")
    if T_eval >= 3 and dt > 0:
        pos_NT2 = traj[:, :, :2]
        vel = np.diff(pos_NT2, axis=1) / dt
        acc = np.diff(vel, axis=1) / dt
        acc_sq = np.sum(acc**2, axis=2).reshape(-1)
        clip_val = float(np.percentile(acc_sq, 95))
        acc_energy = float(np.mean(np.clip(acc_sq, 0.0, clip_val)))

    metrics = {
        "correlation": corr,
        "l2_error": l2_error,
        "coverage_l1": coverage_l1,
        "coverage_l2": coverage_l2,
        "fourier_ergodic": erg,
        "w2": w2,
        "sinkhorn_div_pot": sdiv_pot,
        "acc_energy": acc_energy,
    }

    fig = plt.figure(figsize=(15, 5), dpi=150)
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.pcolormesh(X, Y, tgt, cmap="Purples", shading="auto")
    plt.colorbar(im, ax=ax1, shrink=0.8)
    cmap_t = plt.get_cmap("tab10" if n_agents_eval <= 10 else "hsv")
    for i in range(n_agents_eval):
        c = cmap_t(i % cmap_t.N) if n_agents_eval > 1 else "C1"
        ax1.plot(
            traj[i, :, 0],
            traj[i, :, 1],
            "-",
            color=c,
            lw=0.6 if n_agents_eval == 1 else 0.4,
            alpha=0.8,
        )
    ax1.set_title(
        f"Target + {n_agents_eval} agent trajectories"
        if n_agents_eval > 1
        else "Target density + trajectory"
    )
    ax1.set_aspect("equal")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.pcolormesh(X, Y, hist_emp, cmap="Purples", shading="auto")
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.set_title(f"Empirical density\nCorr={corr:.3f}  L2={l2_error:.4f}")
    ax2.set_aspect("equal")
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")
    txt = (
        f"Correlation:       {corr:.4f}\n"
        f"L2 density error:  {l2_error:.4f}\n"
        f"Coverage L1:       {coverage_l1:.4f}\n"
        f"Coverage L2:       {coverage_l2:.4f}\n"
        f"Fourier ergodic:   {erg:.4e}\n"
        f"W2 ({w2_method}):        {w2:.4f}\n"
        f"Sinkhorn div(POT): {sdiv_pot:.4f}\n"
        f"Accel energy:      {acc_energy:.4f}\n"
    )
    ax3.text(
        0.02,
        0.95,
        txt,
        family="monospace",
        fontsize=11,
        va="top",
        transform=ax3.transAxes,
    )
    ax3.set_title("Metrics")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_eval.png"), bbox_inches="tight")
    plt.close(fig)

    return metrics
