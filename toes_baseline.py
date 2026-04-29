"""User's TOES adapter, de-mangled (markdown-autolink artifacts removed).
Stripped of fair_comparison + run_toes + canonical-export to keep this self
contained. Only the optimisation core is needed for the head-to-head.
"""
from __future__ import annotations

import os, sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as onp

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
_TOES_ROOT = os.path.join(_REPO_ROOT, "time_optimal_ergodic_search")
if _TOES_ROOT not in sys.path:
    sys.path.insert(0, _TOES_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import vmap  # noqa: E402

from time_opt_erg_lib.dynamics import DoubleIntegrator  # noqa: E402
from time_opt_erg_lib.ergodic_metric import ErgodicMetric  # noqa: E402
from time_opt_erg_lib.fourier_utils import BasisFunc, get_ck, get_phik  # noqa: E402
from time_opt_erg_lib.opt_solver import AugmentedLagrangeSolver  # noqa: E402


def phik_from_density_grid(phi_vals, xlim, ylim, basis):
    h, w = phi_vals.shape
    xs_phys = onp.linspace(xlim[0], xlim[1], w)
    ys_phys = onp.linspace(ylim[0], ylim[1], h)
    X_phys, Y_phys = onp.meshgrid(xs_phys, ys_phys)
    X_unit = (X_phys - xlim[0]) / (xlim[1] - xlim[0])
    Y_unit = (Y_phys - ylim[0]) / (ylim[1] - ylim[0])
    pts = onp.stack([X_unit.ravel(), Y_unit.ravel()], axis=1)
    vals = phi_vals.ravel().astype(onp.float64)
    vals = vals / onp.sum(vals)
    return get_phik((jnp.asarray(vals), jnp.asarray(pts)), basis)


def _linear_spline_init(x0, xf, N, jitter=0.0, seed=0):
    ts = onp.linspace(0.0, 1.0, N)[:, None]
    x_seed = (1.0 - ts) * x0[None, :] + ts * xf[None, :]
    if jitter > 0.0:
        rng = onp.random.default_rng(seed)
        x_seed = x_seed + jitter * rng.standard_normal(x_seed.shape)
    return x_seed


@dataclass
class TOESConfig:
    mode: str
    N: int = 200
    n_basis: int = 8
    u_max: float = 1.0
    workspace_bnds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0))
    step_size: float = 1e-3
    c_init: float = 1.0
    max_iter: int = 15000
    convergence_eps: float = 1e-7
    erg_ub: float = 0.01
    tf_init: float = 5.0
    T_total: float = 20.0
    nfz_center_unit: Optional[Tuple[float, float]] = None
    nfz_radius_unit: Optional[float] = None
    nfz_margin_unit: float = 0.05
    lambda_nfz: float = 50.0
    seed: int = 0


def _remap_to_unit(pos_xy, xlim, ylim):
    x = (pos_xy[..., 0] - xlim[0]) / (xlim[1] - xlim[0])
    y = (pos_xy[..., 1] - ylim[0]) / (ylim[1] - ylim[0])
    return onp.stack([x, y], axis=-1)


def _build_solver(cfg, phik, x0_unit, xf_unit):
    basis = BasisFunc(n_basis=[cfg.n_basis, cfg.n_basis])
    erg_metric = ErgodicMetric(basis)
    robot = DoubleIntegrator(dim=2)

    workspace_lo = jnp.array([cfg.workspace_bnds[0][0], cfg.workspace_bnds[1][0]])
    workspace_hi = jnp.array([cfg.workspace_bnds[0][1], cfg.workspace_bnds[1][1]])

    @vmap
    def emap(x):
        return jnp.array([
            (x[0] - workspace_lo[0]) / (workspace_hi[0] - workspace_lo[0]),
            (x[1] - workspace_lo[1]) / (workspace_hi[1] - workspace_lo[1]),
        ])

    def barrier_cost(e):
        return (jnp.maximum(0.0, e - 1.0) + jnp.maximum(0.0, -e)) ** 2

    use_nfz = cfg.nfz_center_unit is not None and cfg.nfz_radius_unit is not None
    if use_nfz:
        c_nfz = jnp.asarray(cfg.nfz_center_unit, dtype=jnp.float64)
        r_nfz = float(cfg.nfz_radius_unit); m_nfz = float(cfg.nfz_margin_unit)
        def nfz_violation(e_batch):
            d = jnp.linalg.norm(e_batch - c_nfz[None, :], axis=1)
            return (r_nfz + m_nfz) - d

    N = cfg.N
    n_x = robot.n * 2
    n_u = robot.m
    args_base = {"N": N, "x0": x0_unit, "xf": xf_unit, "phik": phik}

    if cfg.mode == "time_opt":
        args_base["erg_ub"] = float(cfg.erg_ub)

        def loss(params, args):
            x = params["x"]; tf = params["tf"]
            e = emap(x)
            return jnp.sum(barrier_cost(e)) + tf

        def eq_constr(params, args):
            x, u, tf = params["x"], params["u"], params["tf"]
            dt = tf / args["N"]
            return jnp.vstack([
                (x[0] - args["x0"])[None, :],
                x[1:, :] - (x[:-1, :] + dt * vmap(robot.dfdt)(x[:-1, :], u[:-1, :])),
                (x[-1] - args["xf"])[None, :],
            ])

        def ineq_constr(params, args):
            x, u, tf = params["x"], params["u"], params["tf"]
            dt = tf / args["N"]
            e = emap(x)
            ck = get_ck(e, basis, tf, dt)
            pieces = [
                jnp.array([erg_metric(ck, args["phik"]) - args["erg_ub"], -tf]),
                (jnp.abs(u) - cfg.u_max).flatten(),
            ]
            if use_nfz:
                pieces.append(nfz_violation(e[:, :2]))
            return jnp.concatenate(pieces)

        init_sol = {
            "x": jnp.asarray(onp.hstack([
                _linear_spline_init(onp.asarray(x0_unit[:2]), onp.asarray(xf_unit[:2]),
                                    N, jitter=0.0, seed=cfg.seed),  # jitter disabled for fair test
                onp.zeros((N, n_x - 2)),
            ])),
            "u": jnp.zeros((N, n_u)),
            "tf": jnp.asarray(float(cfg.tf_init)),
        }
    elif cfg.mode == "fixed_time":
        T_total = float(cfg.T_total)
        dt_fixed = T_total / N
        args_base["tf"] = T_total
        args_base["dt"] = dt_fixed

        def loss(params, args):
            x = params["x"]
            e = emap(x)
            ck = get_ck(e, basis, args["tf"], args["dt"])
            return erg_metric(ck, args["phik"]) + jnp.sum(barrier_cost(e))

        def eq_constr(params, args):
            x, u = params["x"], params["u"]
            return jnp.vstack([
                (x[0] - args["x0"])[None, :],
                x[1:, :] - (x[:-1, :] + args["dt"] * vmap(robot.dfdt)(x[:-1, :], u[:-1, :])),
                (x[-1] - args["xf"])[None, :],
            ])

        def ineq_constr(params, args):
            x, u = params["x"], params["u"]
            e = emap(x)
            pieces = [(jnp.abs(u) - cfg.u_max).flatten()]
            if use_nfz:
                pieces.append(nfz_violation(e[:, :2]))
            return jnp.concatenate(pieces)

        init_sol = {
            "x": jnp.asarray(onp.hstack([
                _linear_spline_init(onp.asarray(x0_unit[:2]), onp.asarray(xf_unit[:2]),
                                    N, jitter=0.0, seed=cfg.seed),
                onp.zeros((N, n_x - 2)),
            ])),
            "u": jnp.zeros((N, n_u)),
        }
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    solver = AugmentedLagrangeSolver(
        init_sol, loss, eq_constr, ineq_constr, args_base,
        step_size=cfg.step_size, c=cfg.c_init,
    )
    return solver, basis, erg_metric


@dataclass
class TOESSolution:
    pos_unit: onp.ndarray
    vel_unit: onp.ndarray
    u: onp.ndarray
    tf: float
    dt: float
    erg_metric: float
    n_iter: int
    mode: str


def solve_toes(phi_vals, xlim, ylim, cfg, x0_phys=None, xf_phys=None):
    basis = BasisFunc(n_basis=[cfg.n_basis, cfg.n_basis])
    phik = phik_from_density_grid(phi_vals, xlim, ylim, basis)

    centre_unit = jnp.array([0.5, 0.5, 0.0, 0.0])
    if x0_phys is None:
        x0_unit = centre_unit.at[0].set(0.1).at[1].set(0.1)
    else:
        x0_unit = jnp.asarray(onp.concatenate([
            _remap_to_unit(onp.asarray(x0_phys[:2]), xlim, ylim),
            onp.asarray(x0_phys[2:]) if len(x0_phys) > 2 else onp.zeros(2),
        ]))
    if xf_phys is None:
        xf_unit = centre_unit.at[0].set(0.9).at[1].set(0.9)
    else:
        xf_unit = jnp.asarray(onp.concatenate([
            _remap_to_unit(onp.asarray(xf_phys[:2]), xlim, ylim),
            onp.asarray(xf_phys[2:]) if len(xf_phys) > 2 else onp.zeros(2),
        ]))

    solver, basis_solver, erg_metric_obj = _build_solver(cfg, phik, x0_unit, xf_unit)

    n_iter_run = 0
    _prev_val = None
    for k in range(cfg.max_iter):
        solver.solution, solver.dual_solution, solver.avg_sq_grad, _val = solver.step(
            solver.solution, solver.dual_solution, solver.avg_sq_grad,
            solver.def_args, solver.c,
        )
        solver.c = 1.001 * solver.c
        n_iter_run = k + 1
        if _prev_val is not None and abs(_val - _prev_val) < cfg.convergence_eps:
            break
        _prev_val = _val

    sol = solver.get_solution()
    x = onp.asarray(sol["x"]); u = onp.asarray(sol["u"])
    pos_unit = x[:, :2]; vel_unit = x[:, 2:4] if x.shape[1] >= 4 else onp.zeros_like(pos_unit)

    if cfg.mode == "time_opt":
        tf = float(sol["tf"])
    else:
        tf = float(cfg.T_total)
    dt = tf / cfg.N

    ck = get_ck(jnp.asarray(pos_unit), basis_solver, tf, dt)
    erg_val = float(erg_metric_obj(ck, phik))

    return TOESSolution(pos_unit=pos_unit, vel_unit=vel_unit, u=u, tf=tf, dt=dt,
                       erg_metric=erg_val, n_iter=n_iter_run, mode=cfg.mode)
