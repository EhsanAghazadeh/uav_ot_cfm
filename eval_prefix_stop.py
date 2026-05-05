"""Prefix-stop evaluator: walks one long trajectory and emits the three
experiment curves (fixed-energy → coverage; fixed-coverage → energy;
fixed-N → energy + coverage), with four coverage metrics each.

Coverage metrics (computed at every prefix length):
  - fourier_erg : Σ_k Λ_k (c_k - φ_k)^2 with the same n_basis basis on [0,1]^2
                 used by the planner. Lower is better.
  - density_corr: Pearson correlation between the smoothed visitation
                 histogram (Gaussian KDE on the eval grid) and the target,
                 both flattened. Higher is better.
  - l1          : Σ |hist - target| with hist, target normalised to PMF.
                 Lower is better.
  - l2          : sqrt(Σ (hist - target)^2) with PMFs. Lower is better.

Usage:
  python eval_prefix_stop.py \
      --traj results/toes_milano_long/trajectory.npz \
      --milano-grid data/milano_grid_corrected.npy \
      --smooth-sigma 1.5 \
      --out-dir results/toes_milano_long/curves
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fair_comparison import TrajectoryRecord
from milano_baselines import RotaryWingEnergyModel


# ---------------------------------------------------------------------------
# Cumulative energy
# ---------------------------------------------------------------------------
def cumulative_energy(pos: np.ndarray, dt: float) -> np.ndarray:
    """E[k] = Σ_{t<k} ‖v_t‖² · dt  with  v_t = (p_{t+1} - p_t) / dt.
    Returns shape (T,) with E[0]=0.
    """
    v = (pos[1:] - pos[:-1]) / dt
    inst = np.sum(v * v, axis=1) * dt          # ‖v‖² dt
    cum = np.zeros(pos.shape[0], dtype=np.float64)
    cum[1:] = np.cumsum(inst)
    return cum


# ---------------------------------------------------------------------------
# Coverage metrics
# ---------------------------------------------------------------------------
def _hk(k_vec, basis_n):
    # Same hk normaliser used by upstream BasisFunc (numerical equivalent).
    # We re-implement here so this module has no JAX dependency at eval time.
    h = 1.0
    for ki in k_vec:
        if ki == 0:
            h *= 1.0
        else:
            h *= 0.5
    return np.sqrt(h)


def _build_basis(n_basis: int):
    ks = np.arange(n_basis)
    Kx, Ky = np.meshgrid(ks, ks, indexing="ij")
    K = np.stack([Kx.ravel(), Ky.ravel()], axis=1).astype(np.float64)
    hk = np.array([_hk(k, n_basis) for k in K])  # (B,)
    # Λ_k = (1 + ||k||)^(-(v+1)/2), v=2 → exponent = -1.5
    lam = (1.0 + np.linalg.norm(K, axis=1) ** 2) ** (-1.5)
    return K, hk, lam


def fourier_basis_eval(pts_unit: np.ndarray, K: np.ndarray, hk: np.ndarray):
    """F_k(x) = (1/h_k) ∏_i cos(π k_i x_i). Returns (T, B)."""
    # cos(π k_i x_i) for each (i, k_i, x)
    arg = np.pi * pts_unit[:, None, :] * K[None, :, :]
    fk = np.prod(np.cos(arg), axis=2)
    return fk / hk[None, :]


def phik_from_target(target: np.ndarray, K: np.ndarray, hk: np.ndarray):
    """Match upstream get_phik on the same 50x50 grid the planner used."""
    h, w = target.shape
    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    vals = (target / target.sum()).ravel()
    fk_pts = fourier_basis_eval(pts, K, hk)            # (M, B)
    phik = np.einsum("m,mb->b", vals, fk_pts)           # (B,)  bypasses BLAS
    phik = phik / phik[0]
    return phik


def fourier_ergodic(pos_unit: np.ndarray, dt: float, T_now: float,
                   K, hk, lam, phik):
    """E(c, φ) = Σ Λ_k (c_k - φ_k)^2 with c_k = (1/T) Σ F_k(x_t) dt."""
    if T_now <= 0:
        return np.nan
    fk_t = fourier_basis_eval(pos_unit, K, hk)         # (T, B)
    ck = fk_t.sum(axis=0) * (dt / T_now)
    return float(np.sum(lam * (ck - phik) ** 2))


def visitation_histogram(pos_unit: np.ndarray, target_shape):
    """2D histogram of unit-box positions on the same grid as the target."""
    h, w = target_shape
    # bins are edges, size h+1 / w+1
    H, _, _ = np.histogram2d(
        pos_unit[:, 0], pos_unit[:, 1],
        bins=[w, h], range=[[0, 1], [0, 1]],
    )
    return H.T  # match (h, w)


def coverage_metrics(pos_unit, dt, target_pmf, target_smooth_pmf,
                    K, hk, lam, phik, sigma=1.5):
    """Compute all four metrics at this prefix length."""
    T_now = (pos_unit.shape[0] - 1) * dt
    erg = fourier_ergodic(pos_unit, dt, T_now if T_now > 0 else dt, K, hk, lam, phik)

    H = visitation_histogram(pos_unit, target_pmf.shape)
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        H_smooth = gaussian_filter(H, sigma=sigma)
    else:
        H_smooth = H
    s = H_smooth.sum()
    H_pmf = H_smooth / s if s > 0 else H_smooth
    diff = H_pmf - target_smooth_pmf
    l1 = float(np.abs(diff).sum())
    l2 = float(np.sqrt((diff ** 2).sum()))
    a = H_pmf.ravel(); b = target_smooth_pmf.ravel()
    if a.std() > 0 and b.std() > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = float("nan")
    return dict(fourier_erg=erg, density_corr=corr, l1=l1, l2=l2)


# ---------------------------------------------------------------------------
# NFZ prefix metrics
# ---------------------------------------------------------------------------
def nfz_prefix_metrics(pos_phys: np.ndarray, discs, ks: np.ndarray):
    """Returns (nfz_frac[i], nfz_n_incursions[i]) at each prefix index ks[i].

    discs: list of (center, radius) pairs. A point is inside if inside ANY disc.
    nfz_frac         : fraction of points 0..k inside any NFZ disc.
    nfz_n_incursions : number of outside→inside transitions (union) up to k.
    """
    inside = np.zeros(len(pos_phys), dtype=bool)
    for center, radius in discs:
        c = np.asarray(center, dtype=np.float64)
        inside |= np.linalg.norm(pos_phys - c, axis=1) <= float(radius)
    cum_inside = np.cumsum(inside.astype(np.int64))                # (T,)
    entries = np.zeros(len(pos_phys), dtype=np.int64)
    if len(pos_phys) > 1:
        entries[1:] = (~inside[:-1]) & inside[1:]
    cum_entries = np.cumsum(entries)                               # (T,)
    fracs   = cum_inside[ks].astype(np.float64) / (ks + 1)
    n_incur = cum_entries[ks].astype(np.float64)
    return fracs, n_incur


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True)
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--n-basis", type=int, default=8)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-eval-points", type=int, default=128,
                  help="number of prefix lengths at which to evaluate metrics")
    p.add_argument("--resample-dt", type=float, default=None,
                  help="If set, linearly resample trajectory to this uniform "
                       "dt before computing metrics (apples-to-apples energy "
                       "across methods with different native dt).")
    p.add_argument("--target-T", type=float, default=None,
                  help="If set, linearly rescale times so total horizon = "
                       "target-T (s) before resampling. Use with --resample-dt "
                       "to make all methods share N = target-T / resample-dt.")
    p.add_argument("--time-scale", type=float, default=1.0,
                  help="Multiply native times by this factor (applied after "
                       "--target-T if both given). Pair with the eval_fair "
                       "--time-scale and --domain-size for the rescaled "
                       "UAV-mission protocol.")
    p.add_argument("--nfz", action="append", nargs=3, type=float,
                  metavar=("CX", "CY", "R"), default=[],
                  help="NFZ disc: cx cy radius (repeatable for multiple discs). "
                       "E.g. --nfz 0.19 0.03 0.10 --nfz 0.3 0.3 0.15")
    p.add_argument("--nfz-center", nargs=2, type=float, default=None,
                  help=argparse.SUPPRESS)  # deprecated, use --nfz
    p.add_argument("--nfz-radius", type=float, default=None,
                  help=argparse.SUPPRESS)  # deprecated, use --nfz
    p.add_argument("--mask-nfz-target", action="store_true",
                  help="Compute coverage metrics against the NFZ-masked "
                       "target. Requires at least one --nfz CX CY R disc.")
    args = p.parse_args()
    nfz_discs = [((cx, cy), r) for cx, cy, r in (args.nfz or [])]
    if args.nfz_center is not None and args.nfz_radius is not None:
        nfz_discs.append((tuple(args.nfz_center), float(args.nfz_radius)))
    if args.mask_nfz_target and not nfz_discs:
        raise SystemExit("--mask-nfz-target requires at least one --nfz CX CY R disc.")

    os.makedirs(args.out_dir, exist_ok=True)
    rec = TrajectoryRecord.from_npz(args.traj)
    xlim = (rec.domain[0], rec.domain[1])
    ylim = (rec.domain[2], rec.domain[3])
    pos_phys = rec.pos
    times = rec.times
    native_dt = float(times[1] - times[0])
    native_T = float(times[-1] - times[0])
    if args.target_T is not None and args.target_T > 0:
        scale = args.target_T / max(native_T, 1e-12)
        times = times[0] + (times - times[0]) * scale
        print(f"rescale T     : {native_T:.3f}s -> {args.target_T}s  (×{scale:.4g})")
    if args.time_scale != 1.0:
        times = times * float(args.time_scale)
        print(f"time-scale    : ×{args.time_scale:.4g}  (mission now {times[-1]-times[0]:.1f}s)")
    if args.resample_dt is not None and args.resample_dt > 0:
        T_total = float(times[-1] - times[0])
        n_new = max(2, int(round(T_total / args.resample_dt)) + 1)
        new_times = times[0] + args.resample_dt * np.arange(n_new)
        new_times = np.clip(new_times, times[0], times[-1])
        new_pos = np.column_stack([
            np.interp(new_times, times, pos_phys[:, 0]),
            np.interp(new_times, times, pos_phys[:, 1]),
        ])
        print(f"resample      : dt {native_dt:.5g}s -> {args.resample_dt}s  "
              f"(N {pos_phys.shape[0]} -> {new_pos.shape[0]})")
        pos_phys = new_pos
        times = new_times
    T = pos_phys.shape[0]
    dt = float(times[1] - times[0])
    plan_s = float(rec.meta.get("plan_wall_clock_s", 0.0))

    # Map to unit-box for Fourier metrics
    pos_unit = np.stack([
        (pos_phys[:, 0] - xlim[0]) / (xlim[1] - xlim[0]),
        (pos_phys[:, 1] - ylim[0]) / (ylim[1] - ylim[0]),
    ], axis=1)

    # Target setup
    from target_density import load_milano_target
    mask_discs = nfz_discs if args.mask_nfz_target else []
    if args.mask_nfz_target:
        print(f"  Coverage target: NFZ-masked with {len(mask_discs)} disc(s): {mask_discs}")
    t_target = load_milano_target(args.milano_grid,
                                  smooth_sigma=args.smooth_sigma,
                                  nfz_discs=mask_discs)
    grid_smooth = t_target["grid"]
    target_smooth_pmf = t_target["pmf"]
    # raw (unsmoothed) PMF for L1/L2 reference
    raw = np.load(args.milano_grid).astype(np.float64)
    if t_target["nfz_mask"] is not None:
        raw = np.where(t_target["nfz_mask"], 0.0, raw)
    target_pmf = raw / max(raw.sum(), 1e-12)

    K, hk, lam = _build_basis(args.n_basis)
    phik = phik_from_target(grid_smooth, K, hk)

    # Cumulative energy (Σ‖v‖²·dt) and Zeng-2019 rotary-wing E_total (kJ)
    energy = cumulative_energy(pos_phys, dt)            # (T,)
    method_kind = str(rec.meta.get("method_kind", rec.method))
    energy_model = RotaryWingEnergyModel(time_step=dt)

    # Sweep prefix lengths log-spaced from 32 to T
    ks = np.unique(
        np.round(np.geomspace(32, T - 1, args.n_eval_points)).astype(int)
    )
    nfz_on = bool(nfz_discs)
    if nfz_on:
        nfz_fracs, nfz_incurs = nfz_prefix_metrics(pos_phys, nfz_discs, ks)
        disc_desc = "  ".join(
            f"c=({c[0]:.3f},{c[1]:.3f}) r={r:.3f}" for c, r in nfz_discs)
        print(f"NFZ on ({len(nfz_discs)} disc(s)): {disc_desc}")

    print(f"trajectory    : T={T} samples, dt={dt}s, total horizon={times[-1]:.2f}s")
    print(f"plan wall t   : {plan_s:.2f}s")
    print(f"max energy    : {energy[-1]:.4e}")
    print(f"evaluating {len(ks)} prefix lengths ...")

    rows = []
    eval_t0 = time.perf_counter()
    for i, k in enumerate(ks):
        m = coverage_metrics(pos_unit[:k+1], dt, target_pmf, target_smooth_pmf,
                            K, hk, lam, phik, sigma=args.smooth_sigma)
        e_kj = energy_model.E_total_kJ(pos_phys[:k+1], dt,
                                       method_type=method_kind)
        row = {
            "k": int(k),
            "t_s": float(times[k]),
            "energy": float(energy[k]),
            "E_total_kJ": float(e_kj),
            "n_points": int(k + 1),
            **m,
        }
        if nfz_on:
            row["nfz_frac_inside"]  = float(nfz_fracs[i])
            row["nfz_n_incursions"] = float(nfz_incurs[i])
        rows.append(row)
    eval_s = time.perf_counter() - eval_t0
    print(f"eval wall t   : {eval_s:.2f}s")

    # Pack into arrays
    arr = lambda key: np.array([r[key] for r in rows])
    nfz_arrays = {}
    if nfz_on:
        nfz_arrays["nfz_frac_inside"]  = arr("nfz_frac_inside")
        nfz_arrays["nfz_n_incursions"] = arr("nfz_n_incursions")
    np.savez(
        os.path.join(args.out_dir, "prefix_curve.npz"),
        k=arr("k"), t_s=arr("t_s"),
        energy=arr("energy"),
        E_total_kJ=arr("E_total_kJ"),
        n_points=arr("n_points"),
        fourier_erg=arr("fourier_erg"),
        density_corr=arr("density_corr"),
        l1=arr("l1"), l2=arr("l2"),
        plan_wall_clock_s=plan_s,
        eval_wall_clock_s=eval_s,
        method=rec.method,
        **nfz_arrays,
    )
    with open(os.path.join(args.out_dir, "prefix_curve.json"), "w") as f:
        json.dump({
            "method": rec.method,
            "plan_wall_clock_s": plan_s,
            "eval_wall_clock_s": eval_s,
            "n_eval_points": len(rows),
            "energy_range": [float(arr("energy").min()), float(arr("energy").max())],
            "fourier_erg_range": [float(arr("fourier_erg").min()),
                                  float(arr("fourier_erg").max())],
            "density_corr_range": [float(arr("density_corr").min()),
                                   float(arr("density_corr").max())],
            "rows": rows[:5] + ["..."] + rows[-5:],
        }, f, indent=2)
    print(f"saved         : {os.path.join(args.out_dir, 'prefix_curve.npz')}")


if __name__ == "__main__":
    main()
