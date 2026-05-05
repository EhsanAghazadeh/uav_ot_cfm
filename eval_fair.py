"""Fair-comparison evaluator with native trajectories (no resampling).

Computes scalar coverage and Zeng-energy metrics on each trajectory's
native (t_i, p_i) — no interpolation onto a common grid — then aggregates
across seeds and emits both a human-readable summary and LaTeX-ready
tables for the paper.

Per-trajectory metrics
----------------------
Coverage (computed on the full native trajectory):
    fourier_erg     Σ Λ_k (c_k - φ_k)^2  with Λ_k = (1+‖k‖²)^(-3/2),
                    n_basis x n_basis basis on [0,1]^2.    (lower better)
    density_corr    Pearson corr of smoothed visitation vs target.   (higher)
    l1, l2          PMF distances of smoothed visitation vs target.   (lower)

Energy / kinematics  (no resampling — native (t_i, p_i)):
    T_total_s       t[-1] - t[0]
    N_samples       len(pos)
    arc_length_m    Σ ‖Δp‖ * domain_size
    mean_speed_mps  arc_length_m / T_total_s
    max_speed_mps   max( ‖Δp_i‖ * domain_size / Δt_i )
    E_total_kJ      Σ P_zeng(V_i) Δt_i / 1000
    P_mean_W        E_total_J / T_total_s            ★ headline energy
    E_per_metre_J   E_total_J / arc_length_m

Optional NFZ block (when --nfz-center is given):
    nfz_frac_inside, nfz_dwell_s, nfz_n_incursions,
    nfz_max_depth, nfz_path_inside_m

Outputs
-------
    per_run.json                — one record per --traj
    aggregated.json             — per-method mean / std / min / max
    summary_table.tex           — headline LaTeX table (coverage + energy)
    nfz_table.tex               — NFZ table (if NFZ args given)
    summary.txt                 — human-readable copy of the same tables

Usage
-----
    python eval_fair.py \
        --traj toes:results/toes_seed21/trajectory.npz \
        --traj toes:results/toes_seed42/trajectory.npz \
        --traj otcfm:results/otcfm_seed21/trajectory.npz \
        --traj otcfm:results/otcfm_seed42/trajectory.npz \
        --milano-grid data/milano_grid_corrected.npy \
        --nfz-center 0.19 0.03 --nfz-radius 0.10 \
        --out-dir results/fair_metrics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fair_comparison import TrajectoryRecord
from milano_baselines import RotaryWingEnergyModel
from eval_nofly import violations as nfz_violations


# ---------------------------------------------------------------------------
# Coverage helpers (Fourier ergodic + smoothed-histogram metrics)
# ---------------------------------------------------------------------------
def _hk(k_vec):
    h = 1.0
    for ki in k_vec:
        h *= 1.0 if ki == 0 else 0.5
    return np.sqrt(h)


def _build_basis(n_basis: int):
    ks = np.arange(n_basis)
    Kx, Ky = np.meshgrid(ks, ks, indexing="ij")
    K = np.stack([Kx.ravel(), Ky.ravel()], axis=1).astype(np.float64)
    hk = np.array([_hk(k) for k in K])
    lam = (1.0 + np.linalg.norm(K, axis=1) ** 2) ** (-1.5)
    return K, hk, lam


def _fourier_basis_eval(pts_unit, K, hk):
    arg = np.pi * pts_unit[:, None, :] * K[None, :, :]
    return np.prod(np.cos(arg), axis=2) / hk[None, :]


def _phik_from_target(target, K, hk):
    h, w = target.shape
    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    vals = (target / target.sum()).ravel()
    fk_pts = _fourier_basis_eval(pts, K, hk)
    phik = np.einsum("m,mb->b", vals, fk_pts)
    return phik / phik[0]


def coverage_metrics(pos_unit, dt, target_smooth_pmf, K, hk, lam, phik,
                     sigma=1.5):
    T_now = max((pos_unit.shape[0] - 1) * dt, dt)
    fk_t = _fourier_basis_eval(pos_unit, K, hk)
    ck = fk_t.sum(axis=0) * (dt / T_now)
    erg = float(np.sum(lam * (ck - phik) ** 2))

    h, w = target_smooth_pmf.shape
    H, _, _ = np.histogram2d(pos_unit[:, 0], pos_unit[:, 1],
                             bins=[w, h], range=[[0, 1], [0, 1]])
    H = H.T
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, sigma=sigma)
    s = H.sum()
    H_pmf = H / s if s > 0 else H
    diff = H_pmf - target_smooth_pmf
    a, b = H_pmf.ravel(), target_smooth_pmf.ravel()
    corr = (float(np.corrcoef(a, b)[0, 1])
            if a.std() > 0 and b.std() > 0 else float("nan"))
    return {
        "fourier_erg": erg,
        "density_corr": corr,
        "l1": float(np.abs(diff).sum()),
        "l2": float(np.sqrt((diff ** 2).sum())),
    }


# ---------------------------------------------------------------------------
# Energy / kinematics on NATIVE trajectory (no resampling)
# ---------------------------------------------------------------------------
def kinematics_and_energy(pos, times, domain_size: float,
                          v_clip=(0.01, 100.0),
                          v_clip_realistic=None):
    """Return total time, arc length, speeds, mean/total Zeng energy.

    Parameters
    ----------
    v_clip : (vmin, vmax)
        The numerical-guard clip: caps the V^3 parasitic term beyond the
        Zeng model's extrapolation range. Default (0.01, 100) m/s.
        These values appear in the headline P_mean_W / E_total_kJ.
    v_clip_realistic : float or None
        Optional second clip at a physically realistic UAV cap (e.g.
        30 m/s). When set, two extra fields are returned:
            P_mean_W_realistic, E_total_kJ_realistic
        These quantify the energy under the assumption that no UAV can
        physically exceed the cap; samples requesting V > cap contribute
        Zeng power *at* the cap rather than at the requested V. The gap
        from the headline column reveals teleport-pathology baselines.
        The fraction of samples above the cap (frac_v_over_cap) is also
        reported.

    Returns
    -------
    dict with mission/kinematic fields and one or two energy column sets.
    """
    pos = np.asarray(pos, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if pos.shape[0] < 2:
        raise ValueError("trajectory has < 2 points")

    dx = np.diff(pos, axis=0)
    dt = np.diff(times)
    dt = np.maximum(dt, 1e-12)
    step_m = np.linalg.norm(dx, axis=1) * domain_size  # metres
    v_mps = step_m / dt                                 # m/s per step

    em = RotaryWingEnergyModel()

    # Headline (numerical-guard) cap
    v_clipped = np.clip(v_mps, v_clip[0], v_clip[1])
    P_steps = np.array([em.power(v) for v in v_clipped])
    E_steps = P_steps * dt
    E_total_J = float(E_steps.sum())

    T_total = float(times[-1] - times[0])
    arc_length_m = float(step_m.sum())
    out = {
        "T_total_s": T_total,
        "N_samples": int(pos.shape[0]),
        "arc_length_m": arc_length_m,
        "mean_speed_mps": float(arc_length_m / max(T_total, 1e-12)),
        "max_speed_mps": float(v_mps.max()),
        "E_total_kJ": E_total_J / 1000.0,
        "P_mean_W": float(E_total_J / max(T_total, 1e-12)),
        "E_per_metre_J": float(E_total_J / max(arc_length_m, 1e-12)),
    }

    if v_clip_realistic is not None and v_clip_realistic > 0:
        cap = float(v_clip_realistic)

        # --- "Realistic-cap" power: reported mission time, V capped at cap.
        #     Doesn't penalise the *time* the UAV would have needed to
        #     physically execute a teleport -- only the per-step power.
        v_real = np.clip(v_mps, v_clip[0], cap)
        P_real = np.array([em.power(v) for v in v_real])
        E_real = (P_real * dt).sum()
        frac_over = float((v_mps > cap).mean())
        out["P_mean_W_realistic"] = float(E_real / max(T_total, 1e-12))
        out["E_total_kJ_realistic"] = float(E_real / 1000.0)
        out["E_per_metre_J_realistic"] = float(E_real / max(arc_length_m, 1e-12))
        out["frac_v_over_realistic_cap"] = frac_over
        out["v_clip_realistic"] = cap

        # --- "Realizable-execution" power: for each step that requests
        #     V > cap, extend dt to step_m / cap so V = cap is achievable.
        #     The mission time grows; energy is computed on the extended
        #     trajectory. This is the honest cost of executing a discrete
        #     /teleporting trajectory on a real UAV. Continuous methods are
        #     unaffected (their dt is already long enough).
        dt_real = np.maximum(dt, step_m / max(cap, 1e-9))
        v_real_exec = step_m / np.maximum(dt_real, 1e-12)  # = cap on extended steps
        v_real_exec = np.clip(v_real_exec, v_clip[0], cap)
        P_real_exec = np.array([em.power(v) for v in v_real_exec])
        E_real_exec = (P_real_exec * dt_real).sum()
        T_real_exec = float(dt_real.sum())
        out["T_realizable_s"] = T_real_exec
        out["E_realizable_kJ"] = float(E_real_exec / 1000.0)
        out["P_mean_W_realizable"] = float(E_real_exec / max(T_real_exec, 1e-12))
        out["E_per_metre_J_realizable"] = float(
            E_real_exec / max(arc_length_m, 1e-12))
        # Time-extension factor: 1.0 = no teleport; >> 1 = heavy teleport.
        out["realizable_time_factor"] = float(T_real_exec / max(T_total, 1e-12))

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
SCALAR_KEYS = [
    "fourier_erg", "density_corr", "l1", "l2",
    "T_total_s", "N_samples", "arc_length_m",
    "mean_speed_mps", "max_speed_mps",
    "E_total_kJ", "P_mean_W", "E_per_metre_J",
    "nfz_frac_inside", "nfz_dwell_time_s",
    "nfz_n_incursions", "nfz_max_depth", "nfz_path_inside_norm",
]


def aggregate(per_run):
    """Group by `method`, compute mean/std/min/max per scalar key."""
    by_method = defaultdict(list)
    for r in per_run:
        by_method[r["method"]].append(r)
    out = {}
    for m, rows in by_method.items():
        agg = {"n_seeds": len(rows)}
        for k in SCALAR_KEYS:
            vals = [r[k] for r in rows if k in r and r[k] is not None
                    and not (isinstance(r[k], float) and np.isnan(r[k]))]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            agg[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        out[m] = agg
    return out


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------
def _fmt_latex_num(x: float, prec: int = 3) -> str:
    """Format a float for LaTeX, auto-switching to sci notation."""
    s = f"{x:.{prec}g}"
    if "e" in s:
        mant, exp_str = s.split("e")
        exp = int(exp_str)
        return f"{mant}{{\\cdot}}10^{{{exp}}}"
    return s


def _fmt_mean_std(stats, prec: int = 3, bold_mean: bool = False) -> str:
    r"""Return \ms{mean}{std} — requires \newcommand{\ms}[2]{...} in preamble."""
    if stats is None:
        return "---"
    m, s = stats["mean"], stats["std"]
    m_str = _fmt_latex_num(m, prec)
    s_str = _fmt_latex_num(s, prec)
    if bold_mean:
        m_str = f"\\mathbf{{{m_str}}}"
    return f"\\ms{{{m_str}}}{{{s_str}}}"


def _direction(key):
    """Return +1 if higher is better, -1 otherwise (used to bold the winner)."""
    return +1 if key in ("density_corr",) else -1


def _bold_best(values, key):
    """Return formatted strings; best entry has bold mean inside \\ms{}{}."""
    sign = _direction(key)
    means = [v["mean"] if v is not None else None for v in values]
    valid = [(i, m) for i, m in enumerate(means) if m is not None]
    if not valid:
        return [_fmt_mean_std(v) for v in values]
    best_i = max(valid, key=lambda x: sign * x[1])[0]
    return [_fmt_mean_std(v, bold_mean=(i == best_i))
            for i, v in enumerate(values)]


# Column order matches paper table (T column intentionally omitted — all
# methods share the same mission horizon after the dt normalisation).
HEADLINE_COLUMNS = [
    ("fourier_erg",   r"$\mathcal{E}\downarrow$"),
    ("density_corr",  r"$\rho\uparrow$"),
    ("l1",            r"$\ell_1\downarrow$"),
    ("P_mean_W",      r"$\bar P$ (W)$\downarrow$"),
    ("E_total_kJ",    r"$E_\mathrm{tot}$ (kJ)$\downarrow$"),
    ("arc_length_m",  r"$L$ (m)"),
    ("E_per_metre_J", r"$E/L$ (J/m)$\downarrow$"),
]

NFZ_COLUMNS = [
    ("nfz_frac_inside",     r"frac inside (\%)$\downarrow$"),
    ("nfz_max_depth",       r"max depth$\downarrow$"),
    ("nfz_n_incursions",    r"\# incursions$\downarrow$"),
    ("nfz_dwell_time_s",    r"dwell (s)$\downarrow$"),
    ("nfz_path_inside_norm",r"path-in (norm.)$\downarrow$"),
]


def render_headline_latex(agg, method_order, label="tab:fair"):
    # Requires in paper preamble:
    #   \newcommand{\ms}[2]{$#1_{\pm{\scriptstyle#2}}$}
    from method_style import label as _mlabel
    cols = HEADLINE_COLUMNS
    n = len(cols)
    out = []
    out.append(r"% Requires: \newcommand{\ms}[2]{$#1_{\pm{\scriptstyle#2}}$}")
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{Density matching on Milano (NFZ-masked target). "
               r"Mean$_{\pm\text{std}}$ over 3 seeds; \textbf{best} per column. "
               r"Headline energy metric is mean Zeng power $\bar P$. "
               r"All methods evaluated on the same $\sim\!2540$\,s mission.}")
    out.append(r"\label{" + label + "}")
    out.append(r"\small")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\begin{tabular}{l" + "c" * n + "}")
    out.append(r"\toprule")
    out.append("Method & " + " & ".join(c[1] for c in cols) + r" \\")
    out.append(r"\midrule")
    cells = {}
    for k, _ in cols:
        col_vals = [agg.get(m, {}).get(k) for m in method_order]
        cells[k] = _bold_best(col_vals, k)
    for i, m in enumerate(method_order):
        if m not in agg:
            continue
        row_label = _mlabel(m).replace("&", r"\&")
        row = [row_label]
        for k, _ in cols:
            row.append(cells[k][i] if cells[k][i] is not None else "---")
        out.append(" & ".join(row) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def render_nfz_latex(agg, method_order, nfz_discs, label="tab:nfz"):
    cols = NFZ_COLUMNS
    out = []
    disc_str = "; ".join(
        f"$c=({c[0]:.2f},{c[1]:.2f}),r={r:.2f}$" for c, r in nfz_discs)
    out.append(r"\begin{table}[t]")
    out.append(r"\caption{NFZ violation (" + disc_str + r"). "
               r"\texttt{frac inside} reports the fraction of trajectory "
               r"samples whose position lies in any NFZ disc.}")
    out.append(r"\label{" + label + "}")
    out.append(r"\centering\small")
    out.append(r"\begin{tabular}{l" + "c" * len(cols) + "}")
    out.append(r"\toprule")
    out.append("Method & " + " & ".join(c[1] for c in cols) + r" \\")
    out.append(r"\midrule")
    cells = {}
    for k, _ in cols:
        col_vals = [agg.get(m, {}).get(k) for m in method_order]
        cells[k] = _bold_best(col_vals, k)
    for i, m in enumerate(method_order):
        if m not in agg:
            continue
        if "nfz_frac_inside" not in agg[m]:
            continue
        row = [m.replace("_", r"\_")]
        for k, _ in cols:
            row.append(cells[k][i] if cells[k][i] is not None else "---")
        out.append(" & ".join(row) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def render_text_table(agg, method_order, columns, header):
    rows = [header, ["-" * 8] * len(header)]
    for m in method_order:
        if m not in agg:
            continue
        row = [m]
        for k, _ in columns:
            v = agg[m].get(k)
            if v is None:
                row.append("---")
            else:
                row.append(f"{v['mean']:.4g} ± {v['std']:.2g}")
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", action="append", required=True,
                  help="method:path/to/trajectory.npz (repeatable). "
                       "Multiple entries with the same method are treated "
                       "as different seeds and aggregated.")
    p.add_argument("--milano-grid", default="data/milano_grid_corrected.npy")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--n-basis", type=int, default=8)
    p.add_argument("--domain-size", type=float, default=8.0,
                  help="Physical extent of the unit interval in metres "
                       "(matches RotaryWingEnergyModel default).")
    p.add_argument("--v-clip-max", type=float, default=100.0,
                  help="Cap instantaneous speed before applying Zeng power "
                       "(handles teleporting baselines). Default 100 m/s.")
    p.add_argument("--v-clip-realistic", type=float, default=None,
                  help="If set (e.g. 30.0 m/s = typical rotorcraft max "
                       "cruise), also report a 'realistic' energy column "
                       "computed at this cap. The gap to the headline "
                       "column reveals teleport pathology in baselines.")
    p.add_argument("--time-scale", type=float, default=1.0,
                  help="Multiply native trajectory times by this factor "
                       "before computing kinematics. Use with --domain-size "
                       "to rescale the synthetic 8 m / 0.01 s lab-scale "
                       "experiments to a realistic UAV mission scale "
                       "(e.g. --domain-size 200 --time-scale 10 maps every "
                       "method onto a 200 m x 200 m patrol).")
    p.add_argument("--method-time-scale", action="append", default=[],
                  metavar="METHOD:SCALE",
                  help="Per-method time-scale override (repeatable). "
                       "Syntax: METHOD:SCALE, e.g. --method-time-scale toes:2.0. "
                       "Overrides --time-scale for the named method only. "
                       "Use for dt-dependent methods (TOES, FMEC) whose "
                       "native dt must be preserved while normalising T_eval.")
    p.add_argument("--method-order", nargs="*", default=None,
                  help="Optional explicit method order in tables.")
    p.add_argument("--nfz", action="append", nargs=3, type=float,
                  metavar=("CX", "CY", "R"), default=[],
                  help="NFZ disc: cx cy radius (repeatable).")
    p.add_argument("--nfz-center", nargs=2, type=float, default=None,
                  help=argparse.SUPPRESS)
    p.add_argument("--nfz-radius", type=float, default=None,
                  help=argparse.SUPPRESS)
    p.add_argument("--mask-nfz-target", action="store_true",
                  help="Compute coverage metrics against the NFZ-masked "
                       "target (zero density inside the disc, renormalised). "
                       "Required for fair NFZ-condition comparison: methods "
                       "trained on the masked target should also be evaluated "
                       "against it. Requires --nfz-center / --nfz-radius.")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    nfz_discs = [((cx, cy), r) for cx, cy, r in (args.nfz or [])]
    if args.nfz_center is not None and args.nfz_radius is not None:
        nfz_discs.append((tuple(args.nfz_center), float(args.nfz_radius)))
    nfz_on = bool(nfz_discs)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.mask_nfz_target and not nfz_discs:
        raise SystemExit("--mask-nfz-target requires at least one --nfz CX CY R disc.")

    # ---- Target ----
    from target_density import load_milano_target
    mask_discs = nfz_discs if args.mask_nfz_target else []
    if args.mask_nfz_target:
        print(f"  Coverage target: NFZ-masked with {len(mask_discs)} disc(s): {mask_discs}")
    t_target = load_milano_target(args.milano_grid,
                                  smooth_sigma=args.smooth_sigma,
                                  nfz_discs=mask_discs)
    grid_smooth = t_target["grid"]
    target_smooth_pmf = t_target["pmf"]

    K, hk, lam = _build_basis(args.n_basis)
    phik = _phik_from_target(grid_smooth, K, hk)

    # ---- Per-trajectory metrics ----
    per_run = []
    print(f"\nFair-metric evaluator — domain_size={args.domain_size} m, "
          f"v_clip_max={args.v_clip_max} m/s, n_basis={args.n_basis}\n")
    print(f"{'method':<22} {'seed-id':<8} {'N':>7} {'T(s)':>8} "
          f"{'P̄(W)':>10} {'E(kJ)':>10} {'L(m)':>10} "
          f"{'erg':>9} {'L1':>7} {'rho':>7}")
    print("-" * 110)

    method_ts_overrides: dict = {}
    for spec in args.method_time_scale:
        if ":" not in spec:
            raise SystemExit(f"--method-time-scale must be METHOD:SCALE (got {spec})")
        mname, mscale = spec.rsplit(":", 1)
        method_ts_overrides[mname] = float(mscale)
    if method_ts_overrides:
        print(f"Per-method time-scale overrides: {method_ts_overrides}")

    for spec in args.traj:
        if ":" not in spec:
            raise SystemExit(f"--traj must be method:path (got {spec})")
        method, path = spec.split(":", 1)
        rec = TrajectoryRecord.from_npz(path)
        xlim = (rec.domain[0], rec.domain[1])
        ylim = (rec.domain[2], rec.domain[3])
        pos = rec.pos
        ts = method_ts_overrides.get(method, args.time_scale)
        times = rec.times * float(ts)
        dt_native = float(np.median(np.diff(times)))

        # Map to unit-box for Fourier metrics
        pos_unit = np.stack([
            (pos[:, 0] - xlim[0]) / (xlim[1] - xlim[0]),
            (pos[:, 1] - ylim[0]) / (ylim[1] - ylim[0]),
        ], axis=1)
        cov = coverage_metrics(pos_unit, dt_native, target_smooth_pmf,
                               K, hk, lam, phik, sigma=args.smooth_sigma)
        kin = kinematics_and_energy(pos, times, args.domain_size,
                                    v_clip=(0.01, args.v_clip_max),
                                    v_clip_realistic=args.v_clip_realistic)

        rec_out = {"method": method, "path": path, **cov, **kin}

        if nfz_on:
            v = nfz_violations(pos, dt_native, nfz_discs)
            rec_out.update({
                "nfz_frac_inside":   v["frac_inside"] * 100.0,  # report %
                "nfz_dwell_time_s":  v["dwell_time_s"],
                "nfz_n_incursions":  float(v["n_incursions"]),
                "nfz_max_depth":     v["max_depth"],
                "nfz_path_inside_norm": v.get("path_inside_norm",
                                              v.get("path_inside_m", 0.0)),
            })

        per_run.append(rec_out)
        seed_tag = os.path.basename(os.path.dirname(path))
        print(f"{method:<22} {seed_tag:<8} {kin['N_samples']:>7d} "
              f"{kin['T_total_s']:>8.2f} {kin['P_mean_W']:>10.3g} "
              f"{kin['E_total_kJ']:>10.3g} {kin['arc_length_m']:>10.3g} "
              f"{cov['fourier_erg']:>9.4f} {cov['l1']:>7.3f} "
              f"{cov['density_corr']:>7.3f}")

    # ---- Aggregate ----
    agg = aggregate(per_run)

    # method order: explicit or first-seen
    if args.method_order:
        method_order = list(args.method_order)
    else:
        method_order = []
        for r in per_run:
            if r["method"] not in method_order:
                method_order.append(r["method"])

    # ---- Save outputs ----
    with open(os.path.join(args.out_dir, "per_run.json"), "w") as f:
        json.dump(per_run, f, indent=2, default=float)
    with open(os.path.join(args.out_dir, "aggregated.json"), "w") as f:
        json.dump(agg, f, indent=2, default=float)

    head_tex = render_headline_latex(agg, method_order)
    with open(os.path.join(args.out_dir, "summary_table.tex"), "w") as f:
        f.write(head_tex + "\n")

    summary_txt = ["Fair-comparison summary",
                   "=======================", "",
                   "Coverage and energy (mean ± std over seeds):", ""]
    summary_txt.append(render_text_table(
        agg, method_order, HEADLINE_COLUMNS,
        ["method"] + [c[1].split("$")[0].strip() or c[0] for c in HEADLINE_COLUMNS]))

    if nfz_on:
        nfz_tex = render_nfz_latex(agg, method_order, nfz_discs)
        with open(os.path.join(args.out_dir, "nfz_table.tex"), "w") as f:
            f.write(nfz_tex + "\n")
        summary_txt += ["", "NFZ violation:", ""]
        summary_txt.append(render_text_table(
            agg, method_order, NFZ_COLUMNS,
            ["method"] + [c[0] for c in NFZ_COLUMNS]))

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_txt) + "\n")

    # ---- Print headline + LaTeX ----
    print("\n" + "=" * 70)
    print("LaTeX (summary_table.tex):")
    print("=" * 70)
    print(head_tex)
    if nfz_on:
        print("\n" + "=" * 70)
        print("LaTeX (nfz_table.tex):")
        print("=" * 70)
        print(nfz_tex)

    print(f"\nsaved: {args.out_dir}/per_run.json")
    print(f"saved: {args.out_dir}/aggregated.json")
    print(f"saved: {args.out_dir}/summary_table.tex")
    if nfz_on:
        print(f"saved: {args.out_dir}/nfz_table.tex")
    print(f"saved: {args.out_dir}/summary.txt")


if __name__ == "__main__":
    main()
