"""Appendix: closed-loop tracking-error evaluation.

Rolls every method's reference trajectory through a PD-tracked double
integrator in physical units (paper protocol: domain 200 m, time scale 10,
dt 0.1 s) and reports how coverage and energy change when the flown path is
the tracked one instead of the reference:

    x_ddot = kp (x_ref - x) + kd (v_ref - v),   |x_ddot| <= a_max

with critically damped gains kp = wn^2, kd = 2 wn and a 10 m/s^2
acceleration cap (~1 g horizontal, typical quadrotor). The reference is
the stored trajectory unmodified, so rho_ref reproduces Table 1 exactly
and the tracking error itself exposes how much of each method's reported
coverage survives closed-loop execution. Methods whose references demand
infeasible accelerations (teleporting baselines) show large errors and
coverage collapse; that is the honest realizability statement.

Reported per method (mean over seeds):
    rho_ref / rho_trk    coverage correlation of reference vs tracked path
    P_ref / P_trk        Zeng mean power of reference vs tracked path
    rms_err_m            RMS tracking error in metres
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fair_comparison import TrajectoryRecord
from milano_baselines import RotaryWingEnergyModel
from target_density import load_milano_target
from eval_fair import _build_basis, _phik_from_target, coverage_metrics


def track(pos_m, dt, wn, a_max):
    """PD-tracked double integrator following the reference pos_m (T,2)."""
    kp, kd = wn * wn, 2.0 * wn
    v_ref = np.gradient(pos_m, axis=0) / dt
    x = pos_m[0].copy()
    v = v_ref[0].copy()  # start matched to the reference (launch at speed)
    out = np.empty_like(pos_m)
    out[0] = x
    n_sat = 0
    for i in range(1, len(pos_m)):
        a = kp * (pos_m[i - 1] - x) + kd * (v_ref[i - 1] - v)
        na = np.linalg.norm(a)
        if na > a_max:
            a *= a_max / na
            n_sat += 1
        v = v + a * dt
        x = x + v * dt
        out[i] = x
    return out, n_sat / max(len(pos_m) - 1, 1)


def zeng_power(pos_m, dt, em, v_clip=(0.01, 100.0)):
    step = np.linalg.norm(np.diff(pos_m, axis=0), axis=1)
    v = np.clip(step / dt, v_clip[0], v_clip[1])
    P = em.power(v)
    return float((P * dt).sum() / (dt * len(v)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results_nfz_masked")
    p.add_argument("--domain-size", type=float, default=200.0)
    p.add_argument("--time-scale", type=float, default=10.0)
    p.add_argument("--wn", type=float, default=2.0,
                   help="tracker natural frequency (rad/s)")
    p.add_argument("--a-max", type=float, default=10.0)
    p.add_argument("--nfz", nargs=3, type=float, action="append", default=[])
    p.add_argument("--out", default="results/appendix/tracking/tracking_1d.json")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    mask_discs = [((cx, cy), r) for cx, cy, r in args.nfz]
    t_target = load_milano_target("data/milano_grid_corrected.npy",
                                  smooth_sigma=1.5, nfz_discs=mask_discs)
    pmf = t_target["pmf"]
    K, hk, lam = _build_basis(8)
    phik = _phik_from_target(t_target["grid"], K, hk)
    em = RotaryWingEnergyModel()

    methods = ["toes", "nonuniform_erg", "opt_transport", "sac",
               "fmec_stein", "fmec_sinkhorn",
               "otcfm", "otcfm_nfz_l5", "otcfm_energy_lambda1e-2",
               "otcfm_energy_la1e-4", "otcfm_energy_lambda1e-3_nfz_l500",
               "otcfm_energy_la1e-1_nfz_l500_le1e-3"]
    seeds = [21, 42, 63]

    rows = {}
    print(f"{'method':<38} {'rho_ref':>8} {'rho_trk':>8} {'P_ref':>9} "
          f"{'P_trk':>9} {'rms_err':>8}")
    for m in methods:
        per = []
        for s in seeds:
            path = os.path.join(args.root, f"{m}_seed{s}", "trajectory.npz")
            if not os.path.isfile(path):
                continue
            rec = TrajectoryRecord.from_npz(path)
            pos = np.asarray(rec.pos, dtype=np.float64)
            if np.isnan(pos).any():
                print(f"  !! skipping NaN trajectory: {path}")
                continue
            times = np.asarray(rec.times, dtype=np.float64) * args.time_scale
            dt = float(np.median(np.diff(times)))
            pos_ref = pos * args.domain_size

            trk, frac_sat = track(pos_ref, dt, args.wn, args.a_max)
            err = np.linalg.norm(trk - pos_ref, axis=1)

            xlim = (rec.domain[0], rec.domain[1])
            ylim = (rec.domain[2], rec.domain[3])
            to_unit = lambda q: np.stack([
                (q[:, 0] / args.domain_size - xlim[0]) / (xlim[1] - xlim[0]),
                (q[:, 1] / args.domain_size - ylim[0]) / (ylim[1] - ylim[0]),
            ], axis=1)
            cov_ref = coverage_metrics(to_unit(pos_ref), dt, pmf, K, hk, lam,
                                       phik, sigma=1.5)
            cov_trk = coverage_metrics(to_unit(trk), dt, pmf, K, hk, lam,
                                       phik, sigma=1.5)
            per.append({
                "rho_ref": cov_ref["density_corr"],
                "rho_trk": cov_trk["density_corr"],
                "l1_ref": cov_ref["l1"], "l1_trk": cov_trk["l1"],
                "P_ref": zeng_power(pos_ref, dt, em),
                "P_trk": zeng_power(trk, dt, em),
                "rms_err_m": float(np.sqrt((err ** 2).mean())),
                "max_err_m": float(err.max()),
                "frac_sat": frac_sat,
            })
        if not per:
            continue
        agg = {k: (float(np.mean([r[k] for r in per])),
                   float(np.std([r[k] for r in per]))) for k in per[0]}
        rows[m] = agg
        print(f"{m:<38} {agg['rho_ref'][0]:>8.3f} {agg['rho_trk'][0]:>8.3f} "
              f"{agg['P_ref'][0]:>9.3g} {agg['P_trk'][0]:>9.3g} "
              f"{agg['rms_err_m'][0]:>8.2f}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
