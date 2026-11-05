"""Appendix: turnaround-inclusive energy and mission-time accounting.

Recomputes the Table-1 Milano energy numbers under an INCLUSIVE convention,
same rules for every method, using the exact saved trajectories and the
paper protocol (domain 200 m, time scale 10, Zeng power, clip 0.01-100 m/s):

  E_zeng      speed-based Zeng propulsion energy (the paper's number)
  E_maneuver  kinetic maneuver surcharge: sum over consecutive steps of
              0.5 * m * ||dv_vec||^2 * 1.5 with m = 3.3 kg -- the energy
              model's own accel-jump formula (milano_baselines.py,
              energy_for_acceleration). Charges every velocity change,
              so turnaround reversals and reorientation kinks are paid
              by every method uniformly.
  E_dwell     base-station rest at hover power P_hover = P_0 + P_i for
              t_dwell seconds per latent cycle (OT-CFM variants only:
              baselines never return to a base, so they are not charged).
              Mission time is extended by the dwell.

  P_incl(t_dwell) = (E_zeng + E_maneuver + E_dwell) / (T + T_dwell)

Also prints the closed-form appendix splice-energy trade-off: acceleration
energy of the cubic-Hermite turnaround window vs the smooth-leg integral,
per half-cycle, for eps in {0.02, 0.05, 0.1} at delta in {0.01, 0.05}.
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fair_comparison import TrajectoryRecord
from milano_baselines import RotaryWingEnergyModel


def inclusive_energy(path, domain_size, time_scale, n_cycles, dwells):
    rec = TrajectoryRecord.from_npz(path)
    pos = np.asarray(rec.pos, dtype=np.float64)
    times = np.asarray(rec.times, dtype=np.float64) * time_scale
    dt = np.maximum(np.diff(times), 1e-12)
    step_m = np.linalg.norm(np.diff(pos, axis=0), axis=1) * domain_size
    v = step_m / dt

    em = RotaryWingEnergyModel()
    v_clip = np.clip(v, 0.01, 100.0)
    E_zeng = float((em.power(v_clip) * dt).sum())
    T = float(times[-1] - times[0])

    # velocity vectors per step, maneuver surcharge on consecutive changes
    v_vec = (np.diff(pos, axis=0) * domain_size) / dt[:, None]
    dv_vec = np.diff(v_vec, axis=0)
    dv = np.linalg.norm(dv_vec, axis=1)
    # cap dv at 2x the realistic speed cap so teleporting baselines are
    # charged a bounded reversal, not the square of a numerical artefact
    dv = np.clip(dv, 0.0, 60.0)
    E_man = float(np.sum(em.accel_coeff * dv ** 2 * 1.5))

    # reversal-events-only variant: charge only steps where the heading
    # flips by more than 90 degrees (dot < 0). This isolates turnaround /
    # reorientation impulses from ordinary path curvature.
    dots = np.sum(v_vec[:-1] * v_vec[1:], axis=1)
    rev = dots < 0.0
    E_rev = float(np.sum(em.accel_coeff * dv[rev] ** 2 * 1.5))
    n_rev = int(rev.sum())

    out = {"T_s": T, "E_zeng_kJ": E_zeng / 1e3, "E_man_kJ": E_man / 1e3,
           "E_rev_kJ": E_rev / 1e3, "n_rev": n_rev,
           "P_paper_W": E_zeng / T,
           "P_rev_incl_W": (E_zeng + E_rev) / T}
    for td in dwells:
        E_dw = (n_cycles * td * em.P_hover) if n_cycles else 0.0
        T_dw = (n_cycles * td) if n_cycles else 0.0
        out[f"P_incl_td{td}_W"] = (E_zeng + E_man + E_dw) / (T + T_dw)
    return out


def splice_table():
    print("\n== Appendix splice-energy trade-off (per half-cycle, latent) ==")
    print(f"{'delta':>6} {'eps':>5} {'E_smooth':>10} {'E_splice':>10} {'ratio':>7}")
    for delta in (0.01, 0.05):
        d2 = delta ** 2
        E_smooth = (1 - d2) ** 4 * (1 + d2) / (32 * delta ** 4)
        for eps in (0.02, 0.05, 0.1):
            # cubic Hermite on s in [1-eps, 1]:
            # r~(1-eps)=r(1-eps), r~'(1-eps)=r'(1-eps), r~(1)=1, r~'(1)=0
            s0 = 1 - eps
            r0 = np.sqrt(d2 + (1 - d2) * s0)
            dr0 = (1 - d2) / (2 * r0)
            # Hermite basis on tau in [0,1], s = s0 + eps*tau
            # r~(tau) = h00 r0 + h10 eps dr0 + h01 * 1 + h11 * 0
            tau = np.linspace(0, 1, 20001)
            h00dd = 12 * tau - 6
            h10dd = 6 * tau - 4
            h01dd = -12 * tau + 6
            rdd_tau = h00dd * r0 + h10dd * eps * dr0 + h01dd * 1.0
            rdd_s = rdd_tau / eps ** 2  # d2/ds2 = (1/eps^2) d2/dtau2
            E_spl = float(np.trapezoid(rdd_s ** 2, tau) * eps)
            print(f"{delta:>6} {eps:>5} {E_smooth:>10.4g} {E_spl:>10.4g} "
                  f"{E_spl / E_smooth:>7.2%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results_nfz_masked")
    p.add_argument("--domain-size", type=float, default=200.0)
    p.add_argument("--time-scale", type=float, default=10.0)
    p.add_argument("--dwells", nargs="+", type=float, default=[0, 5, 10, 30])
    p.add_argument("--out", default="results/appendix/turnaround/accounting.json")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    methods = ["toes", "nonuniform_erg", "opt_transport", "sac",
               "fmec_stein", "fmec_sinkhorn",
               "otcfm", "otcfm_nfz_l5", "otcfm_energy_lambda1e-2",
               "otcfm_energy_la1e-4", "otcfm_energy_lambda1e-3_nfz_l500",
               "otcfm_energy_la1e-1_nfz_l500_le1e-3"]
    seeds = [21, 42, 63]

    rows = {}
    for m in methods:
        per_seed = []
        for s in seeds:
            path = os.path.join(args.root, f"{m}_seed{s}", "trajectory.npz")
            if not os.path.isfile(path):
                continue
            meta_p = os.path.join(args.root, f"{m}_seed{s}", "plan_meta.json")
            n_cycles = 0
            if m.startswith("otcfm") and os.path.isfile(meta_p):
                n_cycles = int(json.load(open(meta_p)).get("n_cycles", 128))
            r = inclusive_energy(path, args.domain_size,
                                 args.time_scale, n_cycles, args.dwells)
            if any(np.isnan(v) for v in r.values() if isinstance(v, float)):
                print(f"  !! skipping NaN trajectory: {path}")
                continue
            per_seed.append(r)
        if not per_seed:
            continue
        rows[m] = {k: (float(np.mean([r[k] for r in per_seed])),
                       float(np.std([r[k] for r in per_seed])))
                   for k in per_seed[0]}

    hdr = (f"{'method':<38} {'P_paper':>8} {'n_rev':>6} {'P_rev':>8} {'E_man%':>7}"
           + "".join(f" {'P_td' + str(int(td)):>8}" for td in args.dwells))
    print(hdr)
    for m, r in rows.items():
        man_pct = 100 * r["E_man_kJ"][0] / max(r["E_zeng_kJ"][0], 1e-9)
        line = (f"{m:<38} {r['P_paper_W'][0]:>8.3g} {r['n_rev'][0]:>6.0f} "
                f"{r['P_rev_incl_W'][0]:>8.3g} {man_pct:>6.2f}%")
        for td in args.dwells:
            line += f" {r[f'P_incl_td{td}_W'][0]:>8.3g}"
        print(line)

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    print(f"wrote {args.out}")
    splice_table()


if __name__ == "__main__":
    main()
