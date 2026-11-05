"""No-fly-zone violation evaluator.

Defines a circular no-fly zone (center, radius) inside the Milano domain and
reports per-method violation metrics:

  n_inside        number of trajectory points strictly inside the disc
  frac_inside     n_inside / N
  dwell_time_s    n_inside * dt
  path_inside_m   total path length traversed inside the disc (segment-clip)
  n_incursions    number of contiguous "inside" intervals (entries into zone)
  max_depth       max(radius - ‖p - c‖) over inside points (normalised units)

By default the disc is centred at the smoothed-density peak (≈ (0.19, 0.03))
with radius 0.1, so any method that targets the high-density cluster has to
detour around it.

Usage:
  python -m evaluation.eval_nofly \
      --traj "TOES:results/toes_milano_long_4x/trajectory.npz" \
      --traj "OT-CFM:results/otcfm_milano_4x_v2/trajectory.npz" \
      --center 0.19 0.03 --radius 0.1 \
      --out results/comparison_baselines_4x_v2/nofly.json
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
from ergodic_cfm.fair_comparison import TrajectoryRecord


def _segment_length_inside(p0, p1, c, r):
    """Length of the segment p0→p1 that lies inside the closed disc B(c, r).

    Solves ‖p0 + t(p1-p0) - c‖² = r² for t∈[0,1] and integrates the
    inside-portion length. Closed-form quadratic.
    """
    d = p1 - p0
    f = p0 - c
    a = d @ d
    if a < 1e-24:
        return 0.0 if (f @ f) > r * r else 0.0
    b = 2.0 * (f @ d)
    cc = f @ f - r * r
    disc = b * b - 4.0 * a * cc
    if disc <= 0:
        # no intersection unless both endpoints inside
        if (f @ f) <= r * r and ((p1 - c) @ (p1 - c)) <= r * r:
            return float(np.linalg.norm(d))
        return 0.0
    sd = np.sqrt(disc)
    t1 = (-b - sd) / (2 * a)
    t2 = (-b + sd) / (2 * a)
    lo = max(0.0, min(t1, t2))
    hi = min(1.0, max(t1, t2))
    if hi <= lo:
        return 0.0
    return float((hi - lo) * np.linalg.norm(d))


def violations(pos, dt, discs, scale_to_meters=None):
    """Compute violation metrics for a single trajectory against a union of discs.

    Args:
        pos: (N, 2) positions in normalised [-1, 1]^2 coords.
        dt: timestep (s).
        discs: list of (center, radius) pairs, e.g. [((0.19, 0.03), 0.10), ...].
            A point is "inside" if it is inside ANY disc.
        scale_to_meters: optional float — multiply path lengths by this so the
            "path_inside" output is in metres. If None, output stays normalised.
    """
    pos = np.asarray(pos, dtype=np.float64)

    # Build combined inside mask (union of all discs)
    inside = np.zeros(len(pos), dtype=bool)
    for center, radius in discs:
        c = np.asarray(center, dtype=np.float64)
        inside |= np.linalg.norm(pos - c, axis=1) <= float(radius)

    n_inside = int(inside.sum())
    frac_inside = n_inside / max(len(pos), 1)
    dwell_s = n_inside * float(dt)

    # incursions = transitions outside→inside
    if len(inside) > 0:
        prev_out = np.concatenate([[True], ~inside[:-1]])
        n_incursions = int(((prev_out) & (inside)).sum())
    else:
        n_incursions = 0

    # max penetration depth: max over all discs of (r - dist(p, c)) for inside points
    max_depth = 0.0
    if n_inside:
        depths = np.zeros(len(pos), dtype=np.float64)
        for center, radius in discs:
            c = np.asarray(center, dtype=np.float64)
            d = float(radius) - np.linalg.norm(pos - c, axis=1)
            depths = np.maximum(depths, d)
        max_depth = float(depths[inside].max())

    # path length inside (segment-clip for partial entries, union of all discs)
    L_norm = 0.0
    for i in range(len(pos) - 1):
        # a segment is inside if it overlaps any disc; accumulate max contribution
        seg_len = 0.0
        for center, radius in discs:
            c = np.asarray(center, dtype=np.float64)
            seg_len = max(seg_len,
                          _segment_length_inside(pos[i], pos[i + 1], c, float(radius)))
        L_norm += seg_len
    L_out = L_norm * (scale_to_meters if scale_to_meters else 1.0)

    return {
        "n_inside": n_inside,
        "frac_inside": frac_inside,
        "dwell_time_s": dwell_s,
        "n_incursions": n_incursions,
        "max_depth": max_depth,
        "path_inside_m" if scale_to_meters else "path_inside_norm": L_out,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", action="append", required=True,
                  help="label:path/to/trajectory.npz (repeatable)")
    p.add_argument("--nfz", action="append", nargs=3, type=float,
                  metavar=("CX", "CY", "R"), default=[],
                  help="NFZ disc: cx cy radius (repeatable for multiple discs). "
                       "E.g. --nfz 0.19 0.03 0.10 --nfz 0.3 0.3 0.15. "
                       "When provided, overrides --center / --radius.")
    p.add_argument("--center", nargs=2, type=float, default=[0.19, 0.03],
                  help="No-fly disc centre in normalised coords (default: density peak). "
                       "Used only when --nfz is not given.")
    p.add_argument("--radius", type=float, default=0.1,
                  help="No-fly disc radius in normalised coords (default: 0.1). "
                       "Used only when --nfz is not given.")
    p.add_argument("--meters-per-unit", type=float, default=None,
                  help="Optional scale factor; if given, path_inside is reported in metres.")
    p.add_argument("--out", default=None,
                  help="Optional JSON path to dump full results.")
    args = p.parse_args()

    # Build disc list: --nfz takes priority; fall back to legacy --center / --radius
    if args.nfz:
        nfz_discs = [((cx, cy), r) for cx, cy, r in args.nfz]
    else:
        nfz_discs = [(tuple(args.center), float(args.radius))]

    rows = []
    disc_desc = "  ".join(
        f"c=({c[0]:.3f},{c[1]:.3f}) r={r:.3f}" for c, r in nfz_discs)
    print(f"\nNo-fly zone ({len(nfz_discs)} disc(s)): {disc_desc}\n")
    hdr_path = "path_inside_m" if args.meters_per_unit else "path_inside_norm"
    print(f"{'method':<18} {'N':>7} {'n_in':>7} {'frac_in':>9} "
          f"{'dwell_s':>9} {'incurs':>7} {'maxdepth':>9} {hdr_path:>14}")
    print("-" * 90)

    for spec in args.traj:
        if ":" not in spec:
            raise SystemExit(f"--traj must be label:path (got {spec})")
        label, path = spec.split(":", 1)
        rec = TrajectoryRecord.from_npz(path)
        dt = float(rec.meta.get("dt_s", rec.times[1] - rec.times[0]))
        v = violations(rec.pos, dt, nfz_discs,
                       scale_to_meters=args.meters_per_unit)
        v["method"] = label
        v["N"] = int(len(rec.pos))
        v["dt_s"] = dt
        rows.append(v)
        path_val = v.get("path_inside_m", v.get("path_inside_norm", 0.0))
        print(f"{label:<18} {v['N']:>7d} {v['n_inside']:>7d} "
              f"{v['frac_inside']:>9.4f} {v['dwell_time_s']:>9.2f} "
              f"{v['n_incursions']:>7d} {v['max_depth']:>9.4f} "
              f"{path_val:>14.4f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "nfz_discs": [[[float(c[0]), float(c[1])], float(r)]
                               for c, r in nfz_discs],
                "meters_per_unit": args.meters_per_unit,
                "results": rows,
            }, f, indent=2)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
