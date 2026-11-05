"""Appendix: per-cycle NFZ rejection probability p_v for offline expurgation.

Definition: a latent cycle (200 consecutive samples of the
saved trajectory) counts as violating if ANY of its samples lies inside
an NFZ disc, and p_v is the fraction of violating cycles, averaged over
the 3 training seeds. This is a per-cycle probability, distinct from
Table 1's "frac inside", which is the percentage of individual trajectory
POINTS inside a disc. The numerical closeness of the two for the +NFZ
single-disc case (p_v = 0.011 vs frac inside = 0.011%) is a coincidence
of the cycle length: violating cycles contain only 2-3 violating points
out of 200, so 1% of cycles maps to about 0.01% of points.
"""
import json, os
import numpy as np

CONFIGS = {
    "1d": ("results_nfz_masked", [((0.19, 0.03), 0.10)]),
    "md": ("results_nfz_masked_multi_disc",
           [((0.5, 0.0), 0.2), ((-0.5, 0.0), 0.2),
            ((0.0, 0.5), 0.2), ((0.0, -0.5), 0.2)]),
}
PTS_PER_CYCLE = 200


def pv(root, method, discs):
    vals = []
    for s in (21, 42, 63):
        d = np.load(f"{root}/{method}_seed{s}/trajectory.npz")
        pos = d["pos"]
        if np.isnan(pos).any():
            continue
        n = pos.shape[0] // PTS_PER_CYCLE
        p = pos[:n * PTS_PER_CYCLE].reshape(n, PTS_PER_CYCLE, 2)
        inside = np.zeros(n, dtype=bool)
        for (cx, cy), r in discs:
            inside |= (np.linalg.norm(p - np.array([cx, cy]), axis=2)
                       < r).any(axis=1)
        vals.append(float(inside.mean()))
    return {"per_seed": vals, "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "expected_draws_per_accepted_cycle":
                float(1.0 / (1.0 - np.mean(vals)))}


if __name__ == "__main__":
    out = {}
    for tag, (root, discs) in CONFIGS.items():
        for method in ("otcfm", "otcfm_nfz_l5"):
            r = pv(root, method, discs)
            out[f"{method}_{tag}"] = r
            print(f"{method} {tag}: p_v = {r['mean']:.4f} ± {r['std']:.4f}"
                  f"  overhead = {r['expected_draws_per_accepted_cycle']:.3f}")
    os.makedirs("results/appendix/expurgation", exist_ok=True)
    with open("results/appendix/expurgation/pv.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/appendix/expurgation/pv.json")
