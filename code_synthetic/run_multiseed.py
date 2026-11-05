"""Multi-seed runner for the synthetic experiments (paper Section 6).

Modes:
  exp1 --seed S     Experiment 1 (Gaussian mixture), training seed S
  exp2 --seed S     Experiment 2 (binary 3:1 density), training seed S
  exp3 --seed S     Experiment 3 (no-fly zone at (0.5, -0.5), radius 0.2):
                    unconstrained, +NFZ, +NFZ+acceleration variants

Only the training seed varies; evaluation seeds are fixed. Outputs go to
results/synthetic_runs/<mode>_seed<S>/metrics.json.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "results", "synthetic_runs")


def scalars(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)):
            out[k] = v
        elif isinstance(v, np.generic):
            out[k] = v.item()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["exp1", "exp2", "exp3"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.mode == "exp1":
        import experiment_01_gaussian_mixture as e1
        out = os.path.join(OUT_ROOT, f"exp1_seed{args.seed}")
        os.makedirs(out, exist_ok=True)
        model, losses = e1.train_cfm(seed=args.seed)
        torch.save(model.state_dict(), os.path.join(out, "model.pt"))
        m = e1.evaluate(model, out)
        m = scalars(m)
        m["cfm_loss_final"] = float(np.mean(losses[-50:]))
        m["seed"] = args.seed
    elif args.mode == "exp2":
        import experiment_02_binary_density as e2
        out = os.path.join(OUT_ROOT, f"exp2_seed{args.seed}")
        os.makedirs(out, exist_ok=True)
        model, losses = e2.train_cfm(seed=args.seed)
        torch.save(model.state_dict(), os.path.join(out, "model.pt"))
        m = e2.evaluate(model, out)
        m = scalars(m)
        m["cfm_loss_final"] = float(np.mean(losses[-50:]))
        m["seed"] = args.seed
    else:
        from experiment_03_revised_v2 import (train_constrained,
                                              evaluate_model, NFZ_CENTER)
        from experiment_03_revised_v2 import NFZ_RADIUS as R02
        out = os.path.join(OUT_ROOT, f"exp3_seed{args.seed}")
        os.makedirs(out, exist_ok=True)
        m = {"nfz_center": list(map(float, NFZ_CENTER)), "nfz_radius": R02}
        for tag, (ln, la) in {"unc": (0.0, 0.0), "nfz": (50.0, 0.0),
                              "nfz_acc": (50.0, 0.1)}.items():
            kw = {}
            if ln > 0 or la > 0:
                kw = dict(nfz_center=NFZ_CENTER, nfz_radius=R02)
            model, _ = train_constrained(n_epochs=500, batch_size=1024,
                                         seed=args.seed, lambda_nfz=ln,
                                         lambda_acc=la, label=tag, **kw)
            torch.save(model.state_dict(),
                       os.path.join(out, f"model_{tag}.pt"))
            met = evaluate_model(model, tag, out, nfz_center=NFZ_CENTER,
                                 nfz_radius=R02)
            m[tag] = scalars(met)

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2, default=float)
    print(json.dumps(m, indent=1, default=float))


if __name__ == "__main__":
    main()
