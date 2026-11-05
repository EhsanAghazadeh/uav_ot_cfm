#!/usr/bin/env bash
# Prefix-stop evaluation for FMEC trajectories only (does not modify run_eval_all.sh).
#
# Usage (after sweep_fmec_milano or run_fmec_milano):
#   bash scripts/run_eval_fmec_addon.sh

set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate 2>/dev/null || true

SEEDS=(21 42 63)
METHODS=(fmec_stein fmec_sinkhorn)

for m in "${METHODS[@]}"; do
  for s in "${SEEDS[@]}"; do
    TRAJ="results/${m}_seed${s}/trajectory.npz"
    OUT="results/${m}_seed${s}/curves"
    if [[ -f "$TRAJ" ]]; then
      python -m evaluation.eval_prefix_stop --traj "$TRAJ" \
        --milano-grid data/milano_grid_corrected.npy \
        --smooth-sigma 1.5 --out-dir "$OUT"
    else
      echo "skip missing $TRAJ"
    fi
  done
done

echo "Done. Merge curves into comparison plots manually or extend plot_curves.py ARGS."
