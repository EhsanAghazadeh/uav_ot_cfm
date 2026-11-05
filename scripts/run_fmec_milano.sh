#!/usr/bin/env bash
# Flow Matching Ergodic Coverage baselines (Stein + Sinkhorn) on Milano.
# Requires: pip install -r requirements_fmec.txt (lqrax, ott-jax, tqdm).
#
# Usage (repo root):
#   bash scripts/run_fmec_milano.sh
# Override:
#   SEED=42 FMEC_TSTEPS=6400 bash scripts/run_fmec_milano.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"
FMEC_DT="${FMEC_DT:-0.05}"
FMEC_TSTEPS="${FMEC_TSTEPS:-200}"
STEIN_ITERS="${STEIN_ITERS:-500}"
STEIN_STEP="${STEIN_STEP:-0.01}"
SINK_ITERS="${SINK_ITERS:-200}"
SINK_STEP="${SINK_STEP:-0.005}"
GRID="${MILANO_GRID:-data/milano_grid_corrected.npy}"
OUT_STEIN="${OUT_STEIN:-results/fmec_stein_seed${SEED}}"
OUT_SINK="${OUT_SINK:-results/fmec_sinkhorn_seed${SEED}}"

mkdir -p "$OUT_STEIN" "$OUT_SINK"

python -m experiments.milano.run_baseline_milano_v1 --method fmec_stein \
  --milano-grid "$GRID" --seed "$SEED" \
  --fmec-dt "$FMEC_DT" --fmec-tsteps "$FMEC_TSTEPS" \
  --fmec-num-iters "$STEIN_ITERS" --fmec-step-size "$STEIN_STEP" \
  --fmec-grad-subsample 2000 --fmec-metric-subsample 2000 \
  --out-dir "$OUT_STEIN"

python -m experiments.milano.run_baseline_milano_v1 --method fmec_sinkhorn \
  --milano-grid "$GRID" --seed "$SEED" \
  --fmec-dt "$FMEC_DT" --fmec-tsteps "$FMEC_TSTEPS" \
  --fmec-num-iters "$SINK_ITERS" --fmec-step-size "$SINK_STEP" \
  --fmec-grad-subsample 2000 --fmec-metric-subsample 2000 \
  --out-dir "$OUT_SINK"

echo "Wrote $OUT_STEIN/trajectory.npz and $OUT_SINK/trajectory.npz"
