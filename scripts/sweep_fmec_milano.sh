#!/usr/bin/env bash
#SBATCH --job-name=sweep_fmec_milano
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%A_%a_sweep_fmec_milano.out
#SBATCH --error=./slurm_output/%A_%a_sweep_fmec_milano.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#
# FMEC Stein + Sinkhorn × seeds (2×3 = 6 tasks). Requires requirements_fmec.txt.
#
# Submit: sbatch scripts/sweep_fmec_milano.sh
# NFZ masked target: sbatch --export=ALL,MASK_NFZ=1 scripts/sweep_fmec_milano.sh

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

VARIANTS=(fmec_stein fmec_sinkhorn)
SEEDS=(21 42 63)
GRID="${MILANO_GRID:-data/milano_grid_corrected.npy}"

i=${SLURM_ARRAY_TASK_ID:-0}
nv=${#VARIANTS[@]}
V=${VARIANTS[$((i % nv))]}
S=${SEEDS[$((i / nv))]}

FMEC_DT="${FMEC_DT:-0.05}"
FMEC_TSTEPS="${FMEC_TSTEPS:-200}"
STEIN_ITERS="${STEIN_ITERS:-500}"
STEIN_STEP="${STEIN_STEP:-0.01}"
SINK_ITERS="${SINK_ITERS:-200}"
SINK_STEP="${SINK_STEP:-0.005}"

OUT="results/${V}_seed${S}"
mkdir -p "$OUT"

NFZ_FLAGS=()
if [[ "${MASK_NFZ:-0}" == "1" ]]; then
  NFZ_FLAGS=(--mask-nfz-target --nfz-center 0.19 0.03 --nfz-radius 0.10)
fi

echo "[sweep_fmec] task=$i method=$V seed=$S out=$OUT"

if [[ "$V" == "fmec_stein" ]]; then
  python run_baseline_milano_v1.py --method fmec_stein \
    --milano-grid "$GRID" --seed "$S" \
    --fmec-dt "$FMEC_DT" --fmec-tsteps "$FMEC_TSTEPS" \
    --fmec-num-iters "$STEIN_ITERS" --fmec-step-size "$STEIN_STEP" \
    --fmec-grad-subsample "${FMEC_GRAD_SUB:-2000}" \
    --fmec-metric-subsample "${FMEC_METRIC_SUB:-2000}" \
    "${NFZ_FLAGS[@]}" \
    --out-dir "$OUT"
else
  python run_baseline_milano_v1.py --method fmec_sinkhorn \
    --milano-grid "$GRID" --seed "$S" \
    --fmec-dt "$FMEC_DT" --fmec-tsteps "$FMEC_TSTEPS" \
    --fmec-num-iters "$SINK_ITERS" --fmec-step-size "$SINK_STEP" \
    --fmec-grad-subsample "${FMEC_GRAD_SUB:-2000}" \
    --fmec-metric-subsample "${FMEC_METRIC_SUB:-2000}" \
    "${NFZ_FLAGS[@]}" \
    --out-dir "$OUT"
fi
