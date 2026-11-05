#!/bin/bash
#SBATCH --job-name=sweep_v3_sigma_le1e-2
#SBATCH --output=./slurm_output/%A_%a_sweep_v3_sigma_le1e-2.out
#SBATCH --error=./slurm_output/%A_%a_sweep_v3_sigma_le1e-2.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --array=0-4

# v3 RFF bandwidth sweep on a single seed (42).
# Trains 5 OT-CFM v3 models, varying only --sigma-x. The right value is
# the one whose CFM training loss settles lowest *and* whose Fourier
# ergodic on the eval target is best. Too small -> the network behaves
# like a low-frequency MLP and over-smooths Milano; too large -> the
# Random Fourier Features alias and training noise dominates.
#
# Picks the same hidden-dim / n-layers / EMA decay as run_otcfm_v3.sh
# so the only varying axis is sigma_x.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SIGMAS=(1.0 2.0 5.0 10.0 20.0)
SIGMA=${SIGMAS[$SLURM_ARRAY_TASK_ID]}

SEED=42
OUT=results/sweep_v3_sigma/sigma${SIGMA}_seed${SEED}_le1e-2
mkdir -p "$(dirname "$OUT")"
echo "[sweep_v3_sigma] task=$SLURM_ARRAY_TASK_ID  sigma_x=$SIGMA  out=$OUT"

python run_otcfm_milano.py \
    --arch v3 --hidden-dim 256 --n-layers 12 \
    --n-time-freqs 16 --n-spatial-freqs 16 --sigma-x $SIGMA \
    --ema-decay 0.999 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --energy-lambda 1e-2 --energy-chunk-size 64 --energy-steps 8 \
    --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    --seed $SEED --out-dir $OUT
