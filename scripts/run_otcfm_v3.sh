#!/bin/bash
#SBATCH --job-name=otcfm_v3_train
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%A_%a_otcfm_v3_train.out
#SBATCH --error=./slurm_output/%A_%a_otcfm_v3_train.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --array=0-2

# Train OT-CFM with the v3 backbone (Random Fourier Features on the
# spatial input + FiLM-conditioned residual MLP) and EMA inference
# weights. Same dataset, same horizon, same all other knobs as the v1
# baseline -- only the velocity-net architecture changes, so the
# coverage delta directly attributes to the architecture choice.
#
# Tune RFF bandwidth sigma_x on {1, 2, 5, 10, 20} on a single seed
# first if v3 underperforms; sigma_x=5 is the sensible default.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SEEDS=(21 42 63)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results/otcfm_v3_seed${SEED}

echo "[otcfm_v3] seed=$SEED  out=$OUT"

python run_otcfm_milano.py \
    --arch v3 --hidden-dim 256 --n-layers 12 \
    --n-time-freqs 16 --n-spatial-freqs 16 --sigma-x 5.0 \
    --ema-decay 0.999 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --reuse-model-from results/otcfm_v3_seed${SEED}/model.pt \
    --seed $SEED --out-dir $OUT
