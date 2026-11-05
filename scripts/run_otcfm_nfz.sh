#!/bin/bash
#SBATCH --job-name=otcfm_nfz_l5_train
#SBATCH --output=./slurm_output/%j_otcfm_nfz_l5_seed_train.out
#SBATCH --error=./slurm_output/%j_otcfm_nfz_l5_seed_train.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --array=0-2

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SEEDS=(21 42 63)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results/otcfm_nfz_l5_seed${SEED}

python -m experiments.milano.run_otcfm_milano \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --nfz-center 0.19 0.03 --nfz-radius 0.1 --nfz-margin 0.05 \
    --nfz-lambda 5.0 --nfz-steps 8 --nfz-reject-targets \
    --reuse-model-from results/otcfm_nfz_l5_seed${SEED}/model.pt \
    --seed $SEED --out-dir $OUT
