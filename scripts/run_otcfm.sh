#!/bin/bash
#SBATCH --job-name=otcfm_train_b4096
#SBATCH --output=./slurm_output/%j_otcfm_train_b4096.out
#SBATCH --error=./slurm_output/%j_otcfm_train_b4096.err
#SBATCH --partition=gpu
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
OUT=results/otcfm_seed_b4096${SEED}

python -m experiments.milano.run_otcfm_milano \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 4096 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 60 \
    --reuse-model-from results/otcfm_seed_b4096${SEED}/model.pt \
    --seed $SEED --out-dir $OUT