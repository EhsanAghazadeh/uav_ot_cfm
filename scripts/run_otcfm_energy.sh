#!/bin/bash
#SBATCH --job-name=otcfm_energy_la1e-2_train_rescaled_20
#SBATCH --output=./slurm_output/%j_otcfm_energy_la1e-2_train_rescaled_20.out
#SBATCH --error=./slurm_output/%j_otcfm_energy_la1e-2_train_rescaled_20.err
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
OUT=results/otcfm_energy_la1e-2_seed_rescaled_20${SEED}

python -m experiments.milano.run_otcfm_milano \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 2000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --energy-lambda 1e-2 --energy-chunk-size 64 --energy-steps 8 \
    --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    --domain-size 20.0 \
    --reuse-model-from results/otcfm_energy_la1e-2_seed_rescaled_20${SEED}/model.pt \
    --seed $SEED --out-dir $OUT
