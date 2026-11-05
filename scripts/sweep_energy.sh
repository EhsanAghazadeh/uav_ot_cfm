#!/bin/bash
#SBATCH --job-name=sweep_energy_1_10_100_1000
#SBATCH --output=./slurm_output/%A_%a_sweep_energy_1_10_100_1000.out
#SBATCH --error=./slurm_output/%A_%a_sweep_energy_1_10_100_1000.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --array=0-3

# Combo 2:  L_total = L_cfm + lambda_energy * R_energy   (Zeng power)
# Sweep lambda_energy across 5 log-spaced values around the prior default (1e-4).

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

LAMBDAS=(1 10 100 1000)
LE=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

SEED=42
OUT=results/sweep_energy/le${LE}_seed${SEED}
mkdir -p "$(dirname "$OUT")"
echo "[sweep_energy] task=$SLURM_ARRAY_TASK_ID  lambda_energy=$LE  out=$OUT"

python -m experiments.milano.run_otcfm_milano \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --energy-lambda $LE --energy-chunk-size 64 --energy-steps 8 \
    --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    --domain-size 8.0 \
    --seed $SEED --out-dir $OUT
