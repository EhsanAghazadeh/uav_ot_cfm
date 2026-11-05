#!/bin/bash
#SBATCH --job-name=toes_method
#SBATCH --output=./slurm_output/%j_toes_method.out
#SBATCH --error=./slurm_output/%j_toes_method.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --array=0-2

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SEEDS=(21 42 63)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results/toes_seed${SEED}

python -m experiments.milano.run_baseline_milano --method toes \
    --t-total 1280 --n-knots 25600 --n-basis 50 --u-max 0.05 \
    --max-iter 30000 --seed $SEED \
    --out-dir $OUT
