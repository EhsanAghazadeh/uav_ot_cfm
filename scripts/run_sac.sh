#!/bin/bash
#SBATCH --job-name=sac_train
#SBATCH --output=./slurm_output/%j_sac_train.out
#SBATCH --error=./slurm_output/%j_sac_train.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --array=0-2

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SEEDS=(21 42 63)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results/sac_seed${SEED}

python -m experiments.milano.run_baseline_milano --method sac \
    --num-points 25600 --sac-train-steps 500000 \
    --sac-grid-size 50 --sac-comm-range 0.18 --dt 0.01 \
    --seed $SEED --out-dir $OUT
