#!/bin/bash
#SBATCH --job-name=otcfm_nuerg
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%x_%A_%a.out
#SBATCH --error=./slurm_output/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --array=0-2

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SEEDS=(21 42 63)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results/nonuniform_erg_seed${SEED}

python run_baseline_milano.py --method nonuniform_erg \
    --num-intervals 64 --points-per-interval 200 --dt 0.1 \
    --seed $SEED --out-dir $OUT
