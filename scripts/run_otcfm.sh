#!/bin/bash
#SBATCH --job-name=otcfm_v2
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%x_%A_%a.out
#SBATCH --error=./slurm_output/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --array=0-2

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SEEDS=(21 42 63)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results/otcfm_seed${SEED}

python run_otcfm_milano.py \
    --arch v2 --hidden-dim 512 --n-layers 8 \
    --epochs 3000 --batch-size 2048 --lr 1e-3 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --seed $SEED --out-dir $OUT
