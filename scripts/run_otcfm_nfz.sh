#!/bin/bash
#SBATCH --job-name=otcfm_nfz
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%x_%A_%a.out
#SBATCH --error=./slurm_output/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --array=0-2

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SEEDS=(21 42 63)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results/otcfm_nfz_seed${SEED}

python run_otcfm_milano.py \
    --arch v2 --hidden-dim 256 --n-layers 12 \
    --epochs 2000 --batch-size 1024 --lr 1e-3 \
    --n-cycles 1000 --pts-per-cycle 400 \
    --delta 0.05 --transport-steps 30 \
    --nfz-center 0.19 0.03 --nfz-radius 0.1 --nfz-margin 0.05 \
    --nfz-lambda 5.0 --nfz-steps 8 --nfz-reject-targets \
    --seed $SEED --out-dir $OUT
