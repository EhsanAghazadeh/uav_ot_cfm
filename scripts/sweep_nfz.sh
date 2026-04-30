#!/bin/bash
#SBATCH --job-name=sweep_nfz
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%A_%a_sweep_nfz.out
#SBATCH --error=./slurm_output/%A_%a_sweep_nfz.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --array=0-4

# Combo 1:  L_total = L_cfm + lambda_nfz * R_nfz
# Sweep lambda_nfz across 5 log-spaced values around the prior default (5.0).

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

LAMBDAS=(0.5 5 50 500 5000)
LNFZ=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

SEED=42
OUT=results/sweep_nfz/lnfz${LNFZ}_seed${SEED}
mkdir -p "$(dirname "$OUT")"
echo "[sweep_nfz] task=$SLURM_ARRAY_TASK_ID  lambda_nfz=$LNFZ  out=$OUT"

python run_otcfm_milano.py \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --nfz-center 0.19 0.03 --nfz-radius 0.1 --nfz-margin 0.05 \
    --nfz-lambda $LNFZ --nfz-steps 8 --nfz-reject-targets \
    --seed $SEED --out-dir $OUT
