#!/bin/bash
#SBATCH --job-name=sweep_energy
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%A_%a_sweep_energy.out
#SBATCH --error=./slurm_output/%A_%a_sweep_energy.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --array=0-4

# Combo 2:  L_total = L_cfm + lambda_energy * R_energy   (Zeng power)
# Sweep lambda_energy across 5 log-spaced values around the prior default (1e-4).

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

LAMBDAS=(1e-6 1e-5 1e-4 1e-3 1e-2)
LE=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

SEED=42
OUT=results/sweep_energy/le${LE}_seed${SEED}
mkdir -p "$(dirname "$OUT")"
echo "[sweep_energy] task=$SLURM_ARRAY_TASK_ID  lambda_energy=$LE  out=$OUT"

python run_otcfm_milano.py \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --energy-lambda $LE --energy-chunk-size 64 --energy-steps 8 \
    --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    --domain-size 8.0 \
    --seed $SEED --out-dir $OUT
