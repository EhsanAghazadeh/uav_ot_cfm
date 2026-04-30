#!/bin/bash
#SBATCH --job-name=sweep_nfz_energy
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%A_%a_sweep_nfz_energy.out
#SBATCH --error=./slurm_output/%A_%a_sweep_nfz_energy.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --array=0-4

# Combo 4:  L_total = L_cfm + lambda_nfz * R_nfz + lambda_energy * R_energy
# Paired sweep — i-th task uses i-th value from each axis (5 settings, not 25).

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

LAMBDAS_NFZ=(0.5 5 50 500 5000)
LAMBDAS_E=(1e-6 1e-5 1e-4 1e-3 1e-2)
LNFZ=${LAMBDAS_NFZ[$SLURM_ARRAY_TASK_ID]}
LE=${LAMBDAS_E[$SLURM_ARRAY_TASK_ID]}

SEED=42
OUT=results/sweep_nfz_energy/lnfz${LNFZ}_le${LE}_seed${SEED}
mkdir -p "$(dirname "$OUT")"
echo "[sweep_nfz_energy] task=$SLURM_ARRAY_TASK_ID  lambda_nfz=$LNFZ  lambda_energy=$LE  out=$OUT"

python run_otcfm_milano.py \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --nfz-center 0.19 0.03 --nfz-radius 0.1 --nfz-margin 0.05 \
    --nfz-lambda $LNFZ --nfz-steps 8 --nfz-reject-targets \
    --energy-lambda $LE --energy-chunk-size 64 --energy-steps 8 \
    --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    --domain-size 8.0 \
    --seed $SEED --out-dir $OUT
