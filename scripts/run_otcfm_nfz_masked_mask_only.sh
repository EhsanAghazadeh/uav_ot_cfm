#!/bin/bash
#SBATCH --job-name=otcfm_nfz_mask_only
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%A_%a_otcfm_nfz_mask_only.out
#SBATCH --error=./slurm_output/%A_%a_otcfm_nfz_mask_only.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --array=0-2

# Ablation (a): OT-CFM trained on the NFZ-masked target *without* the
# soft penalty. Tests whether the density-level encoding alone keeps
# samples out of the disc, i.e. whether the soft penalty is necessary.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SEEDS=(21 42 63)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results_nfz_masked/otcfm_mask_only_seed${SEED}

echo "[otcfm_nfz_mask_only] seed=$SEED  out=$OUT"

python run_otcfm_milano.py \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --nfz-center 0.19 0.03 --nfz-radius 0.10 --mask-nfz-target \
    --seed $SEED --out-dir $OUT
