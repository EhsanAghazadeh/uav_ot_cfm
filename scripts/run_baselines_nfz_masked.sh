#!/bin/bash
#SBATCH --job-name=baselines_nfz_masked
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%A_%a_baselines_nfz_masked.out
#SBATCH --error=./slurm_output/%A_%a_baselines_nfz_masked.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=0-11

# NFZ-condition baselines on the *masked* Milano target
# (zero density inside the disc, renormalised). Apples-to-apples with
# OT-CFM +NFZ runs that also train on the masked target.
#
# 4 methods x 3 seeds = 12 array tasks.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

METHODS=(toes nonuniform_erg opt_transport sac)
SEEDS=(21 42 63)

i=$SLURM_ARRAY_TASK_ID
M=${METHODS[$((i / 3))]}
S=${SEEDS[$((i % 3))]}
OUT=results_nfz_masked/${M}_seed${S}

echo "[baselines_nfz_masked] task=$i  method=$M  seed=$S  out=$OUT"

python run_baseline_milano.py --method $M --seed $S \
    --milano-grid data/milano_grid_corrected.npy --smooth-sigma 1.5 \
    --nfz-center 0.19 0.03 --nfz-radius 0.10 --mask-nfz-target \
    --out-dir $OUT
