#!/bin/bash
#SBATCH --job-name=baselines_nfz_masked
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%A_%a_baselines_nfz_masked.out
#SBATCH --error=./slurm_output/%A_%a_baselines_nfz_masked.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --array=0-11

# NFZ-condition baselines on the *masked* Milano target
# (zero density inside the disc, renormalised). Apples-to-apples with
# OT-CFM +NFZ runs that also train on the masked target.
#
# Per-method CLI args mirror the unmasked baseline scripts verbatim
# (run_toes.sh / run_nonuniform_erg.sh / run_opt_transport.sh / run_sac.sh)
# so trajectory length, horizon, and method hyperparameters are identical
# — only the target distribution is different.
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

NFZ_FLAGS="--nfz-center 0.19 0.03 --nfz-radius 0.10 --mask-nfz-target"

echo "[baselines_nfz_masked] task=$i  method=$M  seed=$S  out=$OUT"

case "$M" in
    toes)
        python run_baseline_milano.py --method toes \
            --t-total 1280 --n-knots 25600 --n-basis 50 --u-max 0.05 \
            --max-iter 30000 \
            $NFZ_FLAGS \
            --seed $S --out-dir $OUT
        ;;
    nonuniform_erg)
        python run_baseline_milano.py --method nonuniform_erg \
            --num-intervals 64 --points-per-interval 200 --dt 0.1 \
            $NFZ_FLAGS \
            --seed $S --out-dir $OUT
        ;;
    opt_transport)
        python run_baseline_milano.py --method opt_transport \
            --num-points 25600 --dt 0.1 \
            $NFZ_FLAGS \
            --seed $S --out-dir $OUT
        ;;
    sac)
        python run_baseline_milano.py --method sac \
            --num-points 25600 --sac-train-steps 500000 \
            --sac-grid-size 50 --sac-comm-range 0.18 --dt 0.1 \
            $NFZ_FLAGS \
            --seed $S --out-dir $OUT
        ;;
    *)
        echo "unknown method $M"; exit 1
        ;;
esac
