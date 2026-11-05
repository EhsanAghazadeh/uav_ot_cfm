#!/bin/bash
#SBATCH --job-name=baselines_nfz_masked
#SBATCH --output=./slurm_output/%A_%a_baselines_nfz_masked.out
#SBATCH --error=./slurm_output/%A_%a_baselines_nfz_masked.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --array=0-17

# NFZ-condition baselines on the *masked* Milano target
# (zero density inside the disc, renormalised). Apples-to-apples with
# OT-CFM +NFZ runs that also train on the masked target.
#
# Per-method CLI args mirror the unmasked baseline scripts verbatim
# (run_toes.sh / run_nonuniform_erg.sh / run_opt_transport.sh / run_sac.sh)
# so trajectory length, horizon, and method hyperparameters are identical
# — only the target distribution is different.
#
# 6 methods x 3 seeds = 18 array tasks.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

METHODS=(toes nonuniform_erg opt_transport sac fmec_stein fmec_sinkhorn)
SEEDS=(21 42 63)

i=$SLURM_ARRAY_TASK_ID
M=${METHODS[$((i / 3))]}
S=${SEEDS[$((i % 3))]}
# NFZ configuration of Table 1, chosen with CONFIG (default: 1d):
#   CONFIG=1d sbatch scripts/run_baselines_nfz_masked.sh   # single disc
#   CONFIG=md sbatch scripts/run_baselines_nfz_masked.sh   # four discs
CONFIG=${CONFIG:-1d}
if [ "$CONFIG" = "1d" ]; then
    ROOT=results_nfz_masked
    NFZ_FLAGS="--nfz 0.19 0.03 0.10 --mask-nfz-target"
else
    ROOT=results_nfz_masked_multi_disc
    # repeat --nfz once per disc: cx cy r
    NFZ_FLAGS="--nfz 0.5 0.0 0.2 --nfz -0.5 0.0 0.2 --nfz 0.0 0.5 0.2 --nfz 0.0 -0.5 0.2 --mask-nfz-target"
fi
OUT=${ROOT}/${M}_seed${S}

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
            --num-intervals 64 --points-per-interval 200 --dt 0.01 \
            $NFZ_FLAGS \
            --seed $S --out-dir $OUT
        ;;
    opt_transport)
        python run_baseline_milano.py --method opt_transport \
            --num-points 25600 --dt 0.01 \
            $NFZ_FLAGS \
            --seed $S --out-dir $OUT
        ;;
    sac)
        python run_baseline_milano.py --method sac \
            --num-points 25600 --sac-train-steps 500000 \
            --sac-grid-size 50 --sac-comm-range 0.18 --dt 0.01 \
            $NFZ_FLAGS \
            --seed $S --out-dir $OUT
        ;;
    fmec_stein)
        python run_baseline_milano_v1.py --method fmec_stein \
            --fmec-dt 0.05 --fmec-tsteps 6400 \
            --fmec-num-iters 500 --fmec-step-size 0.01 \
            --fmec-num-samples 1000 --fmec-grad-subsample 2000 \
            $NFZ_FLAGS \
            --seed $S --out-dir $OUT
        ;;
    fmec_sinkhorn)
        python run_baseline_milano_v1.py --method fmec_sinkhorn \
            --fmec-dt 0.05 --fmec-tsteps 6400 \
            --fmec-num-iters 200 --fmec-step-size 0.005 \
            --fmec-num-samples 1000 --fmec-grad-subsample 2000 \
            $NFZ_FLAGS \
            --seed $S --out-dir $OUT
        ;;
    *)
        echo "unknown method $M"; exit 1
        ;;
esac
