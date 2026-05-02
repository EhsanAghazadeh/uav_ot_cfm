#!/bin/bash
#SBATCH --job-name=eval_nfz_masked
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%j_eval_nfz_masked.out
#SBATCH --error=./slurm_output/%j_eval_nfz_masked.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

# Run AFTER the NFZ-condition training jobs finish:
#   - scripts/run_baselines_nfz_masked.sh
#   - scripts/run_otcfm_nfz_masked_mask_only.sh
#   - scripts/run_otcfm_nfz_masked_mask_plus_penalty.sh
#
# Evaluates everything against the *same* NFZ-masked target so coverage
# metrics (Fourier ergodic, density correlation, L1) are apples-to-apples.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

# All methods evaluated for this NFZ condition. Includes both OT-CFM
# ablations (mask_only and mask_plus_penalty) plus the four baselines.
METHODS=(toes nonuniform_erg opt_transport sac
         otcfm_mask_only otcfm_mask_plus_penalty)
SEEDS=(21 42 63)
NFZ_C="0.19 0.03"
NFZ_R="0.10"

ROOT=results_nfz_masked

# 1. Prefix-stop curves on the masked target
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        TRAJ=${ROOT}/${m}_seed${s}/trajectory.npz
        OUT=${ROOT}/${m}_seed${s}/curves
        if [ ! -f "$TRAJ" ]; then echo "skip missing $TRAJ"; continue; fi
        if [ -f "${OUT}/prefix_curve.npz" ]; then echo "skip cached $OUT"; continue; fi
        python eval_prefix_stop.py --traj "$TRAJ" \
            --milano-grid data/milano_grid_corrected.npy \
            --smooth-sigma 1.5 \
            --nfz-center $NFZ_C --nfz-radius $NFZ_R --mask-nfz-target \
            --out-dir "$OUT"
    done
done

# 2. NFZ violation table per seed (no change — still measures incursions
#    against the disc itself, which is method-independent)
for s in "${SEEDS[@]}"; do
    ARGS=""
    for m in "${METHODS[@]}"; do
        T=${ROOT}/${m}_seed${s}/trajectory.npz
        [ -f "$T" ] && ARGS="$ARGS --traj ${m}:${T}"
    done
    if [ -n "$ARGS" ]; then
        python eval_nofly.py $ARGS \
            --center $NFZ_C --radius $NFZ_R \
            --out ${ROOT}/nofly_seed${s}.json
    fi
done

# 3. Cross-method curve plots per seed
for s in "${SEEDS[@]}"; do
    ARGS=""
    for m in "${METHODS[@]}"; do
        C=${ROOT}/${m}_seed${s}/curves/prefix_curve.npz
        [ -f "$C" ] && ARGS="$ARGS --curve ${m}:${C}"
    done
    [ -n "$ARGS" ] && python plot_curves.py $ARGS \
        --out-dir ${ROOT}/comparison_seed${s}
done

# 4. Trajectory grids w/ NFZ overlay
for s in "${SEEDS[@]}"; do
    ARGS=""
    for m in "${METHODS[@]}"; do
        T=${ROOT}/${m}_seed${s}/trajectory.npz
        [ -f "$T" ] && ARGS="$ARGS --traj ${m}:${T}"
    done
    [ -n "$ARGS" ] && python plot_all_methods.py $ARGS \
        --milano-grid data/milano_grid_corrected.npy \
        --nofly $NFZ_C $NFZ_R \
        --suptitle "NFZ-masked Milano (seed=${s})" \
        --out ${ROOT}/comparison_seed${s}/trajectories_all_nfz.png
done

# 5. Fair-comparison metrics on the *masked* target
FAIR_ARGS=""
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        T=${ROOT}/${m}_seed${s}/trajectory.npz
        [ -f "$T" ] && FAIR_ARGS="$FAIR_ARGS --traj ${m}:${T}"
    done
done
if [ -n "$FAIR_ARGS" ]; then
    python eval_fair.py $FAIR_ARGS \
        --milano-grid data/milano_grid_corrected.npy \
        --nfz-center $NFZ_C --nfz-radius $NFZ_R --mask-nfz-target \
        --out-dir ${ROOT}/fair_metrics
    python plot_pareto.py \
        --per-run ${ROOT}/fair_metrics/per_run.json \
        --out-dir ${ROOT}/fair_metrics
fi

echo "NFZ-masked evaluation complete"
