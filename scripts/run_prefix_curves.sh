#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=prefix_curves
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%j_prefix_curves.out
#SBATCH --error=./slurm_output/%j_prefix_curves.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

# Blocks E + F from run_post_eval.sh in isolation:
#   E — eval_prefix_stop.py for every method+seed
#   F — plot_curves.py (aggregated: mean+std band across seeds)

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

GRID="data/milano_grid_corrected.npy"
NFZ_C="0.19 0.03"
NFZ_R="0.10"
SEEDS=(21 42 63)

ROOT=results
METHODS=(toes nonuniform_erg opt_transport sac
         sac_velpen00 sac_velpen010 sac_velpen020
         fmec_stein fmec_sinkhorn
         otcfm
         otcfm_nfz_l5
         otcfm_energy_lambda1e-2
         otcfm_energy_la1e-4
         otcfm_energy_lambda1e-3_nfz_l500
         otcfm_energy_la1e-1_nfz_l500_le1e-3)

ROOT_M=results_nfz_masked
METHODS_M=(toes nonuniform_erg opt_transport sac
           fmec_stein fmec_sinkhorn
           otcfm_mask_only otcfm_mask_plus_penalty)

_ts() { case "$1" in toes*) echo "2.0";; fmec_stein*|fmec_sinkhorn*) echo "7.94";; *) echo "10.0";; esac; }

# ---------------------------------------------------------------------------
# Block E — prefix-stop curves
# ---------------------------------------------------------------------------
echo "=== Block E: prefix-stop curves (results/) ==="
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        TRAJ="${ROOT}/${m}_seed${s}/trajectory.npz"
        OUT="${ROOT}/${m}_seed${s}/curves"
        [ -f "${OUT}/prefix_curve.npz" ] && { echo "  cached $m seed$s"; continue; }
        [ ! -f "$TRAJ" ] && { echo "  skip missing $TRAJ"; continue; }
        echo "  eval $m seed$s"
        python eval_prefix_stop.py --traj "$TRAJ" \
            --milano-grid "$GRID" --smooth-sigma 1.5 \
            --time-scale "$(_ts "$m")" \
            --nfz-center $NFZ_C --nfz-radius $NFZ_R --mask-nfz-target \
            --out-dir "$OUT"
    done
done

echo "=== Block E: prefix-stop curves (results_nfz_masked/) ==="
for m in "${METHODS_M[@]}"; do
    for s in "${SEEDS[@]}"; do
        TRAJ="${ROOT_M}/${m}_seed${s}/trajectory.npz"
        OUT="${ROOT_M}/${m}_seed${s}/curves"
        [ -f "${OUT}/prefix_curve.npz" ] && { echo "  cached $m seed$s"; continue; }
        [ ! -f "$TRAJ" ] && { echo "  skip missing $TRAJ"; continue; }
        echo "  eval $m seed$s"
        python eval_prefix_stop.py --traj "$TRAJ" \
            --milano-grid "$GRID" --smooth-sigma 1.5 \
            --time-scale "$(_ts "$m")" \
            --nfz-center $NFZ_C --nfz-radius $NFZ_R --mask-nfz-target \
            --out-dir "$OUT"
    done
done

# ---------------------------------------------------------------------------
# Block F — aggregated prefix-curve plots (mean + std band across seeds)
# ---------------------------------------------------------------------------
echo "=== Block F: aggregated prefix-curve plots (results/) ==="
CURVE_ARGS=""
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        C="${ROOT}/${m}_seed${s}/curves/prefix_curve.npz"
        [ -f "$C" ] && CURVE_ARGS="$CURVE_ARGS --curve ${m}:${C}"
    done
done
[ -n "$CURVE_ARGS" ] && python plot_curves.py $CURVE_ARGS \
    --out-dir "${ROOT}/comparison_all_seeds"

echo "=== Block F: aggregated prefix-curve plots (results_nfz_masked/) ==="
CURVE_ARGS=""
for m in "${METHODS_M[@]}"; do
    for s in "${SEEDS[@]}"; do
        C="${ROOT_M}/${m}_seed${s}/curves/prefix_curve.npz"
        [ -f "$C" ] && CURVE_ARGS="$CURVE_ARGS --curve ${m}:${C}"
    done
done
[ -n "$CURVE_ARGS" ] && python plot_curves.py $CURVE_ARGS \
    --out-dir "${ROOT_M}/comparison_all_seeds"

echo "prefix_curves complete"
