#!/bin/bash
#SBATCH --job-name=prefix_curves_nfz_center_0_3_0_3
#SBATCH --output=./slurm_output/%j_prefix_curves_nfz_center_0_3_0_3.out
#SBATCH --error=./slurm_output/%j_prefix_curves_nfz_center_0_3_0_3.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

# Prefix-stop curves + aggregated plots for results_nfz_masked/ only.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

GRID="data/milano_grid_corrected.npy"
SEEDS=(21 42 63)

# ── NFZ configuration ─────────────────────────────────────────────────────────
# Single disc:
# NFZ_FLAGS="--nfz 0.19 0.03 0.10"
#
# Multiple discs (repeat --nfz once per disc: cx cy r):
# NFZ_FLAGS="--nfz 0.5 0.0 0.2 --nfz -0.5 0.0 0.2 --nfz 0.0 0.5 0.2 --nfz 0.0 -0.5 0.2"
#
NFZ_FLAGS="--nfz 0.3 0.3 0.3"
# ─────────────────────────────────────────────────────────────────────────────

ROOT_M=results_nfz_masked_multi_disc
# METHODS_M=(toes nonuniform_erg opt_transport sac
#            fmec_stein fmec_sinkhorn
#            otcfm_mask_only otcfm_mask_plus_penalty)

METHODS_M=(toes nonuniform_erg opt_transport sac
         fmec_stein fmec_sinkhorn
         otcfm
         otcfm_nfz_l5
         otcfm_energy_lambda1e-2
         otcfm_energy_la1e-4
         otcfm_energy_lambda1e-3_nfz_l500
         otcfm_energy_la1e-1_nfz_l500_le1e-3)

_ts() { case "$1" in toes*) echo "2.0";; fmec_stein*|fmec_sinkhorn*) echo "7.94";; *) echo "10.0";; esac; }

# ---------------------------------------------------------------------------
# Block E — eval_prefix_stop.py per method+seed
# ---------------------------------------------------------------------------
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
            $NFZ_FLAGS --mask-nfz-target \
            --out-dir "$OUT"
    done
done

# ---------------------------------------------------------------------------
# Block F — aggregated prefix-curve plots (mean + std band across seeds)
# ---------------------------------------------------------------------------
echo "=== Block F: aggregated prefix-curve plots ==="
CURVE_ARGS=""
for m in "${METHODS_M[@]}"; do
    for s in "${SEEDS[@]}"; do
        C="${ROOT_M}/${m}_seed${s}/curves/prefix_curve.npz"
        [ -f "$C" ] && CURVE_ARGS="$CURVE_ARGS --curve ${m}:${C}"
    done
done
[ -n "$CURVE_ARGS" ] && python plot_curves.py $CURVE_ARGS \
    --out-dir "${ROOT_M}/comparison_all_seeds"

echo "prefix_curves complete -> ${ROOT_M}/comparison_all_seeds/"
