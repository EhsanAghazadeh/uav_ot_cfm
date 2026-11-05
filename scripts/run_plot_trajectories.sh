#!/bin/bash
#SBATCH --job-name=plot_trajectories
#SBATCH --output=./slurm_output/%j_plot_trajectories.out
#SBATCH --error=./slurm_output/%j_plot_trajectories.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00

# Plot trajectory grids for all methods.
# Edit ROOT / METHODS / NFZ_DISCS below, then sbatch this script.
#
# Single NFZ disc:
#   NFZ_DISCS="--nofly 0.19 0.03 0.10"
#
# Multiple NFZ discs:
#   NFZ_DISCS="--nofly 0.5 0.0 0.2 --nofly -0.5 0.0 0.2 --nofly 0.0 0.5 0.2 --nofly 0.0 -0.5 0.2"

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

GRID="data/milano_grid_corrected.npy"
SEEDS=(21 42 63)

# ── Edit these ────────────────────────────────────────────────────────────
ROOT=results_nfz_masked
METHODS=(toes nonuniform_erg opt_transport sac
         fmec_stein fmec_sinkhorn
         otcfm
         otcfm_nfz_l5
         otcfm_energy_lambda1e-2
         otcfm_energy_la1e-4
         otcfm_energy_lambda1e-3_nfz_l500
         otcfm_energy_la1e-1_nfz_l500_le1e-3)
OUT_DIR="${ROOT}/trajectory_plots"
NFZ_DISCS="--nofly 0.19 0.03 0.10"
# NFZ_DISCS="--nofly 0.5 0.0 0.2 --nofly -0.5 0.0 0.2 --nofly 0.0 0.5 0.2 --nofly 0.0 -0.5 0.2"
SUPTITLE="NFZ-masked Milano target with multiple NFZ discs"
# ─────────────────────────────────────────────────────────────────────────

mkdir -p "$OUT_DIR"

# One plot per seed: all methods side-by-side
for s in "${SEEDS[@]}"; do
    ARGS=""
    for m in "${METHODS[@]}"; do
        T="${ROOT}/${m}_seed${s}/trajectory.npz"
        [ -f "$T" ] && ARGS="$ARGS --traj ${m}_seed${s}:${T}"
    done
    if [ -n "$ARGS" ]; then
        python plot_all_methods.py $ARGS \
            --milano-grid "$GRID" \
            $NFZ_DISCS \
            --out "${OUT_DIR}/trajectories_seed${s}.png"
    fi
done

# One combined plot: all methods × all seeds (useful for appendix)
ARGS=""
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        T="${ROOT}/${m}_seed${s}/trajectory.npz"
        [ -f "$T" ] && ARGS="$ARGS --traj ${m}_s${s}:${T}"
    done
done
if [ -n "$ARGS" ]; then
    python plot_all_methods.py $ARGS \
        --milano-grid "$GRID" \
        $NFZ_DISCS \
        --ncols 3 \
        --out "${OUT_DIR}/trajectories_all.png"
fi

echo "saved to $OUT_DIR"


# --suptitle "$SUPTITLE  (seed=$s)" \
# --suptitle "$SUPTITLE" \