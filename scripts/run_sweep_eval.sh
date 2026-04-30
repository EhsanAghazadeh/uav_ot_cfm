#!/bin/bash
#SBATCH --job-name=sweep_eval
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%j_sweep_eval.out
#SBATCH --error=./slurm_output/%j_sweep_eval.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

# Run AFTER all five lambda sweeps finish.
# Steps:
#   1. eval_prefix_stop.py per sweep run (writes curves/prefix_curve.npz)
#   2. pick_sweep_winners.py  -> winners.json + LaTeX + per-combo plots
#                                + winners trajectory grid + winners curves

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

COMBOS=(sweep_nfz sweep_energy sweep_acc sweep_nfz_energy sweep_nfz_energy_acc)

# 1. Prefix-stop curves for every sweep run.
for combo in "${COMBOS[@]}"; do
    for run_dir in results/${combo}/*_seed*/; do
        [ -d "$run_dir" ] || continue
        TRAJ="${run_dir}trajectory.npz"
        OUT="${run_dir}curves"
        if [ ! -f "$TRAJ" ]; then
            echo "skip missing $TRAJ"; continue
        fi
        if [ -f "${OUT}/prefix_curve.npz" ]; then
            echo "skip cached $OUT"; continue
        fi
        python eval_prefix_stop.py --traj "$TRAJ" \
            --milano-grid data/milano_grid_corrected.npy \
            --smooth-sigma 1.5 --out-dir "$OUT"
    done
done

# 2. Pick winners + emit LaTeX tables + composite plots.
python pick_sweep_winners.py \
    --sweep-root results \
    --milano-grid data/milano_grid_corrected.npy \
    --nfz-center 0.19 0.03 --nfz-radius 0.10 \
    --out-dir results/sweep_winners

echo "sweep evaluation complete"
