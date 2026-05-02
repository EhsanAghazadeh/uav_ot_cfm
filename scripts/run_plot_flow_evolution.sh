#!/bin/bash
#SBATCH --job-name=flow_evolution
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%A_%a_flow_evolution.out
#SBATCH --error=./slurm_output/%A_%a_flow_evolution.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --array=0-3

# Renders the flow-matching evolution figure (latent annulus -> Milano)
# for each trained OT-CFM model directory listed below. Each array task
# produces one PNG. CPU is sufficient: ~30k particles x 40 RK4 steps
# integrates in a few seconds; the plot is the bottleneck.
#
# Trained model dirs: pre-existing OT-CFM checkpoints, plus the two
# masked-NFZ ablations once those finish. Edit MODEL_DIRS to taste.
#
# Output goes alongside each model: <model-dir>/flow_evolution.png

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

# (model_dir, optional NFZ flags) pairs.  NFZ flags are passed as a single
# string -- empty for unconstrained models, "--nfz-center 0.19 0.03 --nfz-radius 0.10"
# for NFZ-aware ones (puts the hole in the target panel + draws the disc
# on every evolution panel).
MODEL_DIRS=(
    "results/otcfm_seed42"
    "results/otcfm_nfz_l5_seed42"
    "results_nfz_masked/otcfm_mask_only_seed42"
    "results_nfz_masked/otcfm_mask_plus_penalty_seed42"
)
NFZ_FLAGS=(
    ""
    "--nfz-center 0.19 0.03 --nfz-radius 0.10"
    "--nfz-center 0.19 0.03 --nfz-radius 0.10"
    "--nfz-center 0.19 0.03 --nfz-radius 0.10"
)

i=$SLURM_ARRAY_TASK_ID
MD=${MODEL_DIRS[$i]}
NF=${NFZ_FLAGS[$i]}
OUT=${MD}/flow_evolution.png

if [ ! -f "${MD}/model.pt" ]; then
    echo "skip ${MD}: no model.pt"
    exit 0
fi

echo "[flow_evolution] task=$i  model_dir=$MD  out=$OUT"
echo "                  nfz_flags='${NF}'"

python plot_flow_evolution.py \
    --model-dir "$MD" \
    --milano-grid data/milano_grid_corrected.npy \
    --smooth-sigma 1.5 \
    --source uniform --n-particles 30000 \
    --n-frames 8 --n-steps 40 \
    --bins 160 --smooth-hist 1.5 \
    --cmap viridis \
    $NF \
    --seed 42 \
    --out "$OUT"

echo "saved $OUT"
