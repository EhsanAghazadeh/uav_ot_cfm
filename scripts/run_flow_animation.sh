#!/bin/bash
#SBATCH --job-name=flow_animation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00

# Exports the latent -> Milano flow-matching animation assets used by the
# project page (sprite sheets, GIFs, latent/UAV path excerpts).
# Run from the repo root:  sbatch scripts/run_flow_animation.sh
# Adapt the partition / module lines to your cluster.

set -euo pipefail
hostname; date
module load python/3.11.7 2>/dev/null || true
module list 2>&1 || true
source .venv/bin/activate
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -m plotting.make_flow_animation \
    --model results/otcfm_seed42:otcfm \
    --model results_nfz_masked/otcfm_nfz_l5_seed42:otcfm_nfz \
    --n-particles 30000 --n-steps 60 \
    --out results/flow_animation
date
