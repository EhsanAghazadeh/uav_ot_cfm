#!/bin/bash
#SBATCH --job-name=synthetic_multiseed
#SBATCH --output=./slurm_output/%A_%a_synthetic_multiseed.out
#SBATCH --error=./slurm_output/%A_%a_synthetic_multiseed.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=06:00:00
#SBATCH --array=0-14

# Synthetic experiments 1-3 (paper Section 6) over training seeds 0-4.
#   0-4    Experiment 1 (Gaussian mixture)
#   5-9    Experiment 2 (binary 3:1 density)
#   10-14  Experiment 3 (no-fly zone; unconstrained, +NFZ, +NFZ+acc)
# Submit from the repo root:  sbatch scripts/synthetic_multiseed.sh
# Single seed-42 runs: python code_synthetic/experiment_0{1,2}_*.py and
#                      python code_synthetic/experiment_03_revised_v2.py

set -euo pipefail
mkdir -p slurm_output
module purge
module load python/3.11.7
source .venv/bin/activate

i=$SLURM_ARRAY_TASK_ID
MODES=(exp1 exp2 exp3)
python code_synthetic/run_multiseed.py "${MODES[$((i / 5))]}" --seed "$((i % 5))"
