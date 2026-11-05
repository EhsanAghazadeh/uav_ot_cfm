#!/bin/bash
#SBATCH --job-name=appendix_delta_sncap
#SBATCH --output=./slurm_output/%A_%a_appendix_delta_sncap.out
#SBATCH --error=./slurm_output/%A_%a_appendix_delta_sncap.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --array=0-26

# Experiment 1 protocol (train_synth.py), two appendix studies, seeds 0-2:
#   0-17   latent inner radius delta in {0.005,0.01,0.02,0.05,0.1,0.2}
#   18-26  spectral-norm cap L_v^net in {3, 5, 8} (certified end-to-end bound)

set -euo pipefail
mkdir -p slurm_output
module purge
module load python/3.11.7
source .venv/bin/activate

i=$SLURM_ARRAY_TASK_ID
S=$((i % 3))
if [ "$i" -lt 18 ]; then
    DELTAS=(0.005 0.01 0.02 0.05 0.1 0.2)
    D=${DELTAS[$((i / 3))]}
    python -m experiments.appendix.train_synth --delta "$D" --seed "$S" --sn-cap 0 --epochs 1000 \
        --out-dir "results/appendix/delta_ablation/delta${D}_seed${S}"
else
    CAPS=(3 5 8)
    C=${CAPS[$(((i - 18) / 3))]}
    python -m experiments.appendix.train_synth --seed "$S" --sn-cap "$C" --epochs 1000 \
        --out-dir "results/appendix/sn_certified/cap${C}_seed${S}"
fi
