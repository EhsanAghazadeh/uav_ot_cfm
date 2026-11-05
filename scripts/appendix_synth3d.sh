#!/bin/bash
#SBATCH --job-name=appendix_synth3d
#SBATCH --output=./slurm_output/%A_%a_appendix_synth3d.out
#SBATCH --error=./slurm_output/%A_%a_appendix_synth3d.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=03:00:00
#SBATCH --array=0-5

# 3D instantiation of the appendix R^d construction (d = 3):
# delta in {0.05, 0.1} x seeds {0, 1, 2}.

set -euo pipefail
mkdir -p slurm_output
module purge
module load python/3.11.7
source .venv/bin/activate

DELTAS=(0.05 0.1)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
D=${DELTAS[$((i / 3))]}
S=${SEEDS[$((i % 3))]}
python -m experiments.appendix.train_synth3d --delta "$D" --seed "$S" --epochs 1000 \
    --out-dir "results/appendix/synth3d/delta${D}_seed${S}"
