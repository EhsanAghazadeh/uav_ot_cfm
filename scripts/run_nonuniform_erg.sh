#!/bin/bash
#SBATCH --job-name=nonuniform_erg_nfz_center_0_3_0_3   
#SBATCH --output=./slurm_output/%j_nonuniform_erg_nfz_center_0_3_0_3.out
#SBATCH --error=./slurm_output/%j_nonuniform_erg_nfz_center_0_3_0_3.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --array=0-2

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SEEDS=(21 42 63)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results_nfz_masked/nonuniform_erg_nfz_center_0_3_0_3_seed${SEED}

python -m experiments.milano.run_baseline_milano --method nonuniform_erg \
    --num-intervals 64 --points-per-interval 200 --dt 0.01 \
    --nfz-center 0.3 0.3 --nfz-radius 0.3 --mask-nfz-target \
    --seed $SEED --out-dir $OUT
