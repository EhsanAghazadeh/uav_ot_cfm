#!/bin/bash
#SBATCH --job-name=otcfm_nfz_multi_disc_energy_la1e-4
#SBATCH --output=./slurm_output/%A_%a_otcfm_nfz_multi_disc_energy_la1e-4.out
#SBATCH --error=./slurm_output/%A_%a_otcfm_nfz_multi_disc_energy_la1e-4.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --array=0-2

# OT-CFM trained on a 4-disc NFZ configuration:
#   discs at (±0.2, 0) and (0, ±0.2), all with r=0.2
# Uses mask-nfz-target + soft NFZ penalty.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SEEDS=(21 42 63)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results_nfz_masked_multi_disc/otcfm_energy_la1e-4_seed${SEED}

echo "[otcfm_nfz_multi_disc] seed=$SEED  out=$OUT"

python -m experiments.milano.run_otcfm_milano \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --nfz  0.5  0.0 0.2 \
    --nfz -0.5  0.0 0.2 \
    --nfz  0.0  0.5 0.2 \
    --nfz  0.0 -0.5 0.2 \
    --mask-nfz-target \
    --energy-chunk-size 64 --energy-steps 8 \
    --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    --domain-size 8 \
    --acc-lambda 1e-4 \
    --reuse-model-from ${OUT}/model.pt \
    --seed $SEED --out-dir $OUT


#     --acc-lambda 1e-1 \
    # --energy-lambda 1e-3 --energy-chunk-size 64 --energy-steps 8 \
    # --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    # --domain-size 8 \
#     --nfz-lambda 5 --nfz-steps 8 \