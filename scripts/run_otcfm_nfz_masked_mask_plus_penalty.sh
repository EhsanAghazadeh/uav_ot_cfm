#!/bin/bash
#SBATCH --job-name=otcfm_nfz_mask_energy_lambda1e-3_nfz_l500_center_0_3_0_3
#SBATCH --output=./slurm_output/%A_%a_otcfm_nfz_mask_energy_lambda1e-3_nfz_l500_center_0_3_0_3.out
#SBATCH --error=./slurm_output/%A_%a_otcfm_nfz_mask_energy_lambda1e-3_nfz_l500_center_0_3_0_3.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --array=0-0

# Ablation (b): OT-CFM trained on the NFZ-masked target *and* with the
# soft penalty (lambda_nfz=5, the combo-1 winner). Belt-and-braces; the
# expected paper-headline configuration.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SEEDS=(21 42 63)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results_nfz_masked_03_03/otcfm_energy_lambda1e-3_nfz_l500_seed${SEED}

echo "[otcfm_nfz_mask_plus_penalty] seed=$SEED  out=$OUT"

python -m experiments.milano.run_otcfm_milano \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --nfz-center 0.3 0.3 --nfz-radius 0.3 \
    --mask-nfz-target \
    --nfz-lambda 500 --nfz-steps 8 \
    --energy-lambda 1e-3 --energy-chunk-size 64 --energy-steps 8 \
    --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    --domain-size 8 \
    --reuse-model-from results_nfz_masked_03_03/otcfm_energy_lambda1e-3_nfz_l500_seed${SEED}/model.pt \
    --seed $SEED --out-dir $OUT
