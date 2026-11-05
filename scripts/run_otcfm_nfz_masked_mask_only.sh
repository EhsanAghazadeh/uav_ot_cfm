#!/bin/bash
#SBATCH --job-name=otcfm_nfz_mask_only_la1e-4_center_0_3_0_3
#SBATCH --output=./slurm_output/%A_%a_otcfm_nfz_mask_only_la1e-4_center_0_3_0_3.out
#SBATCH --error=./slurm_output/%A_%a_otcfm_nfz_mask_only_la1e-4_center_0_3_0_3.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --array=2-2

# Ablation (a): OT-CFM trained on the NFZ-masked target *without* the
# soft penalty. Tests whether the density-level encoding alone keeps
# samples out of the disc, i.e. whether the soft penalty is necessary.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

SEEDS=(21 42 63)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results_nfz_masked_03_03/otcfm_energy_la1e-4_seed${SEED}

echo "[otcfm_nfz_mask_only] seed=$SEED  out=$OUT"

python -m experiments.milano.run_otcfm_milano \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --nfz-center 0.3 0.3 --nfz-radius 0.3 --mask-nfz-target \
    --acc-lambda 1e-4 \
    --energy-chunk-size 64 --energy-steps 8 \
    --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    --domain-size 8.0 \
    --reuse-model-from results_nfz_masked_03_03/otcfm_energy_la1e-4_seed${SEED}/model.pt \
    --seed $SEED --out-dir $OUT


    # --energy-lambda 1e-2 --energy-chunk-size 64 --energy-steps 8 \
    # --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    # --domain-size 8 \