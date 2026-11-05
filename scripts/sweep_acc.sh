#!/bin/bash
#SBATCH --job-name=sweep_acc
#SBATCH --output=./slurm_output/%A_%a_sweep_acc.out
#SBATCH --error=./slurm_output/%A_%a_sweep_acc.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --array=0-4

# Combo 3:  L_total = L_cfm + lambda_acc * R_acc   (Hessian-Frobenius proxy)
# Sweep lambda_acc across 5 log-spaced values; no prior tuned value, so we
# centre on 1e-2 (where lambda_acc * mean||a||^2 is roughly comparable to L_cfm).

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

LAMBDAS=(1e-4 1e-3 1e-2 1e-1 1.0)
LA=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

SEED=42
OUT=results/sweep_acc/la${LA}_seed${SEED}
mkdir -p "$(dirname "$OUT")"
echo "[sweep_acc] task=$SLURM_ARRAY_TASK_ID  lambda_acc=$LA  out=$OUT"

python -m experiments.milano.run_otcfm_milano \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --acc-lambda $LA \
    --energy-chunk-size 64 --energy-steps 8 \
    --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    --domain-size 8.0 \
    --seed $SEED --out-dir $OUT
