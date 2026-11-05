#!/bin/bash
#SBATCH --job-name=table1_otcfm
#SBATCH --output=./slurm_output/%A_%a_table1_otcfm.out
#SBATCH --error=./slurm_output/%A_%a_table1_otcfm.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --array=0-35%6

# Canonical launcher for the six OT-CFM variants of Table 1 (Milano),
# under both no-fly configurations, seeds 21/42/63:
#   task = 18*cfg + 3*variant + seed_idx
#   cfg 0: single disc (1D) -> results_nfz_masked/
#   cfg 1: four discs  (MD) -> results_nfz_masked_multi_disc/
# Flags reproduce the plan_meta.json of the runs reported in the paper.
# All variants train on the NFZ-masked target; they differ only in the
# penalty weights. Baselines: scripts/run_baselines_nfz_masked.sh.

set -euo pipefail
mkdir -p slurm_output
module purge
module load python/3.11.7
source .venv/bin/activate

i=$SLURM_ARRAY_TASK_ID
CFG=$((i / 18)); V=$(((i % 18) / 3)); SEEDS=(21 42 63); SEED=${SEEDS[$((i % 3))]}

#        name                                   nfz_lambda energy_lambda acc_lambda
NAMES=(  otcfm                                  otcfm_nfz_l5 otcfm_energy_lambda1e-2 otcfm_energy_la1e-4 otcfm_energy_lambda1e-3_nfz_l500 otcfm_energy_la1e-1_nfz_l500_le1e-3 )
LNFZ=(   0                                      5            0                       0                   500                             500 )
LE=(     0                                      0            1e-2                    0                   1e-3                            1e-3 )
LA=(     0                                      0            0                       1e-4                0                               1e-1 )
LABEL=(  "OT-CFM" "+NFZ" "+E" "+A" "+NFZ+E" "+NEA" )

if [ "$CFG" -eq 0 ]; then
    ROOT=results_nfz_masked
    NFZ=(--nfz-center 0.19 0.03 --nfz-radius 0.1)
else
    ROOT=results_nfz_masked_multi_disc
    NFZ=(--nfz 0.5 0.0 0.2 --nfz -0.5 0.0 0.2 --nfz 0.0 0.5 0.2 --nfz 0.0 -0.5 0.2)
fi
OUT=${ROOT}/${NAMES[$V]}_seed${SEED}
echo "[table1] ${LABEL[$V]}  cfg=$CFG  seed=$SEED  out=$OUT"

python -m experiments.milano.run_otcfm_milano \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 2048 \
    --delta 0.05 --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    "${NFZ[@]}" --nfz-margin 0.05 --mask-nfz-target \
    --nfz-lambda "${LNFZ[$V]}" --nfz-steps 8 \
    --energy-lambda "${LE[$V]}" --energy-chunk-size 64 --energy-steps 8 \
    --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    --acc-lambda "${LA[$V]}" \
    --domain-size 8.0 \
    --seed "$SEED" --out-dir "$OUT"
