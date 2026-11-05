#!/bin/bash
#SBATCH --job-name=table1_eval
#SBATCH --output=./slurm_output/%A_%a_table1_eval.out
#SBATCH --error=./slurm_output/%A_%a_table1_eval.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --array=0-1

# Table 1 metrics (coverage rho, mean Zeng power, NFZ violation) and the
# coverage-energy Pareto curves, for both no-fly configurations:
#   task 0: single disc (results_nfz_masked/)
#   task 1: four discs  (results_nfz_masked_multi_disc/)
# Protocol: 200 m domain, time scale 10, 30 m/s speed cap with retiming;
# TOES and FMEC are evaluated at their native time scales.
# Run after scripts/table1_otcfm.sh and scripts/run_baselines_nfz_masked.sh.

set -euo pipefail
mkdir -p slurm_output
module purge
module load python/3.11.7
source .venv/bin/activate

if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
    ROOT=results_nfz_masked
    NFZ=(--nfz 0.19 0.03 0.10)
else
    ROOT=results_nfz_masked_multi_disc
    NFZ=(--nfz 0.5 0.0 0.2 --nfz -0.5 0.0 0.2 --nfz 0.0 0.5 0.2 --nfz 0.0 -0.5 0.2)
fi

METHODS=(toes nonuniform_erg opt_transport sac fmec_stein fmec_sinkhorn
         otcfm otcfm_nfz_l5 otcfm_energy_lambda1e-2 otcfm_energy_la1e-4
         otcfm_energy_lambda1e-3_nfz_l500 otcfm_energy_la1e-1_nfz_l500_le1e-3)
ARGS=()
for m in "${METHODS[@]}"; do
    for s in 21 42 63; do
        T="${ROOT}/${m}_seed${s}/trajectory.npz"
        [ -f "$T" ] && ARGS+=(--traj "${m}:${T}")
    done
done

python eval_fair.py "${ARGS[@]}" \
    --milano-grid data/milano_grid_corrected.npy \
    --domain-size 200.0 --time-scale 10.0 \
    --method-time-scale toes:2.0 \
    --method-time-scale fmec_stein:7.94 \
    --method-time-scale fmec_sinkhorn:7.94 \
    --v-clip-realistic 30.0 \
    "${NFZ[@]}" --mask-nfz-target \
    --out-dir "${ROOT}/fair_metrics_rescaled"
python plot_pareto.py \
    --per-run "${ROOT}/fair_metrics_rescaled/per_run.json" \
    --out-dir "${ROOT}/fair_metrics_rescaled"
