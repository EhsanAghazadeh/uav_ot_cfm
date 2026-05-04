#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=post_eval
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%j_post_eval.out
#SBATCH --error=./slurm_output/%j_post_eval.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

# Post-training omnibus eval. Three blocks, all CPU, all cheap:
#
#   Block A (items 1+4+5):  fair-comparison metrics on the rescaled
#                            UAV-mission protocol (200 m / 0.2 s native ->
#                            time_scale x10 = 200 m / 2 s effective ->
#                            mission ~42 min for OT-CFM, ~7 h for the
#                            2560 s baselines). Computed both on the
#                            unconstrained target (results/) and the
#                            NFZ-masked target (results_nfz_masked/).
#                            FMEC variants are included on both sides.
#
#   Block B (item 6):        multi-agent ablation on the trained vanilla
#                            OT-CFM checkpoint at seed 42, fleet sizes
#                            {1, 2, 5, 10, 20}. Tests Theorem 4 collective
#                            O(1/sqrt(N)) prediction.
#
#   Block C (item 7):        amortization / wall-clock LaTeX table from
#                            existing plan_meta.json files (no compute).
#
# Run AFTER all training jobs (sweep_fmec, run_otcfm_*, run_baselines_*) and
# the masked-target jobs have produced trajectories.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

GRID="data/milano_grid_corrected.npy"
DOMAIN_M=200.0       # physical extent of [-1,1] in metres
TIME_SCALE=10.0      # multiplies native trajectory times
V_CAP_REALISTIC=30.0 # typical rotorcraft max cruise (m/s); reveals teleport
NFZ_C="0.19 0.03"
NFZ_R="0.10"
SEEDS=(21 42 63)


# ---------------------------------------------------------------------------
# Block A.1 -- unmasked: fair eval on rescaled UAV-mission protocol
# ---------------------------------------------------------------------------
ROOT=results
METHODS=(toes nonuniform_erg opt_transport sac
         sac_velpen00 sac_velpen010 sac_velpen020
         fmec_stein fmec_sinkhorn
         otcfm
         otcfm_nfz_l5
         otcfm_energy_lambda1e-2
         otcfm_energy_la1e-4
         otcfm_energy_lambda1e-3_nfz_l500
         otcfm_energy_la1e-1_nfz_l500_le1e-3)

ARGS=""
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        T="${ROOT}/${m}_seed${s}/trajectory.npz"
        [ -f "$T" ] && ARGS="$ARGS --traj ${m}:${T}"
    done
done

if [ -n "$ARGS" ]; then
    echo "=== Block A.1: rescaled fair eval (unmasked) ==="
    python eval_fair.py $ARGS \
        --milano-grid "$GRID" \
        --domain-size $DOMAIN_M --time-scale $TIME_SCALE \
        --method-time-scale toes:2.0 \
        --method-time-scale fmec_stein:7.94 \
        --method-time-scale fmec_sinkhorn:7.94 \
        --v-clip-realistic $V_CAP_REALISTIC \
        --nfz-center $NFZ_C --nfz-radius $NFZ_R --mask-nfz-target \
        --out-dir "${ROOT}/fair_metrics_rescaled"
    python plot_pareto.py \
        --per-run "${ROOT}/fair_metrics_rescaled/per_run.json" \
        --out-dir "${ROOT}/fair_metrics_rescaled"
fi


# ---------------------------------------------------------------------------
# Block A.2 -- masked: fair eval on rescaled UAV-mission protocol
# ---------------------------------------------------------------------------
ROOT_M=results_nfz_masked
METHODS_M=(toes nonuniform_erg opt_transport sac
           fmec_stein fmec_sinkhorn
           otcfm_mask_only otcfm_mask_plus_penalty)

ARGS=""
for m in "${METHODS_M[@]}"; do
    for s in "${SEEDS[@]}"; do
        T="${ROOT_M}/${m}_seed${s}/trajectory.npz"
        [ -f "$T" ] && ARGS="$ARGS --traj ${m}:${T}"
    done
done

if [ -n "$ARGS" ]; then
    echo "=== Block A.2: rescaled fair eval (NFZ-masked target) ==="
    python eval_fair.py $ARGS \
        --milano-grid "$GRID" \
        --domain-size $DOMAIN_M --time-scale $TIME_SCALE \
        --method-time-scale toes:2.0 \
        --method-time-scale fmec_stein:7.94 \
        --method-time-scale fmec_sinkhorn:7.94 \
        --v-clip-realistic $V_CAP_REALISTIC \
        --nfz-center $NFZ_C --nfz-radius $NFZ_R --mask-nfz-target \
        --out-dir "${ROOT_M}/fair_metrics_rescaled"
    python plot_pareto.py \
        --per-run "${ROOT_M}/fair_metrics_rescaled/per_run.json" \
        --out-dir "${ROOT_M}/fair_metrics_rescaled"
fi


# ---------------------------------------------------------------------------
# Block B -- multi-agent ablation (trained vanilla OT-CFM, seed 42)
# ---------------------------------------------------------------------------
MA_MODEL="results/otcfm_seed42"
if [ -f "${MA_MODEL}/model.pt" ] && [ -f "${MA_MODEL}/plan_meta.json" ]; then
    echo "=== Block B: multi-agent ablation ==="
    python eval_multi_agent.py \
        --model-dir "$MA_MODEL" \
        --milano-grid "$GRID" \
        --fleet-sizes 1 2 5 10 20 \
        --domain-size $DOMAIN_M --time-scale $TIME_SCALE \
        --out-dir results/multi_agent
else
    echo "=== Block B: skipping (no $MA_MODEL/model.pt) ==="
fi


# ---------------------------------------------------------------------------
# Block C -- amortization wall-clock table
# ---------------------------------------------------------------------------
echo "=== Block C: amortization + sampling-protocol tables ==="
python eval_amortization.py \
    --root results \
    --out results/fair_metrics_rescaled/amortization.tex
if [ -f "results/fair_metrics_rescaled/per_run.json" ]; then
    python eval_sampling_table.py \
        --per-run results/fair_metrics_rescaled/per_run.json \
        --root results \
        --out results/fair_metrics_rescaled/sampling_protocol.tex
fi
if [ -f "results_nfz_masked/fair_metrics_rescaled/per_run.json" ]; then
    python eval_sampling_table.py \
        --per-run results_nfz_masked/fair_metrics_rescaled/per_run.json \
        --root results_nfz_masked \
        --label tab:sampling_nfz_masked \
        --out results_nfz_masked/fair_metrics_rescaled/sampling_protocol.tex
fi


# ---------------------------------------------------------------------------
# Block D -- per-method velocity distribution (teleport-pathology figure)
# ---------------------------------------------------------------------------
echo "=== Block D: velocity distribution ==="
ARGS=""
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        T="${ROOT}/${m}_seed${s}/trajectory.npz"
        [ -f "$T" ] && ARGS="$ARGS --traj ${m}:${T}"
    done
done
if [ -n "$ARGS" ]; then
    python plot_velocity_distribution.py $ARGS \
        --domain-size $DOMAIN_M --time-scale $TIME_SCALE \
        --v-cap-realistic $V_CAP_REALISTIC \
        --out-dir "${ROOT}/velocity_dist"
fi

# ---------------------------------------------------------------------------
# Block E -- prefix-stop curves (eval_prefix_stop.py per method+seed)
# Skips if prefix_curve.npz already exists (re-delete to force recompute).
# Per-method --time-scale matches the normalisation used in eval_fair.py.
# ---------------------------------------------------------------------------
_ts() {          # $1 = method name → correct time-scale
    case "$1" in
        toes*)           echo "2.0"  ;;
        fmec_stein*|fmec_sinkhorn*) echo "7.94" ;;
        *)               echo "10.0" ;;
    esac
}

echo "=== Block E: prefix-stop curves (results/) ==="
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        TRAJ="${ROOT}/${m}_seed${s}/trajectory.npz"
        OUT="${ROOT}/${m}_seed${s}/curves"
        [ -f "${OUT}/prefix_curve.npz" ] && continue
        [ ! -f "$TRAJ" ] && { echo "  skip missing $TRAJ"; continue; }
        python eval_prefix_stop.py --traj "$TRAJ" \
            --milano-grid "$GRID" --smooth-sigma 1.5 \
            --time-scale "$(_ts "$m")" \
            --nfz-center $NFZ_C --nfz-radius $NFZ_R --mask-nfz-target \
            --out-dir "$OUT"
    done
done

echo "=== Block E: prefix-stop curves (results_nfz_masked/) ==="
for m in "${METHODS_M[@]}"; do
    for s in "${SEEDS[@]}"; do
        TRAJ="${ROOT_M}/${m}_seed${s}/trajectory.npz"
        OUT="${ROOT_M}/${m}_seed${s}/curves"
        [ -f "${OUT}/prefix_curve.npz" ] && continue
        [ ! -f "$TRAJ" ] && { echo "  skip missing $TRAJ"; continue; }
        python eval_prefix_stop.py --traj "$TRAJ" \
            --milano-grid "$GRID" --smooth-sigma 1.5 \
            --time-scale "$(_ts "$m")" \
            --nfz-center $NFZ_C --nfz-radius $NFZ_R --mask-nfz-target \
            --out-dir "$OUT"
    done
done


# ---------------------------------------------------------------------------
# Block F -- aggregated prefix-curve plots (all seeds per method on one plot)
# plot_curves.py groups repeated method names → mean + std band + seed lines.
# ---------------------------------------------------------------------------
echo "=== Block F: aggregated prefix-curve plots (results/) ==="
CURVE_ARGS=""
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        C="${ROOT}/${m}_seed${s}/curves/prefix_curve.npz"
        [ -f "$C" ] && CURVE_ARGS="$CURVE_ARGS --curve ${m}:${C}"
    done
done
[ -n "$CURVE_ARGS" ] && python plot_curves.py $CURVE_ARGS \
    --out-dir "${ROOT}/comparison_all_seeds"

echo "=== Block F: aggregated prefix-curve plots (results_nfz_masked/) ==="
CURVE_ARGS=""
for m in "${METHODS_M[@]}"; do
    for s in "${SEEDS[@]}"; do
        C="${ROOT_M}/${m}_seed${s}/curves/prefix_curve.npz"
        [ -f "$C" ] && CURVE_ARGS="$CURVE_ARGS --curve ${m}:${C}"
    done
done
[ -n "$CURVE_ARGS" ] && python plot_curves.py $CURVE_ARGS \
    --out-dir "${ROOT_M}/comparison_all_seeds"


echo "post_eval complete"