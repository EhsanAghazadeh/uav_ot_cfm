# Single-NFZ-center evaluation pipeline.
#
# Produces, for ONE choice of NFZ disc (NFZ_C, NFZ_R):
#   - prefix-stop curves per (method, seed)
#   - cross-method curve overlay
#   - trajectory grid with NFZ overlay
#   - fair-comparison metrics (rescaled UAV-mission protocol)
#   - Pareto plot (3 views) on the masked target
#   - velocity distribution figure
#
# IMPORTANT: every method listed in METHODS must have been TRAINED
# against the same (NFZ_C, NFZ_R). Mixing methods trained against
# different masked targets in one Pareto is a category error.
#
# Edit the three settings below for each NFZ condition you want to
# report (e.g. high-density disc 0.19 0.03, low-density disc -0.6 -0.6,
# alt-density disc 0.6 0.6) and submit one job per condition.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

# ------------------- Edit these for each condition -------------------------
NFZ_C="0.3 0.3"
NFZ_R="0.30"
OUT_TAG="nfz_high_density_0p3_0p3"      # subdir suffix; one per (NFZ_C, NFZ_R)
METHODS=(
    nonuniform_erg_nfz_center_0_3_0_3
    # add baselines / OT-CFM variants trained against this same center:
    # opt_transport_nfz_center_0_3_0_3
    # toes_nfz_center_-0_6_-0_6
    otcfm_mask_vel_lv0p1_nfz_l10_center_0_3_0_3
    otcfm_mask_vel_lv0p1_center_0_3_0_3
    otcfm_mask_only_la1e-2_center_0_3_0_3
    otcfm_mask_plus_penalty_la10_le1_center_0_3_0_3
    otcfm_mask_only_center_0_3_0_3_radius_0_3
    otcfm_mask_plus_penalty_center_0_3_0_3_radius_0_3
)
# ---------------------------------------------------------------------------

GRID="data/milano_grid_corrected.npy"
DOMAIN_M=200.0
TIME_SCALE=10.0
V_CAP_REALISTIC=30.0
SEEDS=(21 42 63)

ROOT="results_nfz_masked_03_03"
COMPARISON_DIR="${ROOT}/comparison_${OUT_TAG}"
FAIR_DIR="${ROOT}/fair_metrics_${OUT_TAG}"
VEL_DIR="${ROOT}/velocity_dist_${OUT_TAG}"
mkdir -p "$COMPARISON_DIR" "$FAIR_DIR" "$VEL_DIR"

echo "[eval_nfz_focused] NFZ_C=($NFZ_C)  NFZ_R=$NFZ_R  tag=$OUT_TAG"
echo "                   methods: ${METHODS[*]}"

# 1) Prefix-stop curves per (method, seed) -- skipped if cached
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        TRAJ="${ROOT}/${m}_seed${s}/trajectory.npz"
        OUT="${ROOT}/${m}_seed${s}/curves"
        if [ ! -f "$TRAJ" ]; then echo "  skip missing $TRAJ"; continue; fi
        if [ -f "${OUT}/prefix_curve.npz" ]; then
            echo "  skip cached $OUT"; continue
        fi
        python -m evaluation.eval_prefix_stop --traj "$TRAJ" \
            --milano-grid "$GRID" --smooth-sigma 1.5 \
            --time-scale $TIME_SCALE \
            --nfz-center $NFZ_C --nfz-radius $NFZ_R --mask-nfz-target \
            --out-dir "$OUT"
    done
done

# 2) Cross-method curve overlay (one figure across all methods/seeds)
ARGS=""
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        C="${ROOT}/${m}_seed${s}/curves/prefix_curve.npz"
        [ -f "$C" ] && ARGS="$ARGS --curve ${m}:${C}"
    done
done
if [ -n "$ARGS" ]; then
    python -m plotting.plot_curves $ARGS --out-dir "$COMPARISON_DIR"
fi

# 3) Trajectory grid w/ NFZ overlay (one panel per method+seed)
ARGS=""
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        T="${ROOT}/${m}_seed${s}/trajectory.npz"
        [ -f "$T" ] && ARGS="$ARGS --traj ${m}_seed${s}:${T}"
    done
done
if [ -n "$ARGS" ]; then
    python -m plotting.plot_all_methods $ARGS \
        --milano-grid "$GRID" \
        --nofly $NFZ_C $NFZ_R \
        --suptitle "NFZ-masked Milano @ ($NFZ_C), r=$NFZ_R" \
        --out "${COMPARISON_DIR}/trajectories_all.png"
fi

# 4) Fair-comparison metrics on the masked target (rescaled)
ARGS=""
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        T="${ROOT}/${m}_seed${s}/trajectory.npz"
        [ -f "$T" ] && ARGS="$ARGS --traj ${m}:${T}"
    done
done
if [ -n "$ARGS" ]; then
    python -m evaluation.eval_fair $ARGS \
        --milano-grid "$GRID" \
        --domain-size $DOMAIN_M --time-scale $TIME_SCALE \
        --v-clip-realistic $V_CAP_REALISTIC \
        --nfz-center $NFZ_C --nfz-radius $NFZ_R --mask-nfz-target \
        --out-dir "$FAIR_DIR"
    python -m plotting.plot_pareto \
        --per-run "${FAIR_DIR}/per_run.json" \
        --out-dir "$FAIR_DIR"
    # Velocity distribution (3-panel)
    python -m plotting.plot_velocity_distribution $ARGS \
        --domain-size $DOMAIN_M --time-scale $TIME_SCALE \
        --v-cap-realistic $V_CAP_REALISTIC \
        --out-dir "$VEL_DIR"
fi

echo "NFZ-focused evaluation complete -> $FAIR_DIR (and ${COMPARISON_DIR})"