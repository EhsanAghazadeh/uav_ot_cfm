#!/bin/bash
# Appendix: physical-scale sensitivity sweep.
# Re-evaluates the exact Table-1 trajectories at several physical domain
# sizes, holding the paper's time protocol fixed (TIME_SCALE=10, cap 30 m/s).
# The 200 m point matches Table 1 (scripts/table1_eval.sh) for all methods
# except TOES and FMEC, which Table 1 evaluates at their native time scales.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
PY=.venv/bin/python

SEEDS=(21 42 63)
METHODS=(toes nonuniform_erg opt_transport sac
         fmec_stein fmec_sinkhorn
         otcfm
         otcfm_nfz_l5
         otcfm_energy_lambda1e-2
         otcfm_energy_la1e-4
         otcfm_energy_lambda1e-3_nfz_l500
         otcfm_energy_la1e-1_nfz_l500_le1e-3)

run_cfg () {  # $1=ROOT  $2=TAG  $3=NFZ_FLAGS...
    local ROOT=$1 TAG=$2; shift 2
    local NFZ_FLAGS=("$@")
    local ARGS=""
    for m in "${METHODS[@]}"; do
        for s in "${SEEDS[@]}"; do
            T="${ROOT}/${m}_seed${s}/trajectory.npz"
            [ -f "$T" ] && ARGS="$ARGS --traj ${m}:${T}"
        done
    done
    for D in 8 80 200 800 2000; do
        OUT=results/appendix/scale_sweep/${TAG}_D${D}
        mkdir -p "$OUT"
        echo "=== ${TAG} domain=${D} m ==="
        $PY -m evaluation.eval_fair $ARGS \
            --milano-grid data/milano_grid_corrected.npy \
            --domain-size "$D" --time-scale 10.0 \
            --v-clip-realistic 30.0 \
            "${NFZ_FLAGS[@]}" --mask-nfz-target \
            --out-dir "$OUT" > "$OUT/log.txt" 2>&1
    done
    # Speed-preserving check: domain 2000 with time-scale 100 should give
    # the same wattage as domain 200 with time-scale 10.
    OUT=results/appendix/scale_sweep/${TAG}_D2000_ts100
    mkdir -p "$OUT"
    $PY -m evaluation.eval_fair $ARGS \
        --milano-grid data/milano_grid_corrected.npy \
        --domain-size 2000 --time-scale 100.0 \
        --v-clip-realistic 30.0 \
        "${NFZ_FLAGS[@]}" --mask-nfz-target \
        --out-dir "$OUT" > "$OUT/log.txt" 2>&1
}

run_cfg results_nfz_masked 1d --nfz 0.19 0.03 0.10
run_cfg results_nfz_masked_multi_disc md \
    --nfz 0.5 0.0 0.2 --nfz -0.5 0.0 0.2 --nfz 0.0 0.5 0.2 --nfz 0.0 -0.5 0.2
echo DONE
