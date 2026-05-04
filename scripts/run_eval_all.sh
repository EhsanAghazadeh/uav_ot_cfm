#!/bin/bash
#SBATCH --job-name=otcfm_eval
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%j_otcfm_eval.out
#SBATCH --error=./slurm_output/%j_otcfm_eval.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

# Run AFTER all training jobs finish.
# Iterates over each results/<method>_seed<S>/trajectory.npz, runs the
# prefix-stop evaluator, no-fly-zone evaluator, and renders trajectory plots.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

METHODS=(toes nonuniform_erg opt_transport sac otcfm otcfm_nfz_l5 otcfm_energy_lambda1e-2 otcfm_energy_la1e-4 otcfm_energy_lambda1e-3_nfz_l500 otcfm_energy_la1e-1_nfz_l500_le1e-3)
SEEDS=(21 42 63)

_ts() { case "$1" in toes*) echo "2.0";; fmec_stein*|fmec_sinkhorn*) echo "7.94";; *) echo "10.0";; esac; }

# 1. prefix-stop curves
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        TRAJ=results/${m}_seed${s}/trajectory.npz
        OUT=results/${m}_seed${s}/curves
        if [ -f "$TRAJ" ]; then
            python eval_prefix_stop.py --traj "$TRAJ" \
                --milano-grid data/milano_grid_corrected.npy \
                --smooth-sigma 1.5 \
                --time-scale "$(_ts "$m")" \
                --nfz-center 0.19 0.03 --nfz-radius 0.10 --mask-nfz-target \
                --out-dir "$OUT"
        else
            echo "skip missing $TRAJ"
        fi
    done
done

# 2. NFZ violation table per seed (centred at density peak)
for s in "${SEEDS[@]}"; do
    ARGS=""
    for m in "${METHODS[@]}"; do
        T=results/${m}_seed${s}/trajectory.npz
        [ -f "$T" ] && ARGS="$ARGS --traj ${m}:${T}"
    done
    if [ -n "$ARGS" ]; then
        python eval_nofly.py $ARGS \
            --center 0.19 0.03 --radius 0.1 \
            --out results/nofly_seed${s}.json
    fi
done

# 3. Cross-method comparison — per seed AND aggregated (all seeds together)
for s in "${SEEDS[@]}"; do
    ARGS=""
    for m in "${METHODS[@]}"; do
        C=results/${m}_seed${s}/curves/prefix_curve.npz
        [ -f "$C" ] && ARGS="$ARGS --curve ${m}:${C}"
    done
    [ -n "$ARGS" ] && python plot_curves.py $ARGS \
        --out-dir results/comparison_seed${s}
done
# Aggregated: all seeds per method on one plot (mean + std band)
AGG_ARGS=""
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        C=results/${m}_seed${s}/curves/prefix_curve.npz
        [ -f "$C" ] && AGG_ARGS="$AGG_ARGS --curve ${m}:${C}"
    done
done
[ -n "$AGG_ARGS" ] && python plot_curves.py $AGG_ARGS \
    --out-dir results/comparison_all_seeds

# 4. Multi-panel trajectory grids per seed (with NFZ overlay)
for s in "${SEEDS[@]}"; do
    ARGS=""
    for m in "${METHODS[@]}"; do
        T=results/${m}_seed${s}/trajectory.npz
        [ -f "$T" ] && ARGS="$ARGS --traj ${m}:${T}"
    done
    [ -n "$ARGS" ] && python plot_all_methods.py $ARGS \
        --milano-grid data/milano_grid_corrected.npy \
        --nofly 0.19 0.03 0.1 \
        --suptitle "Milano (seed=${s}) — NFZ disc highlighted" \
        --out results/comparison_seed${s}/trajectories_all_nfz.png
done

# 5. Fair-comparison metrics (native trajectories, no resampling).
#    Aggregates all (method, seed) into one table + LaTeX + Pareto plots.
FAIR_ARGS=""
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        T=results/${m}_seed${s}/trajectory.npz
        [ -f "$T" ] && FAIR_ARGS="$FAIR_ARGS --traj ${m}:${T}"
    done
done
if [ -n "$FAIR_ARGS" ]; then
    python eval_fair.py $FAIR_ARGS \
        --milano-grid data/milano_grid_corrected.npy \
        --method-time-scale toes:2.0 \
        --nfz-center 0.19 0.03 --nfz-radius 0.10 --mask-nfz-target \
        --out-dir results/fair_metrics
    python plot_pareto.py \
        --per-run results/fair_metrics/per_run.json \
        --out-dir results/fair_metrics
fi

echo "all-method evaluation complete"
