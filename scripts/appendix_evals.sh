#!/bin/bash
# Appendix evaluations on the saved Table 1 trajectories (CPU, minutes).
# Run from the repo root after the Milano runs in results_nfz_masked/ and
# results_nfz_masked_multi_disc/ exist (or inside a batch job).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Closed-loop tracking error (PD-tracked double integrator)
python -m experiments.appendix.eval_tracking_error --root results_nfz_masked \
    --out results/appendix/tracking/tracking_1d.json
# Turnaround-inclusive energy and mission-time accounting
python -m experiments.appendix.eval_turnaround_accounting --root results_nfz_masked \
    --out results/appendix/turnaround/accounting_1d.json
# Per-cycle NFZ rejection probability for offline expurgation
python -m experiments.appendix.eval_expurgation_pv
# Fleet separation (one shared map, 2-20 agents)
python -m experiments.appendix.eval_fleet_separation --model-dir results/otcfm_seed42 \
    --fleet-sizes 2 5 10 20 --domain-size 200.0 \
    --out-dir results/appendix/fleet_separation
