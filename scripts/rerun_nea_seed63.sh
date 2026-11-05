#!/bin/bash
#SBATCH --job-name=nea_seed63_rerun
#SBATCH --output=./slurm_output/%j_nea_seed63_rerun.out
#SBATCH --error=./slurm_output/%j_nea_seed63_rerun.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=03:00:00

# Rerun of the +NEA single-disc configuration at seed 63, which diverges
# reproducibly (acceleration-penalty feedback within ~200 epochs). It is
# reported in the paper as a characterized failure case.
# Exact config from results_nfz_masked/otcfm_energy_la1e-1_nfz_l500_le1e-3_seed63/
# plan_meta.json (identical to seeds 21/42 except the seed). Written to a
# _rerun directory so the original run is preserved.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

OUT=results_nfz_masked/otcfm_energy_la1e-1_nfz_l500_le1e-3_seed63_rerun
mkdir -p "$OUT"

python run_otcfm_milano.py \
    --arch v1 --hidden-dim 256 --n-layers 12 \
    --epochs 3000 --batch-size 2048 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --nfz-center 0.19 0.03 --nfz-radius 0.1 --nfz-margin 0.05 \
    --nfz-lambda 500 --nfz-steps 8 \
    --energy-lambda 1e-3 --energy-chunk-size 64 --energy-steps 8 \
    --energy-n-cycles 50 --energy-pts-per-cycle 200 \
    --acc-lambda 1e-1 \
    --mask-nfz-target \
    --domain-size 8.0 \
    --force-retrain \
    --seed 63 --out-dir "$OUT"

# Immediately verify the trajectory is finite and report headline metrics.
python - <<'EOF'
import numpy as np, json
d = np.load("results_nfz_masked/otcfm_energy_la1e-1_nfz_l500_le1e-3_seed63_rerun/trajectory.npz")
n_nan = int(np.isnan(d["pos"]).sum())
print(f"[verify] pos NaNs: {n_nan}  shape: {d['pos'].shape}")
if n_nan:
    raise SystemExit("RERUN ALSO PRODUCED NaN — training diverged again")
EOF
echo "[rerun] done, trajectory is finite"
