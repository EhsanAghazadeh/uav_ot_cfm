#!/bin/bash
#SBATCH --job-name=sac_velpen_train
#SBATCH --chdir=/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#SBATCH --output=./slurm_output/%A_%a_sac_velpen_train.out
#SBATCH --error=./slurm_output/%A_%a_sac_velpen_train.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --array=0-8

# Re-train SAC with a velocity-norm reward penalty so it produces
# realistic UAV kinematics instead of the L=18 km circles produced by
# the unregularised reward.  We sweep BETA so the chosen value can be
# defended in the paper:
#
#   BETA=0.0  reproduces the old reward (sanity check)
#   BETA=0.10 mild kinematic regularisation
#   BETA=0.20 matched-magnitude to the data-collection term
#
# 3 betas x 3 seeds = 9 array tasks. Pick the BETA that delivers a
# trajectory with arc length L in [1, 3] km on the headline mission
# horizon (T=2560 s); that's the credible operating point.

set -euo pipefail
mkdir -p slurm_output

module purge
module load python/3.11.7
source .venv/bin/activate

BETAS=(0.0 0.10 0.20)
SEEDS=(21 42 63)

i=$SLURM_ARRAY_TASK_ID
B=${BETAS[$((i / 3))]}
S=${SEEDS[$((i % 3))]}

# Encode beta with no decimal point in the dirname for clean glob
B_TAG=$(printf "%s" "$B" | tr -d '.')
OUT=results/sac_velpen${B_TAG}_seed${S}
mkdir -p "$(dirname "$OUT")"

echo "[sac_velpen] task=$i  beta=$B  seed=$S  out=$OUT"

python run_baseline_milano.py --method sac \
    --num-points 25600 --sac-train-steps 500000 \
    --sac-grid-size 50 --sac-comm-range 0.18 --dt 0.1 \
    --sac-vel-penalty "$B" \
    --seed $S --out-dir $OUT
