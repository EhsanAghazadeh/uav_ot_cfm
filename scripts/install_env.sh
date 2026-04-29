#!/bin/bash
# One-time setup: load python, create venv, install deps, install TOES upstream.
# Run on the head node (or any login node), NOT inside an sbatch job.
#
# Usage:
#   cd /work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
#   bash scripts/install_env.sh

set -euo pipefail

# ---- 1. Find an available Python module ----
# UMass Unity cluster: list candidates with `module avail python`.
# Most environments have one of these.
module purge
module load python/3.11.7   # adjust if `module avail python` shows a different version

# ---- 2. Create venv at repo root ----
cd "$(dirname "$0")/.."
if [ ! -d .venv ]; then
    python -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip

# ---- 3. Install dependencies ----
pip install -r requirements.txt

# ---- 4. Editable install of vendored TOES upstream ----
git submodule update --init --recursive
pip install -e time_optimal_ergodic_search

echo "Environment ready. To activate later:"
echo "    module load python/3.11.7"
echo "    source .venv/bin/activate"
