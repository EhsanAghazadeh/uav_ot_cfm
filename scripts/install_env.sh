#!/bin/bash
# One-time setup: load python, create venv, install deps, install TOES upstream.
# Run on the head node (or any login node), NOT inside an sbatch job.
#
# Usage:
#   cd uav_ot_cfm   (repo root)
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
if [ -f time_optimal_ergodic_search/pyproject.toml ] || [ -f time_optimal_ergodic_search/setup.py ]; then
    pip install -e time_optimal_ergodic_search
else
    # Upstream TOES is source-only here (no setup.py/pyproject.toml), so add it to site-packages.
    python - <<'PY'
import os
import site
import sys

repo_root = os.getcwd()
toes_root = os.path.join(repo_root, "time_optimal_ergodic_search")
pth_name = "time_optimal_ergodic_search.pth"

site_packages = []
try:
    site_packages.extend(site.getsitepackages())
except Exception:
    pass

for sp in site_packages:
    if not sp:
        continue
    # Keep changes scoped to the active virtual environment.
    if sys.prefix not in os.path.abspath(sp):
        continue
    os.makedirs(sp, exist_ok=True)
    pth_path = os.path.join(sp, pth_name)
    with open(pth_path, "w", encoding="utf-8") as f:
        f.write(toes_root + "\n")
    print(f"Wrote {pth_path} -> {toes_root}")
PY
fi

echo "Environment ready. To activate later:"
echo "    module load python/3.11.7"
echo "    source .venv/bin/activate"
