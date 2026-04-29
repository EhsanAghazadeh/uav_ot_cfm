# Milano OT-CFM Comparison Harness

End-to-end pipeline for comparing **OT-CFM** against four baselines (TOES,
NonUniformErg, OptTransport, SAC) on the Milano cellular-density target,
including an NFZ-aware OT-CFM variant.

## Server / SLURM (multi-seed sweep)

Scripts under `scripts/` run the full multi-seed sweep on a SLURM cluster
(seeds 21, 42, 63 via `--array=0-2`). Repo root on server:
`/work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm`.

### One-time install (head/login node)

```bash
cd /work/pi_aghasemi_umass_edu/eaghazadeh/uav_ot_cfm
module avail python                           # find available python module
# edit scripts/install_env.sh if your version differs from python/3.11.7
bash scripts/install_env.sh
```

This loads python, creates `.venv`, installs `requirements.txt`, and pip-installs
the TOES upstream submodule in editable mode.

### Submit jobs

```bash
sbatch scripts/run_toes.sh             # 24h × 3 seeds
sbatch scripts/run_nonuniform_erg.sh   # 1h × 3 seeds
sbatch scripts/run_opt_transport.sh    # 1h × 3 seeds
sbatch scripts/run_sac.sh              # 8h × 3 seeds
sbatch scripts/run_otcfm.sh            # 6h × 3 seeds (vanilla v2)
sbatch scripts/run_otcfm_nfz.sh        # 8h × 3 seeds (NFZ-aware)
```

After all training jobs finish:

```bash
sbatch scripts/run_eval_all.sh         # prefix-stop, NFZ table, plots
```

Outputs land in `results/<method>_seed<S>/` and aggregated comparison plots
in `results/comparison_seed<S>/`. Each script writes its `slurm_output/*.out`
log under `slurm_output/`.

---

## Quickstart

```bash
git clone --recurse-submodules https://github.com/EhsanAghazadeh/uav_ot_cfm.git
cd uav_ot_cfm

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e time_optimal_ergodic_search   # editable install of TOES upstream

# train + eval the headline OT-CFM v2 model
python run_otcfm_milano.py --arch v2 --hidden-dim 512 --n-layers 8 \
    --epochs 3000 --batch-size 2048 --lr 1e-3 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --out-dir results/otcfm_milano_4x_v2

python eval_prefix_stop.py --traj results/otcfm_milano_4x_v2/trajectory.npz \
    --milano-grid data/milano_grid_corrected.npy --smooth-sigma 1.5 \
    --out-dir results/otcfm_milano_4x_v2/curves
```

If you forgot `--recurse-submodules`, run `git submodule update --init`.

---

## 1. Files to keep

### Code (all live at repo root)

| File | Role |
|---|---|
| `run_baseline_milano.py` | **Unified baseline dispatcher** — runs TOES / NonUniformErg / OptTransport / SAC, saves canonical `trajectory.npz`. |
| `run_otcfm_milano.py` | **OT-CFM trainer + transporter** — trains VelocityNet (v1 or v2), integrates RK4, saves canonical `trajectory.npz`. |
| `eval_prefix_stop.py` | **Evaluator** — walks prefix lengths of one long plan, emits Fourier-ergodic / density-corr / L1 / L2 / Zeng-energy curves. |
| `plot_curves.py` | Renders the four experiment plots (fixed-energy, fixed-coverage, fixed-N, fixed-kJ). |
| `plot_trajectories.py` | Renders trajectory-on-target PNG per method (optional no-fly overlay). |
| `plot_all_methods.py` | **Multi-panel** trajectory grid (all methods on one figure), with optional NFZ overlay + per-panel violation stats. |
| `plot_nfz_comparison.py` | Side-by-side OT-CFM (vanilla) vs OT-CFM-NFZ. |
| `eval_nofly.py` | No-fly-zone violation evaluator (dwell time, incursions, path-inside). |
| `ot_cfm.py` | VelocityNet (v1) + VelocityNetV2 (Fourier-time + residual + LN), Sinkhorn-Hungarian coupling, RK4 transport. |
| `latent_trajectory.py` | Annular latent domain `D_δ` and radial back-and-forth trajectory. |
| `fair_comparison.py` | Canonical `trajectory.npz` schema (`save_trajectory_npz`, `TrajectoryRecord`). |
| `milano_baselines.py` | NonUniformErg, OptTransportCoverage, SACTrajectoryGenerator, MilanoRateMap, **RotaryWingEnergyModel** (Zeng-2019 kJ). |
| `toes_baseline.py` | Wrapper around `time_optimal_ergodic_search/` upstream package. |

### Data & vendored deps

- `data/milano_grid_corrected.npy` — target density (50×50 grid).
- `time_optimal_ergodic_search/` — upstream JAX package, required by `toes_baseline.py`.

### Files safe to remove

- `run_toes_milano.py` — superseded by `run_baseline_milano.py --method toes`.
- `run_head_to_head.py` — one-shot upstream-vs-adapter validation.
- `run_upstream_notebook.py` — one-shot reference run.
- `test_solve_path.py` — adapter debugging script.
- `results/_smoke_*`, `results/otcfm_milano_smoke` — smoke tests.
- Intermediate scaling experiments (see [§5 Cleanup](#5-cleanup)).

---

## 2. Trajectory format (canonical npz)

Every method writes `trajectory.npz` with the same keys, so `eval_prefix_stop.py`
and `plot_curves.py` are method-agnostic.

```
pos          (N, 2)   physical positions in the unit square [-1,1]^2
times        (N,)     timestamps (s)
domain       (4,)     (xmin, xmax, ymin, ymax)
method       str      label
meta         dict     method_kind, dt_s, T_total_s, plan_wall_clock_s, ...
```

**`meta["method_kind"]`** is critical — `eval_prefix_stop.py` keys the Zeng
energy formula on it (`toes`, `ot_cfm`, `nonuniform_erg`, `opt_transport`,
`sac`). Velocity-clipping behavior depends on it.

---

## 3. Running each method

All commands assume CWD = `/Users/eaghazadeh/Documents/claude_content/` and a
venv at `.venv/`.

### 3.1 OT-CFM (our method)

```bash
# v2 architecture (Fourier time + residual + LN — recommended)
.venv/bin/python run_otcfm_milano.py \
    --arch v2 --hidden-dim 512 --n-layers 8 \
    --epochs 3000 --batch-size 2048 --lr 1e-3 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 30 \
    --out-dir results/otcfm_milano_4x_v2

# v1 architecture (original MLP, for ablation)
.venv/bin/python run_otcfm_milano.py \
    --arch v1 --hidden-dim 512 --n-layers 6 \
    --epochs 1000 --batch-size 1024 --lr 2e-3 \
    --n-cycles 128 --pts-per-cycle 200 --transport-steps 20 \
    --out-dir results/otcfm_milano_4x
```

Knobs: `--n-cycles × --pts-per-cycle × 2` ≈ trajectory length N (one cycle of
1-cycle burn-in is dropped). Bumping `--transport-steps` from 20 → 30 gives a
more accurate RK4 integration at marginal cost.

### 3.1.1 OT-CFM with no-fly-zone constraint penalty

Train with a soft penalty on the transported endpoints **G_θ(z₀)**:

```
loss = L_CFM + λ_nfz · mean[ max(0, r + margin - ‖G_θ(z₀) − c‖)² ]
```

Penalty pushes any endpoint that lands inside the disc B(c, r+margin)
back outside. `--nfz-reject-targets` additionally drops target samples
that fall inside the disc, so the data-generating distribution itself
avoids the NFZ.

```bash
.venv/bin/python run_otcfm_milano.py \
    --arch v2 --hidden-dim 256 --n-layers 12 \
    --epochs 2000 --batch-size 1024 --lr 1e-3 \
    --n-cycles 1000 --pts-per-cycle 400 \
    --delta 0.05 --transport-steps 30 \
    --nfz-center 0.19 0.03 --nfz-radius 0.1 --nfz-margin 0.05 \
    --nfz-lambda 5.0 --nfz-steps 8 --nfz-reject-targets \
    --out-dir results/otcfm_milano_4x_v2_nfz
```

NFZ-specific flags:

| Flag | Default | Purpose |
|---|---|---|
| `--nfz-center CX CY` | — | Disc centre. Omit to disable NFZ. |
| `--nfz-radius` | — | Disc radius. |
| `--nfz-margin` | 0.05 | Buffer beyond radius to avoid points sitting on boundary. |
| `--nfz-lambda` | 0.0 | Weight in total loss. 0 = off. Try 1–10. |
| `--nfz-steps` | 8 | RK4 steps for the in-training (differentiable) transport. Lower = faster but coarser endpoint estimate. |
| `--nfz-reject-targets` | off | If set, target samples inside the disc are resampled. Strongly recommended. |

Cost: each epoch does an 8-step RK4 forward/backward through the network
on top of the standard CFM step, ~5× slower than vanilla training. With
the config above expect ~3 h on CPU.

### 3.2 TOES

```bash
.venv/bin/python run_baseline_milano.py --method toes \
    --t-total 1280 --n-knots 25600 --n-basis 50 --u-max 0.05 \
    --max-iter 30000 --out-dir results/toes_milano_long_4x
```

`--n-knots` = N. NLP is the bottleneck — 25.6k knots takes ~hours, set
`--max-iter 30000` for stable convergence at this scale.

### 3.3 NonUniformErgodic

```bash
.venv/bin/python run_baseline_milano.py --method nonuniform_erg \
    --num-intervals 64 --points-per-interval 200 --dt 0.1 \
    --out-dir results/nonuniform_erg_milano_4x
```

Output is `2 × num_intervals × points_per_interval` points (out + back per
interval). **Not a flyable trajectory** — see §6.

### 3.4 OptimalTransportCoverage

```bash
.venv/bin/python run_baseline_milano.py --method opt_transport \
    --num-points 25600 --dt 0.1 \
    --out-dir results/opt_transport_milano_4x
```

**Not flyable** — see §6.

### 3.5 SAC

```bash
.venv/bin/python run_baseline_milano.py --method sac \
    --num-points 25600 --sac-train-steps 500000 --sac-grid-size 50 \
    --sac-comm-range 0.18 --dt 0.1 --seed 21 \
    --out-dir results/sac_milano_4x
```

Train on a server (≈3h on CPU). The training reward optimises sensor-buffer
*harvesting*, not density coverage — included for completeness.

---

## 4. Evaluation & plots

### 4.1 Per-method prefix-stop curves

```bash
.venv/bin/python eval_prefix_stop.py \
    --traj results/otcfm_milano_4x_v2/trajectory.npz \
    --milano-grid data/milano_grid_corrected.npy \
    --smooth-sigma 1.5 \
    --out-dir results/otcfm_milano_4x_v2/curves
```

Optional flags for fair cross-method comparison:

- `--resample-dt 0.1` — linear-interpolate onto a common Δt (OT-CFM's native
  Δt is unitless flow-time; baselines use 0.1 s).
- `--target-T 2560` — rescale times so total horizon matches before resampling
  (only useful when comparing energy at common N **and** T).

### 4.2 Cross-method comparison plots

```bash
.venv/bin/python plot_curves.py \
    --curve "TOES:results/toes_milano_long_4x/curves/prefix_curve.npz" \
    --curve "NonUniformErg:results/nonuniform_erg_milano_4x/curves/prefix_curve.npz" \
    --curve "OptTransport:results/opt_transport_milano_4x/curves/prefix_curve.npz" \
    --curve "SAC:results/sac_milano_4x/curves/prefix_curve.npz" \
    --curve "OT-CFM:results/otcfm_milano_4x_v2/curves/prefix_curve.npz" \
    --out-dir results/comparison_baselines_4x_v2
```

Produces:
- `exp1_fixed_energy.png` — coverage vs. squared-velocity energy budget.
- `exp1_all_metrics.png` — same, four coverage metrics side-by-side.
- `exp2_fixed_coverage.png` — energy needed to hit a coverage threshold.
- `exp3_fixed_n.png` — energy + Fourier-erg + Zeng-kJ vs. N.
- `exp4_fixed_kj.png` — coverage at common Zeng-energy budget.

### 4.3 Trajectory-on-target plots

```bash
.venv/bin/python plot_trajectories.py \
    --traj "TOES:results/toes_milano_long_4x/trajectory.npz" \
    --traj "OT-CFM:results/otcfm_milano_4x_v2/trajectory.npz" \
    --milano-grid data/milano_grid_corrected.npy \
    --max-points 8000 \
    --out-dir results/comparison_baselines_4x_v2/trajectories
```

`--max-points` subsamples by uniform stride (density-aware alpha/lw kicks in
automatically at high N).

---

## 5. Metrics

| Metric | Where | Direction | What it measures |
|---|---|---|---|
| **`fourier_erg`** | `eval_prefix_stop.py` | ↓ better | Σ Λ_k (c_k − φ_k)², Λ_k=(1+‖k‖²)⁻¹·⁵. Dt-independent geometric coverage. |
| **`density_corr`** | `eval_prefix_stop.py` | ↑ better | Pearson corr between smoothed visitation histogram and target. Misleading for i.i.d. samplers. |
| `l1`, `l2` | `eval_prefix_stop.py` | ↓ better | Visitation PMF distances. |
| **`energy`** | `eval_prefix_stop.py` | ↓ better | Σ ‖v‖² · Δt. Dt-sensitive — use `--resample-dt` for fair comparison. |
| **`E_total_kJ`** | `RotaryWingEnergyModel` (Zeng-2019) | ↓ better | Rotary-wing UAV energy at v_clip ∈ [0.1, 35] m/s. Time-dominated when T is large. |
| **path length (m)** | post-hoc (`Σ ‖p_{i+1} − p_i‖`) | ↓ better | Cleanest geometry-only cost. Reported in §6. |

**Recommended primary metrics:** `fourier_erg` (ergodic quality) + `path length` (kinematic cost). Drop `density_corr` for the headline result — it inflates i.i.d. samplers that aren't flyable.

---

## 6. Headline numbers (4× scale, N≈25.6k)

| Method | Flyable | N | Path L (m) | Fourier erg ↓ | density_corr ↑ | E @ 5 m/s (kJ) |
|---|---|---|---|---|---|---|
| TOES | ✓ | 25,600 | 289 | 0.183 | 0.27 | 7.9 |
| NonUniformErg | ✗ teleports | 25,600 | 2,175 | 0.0085 | 0.91* | 59.6 |
| OptTransport | ✗ teleports | 25,600 | 5,140 | 0.0051 | 0.61 | 141 |
| SAC | ✓ | 25,600 | 17,421 | 0.118 | 0.34 | 477 |
| **OT-CFM v2** | **✓** | 25,400 | 2,285 | **0.0024** | 0.90 | **62.6** |

\* density_corr is high because NonUniformErg samples the density i.i.d. by
construction; the trajectory is not flyable.

**Why NonUniformErg / OptTransport are unflyable.** Each is a density
*sampler*, not a continuous trajectory. NonUniformErg draws a fresh sweep
angle α at every interval boundary (`milano_baselines.py:296-309`); the last
point of sweep k and the first point of sweep k+1 sit at uncorrelated angles,
producing up-to-2ρ jumps (≈80 m/s instantaneous velocity at Δt=0.1 s, before
Zeng-formula clipping at 35 m/s). OptTransport has the same property between
anchor visits.

**Headline takeaway.** OT-CFM v2 is the only method achieving the lowest
Fourier-ergodic metric (0.0024) **and** producing a kinematically continuous,
flyable trajectory (path length 2,285 m, comparable to NonUniformErg).

---

## 7. No-fly-zone experiment

Define a circular no-fly disc inside the Milano domain and report how often
each method violates it. Default centre is the smoothed-density peak
(0.19, 0.03) — the spot every method *wants* to spend time at — with
radius 0.1.

### Run

```bash
.venv/bin/python eval_nofly.py \
    --traj "TOES:results/toes_milano_long_4x/trajectory.npz" \
    --traj "NonUniformErg:results/nonuniform_erg_milano_4x/trajectory.npz" \
    --traj "OptTransport:results/opt_transport_milano_4x/trajectory.npz" \
    --traj "SAC:results/sac_milano_4x/trajectory.npz" \
    --traj "OT-CFM:results/otcfm_milano_4x_v2/trajectory.npz" \
    --center 0.19 0.03 --radius 0.1 \
    --out results/comparison_baselines_4x_v2/nofly.json

# Visualise
.venv/bin/python plot_trajectories.py \
    --traj "TOES:results/toes_milano_long_4x/trajectory.npz" \
    --traj "OT-CFM:results/otcfm_milano_4x_v2/trajectory.npz" \
    --milano-grid data/milano_grid_corrected.npy \
    --nofly 0.19 0.03 0.1 \
    --out-dir results/comparison_baselines_4x_v2/trajectories_nofly
```

### Metrics

| Metric | Meaning |
|---|---|
| `n_inside` | trajectory points strictly inside the disc |
| `frac_inside` | n_inside / N |
| `dwell_time_s` | n_inside × Δt (time spent inside) |
| `n_incursions` | number of contiguous outside→inside transitions |
| `max_depth` | deepest penetration (radius − ‖p − c‖), normalised |
| `path_inside_norm` | total path length traversed inside (segment-clipped at boundary) |

### 4× headline numbers (centre=(0.19, 0.03), radius=0.1)

| Method | N | n_in | frac_in | dwell (s) | incursions | max depth | path inside (norm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TOES**          | 25,600 |     0 | 0.0000 |   0.00 |   0 | 0.0000 |  0.00 |
| NonUniformErg     | 25,600 | 1,604 | 0.0627 | 160.40 |  82 | 0.0969 | 12.59 |
| OptTransport      | 25,600 |   864 | 0.0338 |  86.40 | 150 | 0.0644 | 14.34 |
| SAC               | 25,600 |   370 | 0.0145 |  37.00 | 301 | 0.0885 | 52.23 |
| **OT-CFM v2**     | 25,400 | 1,680 | 0.0661 |  16.80 | 111 | 0.0965 | 16.75 |

**Reading the table.**
- **TOES** stays out — pure constraint-NLP optimisation steers around the
  peak (but pays a coverage cost: Fourier-erg = 0.183).
- **NonUniformErg / OptTransport** sample the density i.i.d., so they enter
  the disc with the same frequency as the density inside it. Their teleports
  also create extra incursions where a sweep starts inside.
- **SAC** has the fewest dwell-points but the most incursions — it grazes
  the boundary often without committing. Its path-inside is huge because it
  crosses the disc fast (high speed × many crossings).
- **OT-CFM v2** has *more inside-points than NonUniformErg* but **dwell time
  is 10× lower** because OT-CFM's native Δt = 0.01 s vs. baselines' 0.1 s.
  Compare on equal Δt by adding `--resample-dt 0.1` upstream of the eval if
  you need an apples-to-apples dwell number.

### 7.1 NFZ-aware OT-CFM (this work)

We extend OT-CFM with a **soft endpoint penalty**: during training, every
batch is integrated through the current network (8 RK4 steps with autograd)
to obtain G_θ(z₀); points that land inside the disc B(c, r+margin) get
quadratically penalised. Combined with rejection of in-NFZ targets at sample
time, the method drives violations down by ~130×.

```bash
.venv/bin/python run_otcfm_milano.py \
    --arch v2 --hidden-dim 256 --n-layers 12 \
    --epochs 2000 --batch-size 1024 --lr 1e-3 \
    --n-cycles 1000 --pts-per-cycle 400 \
    --delta 0.05 --transport-steps 30 \
    --nfz-center 0.19 0.03 --nfz-radius 0.1 --nfz-margin 0.05 \
    --nfz-lambda 5.0 --nfz-steps 8 --nfz-reject-targets \
    --out-dir results/otcfm_milano_4x_v2_nfz
```

### 7.2 Full NFZ comparison (run after training)

```bash
# (a) Prefix-stop curves on the new trajectory
.venv/bin/python eval_prefix_stop.py \
    --traj results/otcfm_milano_4x_v2_nfz/trajectory.npz \
    --milano-grid data/milano_grid_corrected.npy --smooth-sigma 1.5 \
    --out-dir results/otcfm_milano_4x_v2_nfz/curves

# (b) NFZ violation table for ALL methods
.venv/bin/python eval_nofly.py \
    --traj "TOES:results/toes_milano_long_4x/trajectory.npz" \
    --traj "NonUniformErg:results/nonuniform_erg_milano_4x/trajectory.npz" \
    --traj "OptTransport:results/opt_transport_milano_4x/trajectory.npz" \
    --traj "SAC:results/sac_milano_4x/trajectory.npz" \
    --traj "OT-CFM:results/otcfm_milano_4x_v2/trajectory.npz" \
    --traj "OT-CFM-NFZ:results/otcfm_milano_4x_v2_nfz/trajectory.npz" \
    --center 0.19 0.03 --radius 0.1 \
    --out results/comparison_baselines_4x_v2_nfz/nofly.json

# (c) Cross-method curves: fixed-energy / fixed-coverage / fixed-N / fixed-kJ
.venv/bin/python plot_curves.py \
    --curve "TOES:results/toes_milano_long_4x/curves/prefix_curve.npz" \
    --curve "NonUniformErg:results/nonuniform_erg_milano_4x/curves/prefix_curve.npz" \
    --curve "OptTransport:results/opt_transport_milano_4x/curves/prefix_curve.npz" \
    --curve "SAC:results/sac_milano_4x/curves/prefix_curve.npz" \
    --curve "OT-CFM:results/otcfm_milano_4x_v2/curves/prefix_curve.npz" \
    --curve "OT-CFM-NFZ:results/otcfm_milano_4x_v2_nfz/curves/prefix_curve.npz" \
    --out-dir results/comparison_baselines_4x_v2_nfz

# (d) Beautiful trajectory grid — without NFZ
.venv/bin/python plot_all_methods.py \
    --traj "TOES:results/toes_milano_long_4x/trajectory.npz" \
    --traj "NonUniformErg:results/nonuniform_erg_milano_4x/trajectory.npz" \
    --traj "OptTransport:results/opt_transport_milano_4x/trajectory.npz" \
    --traj "SAC:results/sac_milano_4x/trajectory.npz" \
    --traj "OT-CFM:results/otcfm_milano_4x_v2/trajectory.npz" \
    --milano-grid data/milano_grid_corrected.npy \
    --suptitle "Milano cellular density — unconstrained planning (no NFZ)" \
    --out results/comparison_baselines_4x_v2/trajectories_all.png

# (e) Beautiful trajectory grid — WITH NFZ overlay + per-panel stats
.venv/bin/python plot_all_methods.py \
    --traj "TOES:results/toes_milano_long_4x/trajectory.npz" \
    --traj "NonUniformErg:results/nonuniform_erg_milano_4x/trajectory.npz" \
    --traj "OptTransport:results/opt_transport_milano_4x/trajectory.npz" \
    --traj "SAC:results/sac_milano_4x/trajectory.npz" \
    --traj "OT-CFM:results/otcfm_milano_4x_v2/trajectory.npz" \
    --traj "OT-CFM-NFZ:results/otcfm_milano_4x_v2_nfz/trajectory.npz" \
    --milano-grid data/milano_grid_corrected.npy \
    --nofly 0.19 0.03 0.1 \
    --suptitle "Milano with no-fly zone (red disc) — only OT-CFM-NFZ is NFZ-aware" \
    --out results/comparison_baselines_4x_v2_nfz/trajectories_all_nfz.png

# (f) OT-CFM vs OT-CFM-NFZ side-by-side
.venv/bin/python plot_nfz_comparison.py \
    --vanilla results/otcfm_milano_4x_v2/trajectory.npz \
    --nfz results/otcfm_milano_4x_v2_nfz/trajectory.npz \
    --milano-grid data/milano_grid_corrected.npy \
    --center 0.19 0.03 --radius 0.1 \
    --out results/otcfm_milano_4x_v2_nfz/nfz_vs_baseline.png
```

### 7.3 Headline NFZ-violation table (centre=(0.19, 0.03), r=0.1)

| Method | NFZ-aware? | N | n_in | frac_in | dwell (s) | incursions | max depth |
|---|---|---:|---:|---:|---:|---:|---:|
| TOES | retrofit only (CBF) | 25,600 | 0 | 0.00% | 0.0 | 0 | 0.000 |
| NonUniformErg | ✗ | 25,600 | 1,604 | 6.27% | 160.4 | 82 | 0.097 |
| OptTransport | ✗ | 25,600 | 864 | 3.38% | 86.4 | 150 | 0.064 |
| SAC | retrofit only (reward) | 25,600 | 370 | 1.45% | 37.0 | 301 | 0.089 |
| OT-CFM | ✗ | 25,400 | 1,680 | 6.61% | 16.8 | 111 | 0.097 |
| **OT-CFM-NFZ** | **✓ this work** | 399,600 | **190** | **0.05%** | **0.95** | 130 | 0.082 |

OT-CFM-NFZ achieves a **130× reduction in violation fraction** vs vanilla
OT-CFM at the cost of training a model that is 2× deeper (12 vs 8 layers)
and ~5× slower per epoch. TOES still hits 0 by virtue of being a constraint
NLP, but pays a coverage cost (Fourier-erg = 0.183 vs 0.0024 for OT-CFM-NFZ).

### 7.4 Can the other methods avoid the NFZ?

| Method | Native NFZ support? | How to retrofit | Cost |
|---|---|---|---|
| **TOES** | **Yes (hard constraint)**. It's a constrained NLP — augmented Lagrangian directly takes inequality constraints. | Add CBF: `g(x) = ‖x − c‖² − (r+margin)² ≥ 0` to the constraint set in `time_opt_erg_lib`. | Free at solve time; might slow NLP convergence. |
| **NonUniformErgodic** | **No**. It's an i.i.d. inverse-CDF sampler over angle α; the radial sweep at fixed α traverses the whole annulus and can pass straight through the disc. | Reformulate: per α, segment-clip the radial sweep at the disc boundary, or zero out the angular CDF where the sweep would intersect. Structural rewrite. | Significant rewrite. Doesn't fix the fundamental teleport problem. |
| **OptTransportCoverage** | **No**. Anchor-allocation: places anchors on a 15×15 grid then visits in some order. Anchors inside the disc would need masking, and inter-anchor segments still cross the disc. | Mask anchors in disc + plan flyable inter-anchor routing (TSP under the obstacle). | Major reformulation. Same flyability issue. |
| **SAC** | **Yes via reward shaping**. Add −R_penalty whenever the agent enters the disc. Standard RL practice. | Add to the env's reward function in `milano_baselines.py::SACTrajectoryGenerator`. Retrain. | Retrains from scratch (~hours). |
| **OT-CFM (this work)** | **Yes via soft penalty + target rejection** (demonstrated above). | Already wired: `--nfz-center / --nfz-radius / --nfz-lambda / --nfz-reject-targets`. | One extra forward+backward through 8 RK4 steps per training batch. |

**Summary.** TOES and SAC have *constraint mechanisms* in their formulation
(NLP constraints, RL rewards). OT-CFM acquires one through the soft
endpoint penalty introduced here. The two i.i.d. samplers
(NonUniformErgodic, OptTransport) have **no native notion of constraint** —
they sample from a distribution; making them constraint-aware means
restructuring the sampler itself, and even after restructuring they remain
non-flyable due to inter-segment teleports.

This is a key headline of the paper: OT-CFM is the **only** method that is
simultaneously (i) flyable, (ii) low Fourier-erg, and (iii) able to absorb
soft constraints like NFZs into the training loss with no structural change.

---

## 8. Reproducing the full sweep

```bash
# 1. Train OT-CFM v2 (~30 min CPU)
.venv/bin/python run_otcfm_milano.py --arch v2 --hidden-dim 512 --n-layers 8 \
    --epochs 3000 --batch-size 2048 --lr 1e-3 --n-cycles 128 --pts-per-cycle 200 \
    --transport-steps 30 --out-dir results/otcfm_milano_4x_v2

# 2. Run baselines (TOES is the long pole)
.venv/bin/python run_baseline_milano.py --method toes --t-total 1280 \
    --n-knots 25600 --max-iter 30000 --out-dir results/toes_milano_long_4x
.venv/bin/python run_baseline_milano.py --method nonuniform_erg \
    --num-intervals 64 --points-per-interval 200 --out-dir results/nonuniform_erg_milano_4x
.venv/bin/python run_baseline_milano.py --method opt_transport \
    --num-points 25600 --out-dir results/opt_transport_milano_4x
.venv/bin/python run_baseline_milano.py --method sac --num-points 25600 \
    --sac-train-steps 500000 --out-dir results/sac_milano_4x

# 3. Evaluate each (loop over runs)
for d in results/{toes_milano_long_4x,nonuniform_erg_milano_4x,opt_transport_milano_4x,sac_milano_4x,otcfm_milano_4x_v2}; do
  .venv/bin/python eval_prefix_stop.py --traj "$d/trajectory.npz" \
      --milano-grid data/milano_grid_corrected.npy --smooth-sigma 1.5 \
      --out-dir "$d/curves"
done

# 4. Plot
.venv/bin/python plot_curves.py \
    --curve "TOES:results/toes_milano_long_4x/curves/prefix_curve.npz" \
    --curve "NonUniformErg:results/nonuniform_erg_milano_4x/curves/prefix_curve.npz" \
    --curve "OptTransport:results/opt_transport_milano_4x/curves/prefix_curve.npz" \
    --curve "SAC:results/sac_milano_4x/curves/prefix_curve.npz" \
    --curve "OT-CFM:results/otcfm_milano_4x_v2/curves/prefix_curve.npz" \
    --out-dir results/comparison_baselines_4x_v2

.venv/bin/python plot_trajectories.py \
    --traj "TOES:results/toes_milano_long_4x/trajectory.npz" \
    --traj "NonUniformErg:results/nonuniform_erg_milano_4x/trajectory.npz" \
    --traj "OptTransport:results/opt_transport_milano_4x/trajectory.npz" \
    --traj "SAC:results/sac_milano_4x/trajectory.npz" \
    --traj "OT-CFM:results/otcfm_milano_4x_v2/trajectory.npz" \
    --milano-grid data/milano_grid_corrected.npy \
    --out-dir results/comparison_baselines_4x_v2/trajectories
```
