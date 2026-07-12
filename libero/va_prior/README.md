# Vision-action action-prior experiment

This module tests whether a language-free visual policy trained on multiple goals in the same
scene can produce a useful multi-modal action-chunk prior on LIBERO_GOAL. It lives outside the lifelong
training stack so that trajectory splits and baseline comparisons are explicit.

## Setup and data

Install LIBERO normally, then install the newer vision-model dependency in a compatible
environment:

```bash
pip install -r requirements.txt
pip install -r requirements-va.txt
```

Install the CUDA-enabled PyTorch build appropriate for the server first; like the upstream
LIBERO requirements, `requirements-va.txt` intentionally does not select a CUDA build.

Pass multiple `libero_goal/*_demo.hdf5` files from the shared tabletop scene. The model is
never given their task ids or language; task ids are retained only for grouped metrics. The
script checks the actual action and proprioception dimensions from HDF5. Splits are made by demonstration,
saved in every run directory, and shared by setting the same `--split-seed`. The default
observation history is two frames (`--obs-horizon 2`); this is distinct from the ten-step
future action horizon.

## Train the three comparable heads

Run at least seeds 0, 1, and 2. Change `--horizon 1` for the single-action ablation and
`--backbone siglip` for the frozen-backbone ablation.

```bash
export CUDA_VISIBLE_DEVICES=1

DATASETS=(
  /root/gpufree-data/liumingyu/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA/datasets/libero_goal/*_demo.hdf5
)

echo "找到 ${#DATASETS[@]} 个任务文件"
printf '%s\n' "${DATASETS[@]}"

for seed in 0 1 2; do
  python scripts/va_prior_experiment.py train \
    --dataset "${DATASETS[@]}" \
    --output "runs/va/deterministic_multitask_s${seed}" \
    --head deterministic \
    --backbone dinov2 \
    --seed "$seed" 
done


# Flow
for seed in 0 1 2; do
  python scripts/va_prior_experiment.py train \
    --dataset "${DATASETS[@]}" \
    --output "runs/va/flow_s${seed}" \
    --head flow \
    --backbone dinov2 \
    --seed "$seed"
done
```

`--backbone tiny` is an offline smoke-test option; it is not an experimental result.

## Offline and closed-loop evaluation

```bash
python scripts/va_prior_experiment.py evaluate --checkpoint runs/va/flow_s0/best.pt --output runs/va/flow_s0/test

python scripts/va_prior_experiment.py rollout --checkpoint runs/va/flow_s0/best.pt --bddl-file /data/libero/bddl_files/libero_goal/open_the_top_drawer_and_put_the_bowl_inside.bddl --init-states /data/libero/init_files/libero_goal/open_the_top_drawer_and_put_the_bowl_inside.pruned_init --output runs/va/flow_s0/rollout
```

Offline evaluation writes `metrics.json` and `candidate_pca.png`. In addition to per-goal
metrics, `cross_goal_initial_coverage` uses matched held-out demonstration indices: candidates
from one language-free initial observation are compared with initial action chunks from every
goal. Index-aligned recordings are accepted only when image MAE and proprioception RMSE are
below configurable matching thresholds, so different physical states are not mislabeled as
goal ambiguity. This is the main test of the intended multi-goal prior. Closed-loop evaluation
uses 10 init states, 20 stochastic repeats each, samples from `prior_weights`, executes two
actions, and replans. Collision rate is `null` when the environment does not expose a
collision field; it is never silently reported as zero.

The public prior dictionary contains:

- `candidate_chunks`: `[B, K, H, action_dim]`, denormalized, including gripper;
- `prior_weights`: `[B, K]` (zero for padded clusters);
- `candidate_features`: `[B, K, hidden_dim]`;
- `gripper_chunks`: `[B, K, H]`.

Only offline best-of-K metrics use an oracle. Rollout selection never sees the demonstrated
action. A flow result is useful as a prior if it improves held-out coverage over deterministic
BC and improves coverage or closed-loop success over GMM without relying on excessive
diversity.

After evaluating three seeds per head, aggregate the offline evidence with:

```bash
python scripts/summarize_va_prior.py \
  --deterministic runs/va/deterministic_s{0,1,2}/test/metrics.json \
  --gmm runs/va/gmm_s{0,1,2}/test/metrics.json \
  --flow runs/va/flow_s{0,1,2}/test/metrics.json \
  --output runs/va/summary.json
```

