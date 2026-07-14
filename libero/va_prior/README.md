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

## Train the comparable heads and capacities

Run at least seeds 0, 1, and 2. Change `--horizon 1` for the single-action ablation and
`--backbone siglip` for the frozen-backbone ablation.

`--action-head-size` changes only the continuous-action head width: `small=256`,
`base=512` (the backward-compatible default), and `large=1024`. Encoder and candidate-feature
dimensions stay fixed, so this ablation measures action-head capacity rather than total model size.

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

## Weights & Biases training visualizations

Enable W&B on any training command with:

```bash
python scripts/va_prior_experiment.py train \
  --dataset "${DATASETS[@]}" \
  --output runs/va/flow_s0 \
  --head flow \
  --use-wandb \
  --wandb-project libero-va-prior \
  --wandb-run-name flow_s0 \
  --visualize-every 5 \
  --visualize-k 8
```

Every epoch logs total, continuous-action, and gripper losses for training and validation,
plus learning rate, epoch time, and `val/acc`. Here `val/acc` is macro-averaged best-of-K
candidate recall: a validation frame is a hit when any clustered candidate is within the
data-derived normalized ADE threshold. It uses an oracle only to measure whether the prior
covers the demonstrated motion; it is not closed-loop task success. Configure it with
`--val-k`, `--val-flow-steps`, and `--val-cluster-threshold`.

Every `--visualize-every` epochs (and always on the final epoch), one fixed initial validation
sample per goal is used to log predicted and target histograms for every action degree of
freedom. For each goal the script also restores the demonstration camera and OSC controller,
starts at the recorded end-effector position, integrates controller-scaled candidate and
ground-truth XYZ actions, and projects both trajectories onto a freshly rendered fixed-camera
image. By default, `frontview`, `sideview`, and `birdview` are rendered from the same restored
state. Use, for example, `--visualization-cameras frontview sideview agentview` to choose any
combination. Per-camera PNG files are saved under `OUTPUT/visualizations/epoch_NNNN/` even
when W&B is disabled.

GMM and flow heads sample `--visualize-k` stochastic chunks. A deterministic head has only
one possible chunk and is intentionally plotted once. Visualization uses the raw stochastic
samples before candidate clustering, so all requested chunks contribute to the overlays and
histograms. Use `--wandb-mode offline` to collect a run without uploading immediately, or
`--wandb-mode disabled` for local smoke tests.

## Offline and closed-loop evaluation

```bash
python scripts/va_prior_experiment.py evaluate --checkpoint runs/va/flow_s0/best.pt --output runs/va/flow_s0/test

python scripts/va_prior_experiment.py rollout --checkpoint runs/va/flow_s0/best.pt --bddl-file /data/libero/bddl_files/libero_goal/open_the_top_drawer_and_put_the_bowl_inside.bddl --init-states /data/libero/init_files/libero_goal/open_the_top_drawer_and_put_the_bowl_inside.pruned_init --output runs/va/flow_s0/rollout
```

Add `--use-wandb` to `rollout` to resume the training run stored in the checkpoint and log
`rollout/acc`, `rollout/success_rate`, and per-goal success summaries. Older checkpoints that
do not contain a W&B run id create a separate rollout run linked by checkpoint path.

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

