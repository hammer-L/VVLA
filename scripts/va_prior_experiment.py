"""Train and evaluate a language-free, same-scene multi-goal action prior."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, default_collate

from libero.va_prior.data import (ActionStats, LiberoActionChunkDataset, TaskTaggedDataset,
                                  compute_action_stats_multi, trajectory_split)
from libero.va_prior.model import VAPriorModel


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_model(args, sample, stats):
    return VAPriorModel(
        head=args.head, backbone=args.backbone, proprio_dim=sample["proprio"].shape[-1],
        continuous_dim=len(stats.mean), horizon=args.horizon, hidden_dim=args.hidden_dim,
        num_modes=args.num_modes, action_mean=stats.mean, action_std=stats.std,
    )


def combined_dataset(paths, splits, split_name, stats, horizon, obs_horizon):
    datasets = []
    for task_id, path in enumerate(paths):
        base = LiberoActionChunkDataset(path, splits[str(path)][split_name], stats, horizon,
                                        obs_horizon=obs_horizon)
        datasets.append(TaskTaggedDataset(base, task_id))
    return ConcatDataset(datasets)


def train(args):
    seed_everything(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    paths = [str(Path(x)) for x in args.dataset]
    splits = {path: trajectory_split(path, args.split_seed) for path in paths}
    stats = compute_action_stats_multi(paths, {p: splits[p]["train"] for p in paths})
    stats.save(out / "action_stats.json")
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    ds = combined_dataset(paths, splits, "train", stats, args.horizon, args.obs_horizon)
    val = combined_dataset(paths, splits, "val", stats, args.horizon, args.obs_horizon)
    model = build_model(args, ds[0], stats).to(args.device)
    loader = DataLoader(ds, args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val, args.batch_size, num_workers=args.workers)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    best = float("inf")
    config = vars(args).copy()
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.loss(move(batch, args.device)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); running += loss.item()
        model.eval(); values = []
        with torch.no_grad():
            for batch in val_loader:
                values.append(model.loss(move(batch, args.device))[0].item())
        score = float(np.mean(values)) if values else running / max(len(loader), 1)
        print(json.dumps({"epoch": epoch, "train_loss": running / max(len(loader), 1), "val_loss": score}))
        state = {"model": model.state_dict(), "config": config, "stats": {
            "mean": stats.mean.tolist(), "std": stats.std.tolist(),
            "recall_threshold": stats.recall_threshold}, "splits": splits}
        torch.save(state, out / "last.pt")
        if score < best:
            best = score; torch.save(state, out / "best.pt")


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    cfg = argparse.Namespace(**state["config"])
    stats = ActionStats(np.asarray(state["stats"]["mean"], np.float32),
                        np.asarray(state["stats"]["std"], np.float32),
                        state["stats"]["recall_threshold"])
    paths = [cfg.dataset] if isinstance(cfg.dataset, str) else cfg.dataset
    # Backward compatibility with early single-task checkpoints.
    splits = state["splits"]
    if "train" in splits:
        splits = {str(paths[0]): splits}
    ds = combined_dataset(paths, splits, "test", stats, cfg.horizon, getattr(cfg, "obs_horizon", 2))
    model = build_model(cfg, ds[0], stats).to(device)
    model.load_state_dict(state["model"]); model.eval()
    return model, ds, stats, cfg


def masked_f1(pred, target, mask):
    pred, target = pred[mask].bool(), target[mask].bool()
    tp = (pred & target).sum().item(); fp = (pred & ~target).sum().item(); fn = (~pred & target).sum().item()
    return 2 * tp / max(2 * tp + fp + fn, 1)


@torch.no_grad()
def cross_goal_initial_metrics(model, dataset, stats, args):
    """Can one goal-free initial observation cover actions demonstrated for every goal?"""
    bases = [x.dataset for x in dataset.datasets]
    maps = [{key: i for i, key in enumerate(base.index) if key[1] == 0} for base in bases]
    common = sorted(set.intersection(*(set(x) for x in maps)))
    index_aligned = len(common)
    recalls, errors, target_diversity = [], [], []
    per_goal = {i: [] for i in range(len(bases))}
    for key in common:
        items = [bases[i][maps[i][key]] for i in range(len(bases))]
        image_error = max((x["images"] - items[0]["images"]).abs().mean().item() for x in items)
        proprio_error = max(torch.sqrt(((x["proprio"] - items[0]["proprio"]) ** 2).mean()).item() for x in items)
        if image_error > args.match_image_mae or proprio_error > args.match_proprio_rmse:
            continue
        targets = torch.stack([x["continuous"] for x in items]).to(args.device)
        target_diversity.append(torch.pdist(targets.flatten(1)).mean().item() if len(targets) > 1 else 0.0)
        # Average over each task recording as the visual anchor; goal labels remain hidden.
        for anchor in range(len(bases)):
            batch = move(default_collate([items[anchor]]), args.device)
            candidates = model.candidates(batch, args.k, args.flow_steps, args.cluster_threshold)["candidate_chunks"][0, ..., :-1]
            distance = torch.linalg.vector_norm((candidates[:, None] - targets[None]) / model.action_std, dim=-1).mean(-1)
            best = distance.min(0).values
            errors.extend(best.cpu().tolist())
            hit = best <= stats.recall_threshold
            recalls.extend(hit.float().cpu().tolist())
            for goal in range(len(bases)): per_goal[goal].append(float(hit[goal]))
    return {
        "index_aligned_initial_states": index_aligned,
        "state_matched_initial_states": (len(recalls) // (len(bases) ** 2)) if bases else 0,
        "match_image_mae_threshold": args.match_image_mae,
        "match_proprio_rmse_threshold": args.match_proprio_rmse,
        "macro_recall": float(np.mean(recalls)) if recalls else None,
        "best_of_k_ade": float(np.mean(errors)) if errors else None,
        "demonstrated_goal_action_diversity": float(np.mean(target_diversity)) if target_diversity else None,
        "recall_by_goal_id": {str(k): float(np.mean(v)) for k, v in per_goal.items() if v},
    }


@torch.no_grad()
def evaluate(args):
    model, ds, stats, cfg = load_checkpoint(args.checkpoint, args.device)
    loader = DataLoader(ds, args.batch_size, num_workers=args.workers)
    totals = {k: [] for k in ("ade", "fde", "best_of_k_ade", "best_of_k_fde", "recall", "diversity")}
    phase = {"early": [], "middle": [], "late": [], "gripper_event": []}
    per_task = {}
    all_pred, all_true, grip_pred, grip_true, grip_mask = [], [], [], [], []
    for raw in loader:
        batch = move(raw, args.device)
        result = model.candidates(batch, args.k, args.flow_steps, args.cluster_threshold)
        candidates = result["candidate_chunks"][..., :-1]
        truth = batch["continuous"] * model.action_std + model.action_mean
        mask = batch["mask"]
        distance = torch.linalg.vector_norm((candidates - truth[:, None]) / model.action_std, dim=-1)
        masked = distance * mask[:, None]
        per_candidate_ade = masked.sum(-1) / mask.sum(-1, keepdim=True)
        valid_last = mask.sum(-1).long() - 1
        fde = distance.gather(-1, valid_last[:, None, None].expand(-1, candidates.shape[1], 1)).squeeze(-1)
        best = per_candidate_ade.min(-1).values
        weights = result["prior_weights"]
        chosen = weights.argmax(-1)
        rows = torch.arange(len(chosen), device=chosen.device)
        totals["ade"].extend(per_candidate_ade[rows, chosen].cpu().tolist())
        totals["fde"].extend(fde[rows, chosen].cpu().tolist())
        totals["best_of_k_ade"].extend(best.cpu().tolist()); totals["best_of_k_fde"].extend(fde.min(-1).values.cpu().tolist())
        totals["recall"].extend((best <= stats.recall_threshold).float().cpu().tolist())
        for task_id in raw["task_id"].unique().tolist():
            selected = raw["task_id"] == task_id
            bucket = per_task.setdefault(str(task_id), {"best_of_k_ade": [], "recall": []})
            bucket["best_of_k_ade"].extend(best.cpu()[selected].tolist())
            bucket["recall"].extend((best.cpu()[selected] <= stats.recall_threshold).float().tolist())
        for i in range(len(candidates)):
            valid = candidates[i][weights[i] > 0].flatten(1) / model.action_std.repeat(cfg.horizon)
            div = torch.pdist(valid).mean().item() if len(valid) > 1 else 0.0
            totals["diversity"].append(div)
            bucket = "early" if raw["progress"][i] < 1/3 else "middle" if raw["progress"][i] < 2/3 else "late"
            phase[bucket].append(best[i].item())
            if raw["gripper_event"][i]: phase["gripper_event"].append(best[i].item())
        all_pred.append(candidates.cpu()); all_true.append(truth.cpu())
        selected_grip = result["gripper_chunks"][rows, chosen]
        grip_pred.append(selected_grip.cpu()); grip_true.append(raw["gripper"]); grip_mask.append(raw["mask"])
    report = {k: float(np.mean(v)) for k, v in totals.items()}
    report["gripper_f1"] = masked_f1(torch.cat(grip_pred), torch.cat(grip_true), torch.cat(grip_mask))
    report["best_of_k_ade_by_phase"] = {k: (float(np.mean(v)) if v else None) for k, v in phase.items()}
    task_paths = [cfg.dataset] if isinstance(cfg.dataset, str) else cfg.dataset
    report["by_goal"] = {Path(task_paths[int(k)]).stem.replace("_demo", ""): {
        metric: float(np.mean(values)) for metric, values in v.items()} for k, v in per_task.items()}
    report["cross_goal_initial_coverage"] = cross_goal_initial_metrics(model, ds, stats, args)
    report["num_test_frames"] = len(ds); report["recall_threshold_normalized"] = stats.recall_threshold
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    make_pca(torch.cat(all_pred).numpy(), torch.cat(all_true).numpy(), output / "candidate_pca.png")
    print(json.dumps(report, indent=2))


def observation_frame(obs, expected_proprio_dim):
    from robosuite.utils.transform_utils import quat2axisangle
    images = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        x = np.asarray(obs[key])[..., :3]
        images.append(torch.from_numpy(np.moveaxis(x, -1, 0).copy()).float().div(255.0))
    values = [np.asarray(obs["robot0_joint_pos"]).reshape(-1),
              np.asarray(obs["robot0_gripper_qpos"]).reshape(-1)]
    if sum(map(len, values)) < expected_proprio_dim:
        ee = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"])])
        values.append(ee)
    proprio = np.concatenate(values).astype(np.float32)
    if len(proprio) != expected_proprio_dim:
        raise ValueError(f"Runtime proprio dim {len(proprio)} != training dim {expected_proprio_dim}")
    return torch.stack(images), torch.from_numpy(proprio)


def observation_batch(history, obs_horizon, device):
    frames = ([history[0]] * max(obs_horizon - len(history), 0) + history)[-obs_horizon:]
    return {"images": torch.stack([x[0] for x in frames])[None].to(device),
            "proprio": torch.stack([x[1] for x in frames])[None].to(device)}


@torch.no_grad()
def rollout(args):
    from libero.libero.envs import OffScreenRenderEnv
    model, _, _, cfg = load_checkpoint(args.checkpoint, args.device)
    init_states = torch.load(args.init_states, map_location="cpu")
    env = OffScreenRenderEnv(bddl_file_name=args.bddl_file, camera_heights=128, camera_widths=128)
    records = []
    try:
        for init_id in range(min(args.num_init_states, len(init_states))):
            for repeat in range(args.repeats):
                seed_everything(args.seed + init_id * args.repeats + repeat)
                env.reset(); obs = env.set_init_state(init_states[init_id])
                for _ in range(5): obs, _, _, _ = env.step(np.zeros(model.continuous_dim + 1))
                steps = saturated = collisions = 0; success = False; collision_known = False
                history = [observation_frame(obs, model.encoder.proprio_embed.in_features)]
                while steps < args.max_steps and not success:
                    batch = observation_batch(history, getattr(cfg, "obs_horizon", 2), args.device)
                    result = model.candidates(batch, args.k, args.flow_steps, args.cluster_threshold)
                    probs = result["prior_weights"][0]
                    choice = torch.multinomial(probs, 1).item()
                    chunk = result["candidate_chunks"][0, choice].cpu().numpy()
                    for action in chunk[:args.execute_steps]:
                        saturated += int(np.any(np.abs(action[:-1]) >= args.saturation_threshold))
                        obs, _, done, info = env.step(action); steps += 1
                        history.append(observation_frame(obs, model.encoder.proprio_embed.in_features))
                        success = bool(done) or bool(env.check_success())
                        info = info or {}
                        collision_keys = [k for k in info if "collision" in k.lower()]
                        if collision_keys:
                            collision_known = True; collisions += int(any(info[k] for k in collision_keys))
                        if success or steps >= args.max_steps: break
                records.append({"init_state": init_id, "repeat": repeat, "success": success,
                                "steps": steps, "saturation_rate": saturated / max(steps, 1),
                                "collision_rate": collisions / max(steps, 1) if collision_known else None})
    finally:
        env.close()
    success = np.asarray([r["success"] for r in records], np.float32)
    by_init = [[r["success"] for r in records if r["init_state"] == i] for i in range(min(args.num_init_states, len(init_states)))]
    report = {"success_rate": float(success.mean()),
              "success_rate_std_across_init_states": float(np.std([np.mean(x) for x in by_init])),
              "mean_steps": float(np.mean([r["steps"] for r in records])),
              "saturation_rate": float(np.mean([r["saturation_rate"] for r in records])),
              "collision_rate": (float(np.mean([r["collision_rate"] for r in records if r["collision_rate"] is not None]))
                                 if any(r["collision_rate"] is not None for r in records) else None),
              "rollouts": len(records)}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "rollout_metrics.json").write_text(json.dumps(report, indent=2))
    (output / "rollouts.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(report, indent=2))


def make_pca(candidates, truth, path, max_points=2000):
    import matplotlib.pyplot as plt
    x = candidates.reshape(-1, np.prod(candidates.shape[-2:]))
    y = truth.reshape(-1, np.prod(truth.shape[-2:]))
    rng = np.random.RandomState(0)
    x = x[rng.choice(len(x), min(len(x), max_points), replace=False)]
    y = y[rng.choice(len(y), min(len(y), max_points), replace=False)]
    joined = np.concatenate([x, y]); joined -= joined.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(joined, full_matrices=False); z = joined @ vt[:2].T
    plt.figure(figsize=(7, 6)); plt.scatter(z[:len(x), 0], z[:len(x), 1], s=6, alpha=.25, label="candidates")
    plt.scatter(z[len(x):, 0], z[len(x):, 1], s=8, alpha=.35, label="ground truth")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def parser():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="cuda"); common.add_argument("--batch-size", type=int, default=64)
    common.add_argument("--workers", type=int, default=4)
    t = sub.add_parser("train", parents=[common]); t.add_argument("--dataset", nargs="+", required=True,
        help="One or more same-scene goal HDF5 files; no goal id is passed to the model")
    t.add_argument("--head", choices=["deterministic", "gmm", "flow"], required=True)
    t.add_argument("--backbone", choices=["tiny", "dinov2", "siglip"], default="dinov2")
    t.add_argument("--horizon", type=int, default=10); t.add_argument("--hidden-dim", type=int, default=256)
    t.add_argument("--obs-horizon", type=int, default=2)
    t.add_argument("--num-modes", type=int, default=5); t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--split-seed", type=int, default=0)
    e = sub.add_parser("evaluate", parents=[common]); e.add_argument("--checkpoint", required=True); e.add_argument("--output", required=True)
    e.add_argument("--k", type=int, default=8); e.add_argument("--flow-steps", type=int, default=20)
    e.add_argument("--cluster-threshold", type=float, default=1.0)
    e.add_argument("--match-image-mae", type=float, default=.05)
    e.add_argument("--match-proprio-rmse", type=float, default=.05)
    r = sub.add_parser("rollout", parents=[common]); r.add_argument("--checkpoint", required=True)
    r.add_argument("--bddl-file", required=True); r.add_argument("--init-states", required=True); r.add_argument("--output", required=True)
    r.add_argument("--num-init-states", type=int, default=10); r.add_argument("--repeats", type=int, default=20)
    r.add_argument("--max-steps", type=int, default=600); r.add_argument("--execute-steps", type=int, default=2)
    r.add_argument("--k", type=int, default=8); r.add_argument("--flow-steps", type=int, default=20)
    r.add_argument("--cluster-threshold", type=float, default=1.0); r.add_argument("--saturation-threshold", type=float, default=.99)
    r.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    {"train": train, "evaluate": evaluate, "rollout": rollout}[args.command](args)
