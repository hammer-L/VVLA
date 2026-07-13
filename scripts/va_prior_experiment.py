"""Train and evaluate a language-free, same-scene multi-goal action prior."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, default_collate

from libero.va_prior.data import (ActionStats, LiberoActionChunkDataset, TaskTaggedDataset,
                                  compute_action_stats_multi, trajectory_split)
from libero.va_prior.model import VAPriorModel
from libero.va_prior.visualization import (action_distributions, restore_projection_geometry,
                                           scale_controller_actions,
                                           trajectory_overlay_figure)


ACTION_HEAD_DIMS = {"small": 256, "base": 512, "large": 1024}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_model(args, sample, stats):
    head_size = getattr(args, "action_head_size", "base")
    return VAPriorModel(
        head=args.head, backbone=args.backbone, proprio_dim=sample["proprio"].shape[-1],
        continuous_dim=len(stats.mean), horizon=args.horizon, hidden_dim=args.hidden_dim,
        num_modes=args.num_modes, action_mean=stats.mean, action_std=stats.std,
        action_head_dim=ACTION_HEAD_DIMS[head_size],
    )


def combined_dataset(paths, splits, split_name, stats, horizon, obs_horizon):
    datasets = []
    for task_id, path in enumerate(paths):
        base = LiberoActionChunkDataset(path, splits[str(path)][split_name], stats, horizon,
                                        obs_horizon=obs_horizon)
        datasets.append(TaskTaggedDataset(base, task_id))
    return ConcatDataset(datasets)


def goal_name(path):
    return Path(path).stem.replace("_demo", "")


def fixed_goal_batch(dataset):
    """Return one stable initial validation sample from every non-empty goal."""
    items, goals = [], []
    for tagged in dataset.datasets:
        if len(tagged):
            items.append(tagged[0])
            goals.append(goal_name(tagged.dataset.path))
    return (default_collate(items), goals) if items else (None, [])


@torch.no_grad()
def validation_candidate_metrics(model, loader, stats, args):
    """Compute macro best-of-K recall; this is candidate coverage, not rollout success."""
    per_goal_hits, per_goal_ade = {}, {}
    metric_device = torch.device(args.device)
    cuda_devices = ([metric_device.index if metric_device.index is not None
                     else torch.cuda.current_device()]
                    if metric_device.type == "cuda" else [])
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.seed + 104729)
        if cuda_devices:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(args.seed + 104729)
        for raw in loader:
            batch = move(raw, args.device)
            result = model.candidates(batch, args.val_k, args.val_flow_steps,
                                      args.val_cluster_threshold)
            candidates = result["candidate_chunks"][..., :-1]
            truth = batch["continuous"] * model.action_std + model.action_mean
            mask = batch["mask"]
            distance = torch.linalg.vector_norm(
                (candidates - truth[:, None]) / model.action_std, dim=-1)
            per_candidate = (distance * mask[:, None]).sum(-1) / mask.sum(-1, keepdim=True)
            best = per_candidate.min(-1).values.cpu()
            for task_id in raw["task_id"].unique().tolist():
                selected = raw["task_id"] == task_id
                key = int(task_id)
                values = best[selected]
                per_goal_ade.setdefault(key, []).extend(values.tolist())
                per_goal_hits.setdefault(key, []).extend(
                    (values <= stats.recall_threshold).float().tolist())

    goal_acc = {key: float(np.mean(values)) for key, values in per_goal_hits.items()}
    goal_ade = {key: float(np.mean(values)) for key, values in per_goal_ade.items()}
    return {
        "acc": float(np.mean(list(goal_acc.values()))) if goal_acc else None,
        "best_of_k_ade": float(np.mean(list(goal_ade.values()))) if goal_ade else None,
        "acc_by_goal": goal_acc,
    }


@torch.no_grad()
def training_visualizations(model, raw_batch, goals, args, wandb, output, epoch,
                            geometry_cache):
    """Save balanced trajectory overlays and optionally build W&B media."""
    batch = move(raw_batch, args.device)
    result = model.candidates(batch, args.visualize_k, args.visualize_flow_steps, cluster=False)
    candidate_chunks = result["candidate_chunks"].cpu().numpy()
    prior_weights = result["prior_weights"].cpu().numpy()
    target = (batch["continuous"] * model.action_std + model.action_mean).cpu().numpy()
    values = action_distributions(
        candidate_chunks, prior_weights, target, raw_batch["gripper"].numpy(),
        raw_batch["mask"].numpy())
    log = {}
    if wandb is not None:
        for dof in range(model.continuous_dim):
            prefix = f"action_distribution/dof_{dof}"
            log[f"{prefix}/predicted"] = wandb.Histogram(
                values["predicted_continuous"][:, dof])
            log[f"{prefix}/target"] = wandb.Histogram(
                values["target_continuous"][:, dof])
        log["action_distribution/gripper/predicted"] = wandb.Histogram(
            values["predicted_gripper"])
        log["action_distribution/gripper/target"] = wandb.Histogram(
            values["target_gripper"])

    import matplotlib.pyplot as plt
    image_dir = Path(output) / "visualizations" / f"epoch_{epoch:04d}"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, goal in enumerate(goals):
        image = raw_batch["images"][index, -1, 0].numpy()
        height, width = image.shape[-2:]
        key = (raw_batch["dataset_path"][index], raw_batch["demo_id"][index],
               int(raw_batch["timestep"][index]))
        if key not in geometry_cache:
            geometry_cache[key] = restore_projection_geometry(
                key[0], key[1], key[2], height, width, Path(output) / ".bddl_cache")
        geometry = geometry_cache[key]
        candidate_deltas = scale_controller_actions(candidate_chunks[index], geometry)
        target_deltas = scale_controller_actions(target[index], geometry)
        figure = trajectory_overlay_figure(
            image, candidate_deltas, prior_weights[index], target_deltas,
            raw_batch["mask"][index].numpy(), raw_batch["ee_pos"][index].numpy(),
            geometry["world_to_camera"], title=goal.replace("_", " "))
        path = image_dir / f"{goal}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        if wandb is not None:
            log[f"action_chunks/{goal}/trajectory_overlay"] = wandb.Image(str(path))
        plt.close(figure)
    return log


def train(args):
    if args.visualize_every < 1:
        raise ValueError("--visualize-every must be at least 1")
    if args.visualize_k < 1:
        raise ValueError("--visualize-k must be at least 1")
    if args.visualize_flow_steps < 1:
        raise ValueError("--visualize-flow-steps must be at least 1")
    if args.val_k < 1:
        raise ValueError("--val-k must be at least 1")
    if args.val_flow_steps < 1:
        raise ValueError("--val-flow-steps must be at least 1")
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
    best = float("inf"); best_acc = 0.0
    config = vars(args).copy()
    config.update({
        "action_head_dim": model.action_head_dim,
        "action_head_parameters": model.action_head_parameter_count(),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "wandb_run_id": None,
    })
    wandb = None
    if args.use_wandb:
        import wandb as wandb_module
        wandb = wandb_module
        init_kwargs = {"project": args.wandb_project, "config": config, "mode": args.wandb_mode,
                       "name": args.wandb_run_name or out.name}
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        wandb.init(**init_kwargs)
        config["wandb_run_id"] = wandb.run.id
        wandb.config.update({key: config[key] for key in (
            "action_head_dim", "action_head_parameters", "trainable_parameters",
            "wandb_run_id")}, allow_val_change=True)
        wandb.run.summary["model/action_head_parameters"] = config["action_head_parameters"]
        wandb.run.summary["model/trainable_parameters"] = config["trainable_parameters"]
        wandb.run.summary["model/action_head_size"] = args.action_head_size
        wandb.run.summary["model/action_head_dim"] = model.action_head_dim
        wandb.run.summary["val/recall_threshold_normalized"] = stats.recall_threshold
    fixed_val_batch, visualization_goals = fixed_goal_batch(val)
    task_goals = {index: goal_name(path) for index, path in enumerate(paths)}
    geometry_cache = {}
    try:
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            model.train(); train_totals = {"total": 0.0, "action": 0.0, "gripper": 0.0}
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss, parts = model.loss(move(batch, args.device)); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_totals["total"] += loss.item()
                train_totals["action"] += parts["action"].item()
                train_totals["gripper"] += parts["gripper"].item()
            model.eval(); val_totals = {"total": [], "action": [], "gripper": []}
            with torch.no_grad():
                for batch in val_loader:
                    loss, parts = model.loss(move(batch, args.device))
                    val_totals["total"].append(loss.item())
                    val_totals["action"].append(parts["action"].item())
                    val_totals["gripper"].append(parts["gripper"].item())
            train_metrics = {key: value / max(len(loader), 1) for key, value in train_totals.items()}
            score = float(np.mean(val_totals["total"])) if val_totals["total"] else train_metrics["total"]
            val_metrics = {key: (float(np.mean(value)) if value else train_metrics[key])
                           for key, value in val_totals.items()}
            candidate_metrics = validation_candidate_metrics(model, val_loader, stats, args)
            if candidate_metrics["acc"] is not None:
                best_acc = max(best_acc, candidate_metrics["acc"])
            is_best = score < best
            if is_best:
                best = score
            print(json.dumps({"epoch": epoch, "train_loss": train_metrics["total"],
                              "val_loss": score, "val_acc": candidate_metrics["acc"]}))
            state = {"model": model.state_dict(), "config": config, "stats": {
                "mean": stats.mean.tolist(), "std": stats.std.tolist(),
                "recall_threshold": stats.recall_threshold}, "splits": splits}
            torch.save(state, out / "last.pt")
            if is_best:
                torch.save(state, out / "best.pt")
            log = {}
            if wandb is not None:
                log = {
                    "epoch": epoch,
                    "train/total_loss": train_metrics["total"],
                    "train/action_loss": train_metrics["action"],
                    "train/gripper_loss": train_metrics["gripper"],
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/epoch_time_sec": time.time() - started,
                    "val/total_loss": val_metrics["total"],
                    "val/action_loss": val_metrics["action"],
                    "val/gripper_loss": val_metrics["gripper"],
                    "val/best_total_loss": best,
                    "val/acc": candidate_metrics["acc"],
                    "val/best_of_k_ade": candidate_metrics["best_of_k_ade"],
                    "val/best_acc": best_acc,
                }
                for task_id, value in candidate_metrics["acc_by_goal"].items():
                    log[f"val/acc_by_goal/{task_goals[task_id]}"] = value
            should_visualize = epoch % args.visualize_every == 0 or epoch == args.epochs
            if should_visualize and fixed_val_batch is not None:
                log.update(training_visualizations(
                    model, fixed_val_batch, visualization_goals, args, wandb, out, epoch,
                    geometry_cache))
            if wandb is not None:
                wandb.log(log, step=epoch)
    finally:
        if wandb is not None:
            wandb.finish()


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


def log_rollout_to_wandb(report, args, checkpoint_cfg):
    """Append closed-loop metrics to the originating training run when possible."""
    import wandb

    stored_id = getattr(checkpoint_cfg, "wandb_run_id", None)
    project = args.wandb_project or getattr(checkpoint_cfg, "wandb_project", "libero-va-prior")
    entity = args.wandb_entity or getattr(checkpoint_cfg, "wandb_entity", None)
    mode = args.wandb_mode or getattr(checkpoint_cfg, "wandb_mode", "online")
    run_name = args.wandb_run_name or getattr(checkpoint_cfg, "wandb_run_name", None)
    init_kwargs = {"project": project, "mode": mode,
                   "config": {"rollout_checkpoint": str(args.checkpoint)}}
    if entity:
        init_kwargs["entity"] = entity
    if stored_id:
        init_kwargs.update({"id": stored_id, "resume": "allow"})
    else:
        fallback_name = f"{Path(args.checkpoint).parent.name}-rollout"
        init_kwargs["name"] = run_name or fallback_name
        print("[warning] Checkpoint has no W&B run id; creating a linked rollout run")
    wandb.init(**init_kwargs)
    try:
        goal = Path(args.bddl_file).stem
        log = {
            "rollout/acc": report["success_rate"],
            "rollout/success_rate": report["success_rate"],
            "rollout/success_rate_std_across_init_states":
                report["success_rate_std_across_init_states"],
            "rollout/mean_steps": report["mean_steps"],
            "rollout/saturation_rate": report["saturation_rate"],
        }
        if report["collision_rate"] is not None:
            log["rollout/collision_rate"] = report["collision_rate"]
        wandb.log(log)
        for key, value in log.items():
            wandb.run.summary[key] = value
        wandb.run.summary[f"rollout/by_goal/{goal}/success_rate"] = report["success_rate"]
        wandb.run.summary[f"rollout/by_goal/{goal}/rollouts"] = report["rollouts"]
    finally:
        wandb.finish()


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
    if args.use_wandb:
        log_rollout_to_wandb(report, args, cfg)
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
    t.add_argument("--output", required=True)
    t.add_argument("--head", choices=["deterministic", "gmm", "flow"], required=True)
    t.add_argument("--action-head-size", choices=list(ACTION_HEAD_DIMS), default="base")
    t.add_argument("--backbone", choices=["tiny", "dinov2", "siglip"], default="dinov2")
    t.add_argument("--horizon", type=int, default=10); t.add_argument("--hidden-dim", type=int, default=256)
    t.add_argument("--obs-horizon", type=int, default=2)
    t.add_argument("--num-modes", type=int, default=5); t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--split-seed", type=int, default=0)
    t.add_argument("--use-wandb", action="store_true")
    t.add_argument("--wandb-project", default="libero-va-prior")
    t.add_argument("--wandb-entity", default=None)
    t.add_argument("--wandb-run-name", default=None)
    t.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    t.add_argument("--visualize-every", type=int, default=5)
    t.add_argument("--visualize-k", type=int, default=8)
    t.add_argument("--visualize-flow-steps", type=int, default=20)
    t.add_argument("--val-k", type=int, default=8)
    t.add_argument("--val-flow-steps", type=int, default=20)
    t.add_argument("--val-cluster-threshold", type=float, default=1.0)
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
    r.add_argument("--use-wandb", action="store_true")
    r.add_argument("--wandb-project", default=None)
    r.add_argument("--wandb-entity", default=None)
    r.add_argument("--wandb-run-name", default=None)
    r.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    {"train": train, "evaluate": evaluate, "rollout": rollout}[args.command](args)
