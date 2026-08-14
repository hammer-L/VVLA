# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.classifier_metrics import (
    binary_classifier_metrics,
    classifier_record_report,
    classifier_records_from_batch,
    gather_classifier_records,
)
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    vla_val_dataloader = None
    vla_test_dataloader = None
    if cfg.datasets.vla_data.get("language_overlay_meta", None):
        vla_val_dataloader = build_dataloader(
            cfg=cfg,
            dataset_py=cfg.datasets.vla_data.dataset_py,
            overlay_split="val",
            overlay_mode="exhaustive_eval",
        )
        vla_test_dataloader = build_dataloader(
            cfg=cfg,
            dataset_py=cfg.datasets.vla_data.dataset_py,
            overlay_split="test",
            overlay_mode="exhaustive_eval",
        )

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader, vla_val_dataloader, vla_test_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        vla_val_dataloader=None,
        vla_test_dataloader=None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_val_dataloader = vla_val_dataloader
        self.vla_test_dataloader = vla_test_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self.best_classifier_score = None
        self.best_classifier_checkpoint = None
        self.classifier_threshold = None

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        components = [self.model, self.optimizer, self.vla_train_dataloader]
        if self.vla_val_dataloader is not None:
            components.append(self.vla_val_dataloader)
        if self.vla_test_dataloader is not None:
            components.append(self.vla_test_dataloader)
        prepared = self.setup_distributed_training(self.accelerator, *components)
        self.model, self.optimizer, self.vla_train_dataloader = prepared[:3]
        offset = 3
        if self.vla_val_dataloader is not None:
            self.vla_val_dataloader = prepared[offset]
            offset += 1
        if self.vla_test_dataloader is not None:
            self.vla_test_dataloader = prepared[offset]

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases (best-effort; must not block training)."""
        self._wandb_enabled = False
        if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            try:
                wandb.init(
                    name=self.config.run_id,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                    config=OmegaConf.to_container(
                        self.config.unwrap() if isinstance(self.config, AccessTrackedConfig) else self.config,
                        resolve=True,
                    ),
                )
                wandb.define_metric("eval/loss", summary="min")
                wandb.define_metric("eval/auroc", summary="max")
                wandb.define_metric("eval/average_precision", summary="max")
                wandb.define_metric("eval/f1", summary="max")
                wandb.define_metric("eval/paired_accuracy", summary="max")
                self._wandb_enabled = True
            except Exception as exc:
                logger.warning(f"W&B init failed; continuing without W&B: {exc}")
                self._wandb_enabled = False
        # Rendezvous after rank-0 W&B init. Otherwise a slow or failing init on
        # rank 0 lets the other ranks reach the first collective alone and
        # eventually hit an NCCL watchdog timeout.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        has_eval_metrics = any(key.startswith("eval/") for key in metrics)
        should_log = self.completed_steps % self.config.trainer.logging_frequency == 0 or has_eval_metrics
        if should_log and rank == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            if getattr(self, "_wandb_enabled", False):
                try:
                    wandb.log(metrics, step=self.completed_steps)
                except Exception as exc:
                    self._wandb_enabled = False
                    logger.warning(f"W&B log failed; disabling W&B: {exc}")
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if (
                self.accelerator.sync_gradients
                and self.completed_steps > 0
                and self.completed_steps % self.config.trainer.eval_interval == 0
            ):
                if hasattr(self.accelerator.unwrap_model(self.model), "language_classifier"):
                    step_metrics = self.eval_classifier(step_metrics)
                else:
                    step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            step_metrics["train/steps_per_second"] = 1.0 / max(t_end_model - t_start_model, 1e-12)
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        if hasattr(self.accelerator.unwrap_model(self.model), "language_classifier"):
            if self.vla_val_dataloader is not None and not (Path(self.checkpoint_dir) / "best_classifier.json").exists():
                self.eval_classifier({})
        if (
            hasattr(self.accelerator.unwrap_model(self.model), "language_classifier")
            and self.vla_test_dataloader is not None
        ):
            self._evaluate_selected_classifier_test()
        self._finalize_training()

    def _evaluate_selected_classifier_test(self) -> None:
        """Load the selected validation checkpoint and evaluate test exactly once."""
        self.accelerator.wait_for_everyone()
        selection_path = Path(self.checkpoint_dir) / "best_classifier.json"
        if not selection_path.exists():
            raise RuntimeError("classifier test requested but no validation-selected checkpoint exists")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        checkpoint = selection["checkpoint"]
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.accelerator.unwrap_model(self.model).load_state_dict(state_dict)
        self.best_classifier_checkpoint = checkpoint
        self.classifier_threshold = float(selection["threshold"])
        self._evaluate_classifier_loader(
            self.vla_test_dataloader,
            prefix="test",
            threshold=self.classifier_threshold,
            select_checkpoint=False,
        )

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    @torch.inference_mode()
    def eval_classifier(self, step_metrics: dict = None) -> dict:
        """Traverse the complete, independent exhaustive validation split."""
        if self.vla_val_dataloader is None:
            raise RuntimeError("classifier evaluation requires an independent validation dataloader")
        return self._evaluate_classifier_loader(
            self.vla_val_dataloader,
            prefix="eval",
            step_metrics=step_metrics,
            threshold=None,
            select_checkpoint=True,
        )

    @torch.inference_mode()
    def _evaluate_classifier_loader(
        self,
        dataloader,
        *,
        prefix: str,
        step_metrics: dict | None = None,
        threshold: float | None,
        select_checkpoint: bool,
    ) -> dict:
        """Evaluate all local batches, then object-gather/deduplicate on rank 0."""
        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()
        local_records = []
        try:
            for examples in dataloader:
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                with torch.autocast(device_type, dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    output = model.forward(examples)
                local_records.extend(
                    classifier_records_from_batch(
                        output["classifier_logits"],
                        output["classifier_labels"],
                        examples,
                    )
                )
        finally:
            if was_training:
                model.train()
        records = gather_classifier_records(local_records)
        result = dict(step_metrics or {})
        report = None
        report_path = Path(self.config.output_dir) / f"{prefix}_classifier_report_step_{self.completed_steps}.json"
        should_save_best = False
        if self.accelerator.is_main_process:
            bootstrap_samples = int(model.config.framework.classifier.get("bootstrap_samples", 10_000))
            report = classifier_record_report(
                records,
                prefix=prefix,
                threshold=threshold,
                bootstrap_samples=bootstrap_samples,
                seed=int(getattr(self.config, "seed", 42)),
            )
            report.update(self._evaluation_provenance(dataloader, prefix))
            result.update(report["metrics"])
            if prefix == "eval":
                self.classifier_threshold = float(report["threshold"])
            if select_checkpoint:
                score = self._classifier_checkpoint_score(report)
                should_save_best = self.best_classifier_score is None or score > self.best_classifier_score
                if should_save_best:
                    self.best_classifier_score = score
        decision = [should_save_best]
        if dist.is_initialized():
            dist.broadcast_object_list(decision, src=0)
        if select_checkpoint and decision[0]:
            # DeepSpeed/ZeRO state consolidation may itself be collective, so
            # every rank participates even though only rank zero writes.
            state_dict = self.accelerator.get_state_dict(self.model)
            if self.accelerator.is_main_process:
                self._save_best_classifier(report, report_path, state_dict)
        if self.accelerator.is_main_process:
            if select_checkpoint:
                report["checkpoint"] = self.best_classifier_checkpoint
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        self.accelerator.wait_for_everyone()
        return result

    def _evaluation_provenance(self, dataloader, prefix: str) -> dict:
        dataset = getattr(dataloader, "dataset", None)
        while dataset is not None and not hasattr(dataset, "meta_digest") and hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
            ).strip()
        except Exception:
            git_commit = None
        meta_dir = self.config.datasets.vla_data.get("language_overlay_meta", None)
        quarantine_count = 0
        if meta_dir:
            quarantine_path = Path(str(meta_dir)) / "quarantine.jsonl"
            if quarantine_path.exists():
                with quarantine_path.open(encoding="utf-8") as handle:
                    quarantine_count = sum(1 for line in handle if line.strip())
        return {
            "split": prefix,
            "step": self.completed_steps,
            "checkpoint": self.best_classifier_checkpoint,
            "meta_digest": getattr(dataset, "meta_digest", None),
            "quarantine_count": quarantine_count,
            "git_commit": git_commit,
        }

    @staticmethod
    def _classifier_checkpoint_score(report: dict) -> tuple[float, float, float]:
        metrics = report["metrics"]
        return (
            float(metrics.get("eval/auroc", float("-inf"))),
            float(metrics.get("eval/average_precision", float("-inf"))),
            -float(metrics.get("eval/loss", float("inf"))),
        )

    def _save_best_classifier(self, report: dict, report_path: Path, state_dict: dict) -> None:
        score = self._classifier_checkpoint_score(report)
        checkpoint = Path(self.checkpoint_dir) / "best_classifier_pytorch_model.pt"
        torch.save(state_dict, checkpoint)
        selection = {
            "step": self.completed_steps,
            "score": {"auroc": score[0], "average_precision": score[1], "loss": -score[2]},
            "threshold": report["threshold"],
            "threshold_source": report["threshold_source"],
            "report": str(report_path),
            "checkpoint": str(checkpoint),
        }
        selection_path = Path(self.checkpoint_dir) / "best_classifier.json"
        with selection_path.open("w", encoding="utf-8") as handle:
            json.dump(selection, handle, indent=2, sort_keys=True)
            handle.write("\n")
        self.best_classifier_checkpoint = str(checkpoint)

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                if "classifier_loss" in output_dict:
                    loss_name = "classifier_loss"
                elif "action_loss" in output_dict:
                    loss_name = "action_loss"
                else:
                    raise KeyError("Model output must contain 'action_loss' or 'classifier_loss'")
                total_loss = output_dict[loss_name]

            self.accelerator.backward(total_loss)

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            classifier = getattr(unwrapped_model, "language_classifier", None)
            if classifier is not None:
                classifier_grad_norm = self._gradient_norm(classifier.parameters())
                action_encoder_grad_norm = self._gradient_norm(classifier.action_encoder.parameters())
                vl_projector_grad_norm = self._gradient_norm(classifier.vl_projector.parameters())
                vlm_grad_norm = self._gradient_norm(unwrapped_model.qwen_vl_interface.parameters())
            else:
                classifier_grad_norm = action_encoder_grad_norm = vl_projector_grad_norm = vlm_grad_norm = None

            if self.config.trainer.gradient_clipping is not None:
                total_grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )
            else:
                total_grad_norm = self._gradient_norm(self.model.parameters())

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        logged_loss_name = "action_dit_loss" if loss_name == "action_loss" else loss_name
        metrics = {
            logged_loss_name: total_loss.item(),
            "train/grad_norm": float(total_grad_norm),
        }
        if "classifier_logits" in output_dict:
            logits = self.accelerator.gather_for_metrics(output_dict["classifier_logits"])
            labels = self.accelerator.gather_for_metrics(output_dict["classifier_labels"])
            shuffled = output_dict.get("shuffled_action_logits")
            if shuffled is not None:
                shuffled = self.accelerator.gather_for_metrics(shuffled)
            classifier_cfg = self.accelerator.unwrap_model(self.model).config.framework.classifier
            metrics.update(
                binary_classifier_metrics(
                    logits,
                    labels,
                    prefix="train",
                    threshold=float(classifier_cfg.get("threshold", 0.5)),
                    ece_bins=int(classifier_cfg.get("ece_bins", 10)),
                    shuffled_logits=shuffled,
                )
            )
            metrics.update(
                {
                    "train/grad_norm_classifier": classifier_grad_norm,
                    "train/grad_norm_action_encoder": action_encoder_grad_norm,
                    "train/grad_norm_vl_projector": vl_projector_grad_norm,
                    "train/grad_norm_vlm": vlm_grad_norm,
                }
            )
        if torch.cuda.is_available():
            metrics["system/gpu_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            metrics["system/gpu_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
        return metrics

    @staticmethod
    def _gradient_norm(parameters) -> float:
        squared_norm = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                squared_norm += float(parameter.grad.detach().float().norm(2)) ** 2
        return squared_norm**0.5

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    overlay_meta = cfg.datasets.vla_data.get("language_overlay_meta", None)
    if overlay_meta:
        from starVLA.dataloader.language_overlay import validate_language_overlay_metadata

        validate_language_overlay_metadata(overlay_meta)

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader, vla_val_dataloader, vla_test_dataloader = prepare_data(
        cfg=cfg, accelerator=accelerator, output_dir=output_dir
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        vla_val_dataloader=vla_val_dataloader,
        vla_test_dataloader=vla_test_dataloader,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_tighten.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
