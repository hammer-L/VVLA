"""Qwen-GR00T with an auxiliary language-action compatibility classifier."""

from dataclasses import dataclass, field
import time
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig, Qwen_GR00T
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.language_action_classifier import LanguageActionClassifier
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def select_action_candidates(
    candidate_actions: torch.Tensor, candidate_logits: torch.Tensor, rerank: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one ``[T,A]`` candidate per sample from ``[B,K,T,A]``."""
    if candidate_actions.ndim != 4 or candidate_logits.shape != candidate_actions.shape[:2]:
        raise ValueError("candidate actions/logits must have shapes [B,K,T,A] and [B,K]")
    indices = (
        candidate_logits.argmax(dim=1)
        if rerank
        else torch.zeros(candidate_actions.shape[0], dtype=torch.long, device=candidate_actions.device)
    )
    batch_indices = torch.arange(candidate_actions.shape[0], device=candidate_actions.device)
    return candidate_actions[batch_indices, indices], indices


@dataclass
class QwenGR00TClassifierDefaultConfig(QwenGR00TDefaultConfig):
    """Defaults for protocol-v2 token/action cross-attention."""

    name: str = "QwenGR00TClassifier"
    classifier: dict = field(
        default_factory=lambda: {
            "hidden_dim": 512,
            "dropout": 0.1,
            "num_heads": 8,
            "label_key": "classifier_label",
            "pos_weight": 2.0,
            "protocol_version": "2.0.0",
            "threshold": 0.5,
            "ece_bins": 10,
            "pair_id_key": "pair_id",
            "negative_type_key": "negative_type",
            "bootstrap_samples": 10000,
            "freeze_backbones": True,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00TClassifier")
class Qwen_GR00T_Classifier(Qwen_GR00T):
    """Reuse QwenGR00T weights and add a binary compatibility-scoring branch."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        # Merge classifier defaults first. Qwen_GR00T will merge its own defaults
        # again without removing the extra classifier configuration.
        config = merge_framework_config(QwenGR00TClassifierDefaultConfig, config)
        super().__init__(config=config, **kwargs)

        classifier_cfg = self.config.framework.classifier
        if str(classifier_cfg.get("protocol_version", "")) != "2.0.0":
            raise ValueError("QwenGR00TClassifier requires metadata/checkpoint protocol 2.0.0")
        if float(classifier_cfg.get("pos_weight", 2.0)) != 2.0:
            raise ValueError("protocol 2.0.0 fixes classifier pos_weight at 2.0")
        if not bool(classifier_cfg.get("freeze_backbones", True)):
            raise ValueError("protocol 2.0.0 requires the VLM and original action model to remain frozen")
        action_dim = int(self.config.framework.action_model.action_dim)
        if self.action_horizon != 8 or action_dim != 7:
            raise ValueError(
                "QwenGR00TClassifier LIBERO baseline requires action shape [8, 7], "
                f"got [{self.action_horizon}, {action_dim}]"
            )
        self.language_classifier = LanguageActionClassifier(
            vlm_hidden_dim=int(self.qwen_vl_interface.model.config.hidden_size),
            action_horizon=self.action_horizon,
            action_dim=action_dim,
            hidden_dim=int(classifier_cfg.hidden_dim),
            dropout=float(classifier_cfg.dropout),
            num_heads=int(classifier_cfg.get("num_heads", 8)),
        )
        if bool(classifier_cfg.get("freeze_backbones", True)):
            for module in (self.qwen_vl_interface, self.action_model):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def _encode_examples(self, examples: List[dict]):
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        attention_mask = qwen_inputs.get("attention_mask")
        outputs = self.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1], attention_mask

    def _actions_from_examples(
        self, examples: List[dict], reference: torch.Tensor, key: str = "action"
    ) -> torch.Tensor:
        missing = [index for index, example in enumerate(examples) if key not in example]
        if missing:
            raise KeyError(f"Missing action key {key!r} in examples at indices {missing}")
        actions = torch.as_tensor(
            np.asarray([example[key] for example in examples]),
            device=reference.device,
            dtype=reference.dtype,
        )
        if actions.ndim != 3:
            raise ValueError(f"example[{key!r}] must form [B, T, A], got {tuple(actions.shape)}")
        if actions.shape[1] < self.action_horizon:
            raise ValueError(
                f"Action chunk has {actions.shape[1]} steps, but action_horizon={self.action_horizon}"
            )
        return actions[:, -self.action_horizon :, :]

    def score_actions(
        self,
        last_hidden: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return differentiable compatibility logits for candidate actions."""
        return self.language_classifier.score_main(last_hidden, actions, attention_mask)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        if not examples:
            raise ValueError("examples must be a non-empty list")

        last_hidden, attention_mask = self._encode_examples(examples)
        actions = self._actions_from_examples(examples, last_hidden)
        classifier_output = self.language_classifier(last_hidden, actions, attention_mask)
        logits = classifier_output["main_logits"]

        label_key = str(self.config.framework.classifier.label_key)
        missing = [index for index, example in enumerate(examples) if label_key not in example]
        if missing:
            raise KeyError(f"Missing classifier label key {label_key!r} in examples at indices {missing}")
        labels = torch.as_tensor(
            [example[label_key] for example in examples],
            device=logits.device,
            dtype=logits.dtype,
        ).reshape(-1)
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError(f"{label_key!r} values must be binary (0 or 1)")

        roles = [example.get("contrastive_role") for example in examples]
        roles = roles if all(role is not None for role in roles) else None
        losses = self.language_classifier.loss(
            classifier_output,
            labels,
            roles=roles,
            pos_weight=self.config.framework.classifier.get("pos_weight", None),
        )
        audit_actions = self._actions_from_examples(examples, last_hidden, key="audit_action")
        mean_actions = self._actions_from_examples(examples, last_hidden, key="mean_action")
        with torch.no_grad():
            donor_action_logits = self.score_actions(last_hidden, audit_actions, attention_mask)
            mean_action_logits = self.score_actions(last_hidden, mean_actions, attention_mask)
        return {
            "classifier_loss": losses["loss"],
            "classifier_bce_loss": losses["bce"].detach(),
            "classifier_language_rank_loss": losses["language_rank"].detach(),
            "classifier_action_rank_loss": losses["action_rank"].detach(),
            "classifier_vl_probe_loss": losses["vl_probe"].detach(),
            "classifier_action_probe_loss": losses["action_probe"].detach(),
            "classifier_logits": logits.detach(),
            "classifier_labels": labels.detach(),
            "vl_probe_logits": classifier_output["vl_probe_logits"].detach(),
            "action_probe_logits": classifier_output["action_probe_logits"].detach(),
            "donor_action_logits": donor_action_logits.detach(),
            "mean_action_logits": mean_action_logits.detach(),
        }

    def compute_loss(self, tag: str, batch, loss_scale: dict | None = None) -> dict | None:
        """Expose only the loss tensor to generic multi-dataloader trainers."""
        if tag != "vla":
            return super().compute_loss(tag, batch, loss_scale)
        scale = (loss_scale or {}).get(tag, 1.0)
        return {"classifier_loss": self.forward(batch)["classifier_loss"] * scale}

    @torch.inference_mode()
    def predict_compatibility(self, examples: List[dict]) -> dict:
        """Return logits and probabilities for already-normalized action chunks."""
        if not examples:
            raise ValueError("examples must be a non-empty list")
        last_hidden, attention_mask = self._encode_examples(examples)
        actions = self._actions_from_examples(examples, last_hidden)
        output = self.language_classifier(last_hidden, actions, attention_mask)
        logits = output["main_logits"]
        return {
            "compatibility_logits": logits.detach().cpu().numpy(),
            "compatibility_scores": logits.sigmoid().detach().cpu().numpy(),
            "vl_probe_logits": output["vl_probe_logits"].detach().cpu().numpy(),
            "action_probe_logits": output["action_probe_logits"].detach().cpu().numpy(),
        }

    def _encode_prediction_examples(self, examples: List[dict]):
        batch_images = [[to_pil_preserve(image) for image in example["image"]] for example in examples]
        train_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_size:
            batch_images = resize_images(batch_images, target_size=train_size)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=[example["lang"] for example in examples],
        )
        attention_mask = qwen_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        device_type = "cuda" if next(self.parameters()).is_cuda else "cpu"
        with torch.no_grad(), torch.autocast(
            device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"
        ):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs.hidden_states[-1].detach(), attention_mask

    def predict_action(
        self,
        examples: List[dict],
        *,
        classifier_mode: str = "off",
        num_candidates: int = 1,
        guidance_scale: float = 0.0,
        seed: int | None = None,
        **kwargs,
    ) -> dict:
        """Generate actions with optional classifier reranking and/or guidance."""
        del kwargs
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("examples must be a non-empty list")
        valid_modes = {"off", "rerank", "gradient", "gradient_rerank"}
        if classifier_mode not in valid_modes:
            raise ValueError(f"classifier_mode must be one of {sorted(valid_modes)}, got {classifier_mode!r}")
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1")
        if guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")

        started = time.perf_counter()
        last_hidden, attention_mask = self._encode_prediction_examples(examples)
        batch_size = last_hidden.shape[0]
        state_values = [example["state"] for example in examples] if "state" in examples[0] else None
        state = (
            torch.as_tensor(np.asarray(state_values), device=last_hidden.device, dtype=last_hidden.dtype)
            if state_values is not None
            else None
        )

        uses_rerank = classifier_mode in {"rerank", "gradient_rerank"}
        uses_gradient = classifier_mode in {"gradient", "gradient_rerank"} and guidance_scale != 0.0
        candidate_count = int(num_candidates) if uses_rerank else 1
        generator = None
        if seed is not None:
            generator = torch.Generator(device=last_hidden.device)
            generator.manual_seed(int(seed))

        # Candidate-major layout makes the first B noise tensors exactly the
        # tensors sampled by an off-mode B-item request with the same seed.
        initial_actions = torch.randn(
            (candidate_count, batch_size, self.action_horizon, self.action_model.action_dim),
            device=last_hidden.device,
            dtype=last_hidden.dtype,
            generator=generator,
        ).flatten(0, 1)
        repeated_hidden = last_hidden.repeat(candidate_count, 1, 1)
        repeated_mask = attention_mask.repeat(candidate_count, 1) if attention_mask is not None else None
        repeated_state = state.repeat(candidate_count, 1, 1) if state is not None else None

        def score(candidate_actions: torch.Tensor) -> torch.Tensor:
            return self.score_actions(repeated_hidden, candidate_actions, repeated_mask)

        lower = torch.full(
            (1, 1, self.action_model.action_dim), -1.0, device=last_hidden.device, dtype=last_hidden.dtype
        )
        upper = torch.ones_like(lower)
        lower[..., -1] = 0.0
        device_type = "cuda" if last_hidden.is_cuda else "cpu"
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            sampled = self.action_model.predict_action(
                repeated_hidden,
                repeated_state,
                encoder_attention_mask=repeated_mask,
                guidance_callback=score if uses_gradient else None,
                guidance_scale=float(guidance_scale) if uses_gradient else 0.0,
                initial_actions=initial_actions,
                action_bounds=(lower, upper) if uses_gradient else None,
                return_diagnostics=True,
            )
            candidate_actions, sampler_diagnostics = sampled
            with torch.no_grad():
                candidate_logits_flat = score(candidate_actions)

        candidate_actions = candidate_actions.reshape(
            candidate_count, batch_size, self.action_horizon, self.action_model.action_dim
        ).transpose(0, 1)
        candidate_logits = candidate_logits_flat.reshape(candidate_count, batch_size).transpose(0, 1)
        selected_actions, selected_indices = select_action_candidates(
            candidate_actions, candidate_logits, uses_rerank
        )

        diagnostics = {
            "classifier_mode": classifier_mode,
            "seed": seed,
            "num_candidates": candidate_count,
            "guidance_scale": float(guidance_scale),
            "candidate_logits": candidate_logits.detach().float().cpu().tolist(),
            "selected_indices": selected_indices.detach().cpu().tolist(),
            **sampler_diagnostics,
            "inference_latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        return {
            "normalized_actions": selected_actions.detach().cpu().numpy(),
            "classifier_diagnostics": diagnostics,
        }
