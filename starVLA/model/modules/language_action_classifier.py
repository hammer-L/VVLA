"""Lightweight language-action compatibility classifier.

The classifier is deliberately independent from a particular VLM.  It consumes
the VLM's final hidden states and a fixed-length, normalized action chunk.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Pool ``[B, L, H]`` hidden states without including padding tokens."""
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must have shape [B, L, H], got {tuple(hidden_states.shape)}")

    if attention_mask is None:
        return hidden_states.mean(dim=1)
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            "attention_mask must match the first two hidden-state dimensions; "
            f"got mask={tuple(attention_mask.shape)}, hidden_states={tuple(hidden_states.shape)}"
        )

    mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    token_count = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / token_count


class LanguageActionClassifier(nn.Module):
    """Score whether an action chunk is compatible with vision-language context.

    Fusion uses ``[z_vl, z_action, z_vl * z_action]`` so the head has both the
    original modality features and an inexpensive element-wise interaction.
    The returned value is a binary-classification logit, not a probability.
    """

    def __init__(
        self,
        vlm_hidden_dim: int,
        action_horizon: int,
        action_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(vlm_hidden_dim, action_horizon, action_dim, hidden_dim) <= 0:
            raise ValueError("All classifier dimensions must be positive")

        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)

        self.vl_projector = nn.Sequential(
            nn.LayerNorm(vlm_hidden_dim),
            nn.Linear(vlm_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(self.action_horizon * self.action_dim),
            nn.Linear(self.action_horizon * self.action_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected_action_shape = (hidden_states.shape[0], self.action_horizon, self.action_dim)
        if tuple(actions.shape) != expected_action_shape:
            raise ValueError(
                f"actions must have shape {expected_action_shape}, got {tuple(actions.shape)}"
            )

        vl_features = self.vl_projector(masked_mean_pool(hidden_states, attention_mask))
        action_features = self.action_encoder(actions.flatten(start_dim=1).to(dtype=hidden_states.dtype))
        fused = torch.cat((vl_features, action_features, vl_features * action_features), dim=-1)
        return self.classifier(fused).squeeze(-1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float | None = None) -> torch.Tensor:
        """Binary cross-entropy for logits and 0/1 labels."""
        labels = labels.to(device=logits.device, dtype=logits.dtype)
        weight = None if pos_weight is None else logits.new_tensor(pos_weight)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
