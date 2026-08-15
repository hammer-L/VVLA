"""Action-aware language/action compatibility classifier.

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


class _CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        states = queries + attended
        return states + self.mlp(self.output_norm(states))


class LanguageActionClassifier(nn.Module):
    """Score compatibility only after sequential VL and action interaction.

    The deployment head sees two learned query states after they cross-attend
    token-level VLM states and then eight action tokens. Detached auxiliary
    probes measure single-modality leakage and cannot update either encoder.
    """

    def __init__(
        self,
        vlm_hidden_dim: int,
        action_horizon: int,
        action_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if min(vlm_hidden_dim, action_horizon, action_dim, hidden_dim, num_heads) <= 0:
            raise ValueError("All classifier dimensions must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)

        self.vl_projector = nn.Sequential(
            nn.LayerNorm(vlm_hidden_dim),
            nn.Linear(vlm_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.action_position = nn.Parameter(torch.empty(1, self.action_horizon, hidden_dim))
        self.classifier_queries = nn.Parameter(torch.empty(1, 2, hidden_dim))
        self.vl_cross_attention = _CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.action_cross_attention = _CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.main_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.vl_audit_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.action_audit_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        nn.init.normal_(self.action_position, std=0.02)
        nn.init.normal_(self.classifier_queries, std=0.02)

    def _encode(
        self,
        hidden_states: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must have shape [B, L, H], got {tuple(hidden_states.shape)}")
        expected_action_shape = (hidden_states.shape[0], self.action_horizon, self.action_dim)
        if tuple(actions.shape) != expected_action_shape:
            raise ValueError(f"actions must have shape {expected_action_shape}, got {tuple(actions.shape)}")
        if attention_mask is not None and tuple(attention_mask.shape) != tuple(hidden_states.shape[:2]):
            raise ValueError("attention_mask must match the first two hidden-state dimensions")
        vl_tokens = self.vl_projector(hidden_states)
        action_tokens = self.action_encoder(actions.to(dtype=hidden_states.dtype))
        action_tokens = action_tokens + self.action_position.to(dtype=action_tokens.dtype)
        padding_mask = None if attention_mask is None else ~attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        return vl_tokens, action_tokens, padding_mask

    def _main_from_tokens(
        self,
        vl_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        queries = self.classifier_queries.expand(vl_tokens.shape[0], -1, -1)
        queries = self.vl_cross_attention(queries, vl_tokens, padding_mask)
        queries = self.action_cross_attention(queries, action_tokens)
        return self.main_head(queries.mean(dim=1)).squeeze(-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        vl_tokens, action_tokens, padding_mask = self._encode(hidden_states, actions, attention_mask)
        main_logits = self._main_from_tokens(vl_tokens, action_tokens, padding_mask)
        valid_mask = None if padding_mask is None else ~padding_mask
        z_vl = masked_mean_pool(vl_tokens, valid_mask)
        z_action = action_tokens.mean(dim=1)
        return {
            "main_logits": main_logits,
            "vl_probe_logits": self.vl_audit_head(z_vl.detach()).squeeze(-1),
            "action_probe_logits": self.action_audit_head(z_action.detach()).squeeze(-1),
        }

    def score_main(
        self,
        hidden_states: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vl_tokens, action_tokens, padding_mask = self._encode(hidden_states, actions, attention_mask)
        return self._main_from_tokens(vl_tokens, action_tokens, padding_mask)

    @staticmethod
    def loss(
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        roles: list[str] | tuple[str, ...] | None = None,
        pos_weight: float = 2.0,
    ) -> dict[str, torch.Tensor]:
        logits = outputs["main_logits"]
        labels = labels.to(device=logits.device, dtype=logits.dtype)
        weight = logits.new_tensor(float(pos_weight))
        bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
        vl_probe = F.binary_cross_entropy_with_logits(outputs["vl_probe_logits"], labels)
        action_probe = F.binary_cross_entropy_with_logits(outputs["action_probe_logits"], labels)

        zero = logits.sum() * 0.0
        language_rank = zero
        action_rank = zero
        if roles is not None:
            if len(roles) != logits.numel():
                raise ValueError("contrastive roles and logits must have equal lengths")
            grouped: dict[str, list[int]] = {"positive": [], "language_negative": [], "action_negative": []}
            for index, role in enumerate(roles):
                if role not in grouped:
                    raise ValueError(f"unknown contrastive role: {role!r}")
                grouped[role].append(index)
            sizes = {len(indices) for indices in grouped.values()}
            if len(sizes) != 1:
                raise ValueError("each training batch must contain complete positive/language/action triplets")
            positive = logits[grouped["positive"]]
            language_negative = logits[grouped["language_negative"]]
            action_negative = logits[grouped["action_negative"]]
            language_rank = F.softplus(1.0 - (positive - language_negative)).mean()
            action_rank = F.softplus(1.0 - (positive - action_negative)).mean()
        total = bce + 0.5 * language_rank + 0.5 * action_rank + 0.1 * vl_probe + 0.1 * action_probe
        return {
            "loss": total,
            "bce": bce,
            "language_rank": language_rank,
            "action_rank": action_rank,
            "vl_probe": vl_probe,
            "action_probe": action_probe,
        }
