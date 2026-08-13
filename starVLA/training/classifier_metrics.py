"""Metrics for binary language-action compatibility classification."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable

import torch
import torch.nn.functional as F


def _safe_metric_name(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _auroc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    """Tie-aware probability that a positive scores above a negative."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean())


def _average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    positive_count = int((labels == 1).sum())
    if positive_count == 0:
        return None
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    precision_at_rank = sorted_labels.cumsum(0) / torch.arange(
        1, sorted_labels.numel() + 1, dtype=torch.float32
    )
    return float((precision_at_rank * sorted_labels).sum() / positive_count)


def _expected_calibration_error(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    num_bins: int,
) -> float:
    ece = probabilities.new_tensor(0.0)
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = (probabilities >= lower) & (
            probabilities <= upper if index == num_bins - 1 else probabilities < upper
        )
        if in_bin.any():
            confidence = probabilities[in_bin].mean()
            accuracy = labels[in_bin].mean()
            ece += in_bin.float().mean() * (confidence - accuracy).abs()
    return float(ece)


def binary_classifier_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    prefix: str,
    threshold: float = 0.5,
    ece_bins: int = 10,
    shuffled_logits: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute scalar metrics without requiring scikit-learn."""
    logits = logits.detach().float().reshape(-1).cpu()
    labels = labels.detach().float().reshape(-1).cpu()
    if logits.shape != labels.shape:
        raise ValueError(f"logits and labels must have the same shape, got {logits.shape} and {labels.shape}")
    if logits.numel() == 0:
        return {}

    probabilities = logits.sigmoid()
    predictions = (probabilities >= threshold).float()
    predictions_at_half = (probabilities >= 0.5).float()
    positive = labels == 1
    negative = labels == 0
    tp = int((predictions.bool() & positive).sum())
    fp = int((predictions.bool() & negative).sum())
    fn = int(((~predictions.bool()) & positive).sum())
    tn = int(((~predictions.bool()) & negative).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    half_tp = int((predictions_at_half.bool() & positive).sum())
    half_fp = int((predictions_at_half.bool() & negative).sum())
    half_fn = int(((~predictions_at_half.bool()) & positive).sum())
    half_precision = half_tp / max(half_tp + half_fp, 1)
    half_recall = half_tp / max(half_tp + half_fn, 1)
    half_f1 = 2 * half_precision * half_recall / max(half_precision + half_recall, 1e-12)

    best_threshold = 0.5
    best_f1 = -1.0
    for candidate in torch.unique(probabilities).tolist():
        candidate_predictions = probabilities >= candidate
        candidate_tp = int((candidate_predictions & positive).sum())
        candidate_fp = int((candidate_predictions & negative).sum())
        candidate_fn = int(((~candidate_predictions) & positive).sum())
        candidate_precision = candidate_tp / max(candidate_tp + candidate_fp, 1)
        candidate_recall = candidate_tp / max(candidate_tp + candidate_fn, 1)
        candidate_f1 = 2 * candidate_precision * candidate_recall / max(
            candidate_precision + candidate_recall, 1e-12
        )
        if candidate_f1 > best_f1:
            best_f1 = candidate_f1
            best_threshold = float(candidate)

    metrics = {
        f"{prefix}/loss": float(F.binary_cross_entropy_with_logits(logits, labels)),
        f"{prefix}/accuracy": float((predictions == labels).float().mean()),
        f"{prefix}/precision": precision,
        f"{prefix}/recall": recall,
        f"{prefix}/f1": 2 * precision * recall / max(precision + recall, 1e-12),
        f"{prefix}/accuracy_at_0.5": float((predictions_at_half == labels).float().mean()),
        f"{prefix}/f1_at_0.5": half_f1,
        f"{prefix}/best_f1": best_f1,
        f"{prefix}/best_threshold": best_threshold,
        f"{prefix}/brier_score": float(((probabilities - labels) ** 2).mean()),
        f"{prefix}/ece": _expected_calibration_error(labels, probabilities, max(int(ece_bins), 1)),
        f"{prefix}/threshold": float(threshold),
        f"{prefix}/positive_count": float(positive.sum()),
        f"{prefix}/negative_count": float(negative.sum()),
        f"{prefix}/true_positive_rate": recall,
        f"{prefix}/false_positive_rate": fp / max(fp + tn, 1),
        f"{prefix}/true_negative_rate": tn / max(tn + fp, 1),
        f"{prefix}/probability_mean": float(probabilities.mean()),
        f"{prefix}/logit_mean": float(logits.mean()),
        f"{prefix}/logit_std": float(logits.std(unbiased=False)),
    }

    auroc = _auroc(labels, logits)
    average_precision = _average_precision(labels, logits)
    if auroc is not None:
        metrics[f"{prefix}/auroc"] = auroc
    if average_precision is not None:
        metrics[f"{prefix}/average_precision"] = average_precision

    if positive.any():
        positive_logits = logits[positive]
        metrics[f"{prefix}/positive_loss"] = float(
            F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
        )
        metrics[f"{prefix}/positive_logit_mean"] = float(positive_logits.mean())
        metrics[f"{prefix}/positive_logit_std"] = float(positive_logits.std(unbiased=False))
    if negative.any():
        negative_logits = logits[negative]
        metrics[f"{prefix}/negative_loss"] = float(
            F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
        )
        metrics[f"{prefix}/negative_logit_mean"] = float(negative_logits.mean())
        metrics[f"{prefix}/negative_logit_std"] = float(negative_logits.std(unbiased=False))
    if positive.any() and negative.any():
        metrics[f"{prefix}/logit_margin"] = float(logits[positive].mean() - logits[negative].mean())

    if shuffled_logits is not None and logits.numel() > 1:
        shuffled_logits = shuffled_logits.detach().float().reshape(-1).cpu()
        shuffled_auroc = _auroc(labels, shuffled_logits)
        if shuffled_auroc is not None:
            metrics[f"{prefix}/auroc_shuffled_action"] = shuffled_auroc
            if auroc is not None:
                metrics[f"{prefix}/action_auroc_drop"] = auroc - shuffled_auroc
        metrics[f"{prefix}/action_sensitivity"] = float((logits - shuffled_logits).abs().mean())

    return {key: value for key, value in metrics.items() if math.isfinite(value)}


def example_group_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    examples: Iterable[dict],
    *,
    prefix: str,
    threshold: float,
    pair_id_key: str,
    negative_type_key: str,
) -> dict[str, float]:
    """Compute paired ranking and per-negative-type diagnostics when metadata exists."""
    logits = logits.detach().float().reshape(-1).cpu()
    labels = labels.detach().float().reshape(-1).cpu()
    examples = list(examples)
    if len(examples) != logits.numel():
        return {}

    metrics: dict[str, float] = {}
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        if pair_id_key in example:
            groups[example[pair_id_key]].append(index)

    paired_scores = []
    paired_margins = []
    for indices in groups.values():
        group_labels = labels[indices]
        if (group_labels == 1).any() and (group_labels == 0).any():
            positive_score = logits[indices][group_labels == 1].mean()
            negative_score = logits[indices][group_labels == 0].mean()
            margin = positive_score - negative_score
            paired_scores.append(float((margin > 0).float() + 0.5 * (margin == 0).float()))
            paired_margins.append(float(margin))
    if paired_scores:
        metrics[f"{prefix}/paired_accuracy"] = sum(paired_scores) / len(paired_scores)
        margin_tensor = torch.tensor(paired_margins)
        metrics[f"{prefix}/paired_margin_mean"] = float(margin_tensor.mean())
        metrics[f"{prefix}/paired_margin_std"] = float(margin_tensor.std(unbiased=False))

    negative_types: dict[str, list[int]] = defaultdict(list)
    positive_indices = (labels == 1).nonzero(as_tuple=False).flatten().tolist()
    for index, example in enumerate(examples):
        if labels[index] == 0 and negative_type_key in example:
            negative_types[_safe_metric_name(example[negative_type_key])].append(index)
    for negative_type, negative_indices in negative_types.items():
        type_probabilities = logits[negative_indices].sigmoid()
        metrics[f"{prefix}/negative_accuracy/{negative_type}"] = float(
            (type_probabilities < threshold).float().mean()
        )
        if positive_indices:
            subset_indices = positive_indices + negative_indices
            type_auroc = _auroc(labels[subset_indices], logits[subset_indices])
            if type_auroc is not None:
                metrics[f"{prefix}/auroc/{negative_type}"] = type_auroc
    return metrics
