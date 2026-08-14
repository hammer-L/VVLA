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
            positive_scores = logits[indices][group_labels == 1]
            negative_scores = logits[indices][group_labels == 0]
            margins = positive_scores[:, None] - negative_scores[None, :]
            wins = (margins > 0).float() + 0.5 * (margins == 0).float()
            paired_scores.append(float(wins.mean()))
            paired_margins.append(float(margins.mean()))
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


EXPECTED_VARIANTS = (
    "canonical",
    "paraphrase_1",
    "paraphrase_2",
    "wrong_object",
    "wrong_destination_or_relation",
    "wrong_verb_or_state",
    "wrong_order_direction_or_feasible_alternative",
)


def classifier_records_from_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    examples: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert a prediction batch into serializable distributed records."""
    logits = logits.detach().float().reshape(-1).cpu().tolist()
    labels = labels.detach().float().reshape(-1).cpu().tolist()
    examples = list(examples)
    if len(examples) != len(logits) or len(logits) != len(labels):
        raise ValueError("examples, logits and labels must have equal lengths")
    records = []
    for logit, label, example in zip(logits, labels, examples):
        pair_id = str(example["pair_id"])
        variant_id = str(example["variant_id"])
        records.append(
            {
                "variant_instance_id": str(example.get("variant_instance_id", f"{pair_id}:{variant_id}")),
                "logit": float(logit),
                "label": int(label),
                "variant_id": variant_id,
                "pair_id": pair_id,
                "source_dataset": str(example.get("source_dataset", "unknown")),
                "episode_index": int(example.get("source_episode_index", example.get("episode_index", -1))),
                "task_index": int(example.get("source_task_index", example.get("task_index", -1))),
                "negative_type": example.get("negative_type"),
                "anchor_type": example.get("anchor_type"),
            }
        )
    return records


def gather_classifier_records(local_records: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Object-gather records and remove DistributedSampler padding on rank zero."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        gathered: list[list[dict[str, Any]] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_records)
        if dist.get_rank() != 0:
            return None
        candidates = [record for rank_records in gathered for record in (rank_records or [])]
    else:
        candidates = list(local_records)
    return deduplicate_classifier_records(candidates)


def deduplicate_classifier_records(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove sampler padding by stable variant instance ID."""
    unique: dict[str, dict[str, Any]] = {}
    for record in candidates:
        instance_id = record["variant_instance_id"]
        previous = unique.get(instance_id)
        if previous is not None and previous != record:
            raise ValueError(f"conflicting duplicate classifier instance: {instance_id}")
        unique[instance_id] = record
    return sorted(unique.values(), key=lambda record: (record["pair_id"], EXPECTED_VARIANTS.index(record["variant_id"])))


def assert_complete_seven_variant_groups(records: Iterable[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["pair_id"]].append(record)
    for pair_id, group in groups.items():
        variants = tuple(record["variant_id"] for record in sorted(group, key=lambda record: EXPECTED_VARIANTS.index(record["variant_id"])))
        if len(group) != 7 or variants != EXPECTED_VARIANTS:
            raise ValueError(f"pair {pair_id!r} must contain exactly the ordered 3+4 variants; got {variants}")
        labels = {record["variant_id"]: int(record["label"]) for record in group}
        if any(labels[variant] != int(index < 3) for index, variant in enumerate(EXPECTED_VARIANTS)):
            raise ValueError(f"pair {pair_id!r} has an invalid 3-positive/4-negative label pattern")


def best_f1_threshold(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    metrics = binary_classifier_metrics(logits, labels, prefix="threshold")
    return metrics.get("threshold/best_threshold", 0.5), metrics.get("threshold/best_f1", 0.0)


def _scalar_metrics(records: list[dict[str, Any]], *, threshold: float) -> dict[str, float]:
    logits = torch.tensor([record["logit"] for record in records], dtype=torch.float32)
    labels = torch.tensor([record["label"] for record in records], dtype=torch.float32)
    raw = binary_classifier_metrics(logits, labels, prefix="x", threshold=threshold)
    return {key.split("/", 1)[1]: value for key, value in raw.items()}


def _paired_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["pair_id"]].append(record)
    wins, margins = [], []
    for group in groups.values():
        positives = [record["logit"] for record in group if record["label"] == 1]
        negatives = [record["logit"] for record in group if record["label"] == 0]
        for positive in positives:
            for negative in negatives:
                difference = positive - negative
                wins.append(1.0 if difference > 0 else 0.5 if difference == 0 else 0.0)
                margins.append(difference)
    return {
        "paired_accuracy": sum(wins) / len(wins),
        "paired_margin": sum(margins) / len(margins),
    } if wins else {}


def _macro_metrics(
    records: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
    threshold: float,
) -> dict[str, float]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    values: dict[str, list[float]] = defaultdict(list)
    for group in grouped.values():
        metrics = _scalar_metrics(group, threshold=threshold) | _paired_metrics(group)
        for name, value in metrics.items():
            if math.isfinite(value):
                values[name].append(value)
    return {name: sum(items) / len(items) for name, items in values.items() if items}


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def episode_cluster_bootstrap(
    records: list[dict[str, Any]],
    *,
    threshold: float,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Episode-cluster bootstrap CIs for primary scalar and paired metrics."""
    episodes: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        episodes[(record["source_dataset"], record["episode_index"])].append(record)
    keys = sorted(episodes)
    if not keys or samples <= 0:
        return {}
    metric_names = ("accuracy", "auroc", "average_precision", "f1", "ece", "paired_accuracy", "paired_margin")
    # Cluster summaries make 10,000 resamples practical during validation while
    # keeping every anchor from an episode together. The estimand is the
    # episode-macro metric, matching the report's episode_macro section.
    rows = []
    for key in keys:
        metrics = _scalar_metrics(episodes[key], threshold=threshold) | _paired_metrics(episodes[key])
        rows.append([metrics.get(name, float("nan")) for name in metric_names])
    values = torch.tensor(rows, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    distributions: dict[str, list[float]] = {name: [] for name in metric_names}
    remaining = samples
    while remaining:
        chunk = min(remaining, 512)
        indices = torch.randint(len(keys), (chunk, len(keys)), generator=generator)
        selected = values[indices]
        means = torch.nanmean(selected, dim=1)
        for column, name in enumerate(metric_names):
            distributions[name].extend(means[:, column].tolist())
        remaining -= chunk
    return {
        name: {"low": _percentile(values, 0.025), "high": _percentile(values, 0.975)}
        for name, values in distributions.items() if values and any(math.isfinite(value) for value in values)
    }


def classifier_record_report(
    records: list[dict[str, Any]],
    *,
    prefix: str,
    threshold: float | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute strict seven-variant micro/macro/paired evaluation report."""
    assert_complete_seven_variant_groups(records)
    logits = torch.tensor([record["logit"] for record in records], dtype=torch.float32)
    labels = torch.tensor([record["label"] for record in records], dtype=torch.float32)
    selected_threshold, selected_f1 = best_f1_threshold(logits, labels)
    if threshold is None:
        threshold = selected_threshold
        threshold_source = f"{prefix}_max_f1"
    else:
        threshold_source = "frozen_validation"
    micro = _scalar_metrics(records, threshold=threshold) | _paired_metrics(records)
    negative_type_auroc = {}
    positives = [record for record in records if record["label"] == 1]
    for negative_type in EXPECTED_VARIANTS[3:]:
        subset = positives + [record for record in records if record["negative_type"] == negative_type]
        negative_type_auroc[negative_type] = _scalar_metrics(subset, threshold=threshold).get("auroc")
    metrics: dict[str, float] = {}
    for name, value in micro.items():
        metrics[f"{prefix}/micro/{name}"] = value
        metrics[f"{prefix}/{name}"] = value  # compatibility with existing dashboards/checkpoint rules
    for level, keys in (
        ("anchor", ("pair_id",)),
        ("episode_macro", ("source_dataset", "episode_index")),
        ("task_macro", ("source_dataset", "task_index")),
    ):
        for name, value in _macro_metrics(records, keys=keys, threshold=threshold).items():
            metrics[f"{prefix}/{level}/{name}"] = value
    for negative_type, value in negative_type_auroc.items():
        if value is not None:
            metrics[f"{prefix}/negative_auroc/{negative_type}"] = value
    return {
        "metrics": metrics,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "best_f1": float(selected_f1),
        "counts": {
            "samples": len(records),
            "anchors": len({record["pair_id"] for record in records}),
            "episodes": len({(record["source_dataset"], record["episode_index"]) for record in records}),
            "tasks": len({(record["source_dataset"], record["task_index"]) for record in records}),
            "positive": int(labels.sum()),
            "negative": int((labels == 0).sum()),
        },
        "confidence_intervals_95": episode_cluster_bootstrap(
            records, threshold=threshold, samples=bootstrap_samples, seed=seed
        ),
    }
