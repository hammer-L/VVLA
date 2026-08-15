"""Protocol-v2 metrics for action-aware compatibility classification."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable

import torch
import torch.nn.functional as F


POSITIVE_VARIANTS = ("canonical", "paraphrase_1", "paraphrase_2")
LANGUAGE_NEGATIVES = (
    "wrong_object", "wrong_destination_or_relation", "wrong_verb_or_state",
    "wrong_order_direction_or_feasible_alternative",
)
ACTION_NEGATIVES = ("wrong_phase", "wrong_task_hard")
EXPECTED_COMBINATIONS = (
    *((variant, "positive") for variant in POSITIVE_VARIANTS),
    *((variant, "positive") for variant in LANGUAGE_NEGATIVES),
    *((variant, "wrong_phase") for variant in POSITIVE_VARIANTS),
    *((variant, "wrong_task_hard") for variant in POSITIVE_VARIANTS),
)


def _safe_metric_name(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _auroc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    positive, negative = scores[labels == 1], scores[labels == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    differences = positive[:, None] - negative[None, :]
    return float(((differences > 0).float() + 0.5 * (differences == 0).float()).mean())


def _average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    positive_count = int((labels == 1).sum())
    if not positive_count:
        return None
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    precision = sorted_labels.cumsum(0) / torch.arange(1, len(labels) + 1, dtype=torch.float32)
    return float((precision * sorted_labels).sum() / positive_count)


def _ece(labels: torch.Tensor, probabilities: torch.Tensor, bins: int) -> float:
    result = probabilities.new_tensor(0.0)
    boundaries = torch.linspace(0, 1, bins + 1)
    for index in range(bins):
        selected = (probabilities >= boundaries[index]) & (
            probabilities <= boundaries[index + 1] if index == bins - 1 else probabilities < boundaries[index + 1]
        )
        if selected.any():
            result += selected.float().mean() * (probabilities[selected].mean() - labels[selected].mean()).abs()
    return float(result)


def binary_classifier_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    prefix: str,
    threshold: float = 0.5,
    ece_bins: int = 10,
    shuffled_logits: torch.Tensor | None = None,
) -> dict[str, float]:
    logits = logits.detach().float().reshape(-1).cpu()
    labels = labels.detach().float().reshape(-1).cpu()
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have equal shapes")
    if not logits.numel():
        return {}
    probabilities = logits.sigmoid()
    predictions = (probabilities >= threshold).float()
    positive, negative = labels == 1, labels == 0
    tp = int((predictions.bool() & positive).sum())
    fp = int((predictions.bool() & negative).sum())
    fn = int(((~predictions.bool()) & positive).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    best_threshold, best_f1 = 0.5, -1.0
    for candidate in torch.unique(probabilities).tolist():
        predicted = probabilities >= candidate
        ctp = int((predicted & positive).sum())
        cfp = int((predicted & negative).sum())
        cfn = int(((~predicted) & positive).sum())
        cp, cr = ctp / max(ctp + cfp, 1), ctp / max(ctp + cfn, 1)
        f1 = 2 * cp * cr / max(cp + cr, 1e-12)
        if f1 > best_f1:
            best_threshold, best_f1 = float(candidate), f1
    values = {
        f"{prefix}/loss": float(F.binary_cross_entropy_with_logits(logits, labels)),
        f"{prefix}/accuracy": float((predictions == labels).float().mean()),
        f"{prefix}/precision": precision,
        f"{prefix}/recall": recall,
        f"{prefix}/f1": 2 * precision * recall / max(precision + recall, 1e-12),
        f"{prefix}/best_threshold": best_threshold,
        f"{prefix}/best_f1": best_f1,
        f"{prefix}/brier_score": float(((probabilities - labels) ** 2).mean()),
        f"{prefix}/ece": _ece(labels, probabilities, max(int(ece_bins), 1)),
        f"{prefix}/positive_count": float(positive.sum()),
        f"{prefix}/negative_count": float(negative.sum()),
        f"{prefix}/logit_mean": float(logits.mean()),
        f"{prefix}/logit_std": float(logits.std(unbiased=False)),
    }
    auroc, ap = _auroc(labels, logits), _average_precision(labels, logits)
    if auroc is not None:
        values[f"{prefix}/auroc"] = auroc
    if ap is not None:
        values[f"{prefix}/average_precision"] = ap
    if positive.any() and negative.any():
        values[f"{prefix}/logit_margin"] = float(logits[positive].mean() - logits[negative].mean())
    if shuffled_logits is not None:
        shuffled_logits = shuffled_logits.detach().float().reshape(-1).cpu()
        shuffled_auroc = _auroc(labels, shuffled_logits)
        if shuffled_auroc is not None:
            values[f"{prefix}/auroc_shuffled_action"] = shuffled_auroc
            if auroc is not None:
                values[f"{prefix}/action_auroc_drop"] = auroc - shuffled_auroc
        values[f"{prefix}/action_sensitivity"] = float((logits - shuffled_logits).abs().mean())
    return {key: value for key, value in values.items() if math.isfinite(value)}


def example_group_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    examples: Iterable[dict],
    *, prefix: str, threshold: float, pair_id_key: str, negative_type_key: str,
) -> dict[str, float]:
    """Small non-strict helper retained for live training dashboards."""
    logits, labels, examples = logits.detach().float().cpu(), labels.detach().float().cpu(), list(examples)
    if len(examples) != logits.numel():
        return {}
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[example.get(pair_id_key)].append(index)
    wins, margins = [], []
    for indices in groups.values():
        pos, neg = logits[indices][labels[indices] == 1], logits[indices][labels[indices] == 0]
        if pos.numel() and neg.numel():
            diff = pos[:, None] - neg[None, :]
            wins.extend((((diff > 0).float() + 0.5 * (diff == 0).float()).flatten()).tolist())
            margins.extend(diff.flatten().tolist())
    metrics = {}
    if wins:
        metrics[f"{prefix}/paired_accuracy"] = sum(wins) / len(wins)
        metrics[f"{prefix}/paired_margin_mean"] = sum(margins) / len(margins)
    by_type: dict[str, list[int]] = defaultdict(list)
    positives = (labels == 1).nonzero().flatten().tolist()
    for index, example in enumerate(examples):
        if labels[index] == 0:
            by_type[_safe_metric_name(example.get(negative_type_key))].append(index)
    for name, indices in by_type.items():
        metrics[f"{prefix}/negative_accuracy/{name}"] = float((logits[indices].sigmoid() < threshold).float().mean())
        type_auroc = _auroc(labels[positives + indices], logits[positives + indices])
        if type_auroc is not None:
            metrics[f"{prefix}/auroc/{name}"] = type_auroc
    return metrics


def classifier_records_from_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    examples: Iterable[dict[str, Any]],
    *,
    vl_probe_logits: torch.Tensor | None = None,
    action_probe_logits: torch.Tensor | None = None,
    donor_action_logits: torch.Tensor | None = None,
    mean_action_logits: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    examples = list(examples)
    arrays = {
        "logit": logits, "label": labels, "vl_probe_logit": vl_probe_logits,
        "action_probe_logit": action_probe_logits, "donor_action_logit": donor_action_logits,
        "mean_action_logit": mean_action_logits,
    }
    converted = {
        name: None if value is None else value.detach().float().reshape(-1).cpu().tolist()
        for name, value in arrays.items()
    }
    if any(values is not None and len(values) != len(examples) for values in converted.values()):
        raise ValueError("examples and every classifier output must have equal lengths")
    records = []
    for index, example in enumerate(examples):
        pair_id, variant_id = str(example["pair_id"]), str(example["variant_id"])
        record = {
            "variant_instance_id": str(example.get("variant_instance_id")),
            "pair_id": pair_id,
            "variant_id": variant_id,
            "action_variant_id": str(example["action_variant_id"]),
            "label": int(converted["label"][index]),
            "logit": float(converted["logit"][index]),
            "source_dataset": str(example.get("source_dataset", "unknown")),
            "episode_index": int(example.get("source_episode_index", -1)),
            "task_index": int(example.get("source_task_index", -1)),
            "negative_type": example.get("negative_type"),
            "donor_type": example.get("donor_type"),
            "anchor_type": example.get("anchor_type"),
        }
        for name in ("vl_probe_logit", "action_probe_logit", "donor_action_logit", "mean_action_logit"):
            if converted[name] is not None:
                record[name] = float(converted[name][index])
        records.append(record)
    return records


def gather_classifier_records(local_records: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
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


def _combination_index(record: dict[str, Any]) -> int:
    combination = (record.get("variant_id"), record.get("action_variant_id"))
    try:
        return EXPECTED_COMBINATIONS.index(combination)
    except ValueError as exc:
        raise ValueError(f"record has a non-v2 language/action combination: {combination}") from exc


def deduplicate_classifier_records(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for record in candidates:
        instance_id = record["variant_instance_id"]
        previous = unique.get(instance_id)
        if previous is not None and previous != record:
            raise ValueError(f"conflicting duplicate classifier instance: {instance_id}")
        unique[instance_id] = record
    return sorted(unique.values(), key=lambda record: (record["pair_id"], _combination_index(record)))


def assert_complete_thirteen_variant_groups(records: Iterable[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["pair_id"]].append(record)
    for pair_id, group in groups.items():
        combinations = tuple(
            (record["variant_id"], record["action_variant_id"])
            for record in sorted(group, key=_combination_index)
        )
        if len(group) != 13 or combinations != EXPECTED_COMBINATIONS:
            raise ValueError(f"pair {pair_id!r} must contain exactly the ordered 13 protocol-v2 items; got {combinations}")
        expected_labels = [1, 1, 1] + [0] * 10
        if [int(record["label"]) for record in sorted(group, key=_combination_index)] != expected_labels:
            raise ValueError(f"pair {pair_id!r} has an invalid 3-positive/10-negative label pattern")


def best_f1_threshold(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    metrics = binary_classifier_metrics(logits, labels, prefix="threshold")
    return metrics.get("threshold/best_threshold", 0.5), metrics.get("threshold/best_f1", 0.0)


def _scalar_metrics(records: list[dict[str, Any]], threshold: float, logit_key: str = "logit") -> dict[str, float]:
    logits = torch.tensor([record[logit_key] for record in records])
    labels = torch.tensor([record["label"] for record in records], dtype=torch.float32)
    raw = binary_classifier_metrics(logits, labels, prefix="x", threshold=threshold)
    return {key.split("/", 1)[1]: value for key, value in raw.items()}


def _win_margin(positive: float, negative: float) -> tuple[float, float]:
    difference = positive - negative
    return (1.0 if difference > 0 else 0.5 if difference == 0 else 0.0), difference


def _paired_metrics(records: list[dict[str, Any]], logit_key: str = "logit") -> dict[str, float]:
    groups: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for record in records:
        groups[record["pair_id"]][(record["variant_id"], record["action_variant_id"])] = record
    language_wins, language_margins, action_wins, action_margins = [], [], [], []
    by_donor: dict[str, list[float]] = defaultdict(list)
    by_donor_margin: dict[str, list[float]] = defaultdict(list)
    for group in groups.values():
        for positive_variant in POSITIVE_VARIANTS:
            positive = group[(positive_variant, "positive")][logit_key]
            for negative_variant in LANGUAGE_NEGATIVES:
                win, margin = _win_margin(positive, group[(negative_variant, "positive")][logit_key])
                language_wins.append(win); language_margins.append(margin)
            for donor in ACTION_NEGATIVES:
                win, margin = _win_margin(positive, group[(positive_variant, donor)][logit_key])
                action_wins.append(win); action_margins.append(margin)
                by_donor[donor].append(win); by_donor_margin[donor].append(margin)
    result = {
        "language_paired_accuracy": sum(language_wins) / len(language_wins),
        "language_paired_margin": sum(language_margins) / len(language_margins),
        "action_paired_accuracy": sum(action_wins) / len(action_wins),
        "action_paired_margin": sum(action_margins) / len(action_margins),
    }
    for donor in ACTION_NEGATIVES:
        result[f"action_paired_accuracy/{donor}"] = sum(by_donor[donor]) / len(by_donor[donor])
        result[f"action_paired_margin/{donor}"] = sum(by_donor_margin[donor]) / len(by_donor_margin[donor])
    return result


def _protocol_loss(records: list[dict[str, Any]]) -> tuple[float, float]:
    """Return the fixed v2 objective and its weighted BCE component."""
    logits = torch.tensor([record["logit"] for record in records], dtype=torch.float32)
    labels = torch.tensor([record["label"] for record in records], dtype=torch.float32)
    weighted_bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=torch.tensor(2.0))
    language_terms, action_terms = [], []
    groups: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for record in records:
        groups[record["pair_id"]][(record["variant_id"], record["action_variant_id"])] = record
    for group in groups.values():
        for variant in POSITIVE_VARIANTS:
            positive = torch.tensor(group[(variant, "positive")]["logit"])
            language_terms.extend(
                F.softplus(1.0 - (positive - torch.tensor(group[(negative, "positive")]["logit"])))
                for negative in LANGUAGE_NEGATIVES
            )
            action_terms.extend(
                F.softplus(1.0 - (positive - torch.tensor(group[(variant, donor)]["logit"])))
                for donor in ACTION_NEGATIVES
            )
    language_rank = torch.stack(language_terms).mean()
    action_rank = torch.stack(action_terms).mean()
    probe_losses = logits.new_tensor(0.0)
    if all("vl_probe_logit" in record and "action_probe_logit" in record for record in records):
        probe_losses = 0.1 * F.binary_cross_entropy_with_logits(
            torch.tensor([record["vl_probe_logit"] for record in records]), labels
        ) + 0.1 * F.binary_cross_entropy_with_logits(
            torch.tensor([record["action_probe_logit"] for record in records]), labels
        )
    total = weighted_bce + 0.5 * language_rank + 0.5 * action_rank + probe_losses
    return float(total), float(weighted_bce)


def _macro_metrics(records: list[dict[str, Any]], keys: tuple[str, ...], threshold: float) -> dict[str, float]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    values: dict[str, list[float]] = defaultdict(list)
    for group in grouped.values():
        metrics = _scalar_metrics(group, threshold) | _paired_metrics(group)
        for name, value in metrics.items():
            if math.isfinite(value):
                values[name].append(value)
    return {name: sum(items) / len(items) for name, items in values.items() if items}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return float("nan")
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def episode_cluster_bootstrap(
    records: list[dict[str, Any]], *, threshold: float, samples: int = 10_000, seed: int = 42,
) -> dict[str, dict[str, float]]:
    episodes: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        episodes[(record["source_dataset"], record["episode_index"])].append(record)
    keys = sorted(episodes)
    if not keys or samples <= 0:
        return {}
    metric_names = ("accuracy", "auroc", "language_paired_accuracy", "language_paired_margin", "action_paired_accuracy", "action_paired_margin")
    rows = []
    for key in keys:
        metrics = _scalar_metrics(episodes[key], threshold) | _paired_metrics(episodes[key])
        rows.append([metrics.get(name, float("nan")) for name in metric_names])
    values = torch.tensor(rows, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    distributions = {name: [] for name in metric_names}
    remaining = samples
    while remaining:
        count = min(remaining, 512)
        selected = values[torch.randint(len(keys), (count, len(keys)), generator=generator)]
        means = torch.nanmean(selected, dim=1)
        for column, name in enumerate(metric_names):
            distributions[name].extend(means[:, column].tolist())
        remaining -= count
    return {
        name: {"low": _percentile(items, 0.025), "high": _percentile(items, 0.975)}
        for name, items in distributions.items()
    }


def classifier_record_report(
    records: list[dict[str, Any]], *, prefix: str, threshold: float | None = None,
    bootstrap_samples: int = 10_000, seed: int = 42,
) -> dict[str, Any]:
    assert_complete_thirteen_variant_groups(records)
    logits = torch.tensor([record["logit"] for record in records])
    labels = torch.tensor([record["label"] for record in records], dtype=torch.float32)
    selected_threshold, selected_f1 = best_f1_threshold(logits, labels)
    threshold_source = f"{prefix}_max_f1" if threshold is None else "frozen_validation"
    threshold = selected_threshold if threshold is None else float(threshold)
    micro = _scalar_metrics(records, threshold) | _paired_metrics(records)
    micro["binary_bce_loss"] = micro["loss"]
    micro["loss"], micro["weighted_bce_loss"] = _protocol_loss(records)
    metrics: dict[str, float] = {}
    for name, value in micro.items():
        metrics[f"{prefix}/micro/{name}"] = value
        metrics[f"{prefix}/{name}"] = value
    for level, keys in (
        ("anchor", ("pair_id",)),
        ("episode_macro", ("source_dataset", "episode_index")),
        ("task_macro", ("source_dataset", "task_index")),
    ):
        for name, value in _macro_metrics(records, keys, threshold).items():
            metrics[f"{prefix}/{level}/{name}"] = value

    positives = [record for record in records if record["label"] == 1]
    for donor in ACTION_NEGATIVES:
        subset = positives + [record for record in records if record["action_variant_id"] == donor]
        donor_metrics = _scalar_metrics(subset, threshold)
        if "auroc" in donor_metrics:
            metrics[f"{prefix}/action_negative_auroc/{donor}"] = donor_metrics["auroc"]
    for negative in LANGUAGE_NEGATIVES:
        subset = positives + [record for record in records if record["variant_id"] == negative and record["action_variant_id"] == "positive"]
        values = _scalar_metrics(subset, threshold)
        if "auroc" in values:
            metrics[f"{prefix}/language_negative_auroc/{negative}"] = values["auroc"]

    auxiliary_keys = {
        "vl_probe": "vl_probe_logit", "action_probe": "action_probe_logit",
        "donor_action": "donor_action_logit", "mean_action": "mean_action_logit",
    }
    for namespace, key in auxiliary_keys.items():
        if all(key in record for record in records):
            aux = _scalar_metrics(records, threshold, key) | _paired_metrics(records, key)
            for name, value in aux.items():
                metrics[f"{prefix}/{namespace}/{name}"] = value
            metrics[f"{prefix}/{namespace}/logit_sensitivity"] = sum(
                abs(record["logit"] - record[key]) for record in records
            ) / len(records)

    cis = episode_cluster_bootstrap(records, threshold=threshold, samples=bootstrap_samples, seed=seed)
    acceptance = None
    if prefix == "test":
        language_pra, action_pra = micro["language_paired_accuracy"], micro["action_paired_accuracy"]
        vl_action = metrics.get("test/vl_probe/action_paired_accuracy", float("nan"))
        action_language = metrics.get("test/action_probe/language_paired_accuracy", float("nan"))
        donor_action_pra = metrics.get("test/donor_action/action_paired_accuracy", float("nan"))
        donor_auroc = metrics.get("test/donor_action/auroc", float("nan"))
        checks = {
            "language_pra": language_pra > 0.65,
            "action_pra": action_pra > 0.65,
            "language_ci_low": cis.get("language_paired_accuracy", {}).get("low", -math.inf) > 0.5,
            "action_ci_low": cis.get("action_paired_accuracy", {}).get("low", -math.inf) > 0.5,
            "full_over_vl_action": action_pra - vl_action >= 0.15,
            "full_over_action_language": language_pra - action_language >= 0.15,
            "audit_action_drop": max(action_pra - donor_action_pra, micro.get("auroc", float("nan")) - donor_auroc) >= 0.10,
            "wrong_phase_pra": micro["action_paired_accuracy/wrong_phase"] > 0.60,
            "wrong_task_hard_pra": micro["action_paired_accuracy/wrong_task_hard"] > 0.60,
        }
        acceptance = {"passed": all(checks.values()), "checks": checks}
    return {
        "protocol_version": "2.0.0",
        "metrics": metrics,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "best_f1": float(selected_f1),
        "counts": {
            "samples": len(records), "anchors": len({record["pair_id"] for record in records}),
            "episodes": len({(record["source_dataset"], record["episode_index"]) for record in records}),
            "tasks": len({(record["source_dataset"], record["task_index"]) for record in records}),
            "positive": int(labels.sum()), "negative": int((labels == 0).sum()),
        },
        "confidence_intervals_95": cis,
        "acceptance": acceptance,
        # Persist every normal and counterfactual logit in the report so the
        # aggregate metrics remain independently auditable.
        "predictions": records,
    }


def classifier_checkpoint_score(report: dict[str, Any], prefix: str = "eval") -> tuple[float, float, float, float]:
    """Lexicographic v2 checkpoint key: minimum PRA, harmonic PRA, AUROC, -loss."""
    metrics = report["metrics"]
    language_pra = float(metrics.get(f"{prefix}/language_paired_accuracy", float("-inf")))
    action_pra = float(metrics.get(f"{prefix}/action_paired_accuracy", float("-inf")))
    harmonic = (
        2.0 * language_pra * action_pra / (language_pra + action_pra)
        if language_pra > 0 and action_pra > 0 else float("-inf")
    )
    return (
        min(language_pra, action_pra),
        harmonic,
        float(metrics.get(f"{prefix}/auroc", float("-inf"))),
        -float(metrics.get(f"{prefix}/loss", float("inf"))),
    )
