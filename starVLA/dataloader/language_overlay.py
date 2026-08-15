"""Protocol-v2 metadata overlay for action-aware LIBERO classification."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, Dataset


PROTOCOL_VERSION = "2.0.0"
ACTION_HORIZON = 8
ACTION_DIM = 7
POSITIVE_VARIANTS = ("canonical", "paraphrase_1", "paraphrase_2")
NEGATIVE_VARIANTS = (
    "wrong_object",
    "wrong_destination_or_relation",
    "wrong_verb_or_state",
    "wrong_order_direction_or_feasible_alternative",
)
ALL_VARIANTS = POSITIVE_VARIANTS + NEGATIVE_VARIANTS
ACTION_VARIANTS = ("positive", "wrong_phase", "wrong_task_hard")
EXHAUSTIVE_VARIANTS = (
    *((variant, "positive") for variant in POSITIVE_VARIANTS),
    *((variant, "positive") for variant in NEGATIVE_VARIANTS),
    *((variant, "wrong_phase") for variant in POSITIVE_VARIANTS),
    *((variant, "wrong_task_hard") for variant in POSITIVE_VARIANTS),
)
VALID_MODES = ("contrastive_train", "exhaustive_eval")
REQUIRED_META_FILES = ("benchmark.json", "anchors.jsonl", "language_bank.jsonl", "splits.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def _stable_rank_key(seed: int, *parts: Any) -> str:
    value = "\x1f".join(str(item) for item in (seed, *parts))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def metadata_digest(meta_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "benchmark.json", "splits.jsonl", "video_evidence.jsonl",
        "language_bank.jsonl", "anchors.jsonl", "quarantine.jsonl",
    ):
        path = meta_dir / name
        if path.exists():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_language_overlay_metadata(
    meta_dir: str | Path,
    *,
    required_splits: Sequence[str] = ("train", "val", "test"),
) -> None:
    """Reject stale, incomplete, cross-split, or malformed v2 metadata."""
    meta_dir = Path(meta_dir)
    missing_files = [name for name in REQUIRED_META_FILES if not (meta_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"language overlay metadata files are missing in {meta_dir}: {missing_files}")
    try:
        benchmark = json.loads((meta_dir / "benchmark.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{meta_dir / 'benchmark.json'}: invalid JSON") from exc
    if benchmark.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"language overlay requires protocol {PROTOCOL_VERSION}; v1 metadata/checkpoints are incompatible "
            f"(got {benchmark.get('protocol_version')!r})"
        )
    if benchmark.get("build_status") != "validated":
        raise ValueError(
            "language overlay metadata is not ready for training: "
            f"build_status={benchmark.get('build_status')!r}; run `build.py validate --require-complete`"
        )
    if benchmark.get("action_horizon") != ACTION_HORIZON or benchmark.get("action_dim") != ACTION_DIM:
        raise ValueError(f"protocol v2 requires action shape [{ACTION_HORIZON},{ACTION_DIM}]")

    required_splits = tuple(str(split) for split in required_splits)
    splits = _read_jsonl(meta_dir / "splits.jsonl")
    split_lookup = {(row["source_dataset"], int(row["episode_index"])): row for row in splits}
    anchors = [row for row in _read_jsonl(meta_dir / "anchors.jsonl") if row.get("split") in required_splits]
    language_rows = [
        row for row in _read_jsonl(meta_dir / "language_bank.jsonl") if row.get("split") in required_splits
    ]
    language_by_key = {
        (row["source_dataset"], int(row["task_index"]), row["split"]): row for row in language_rows
    }
    if not language_by_key:
        raise ValueError(f"language overlay has no language groups in {meta_dir / 'language_bank.jsonl'}")
    for anchor in anchors:
        key = (anchor["source_dataset"], int(anchor["task_index"]), anchor["split"])
        group = language_by_key.get(key)
        if group is None:
            raise KeyError(f"language overlay is missing language group {key}")
        variants = group.get("variants", [])
        if len(variants) != 7 or tuple(item.get("variant_id") for item in variants) != ALL_VARIANTS:
            raise ValueError(f"language group is not ordered 3+4: {group.get('language_group_id')}")
        if [int(item.get("label", -1)) for item in variants] != [1, 1, 1, 0, 0, 0, 0]:
            raise ValueError(f"language group has invalid labels: {group.get('language_group_id')}")
        if int(anchor.get("action_end", -1)) - int(anchor.get("action_start", -1)) != ACTION_HORIZON:
            raise ValueError(f"anchor action shape is not [{ACTION_HORIZON},{ACTION_DIM}]: {anchor.get('pair_id')}")
        donors = anchor.get("action_donors")
        if not isinstance(donors, dict) or set(donors) != {"wrong_phase", "wrong_task_hard"}:
            raise ValueError(f"anchor lacks both action donors; exhaustive 13-item group is incomplete: {anchor.get('pair_id')}")
        for donor_type, donor in donors.items():
            if donor.get("pair_id") == anchor.get("pair_id"):
                raise ValueError(f"donor must have a different pair_id: {anchor.get('pair_id')}:{donor_type}")
            donor_split = split_lookup.get((donor.get("source_dataset"), int(donor.get("episode_index", -1))))
            if donor_split is None or donor.get("split") != anchor.get("split") or donor_split.get("split") != anchor.get("split"):
                raise ValueError(f"cross-split action donor: {anchor.get('pair_id')}:{donor_type}")
            if donor.get("source_dataset") != anchor.get("source_dataset"):
                raise ValueError(f"cross-dataset action donor: {anchor.get('pair_id')}:{donor_type}")
            if donor_type == "wrong_phase":
                if (
                    int(donor["episode_index"]) != int(anchor["episode_index"])
                    or donor.get("anchor_type") == anchor.get("anchor_type")
                    or abs(int(donor["step"]) - int(anchor["anchor_step"])) < ACTION_HORIZON
                ):
                    raise ValueError(f"wrong_phase donor violates phase/distance constraints: {anchor.get('pair_id')}")
            elif int(donor["task_index"]) == int(anchor["task_index"]) or donor.get("anchor_type") != anchor.get("anchor_type"):
                raise ValueError(f"wrong_task_hard donor violates task/type constraints: {anchor.get('pair_id')}")


class LanguageOverlayDataset(Dataset):
    """Construct deterministic contrastive triplets or strict 13-item groups."""

    def __init__(
        self,
        source_datasets: Mapping[str, Any] | Sequence[Any],
        meta_dir: str | Path,
        split: str,
        mode: str,
        *,
        seed: int = 42,
        base_dataset: Any | None = None,
    ) -> None:
        self.meta_dir = Path(meta_dir)
        self.split = str(split)
        self.mode = str(mode)
        self.seed = int(seed)
        self.epoch = 0
        self.base_dataset = base_dataset
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"language overlay split must be train/val/test, got {self.split!r}")
        if self.mode not in VALID_MODES:
            raise ValueError(f"language overlay mode must be one of {VALID_MODES}, got {self.mode!r}")
        self.source_datasets = (
            dict(source_datasets) if isinstance(source_datasets, Mapping)
            else {dataset.dataset_name: dataset for dataset in source_datasets}
        )
        self.anchors = sorted(
            (row for row in _read_jsonl(self.meta_dir / "anchors.jsonl") if row["split"] == self.split),
            key=lambda row: (row["source_dataset"], int(row["episode_index"]), int(row["anchor_step"])),
        )
        if not self.anchors:
            raise ValueError(f"no language overlay anchors for split={self.split!r}")
        missing_sources = sorted({row["source_dataset"] for row in self.anchors} - self.source_datasets.keys())
        if missing_sources:
            raise KeyError(f"overlay sources are absent from the LeRobot mixture: {missing_sources}")
        self.language_groups = {
            (row["source_dataset"], int(row["task_index"]), row["split"]): row
            for row in _read_jsonl(self.meta_dir / "language_bank.jsonl")
        }
        for anchor in self.anchors:
            if (anchor["source_dataset"], int(anchor["task_index"]), self.split) not in self.language_groups:
                raise KeyError(f"overlay language group is missing for {anchor['pair_id']}")
            if set(anchor.get("action_donors", {})) != {"wrong_phase", "wrong_task_hard"}:
                raise ValueError(f"protocol-v2 action donors are missing: {anchor['pair_id']}")
        ordered = sorted(
            range(len(self.anchors)),
            key=lambda index: _stable_rank_key(self.seed, self.anchors[index]["pair_id"]),
        )
        self._stable_rank = {anchor_index: rank for rank, anchor_index in enumerate(ordered)}
        self._mean_actions = {
            name: self._normalized_mean_action(dataset)
            for name, dataset in self.source_datasets.items()
        }
        self.meta_digest = metadata_digest(self.meta_dir)

    @property
    def group_size(self) -> int:
        return 3 if self.mode == "contrastive_train" else 13

    def __len__(self) -> int:
        return len(self.anchors) * self.group_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        for dataset in self.source_datasets.values():
            setter = getattr(dataset, "set_epoch", None)
            if callable(setter):
                setter(self.epoch)

    def save_dataset_statistics(self, save_path: str | Path) -> None:
        if self.base_dataset is None or not callable(getattr(self.base_dataset, "save_dataset_statistics", None)):
            raise TypeError("base LeRobot mixture does not expose save_dataset_statistics")
        self.base_dataset.save_dataset_statistics(save_path)

    def _resolve_index(self, index: int) -> tuple[int, int, str, str | None]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        anchor_index, offset = divmod(index, self.group_size)
        if self.mode == "exhaustive_eval":
            language_id, action_variant = EXHAUSTIVE_VARIANTS[offset]
            return anchor_index, ALL_VARIANTS.index(language_id), action_variant, None
        cycle = self.epoch + self._stable_rank[anchor_index]
        positive_index = cycle % len(POSITIVE_VARIANTS)
        negative_index = len(POSITIVE_VARIANTS) + cycle % len(NEGATIVE_VARIANTS)
        action_variant = ACTION_VARIANTS[1 + cycle % 2]
        if offset == 0:
            return anchor_index, positive_index, "positive", "positive"
        if offset == 1:
            return anchor_index, negative_index, "positive", "language_negative"
        return anchor_index, positive_index, action_variant, "action_negative"

    def _load_sample(self, ref: Mapping[str, Any]) -> dict[str, Any]:
        dataset = self.source_datasets[str(ref["source_dataset"])]
        raw = dataset.get_step_data(int(ref["episode_index"]), int(ref["step"]))
        return dataset._pack_sample(dataset.transforms(raw))

    @staticmethod
    def _action_chunk(sample: Mapping[str, Any], context: str) -> np.ndarray:
        actions = np.asarray(sample["action"])
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] < ACTION_HORIZON:
            raise ValueError(f"{context} action must provide [{ACTION_HORIZON},{ACTION_DIM}], got {actions.shape}")
        return actions[-ACTION_HORIZON:].copy()

    @staticmethod
    def _normalized_mean_action(dataset: Any) -> np.ndarray:
        """Read the source loader's statistics and apply LIBERO normalization."""
        try:
            keys = list(dataset.modality_keys["action"])
            statistics = dataset.metadata.statistics.action
            means = []
            for dimension, key in enumerate(keys):
                stat = statistics[key.split(".", 1)[-1]]
                mean = float(np.asarray(stat.mean).reshape(-1)[0])
                if dimension < 6:
                    minimum = float(np.asarray(stat.min).reshape(-1)[0])
                    maximum = float(np.asarray(stat.max).reshape(-1)[0])
                    mean = 0.0 if maximum == minimum else 2.0 * (mean - minimum) / (maximum - minimum) - 1.0
                means.append(mean)
            if len(means) != ACTION_DIM:
                raise ValueError("unexpected action dimension")
            return np.broadcast_to(np.asarray(means, dtype=np.float32), (ACTION_HORIZON, ACTION_DIM)).copy()
        except (AttributeError, KeyError, TypeError, ValueError):
            # Lightweight test/custom sources may omit metadata. Zero is the
            # neutral normalized mean fallback; production LIBERO sources take
            # the statistics path above.
            return np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor_index, variant_index, action_variant, role = self._resolve_index(index)
        anchor = self.anchors[anchor_index]
        source_ref = {
            "source_dataset": anchor["source_dataset"],
            "episode_index": anchor["episode_index"],
            "step": anchor["anchor_step"],
        }
        sample = self._load_sample(source_ref)
        positive_action = self._action_chunk(sample, anchor["pair_id"])
        donors = anchor["action_donors"]
        if action_variant == "positive":
            action = positive_action
            donor_type = None
            action_pair_id = anchor["pair_id"]
        else:
            donor = donors[action_variant]
            action = self._action_chunk(self._load_sample(donor), donor["pair_id"])
            donor_type = action_variant
            action_pair_id = donor["pair_id"]
        audit_variant = "wrong_task_hard" if action_variant != "wrong_task_hard" else "wrong_phase"
        audit_ref = donors[audit_variant]
        audit_action = self._action_chunk(self._load_sample(audit_ref), audit_ref["pair_id"])
        if audit_ref["pair_id"] == action_pair_id:
            raise ValueError(f"audit action must have a different pair_id: {anchor['pair_id']}")

        group = self.language_groups[(anchor["source_dataset"], int(anchor["task_index"]), self.split)]
        variants = group["variants"]
        if len(variants) != 7 or tuple(item["variant_id"] for item in variants) != ALL_VARIANTS:
            raise ValueError(f"language group is not ordered 3+4: {group.get('language_group_id')}")
        variant = variants[variant_index]
        label = int(variant_index < 3 and action_variant == "positive")
        sample["action"] = action
        sample["audit_action"] = audit_action
        sample["mean_action"] = self._mean_actions[anchor["source_dataset"]].copy()
        sample["lang"] = variant["text"]
        sample.update({
            "classifier_label": label,
            "pair_id": anchor["pair_id"],
            "variant_instance_id": f"{anchor['pair_id']}:{variant['variant_id']}:{action_variant}",
            "variant_id": variant["variant_id"],
            "action_variant_id": action_variant,
            "action_pair_id": action_pair_id,
            "audit_action_pair_id": audit_ref["pair_id"],
            "donor_type": donor_type,
            "negative_type": variant.get("negative_type") if variant_index >= 3 else donor_type,
            "contrastive_role": role,
            "anchor_type": anchor["anchor_type"],
            "source_dataset": anchor["source_dataset"],
            "source_task_index": int(anchor["task_index"]),
            "source_episode_index": int(anchor["episode_index"]),
            "source_step": int(anchor["anchor_step"]),
            "language_group_id": group["language_group_id"],
        })
        return sample


class GroupedDistributedBatchSampler(BatchSampler):
    """Assign complete contrastive triplets to one batch and one rank."""

    def __init__(
        self,
        dataset: LanguageOverlayDataset,
        batch_size: int,
        *,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if dataset.mode != "contrastive_train":
            raise ValueError("grouped sampler is only valid for contrastive_train")
        if batch_size < 3 or batch_size % 3:
            raise ValueError("per-device batch_size must be a positive multiple of the triplet size 3")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.groups_per_batch = self.batch_size // 3
        self.num_replicas = int(num_replicas if num_replicas is not None else (dist.get_world_size() if dist.is_initialized() else 1))
        self.rank = int(rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0))
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def _rank_groups(self) -> list[int]:
        groups = list(range(len(self.dataset.anchors)))
        if self.shuffle:
            groups.sort(key=lambda index: _stable_rank_key(self.seed, self.epoch, self.dataset.anchors[index]["pair_id"]))
        if self.drop_last:
            total = len(groups) - len(groups) % self.num_replicas
            groups = groups[:total]
        elif groups:
            total = math.ceil(len(groups) / self.num_replicas) * self.num_replicas
            groups += groups[: total - len(groups)]
        return groups[self.rank::self.num_replicas]

    def __iter__(self) -> Iterator[list[int]]:
        rank_groups = self._rank_groups()
        for start in range(0, len(rank_groups), self.groups_per_batch):
            selected = rank_groups[start : start + self.groups_per_batch]
            if self.drop_last and len(selected) < self.groups_per_batch:
                continue
            yield [group * 3 + offset for group in selected for offset in range(3)]

    def __len__(self) -> int:
        count = len(self._rank_groups())
        return count // self.groups_per_batch if self.drop_last else math.ceil(count / self.groups_per_batch)


def wrap_language_overlay(base_dataset: Any, data_cfg: Any, *, split: str | None = None, mode: str | None = None):
    meta_dir = data_cfg.get("language_overlay_meta", None)
    if not meta_dir:
        return base_dataset
    sources = getattr(base_dataset, "datasets", None)
    if sources is None:
        raise TypeError("language overlay requires a LeRobot mixture exposing .datasets")
    return LanguageOverlayDataset(
        sources,
        meta_dir,
        split=split or data_cfg.get("language_overlay_split", "train"),
        mode=mode or data_cfg.get("language_overlay_mode", "contrastive_train"),
        seed=int(data_cfg.get("language_overlay_seed", 42)),
        base_dataset=base_dataset,
    )
