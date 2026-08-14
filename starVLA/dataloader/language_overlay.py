"""Metadata-only language overlay for LIBERO compatibility classification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch.utils.data import Dataset


POSITIVE_VARIANTS = ("canonical", "paraphrase_1", "paraphrase_2")
NEGATIVE_VARIANTS = (
    "wrong_object",
    "wrong_destination_or_relation",
    "wrong_verb_or_state",
    "wrong_order_direction_or_feasible_alternative",
)
ALL_VARIANTS = POSITIVE_VARIANTS + NEGATIVE_VARIANTS
VALID_MODES = ("balanced_train", "exhaustive_eval")
REQUIRED_META_FILES = ("benchmark.json", "anchors.jsonl", "language_bank.jsonl")


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


def _stable_rank_key(seed: int, anchor: Mapping[str, Any]) -> str:
    value = f"{seed}\x1f{anchor['source_dataset']}\x1f{anchor['episode_index']}\x1f{anchor['anchor_step']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def metadata_digest(meta_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("benchmark.json", "splits.jsonl", "video_evidence.jsonl", "language_bank.jsonl", "anchors.jsonl", "quarantine.jsonl"):
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
    """Fail fast when an overlay build is incomplete or internally inconsistent."""
    meta_dir = Path(meta_dir)
    missing_files = [name for name in REQUIRED_META_FILES if not (meta_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"language overlay metadata files are missing in {meta_dir}: {missing_files}")

    try:
        benchmark = json.loads((meta_dir / "benchmark.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{meta_dir / 'benchmark.json'}: invalid JSON") from exc
    build_status = benchmark.get("build_status")
    if build_status != "validated":
        raise ValueError(
            "language overlay metadata is not ready for training: "
            f"build_status={build_status!r} in {meta_dir / 'benchmark.json'}; "
            "finish the benchmark language and anchors stages, then run "
            "`build.py validate --require-complete`"
        )

    required_splits = tuple(str(split) for split in required_splits)
    anchors = [row for row in _read_jsonl(meta_dir / "anchors.jsonl") if row.get("split") in required_splits]
    language_rows = [
        row for row in _read_jsonl(meta_dir / "language_bank.jsonl") if row.get("split") in required_splits
    ]
    if not language_rows:
        raise ValueError(f"language overlay has no language groups in {meta_dir / 'language_bank.jsonl'}")

    language_keys = {
        (row["source_dataset"], int(row["task_index"]), row["split"])
        for row in language_rows
    }
    anchor_keys = {
        (row["source_dataset"], int(row["task_index"]), row["split"])
        for row in anchors
    }
    missing_groups = sorted(anchor_keys - language_keys)
    if missing_groups:
        raise KeyError(
            f"language overlay is missing {len(missing_groups)} language groups required by anchors; "
            f"examples: {missing_groups[:5]}"
        )


class LanguageOverlayDataset(Dataset):
    """Read source steps through their original dataset and replace only language.

    ``source_datasets`` must map the exact persisted source dataset name to the
    corresponding ``LeRobotSingleDataset`` instance.
    """

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
        if isinstance(source_datasets, Mapping):
            self.source_datasets = dict(source_datasets)
        else:
            self.source_datasets = {dataset.dataset_name: dataset for dataset in source_datasets}

        anchors = [row for row in _read_jsonl(self.meta_dir / "anchors.jsonl") if row["split"] == self.split]
        self.anchors = sorted(
            anchors,
            key=lambda row: (row["source_dataset"], int(row["episode_index"]), int(row["anchor_step"])),
        )
        if not self.anchors:
            raise ValueError(f"no language overlay anchors for split={self.split!r}")
        missing_sources = sorted({row["source_dataset"] for row in self.anchors} - self.source_datasets.keys())
        if missing_sources:
            raise KeyError(f"overlay sources are absent from the LeRobot mixture: {missing_sources}")

        language_rows = _read_jsonl(self.meta_dir / "language_bank.jsonl")
        self.language_groups = {
            (row["source_dataset"], int(row["task_index"]), row["split"]): row for row in language_rows
        }
        missing_groups = sorted(
            {
                (row["source_dataset"], int(row["task_index"]), self.split)
                for row in self.anchors
                if (row["source_dataset"], int(row["task_index"]), self.split) not in self.language_groups
            }
        )
        if missing_groups:
            raise KeyError(f"overlay language groups are missing: {missing_groups[:5]}")
        for row in self.anchors:
            if int(row["action_start"]) != int(row["anchor_step"]):
                raise ValueError(f"anchor action_start must equal anchor_step: {row['pair_id']}")
            if int(row["action_end"]) - int(row["action_start"]) != 8:
                raise ValueError(f"classifier action horizon must be 8: {row['pair_id']}")

        ordered = sorted(range(len(self.anchors)), key=lambda index: _stable_rank_key(self.seed, self.anchors[index]))
        self._balanced_rank = {anchor_index: rank for rank, anchor_index in enumerate(ordered)}
        self.meta_digest = metadata_digest(self.meta_dir)

    def __len__(self) -> int:
        return len(self.anchors) if self.mode == "balanced_train" else len(self.anchors) * 7

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        for dataset in self.source_datasets.values():
            setter = getattr(dataset, "set_epoch", None)
            if callable(setter):
                setter(self.epoch)

    def save_dataset_statistics(self, save_path: str | Path) -> None:
        """Delegate source normalization statistics to the original mixture."""
        if self.base_dataset is None or not callable(getattr(self.base_dataset, "save_dataset_statistics", None)):
            raise TypeError("base LeRobot mixture does not expose save_dataset_statistics")
        self.base_dataset.save_dataset_statistics(save_path)

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.mode == "exhaustive_eval":
            return divmod(index, 7)
        anchor_index = index
        rank = self._balanced_rank[anchor_index]
        positive = (rank + self.epoch) % 2 == 0
        cycle = self.epoch // 2 + rank
        if positive:
            variant_index = cycle % len(POSITIVE_VARIANTS)
        else:
            variant_index = len(POSITIVE_VARIANTS) + cycle % len(NEGATIVE_VARIANTS)
        return anchor_index, variant_index

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor_index, variant_index = self._resolve_index(index)
        anchor = self.anchors[anchor_index]
        dataset = self.source_datasets[anchor["source_dataset"]]
        raw_data = dataset.get_step_data(int(anchor["episode_index"]), int(anchor["anchor_step"]))
        transformed = dataset.transforms(raw_data)
        sample = dataset._pack_sample(transformed)
        group = self.language_groups[(anchor["source_dataset"], int(anchor["task_index"]), self.split)]
        variants = group["variants"]
        if len(variants) != 7 or tuple(item["variant_id"] for item in variants) != ALL_VARIANTS:
            raise ValueError(f"language group is not ordered 3+4: {group.get('language_group_id')}")
        variant = variants[variant_index]
        label = int(variant["label"])
        expected_label = int(variant_index < 3)
        if label != expected_label:
            raise ValueError(f"variant label disagrees with fixed ordering: {variant['variant_id']}")
        sample["lang"] = variant["text"]
        sample.update(
            {
                "classifier_label": label,
                "pair_id": anchor["pair_id"],
                "variant_instance_id": f"{anchor['pair_id']}:{variant['variant_id']}",
                "variant_id": variant["variant_id"],
                "negative_type": variant.get("negative_type"),
                "anchor_type": anchor["anchor_type"],
                "source_dataset": anchor["source_dataset"],
                "source_task_index": int(anchor["task_index"]),
                "source_episode_index": int(anchor["episode_index"]),
                "source_step": int(anchor["anchor_step"]),
                "action_start": int(anchor["action_start"]),
                "action_end": int(anchor["action_end"]),
                "video_evidence_ids": list(variant["video_evidence_ids"]),
                "language_group_id": group["language_group_id"],
            }
        )
        return sample


def wrap_language_overlay(base_dataset: Any, data_cfg: Any, *, split: str | None = None, mode: str | None = None):
    """Wrap a LeRobot mixture when overlay configuration is present."""
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
        mode=mode or data_cfg.get("language_overlay_mode", "balanced_train"),
        seed=int(data_cfg.get("language_overlay_seed", 42)),
        base_dataset=base_dataset,
    )
