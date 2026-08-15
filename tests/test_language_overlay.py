import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from starVLA.dataloader.language_overlay import (
    ACTION_VARIANTS,
    ALL_VARIANTS,
    EXHAUSTIVE_VARIANTS,
    GroupedDistributedBatchSampler,
    LanguageOverlayDataset,
    validate_language_overlay_metadata,
)
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


class FakeSourceDataset:
    dataset_name = "source_a"

    def __init__(self):
        self.transforms = lambda raw: raw
        self.epochs = []

    def get_step_data(self, trajectory_id, base_index):
        value = trajectory_id * 100 + base_index
        return {
            "action": np.arange(56, dtype=np.float32).reshape(8, 7) + value,
            "image": np.full((4, 4, 3), value % 255, dtype=np.uint8),
            "lang": "source language",
        }

    def _pack_sample(self, data):
        return {
            "action": data["action"].copy(),
            "image": [Image.fromarray(data["image"])],
            "lang": data["lang"],
            "robot_tag": "franka",
        }

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class LanguageOverlayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.meta = Path(self.temp.name)
        anchors, splits = [], []
        for episode in range(5):
            anchor = {
                "source_dataset": "source_a", "task_index": 0, "episode_index": episode,
                "split": "train", "anchor_type": "interaction_transport", "anchor_step": 2,
                "action_start": 2, "action_end": 10, "pair_id": f"source_a:{episode}:2",
                "action_donors": {
                    "wrong_phase": {
                        "donor_type": "wrong_phase", "action_variant_id": "wrong_phase",
                        "source_dataset": "source_a", "task_index": 0, "episode_index": episode,
                        "split": "train", "anchor_type": "place_or_finalize", "step": 12,
                        "pair_id": f"source_a:{episode}:12",
                    },
                    "wrong_task_hard": {
                        "donor_type": "wrong_task_hard", "action_variant_id": "wrong_task_hard",
                        "source_dataset": "source_a", "task_index": 1, "episode_index": 100 + episode,
                        "split": "train", "anchor_type": "interaction_transport", "step": 3,
                        "pair_id": f"source_a:{100 + episode}:3",
                    },
                },
            }
            anchors.append(anchor)
            splits.extend([
                {"source_dataset": "source_a", "task_index": 0, "episode_index": episode, "split": "train"},
                {"source_dataset": "source_a", "task_index": 1, "episode_index": 100 + episode, "split": "train"},
            ])
        variants = [
            {
                "variant_id": variant_id, "text": f"instruction {variant_id}",
                "label": int(index < 3), "negative_type": None if index < 3 else variant_id,
                "video_evidence_ids": ["ve_1"], "explanation": "supported",
            }
            for index, variant_id in enumerate(ALL_VARIANTS)
        ]
        write_jsonl(self.meta / "anchors.jsonl", anchors)
        write_jsonl(self.meta / "splits.jsonl", splits)
        write_jsonl(self.meta / "language_bank.jsonl", [{
            "source_dataset": "source_a", "task_index": 0, "split": "train",
            "language_group_id": "lg_1", "variants": variants,
        }])
        (self.meta / "benchmark.json").write_text(json.dumps({
            "protocol_version": "2.0.0", "build_status": "validated",
            "action_horizon": 8, "action_dim": 7,
        }) + "\n")
        self.source = FakeSourceDataset()

    def tearDown(self):
        self.temp.cleanup()

    def test_triplet_preserves_observation_and_changes_only_requested_counterfactual(self):
        overlay = LanguageOverlayDataset({"source_a": self.source}, self.meta, "train", "contrastive_train")
        positive, language_negative, action_negative = (overlay[index] for index in range(3))
        self.assertEqual([positive["classifier_label"], language_negative["classifier_label"], action_negative["classifier_label"]], [1, 0, 0])
        self.assertEqual([positive["contrastive_role"], language_negative["contrastive_role"], action_negative["contrastive_role"]], ["positive", "language_negative", "action_negative"])
        np.testing.assert_array_equal(positive["action"], language_negative["action"])
        self.assertFalse(np.array_equal(positive["action"], action_negative["action"]))
        self.assertEqual(positive["lang"], action_negative["lang"])
        self.assertNotEqual(positive["lang"], language_negative["lang"])
        np.testing.assert_array_equal(np.asarray(positive["image"][0]), np.asarray(language_negative["image"][0]))
        self.assertNotEqual(positive["action_pair_id"], positive["audit_action_pair_id"])

    def test_epoch_rotation_covers_languages_and_both_action_donors(self):
        overlay = LanguageOverlayDataset({"source_a": self.source}, self.meta, "train", "contrastive_train")
        positives, negatives, actions = set(), set(), set()
        for epoch in range(12):
            overlay.set_epoch(epoch)
            positives.add(overlay[0]["variant_id"])
            negatives.add(overlay[1]["variant_id"])
            actions.add(overlay[2]["action_variant_id"])
        self.assertEqual(positives, set(ALL_VARIANTS[:3]))
        self.assertEqual(negatives, set(ALL_VARIANTS[3:]))
        self.assertEqual(actions, set(ACTION_VARIANTS[1:]))

    def test_exhaustive_fixed_thirteen_order(self):
        overlay = LanguageOverlayDataset({"source_a": self.source}, self.meta, "train", "exhaustive_eval")
        self.assertEqual(len(overlay), 65)
        combinations = tuple((overlay[index]["variant_id"], overlay[index]["action_variant_id"]) for index in range(13))
        self.assertEqual(combinations, EXHAUSTIVE_VARIANTS)
        self.assertEqual(sum(overlay[index]["classifier_label"] for index in range(13)), 3)

    def test_grouped_sampler_keeps_triplets_in_one_batch_and_rank(self):
        overlay = LanguageOverlayDataset({"source_a": self.source}, self.meta, "train", "contrastive_train")
        rank_batches = [list(GroupedDistributedBatchSampler(overlay, 6, num_replicas=2, rank=rank, shuffle=False)) for rank in (0, 1)]
        rank_groups = []
        for batches in rank_batches:
            groups = set()
            for batch in batches:
                self.assertEqual(len(batch) % 3, 0)
                for start in range(0, len(batch), 3):
                    self.assertEqual(batch[start:start + 3], [batch[start] // 3 * 3 + offset for offset in range(3)])
                    groups.add(batch[start] // 3)
            rank_groups.append(groups)
        self.assertTrue(rank_groups[0].isdisjoint(rank_groups[1] - {0}))  # only deterministic padding may repeat

    def test_dataloader_reset_updates_dataset_and_sampler_epoch(self):
        overlay = LanguageOverlayDataset({"source_a": self.source}, self.meta, "train", "contrastive_train")
        sampler = GroupedDistributedBatchSampler(overlay, 3)
        class Loader:
            def __iter__(self): return iter(())
        loader = Loader()
        loader.dataset = overlay
        loader.batch_sampler = sampler
        loader.sampler = sampler
        _, epoch = TrainerUtils._reset_dataloader(loader, 0)
        self.assertEqual(epoch, 1)
        self.assertEqual(overlay.epoch, 1)

    def test_preflight_rejects_v1_and_accepts_complete_v2(self):
        validate_language_overlay_metadata(self.meta, required_splits=("train",))
        benchmark = json.loads((self.meta / "benchmark.json").read_text())
        benchmark["protocol_version"] = "1.0.0"
        (self.meta / "benchmark.json").write_text(json.dumps(benchmark))
        with self.assertRaisesRegex(ValueError, "v1 metadata"):
            validate_language_overlay_metadata(self.meta, required_splits=("train",))

    def test_preflight_rejects_same_pair_and_cross_split_donors(self):
        anchors = [json.loads(line) for line in (self.meta / "anchors.jsonl").read_text().splitlines()]
        anchors[0]["action_donors"]["wrong_phase"]["pair_id"] = anchors[0]["pair_id"]
        write_jsonl(self.meta / "anchors.jsonl", anchors)
        with self.assertRaisesRegex(ValueError, "different pair_id"):
            validate_language_overlay_metadata(self.meta, required_splits=("train",))


if __name__ == "__main__":
    unittest.main()
