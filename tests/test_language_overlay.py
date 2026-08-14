import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from starVLA.dataloader.language_overlay import (
    ALL_VARIANTS,
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
        return {
            "action": np.arange(56, dtype=np.float32).reshape(8, 7) + base_index,
            "image": np.full((4, 4, 3), trajectory_id + base_index, dtype=np.uint8),
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
        anchors = []
        for episode in range(5):
            anchors.append(
                {
                    "source_dataset": "source_a",
                    "task_index": 0,
                    "episode_index": episode,
                    "split": "train",
                    "anchor_type": "interaction_transport",
                    "anchor_step": 2,
                    "action_start": 2,
                    "action_end": 10,
                    "pair_id": f"source_a:{episode}:2",
                }
            )
        variants = []
        for index, variant_id in enumerate(ALL_VARIANTS):
            variants.append(
                {
                    "variant_id": variant_id,
                    "text": f"instruction {variant_id}",
                    "label": int(index < 3),
                    "negative_type": None if index < 3 else variant_id,
                    "video_evidence_ids": ["ve_1"],
                    "explanation": "supported",
                }
            )
        write_jsonl(self.meta / "anchors.jsonl", anchors)
        write_jsonl(
            self.meta / "language_bank.jsonl",
            [{
                "source_dataset": "source_a",
                "task_index": 0,
                "split": "train",
                "language_group_id": "lg_1",
                "variants": variants,
            }],
        )
        (self.meta / "benchmark.json").write_text("{}\n")
        self.source = FakeSourceDataset()

    def tearDown(self):
        self.temp.cleanup()

    def test_overlay_preserves_source_modalities(self):
        overlay = LanguageOverlayDataset({"source_a": self.source}, self.meta, "train", "balanced_train")
        anchor_index, _ = overlay._resolve_index(0)
        anchor = overlay.anchors[anchor_index]
        raw = self.source.get_step_data(anchor["episode_index"], anchor["anchor_step"])
        source = self.source._pack_sample(self.source.transforms(raw))
        sample = overlay[0]
        np.testing.assert_array_equal(sample["action"], source["action"])
        np.testing.assert_array_equal(np.asarray(sample["image"][0]), np.asarray(source["image"][0]))
        self.assertEqual(sample["robot_tag"], source["robot_tag"])
        self.assertNotEqual(sample["lang"], source["lang"])

    def test_balanced_two_epoch_and_variant_rotation(self):
        overlay = LanguageOverlayDataset({"source_a": self.source}, self.meta, "train", "balanced_train")
        counts = []
        for epoch in (0, 1):
            overlay.set_epoch(epoch)
            labels = [overlay[index]["classifier_label"] for index in range(len(overlay))]
            counts.append(sum(labels))
            self.assertLessEqual(abs(sum(labels) - (len(labels) - sum(labels))), 1)
        self.assertEqual(sum(counts), len(overlay))

        positive_variants = []
        negative_variants = []
        for epoch in range(8):
            overlay.set_epoch(epoch)
            item = overlay[0]
            (positive_variants if item["classifier_label"] else negative_variants).append(item["variant_id"])
        self.assertEqual(len(set(positive_variants[:3])), 3)
        self.assertEqual(len(set(negative_variants[:4])), 4)

    def test_exhaustive_order_and_length(self):
        overlay = LanguageOverlayDataset({"source_a": self.source}, self.meta, "train", "exhaustive_eval")
        self.assertEqual(len(overlay), 35)
        self.assertEqual(tuple(overlay[index]["variant_id"] for index in range(7)), ALL_VARIANTS)
        self.assertEqual(sum(overlay[index]["classifier_label"] for index in range(7)), 3)

    def test_dataloader_reset_updates_dataset_epoch(self):
        overlay = LanguageOverlayDataset({"source_a": self.source}, self.meta, "train", "balanced_train")

        class Sampler:
            def set_epoch(self, epoch):
                self.epoch = epoch

        class Loader:
            dataset = overlay
            sampler = Sampler()

            def __iter__(self):
                return iter(())

        _, epoch = TrainerUtils._reset_dataloader(Loader(), 0)
        self.assertEqual(epoch, 1)
        self.assertEqual(overlay.epoch, 1)

    def test_preflight_rejects_incomplete_build_before_training(self):
        (self.meta / "benchmark.json").write_text(
            json.dumps({"build_status": "split"}) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "build_status='split'"):
            validate_language_overlay_metadata(self.meta)

    def test_preflight_accepts_complete_metadata(self):
        (self.meta / "benchmark.json").write_text(
            json.dumps({"build_status": "validated"}) + "\n", encoding="utf-8"
        )
        validate_language_overlay_metadata(self.meta, required_splits=("train",))


if __name__ == "__main__":
    unittest.main()
