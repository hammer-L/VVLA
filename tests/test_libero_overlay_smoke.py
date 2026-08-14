"""Opt-in real-data smoke test.

Run with LIBERO_SOURCE_ROOT=/path/to/libero pytest tests/test_libero_overlay_smoke.py.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from starVLA.dataloader.language_overlay import ALL_VARIANTS, LanguageOverlayDataset
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.modules.language_action_classifier import LanguageActionClassifier
from starVLA.training.classifier_metrics import classifier_record_report


SOURCE_ROOT = os.environ.get("LIBERO_SOURCE_ROOT")


@unittest.skipUnless(SOURCE_ROOT, "set LIBERO_SOURCE_ROOT to run the real LIBERO smoke test")
class RealLiberoOverlaySmokeTest(unittest.TestCase):
    def test_real_episode_to_seven_tuple_and_training_step(self):
        source_name = "libero_goal_no_noops_1.0.0_lerobot"
        source = make_LeRobotSingleDataset(
            Path(SOURCE_ROOT),
            source_name,
            "libero_franka",
            # starVLA names the LeRobot 2.x reader branch "v2.0" even when
            # source info.json reports codebase_version v2.1.
            data_cfg={"lerobot_version": "v2.0", "video_backend": "torchvision_av"},
        )
        with tempfile.TemporaryDirectory() as directory:
            meta = Path(directory)
            pair_id = f"{source_name}:0:8"
            anchor = {
                "source_dataset": source_name, "task_index": 0, "episode_index": 0,
                "split": "test", "anchor_type": "interaction_transport", "anchor_step": 8,
                "action_start": 8, "action_end": 16, "pair_id": pair_id,
            }
            variants = [
                {
                    "variant_id": variant,
                    "text": f"smoke instruction {variant}",
                    "label": int(index < 3),
                    "negative_type": None if index < 3 else variant,
                    "video_evidence_ids": ["ve_smoke"],
                    "explanation": "smoke fixture",
                }
                for index, variant in enumerate(ALL_VARIANTS)
            ]
            (meta / "anchors.jsonl").write_text(json.dumps(anchor) + "\n", encoding="utf-8")
            (meta / "language_bank.jsonl").write_text(
                json.dumps({
                    "source_dataset": source_name, "task_index": 0, "split": "test",
                    "language_group_id": "lg_smoke", "variants": variants,
                }) + "\n",
                encoding="utf-8",
            )
            (meta / "benchmark.json").write_text("{}\n", encoding="utf-8")
            overlay = LanguageOverlayDataset([source], meta, "test", "exhaustive_eval")
            samples = [overlay[index] for index in range(7)]
            self.assertEqual(tuple(sample["variant_id"] for sample in samples), ALL_VARIANTS)
            actions = torch.as_tensor(np.asarray([sample["action"] for sample in samples])).float()
            self.assertEqual(tuple(actions.shape), (7, 8, 7))

            classifier = LanguageActionClassifier(16, 8, 7, hidden_dim=12, dropout=0.0)
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=1.0e-3)
            hidden = torch.randn(7, 5, 16)
            logits = classifier(hidden, actions, torch.ones(7, 5, dtype=torch.bool))
            labels = torch.as_tensor([sample["classifier_label"] for sample in samples]).float()
            parameter_before = classifier.classifier[-1].weight.detach().clone()
            loss = classifier.loss(logits, labels)
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad()
            loss.backward()
            self.assertGreater(
                sum(
                    float(parameter.grad.detach().norm())
                    for parameter in classifier.parameters()
                    if parameter.grad is not None
                ),
                0.0,
            )
            optimizer.step()
            self.assertFalse(torch.equal(parameter_before, classifier.classifier[-1].weight.detach()))
            records = []
            for logit, sample in zip(logits, samples):
                records.append({
                    "variant_instance_id": sample["variant_instance_id"],
                    "pair_id": pair_id,
                    "variant_id": sample["variant_id"],
                    "label": sample["classifier_label"],
                    "logit": float(logit.detach()),
                    "negative_type": sample["negative_type"],
                    "source_dataset": source_name,
                    "episode_index": 0,
                    "task_index": 0,
                    "anchor_type": sample["anchor_type"],
                })
            report = classifier_record_report(records, prefix="test", bootstrap_samples=20)
            self.assertEqual(report["counts"]["samples"], 7)
            self.assertIn("test/paired_accuracy", report["metrics"])


if __name__ == "__main__":
    unittest.main()
