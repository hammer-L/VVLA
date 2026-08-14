import unittest
import json
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp

from starVLA.model.modules.language_action_classifier import LanguageActionClassifier, masked_mean_pool
from starVLA.training.classifier_metrics import (
    binary_classifier_metrics,
    classifier_record_report,
    deduplicate_classifier_records,
    example_group_metrics,
)


STRICT_VARIANTS = (
    "canonical", "paraphrase_1", "paraphrase_2", "wrong_object",
    "wrong_destination_or_relation", "wrong_verb_or_state",
    "wrong_order_direction_or_feasible_alternative",
)


def _strict_records(episodes=2):
    records = []
    for episode in range(episodes):
        pair_id = f"source:{episode}:4"
        for index, variant in enumerate(STRICT_VARIANTS):
            records.append({
                "variant_instance_id": f"{pair_id}:{variant}",
                "pair_id": pair_id,
                "variant_id": variant,
                "label": int(index < 3),
                "logit": float(4 - index),
                "negative_type": None if index < 3 else variant,
                "source_dataset": "source",
                "episode_index": episode,
                "task_index": 0,
                "anchor_type": "interaction_transport",
            })
    return records


def _distributed_gather_worker(rank, init_path, output_path):
    import torch.distributed as dist
    from starVLA.training.classifier_metrics import classifier_record_report, gather_classifier_records

    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        records = _strict_records(episodes=1)
        local = records[:4] if rank == 0 else records[4:] + [records[0]]  # sampler padding
        gathered = gather_classifier_records(local)
        if rank == 0:
            report = classifier_record_report(gathered, prefix="val", bootstrap_samples=0)
            Path(output_path).write_text(json.dumps(report["metrics"]), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


class MaskedMeanPoolTest(unittest.TestCase):
    def test_padding_tokens_do_not_affect_pool(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = masked_mean_pool(hidden, mask)
        torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0]]))

    def test_rejects_mismatched_mask(self):
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            masked_mean_pool(torch.randn(2, 3, 4), torch.ones(2, 2))


class LanguageActionClassifierTest(unittest.TestCase):
    def test_forward_and_backward(self):
        model = LanguageActionClassifier(
            vlm_hidden_dim=8,
            action_horizon=4,
            action_dim=3,
            hidden_dim=6,
            dropout=0.0,
        )
        hidden = torch.randn(2, 5, 8, requires_grad=True)
        actions = torch.randn(2, 4, 3, requires_grad=True)
        logits = model(hidden, actions, torch.ones(2, 5, dtype=torch.bool))
        self.assertEqual(tuple(logits.shape), (2,))

        loss = model.loss(logits, torch.tensor([1.0, 0.0]))
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(actions.grad)

    def test_rejects_wrong_action_shape(self):
        model = LanguageActionClassifier(8, 4, 3, hidden_dim=6)
        with self.assertRaisesRegex(ValueError, "actions must have shape"):
            model(torch.randn(2, 5, 8), torch.randn(2, 3, 3))


class ClassifierMetricsTest(unittest.TestCase):
    def test_perfect_binary_metrics(self):
        logits = torch.tensor([4.0, -3.0, 2.0, -2.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        shuffled = torch.tensor([-1.0, 1.0, -1.0, 1.0])
        metrics = binary_classifier_metrics(
            logits, labels, prefix="val", shuffled_logits=shuffled
        )
        self.assertEqual(metrics["val/accuracy"], 1.0)
        self.assertEqual(metrics["val/auroc"], 1.0)
        self.assertEqual(metrics["val/average_precision"], 1.0)
        self.assertEqual(metrics["val/best_f1"], 1.0)
        self.assertIn("val/best_threshold", metrics)
        self.assertGreater(metrics["val/action_auroc_drop"], 0.0)

    def test_pair_and_negative_type_metrics(self):
        examples = [
            {"pair_id": "a", "negative_type": "positive"},
            {"pair_id": "a", "negative_type": "wrong language"},
            {"pair_id": "b", "negative_type": "positive"},
            {"pair_id": "b", "negative_type": "wrong action"},
        ]
        metrics = example_group_metrics(
            torch.tensor([3.0, -2.0, 2.0, -1.0]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            examples,
            prefix="val",
            threshold=0.5,
            pair_id_key="pair_id",
            negative_type_key="negative_type",
        )
        self.assertEqual(metrics["val/paired_accuracy"], 1.0)
        self.assertEqual(metrics["val/negative_accuracy/wrong_language"], 1.0)
        self.assertEqual(metrics["val/auroc/wrong_action"], 1.0)

    def test_strict_seven_variant_metrics_and_padding_dedup(self):
        records = _strict_records(episodes=2)
        padded = records + [dict(records[0]), dict(records[-1])]
        deduplicated = deduplicate_classifier_records(padded)
        self.assertEqual(len(deduplicated), len(records))
        single = classifier_record_report(records, prefix="val", bootstrap_samples=20)
        simulated_two_rank = classifier_record_report(deduplicated, prefix="val", bootstrap_samples=20)
        self.assertEqual(single["metrics"], simulated_two_rank["metrics"])
        self.assertEqual(single["counts"]["samples"], 14)
        self.assertEqual(single["counts"]["anchors"], 2)
        self.assertEqual(single["metrics"]["val/paired_accuracy"], 1.0)

    @unittest.skipUnless(torch.distributed.is_available(), "torch.distributed is unavailable")
    def test_two_rank_object_gather_matches_single_process(self):
        records = _strict_records(episodes=1)
        expected = classifier_record_report(records, prefix="val", bootstrap_samples=0)["metrics"]
        with tempfile.TemporaryDirectory() as directory:
            init_path = str(Path(directory) / "dist-init")
            output_path = str(Path(directory) / "metrics.json")
            mp.spawn(_distributed_gather_worker, args=(init_path, output_path), nprocs=2, join=True)
            actual = json.loads(Path(output_path).read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)

    def test_strict_group_rejects_incomplete_pair(self):
        record = {
            "variant_instance_id": "p:canonical", "pair_id": "p", "variant_id": "canonical",
            "label": 1, "logit": 1.0, "negative_type": None, "source_dataset": "s",
            "episode_index": 0, "task_index": 0, "anchor_type": "a",
        }
        with self.assertRaisesRegex(ValueError, "exactly"):
            classifier_record_report([record], prefix="val", bootstrap_samples=0)


if __name__ == "__main__":
    unittest.main()
