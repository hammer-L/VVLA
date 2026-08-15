import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.multiprocessing as mp

from starVLA.model.modules.language_action_classifier import LanguageActionClassifier, masked_mean_pool
from starVLA.training.classifier_metrics import (
    ACTION_NEGATIVES,
    EXPECTED_COMBINATIONS,
    binary_classifier_metrics,
    classifier_checkpoint_score,
    classifier_record_report,
    deduplicate_classifier_records,
)


def _strict_records(episodes=2):
    main_scores = {
        ("canonical", "positive"): 5.0, ("paraphrase_1", "positive"): 4.5,
        ("paraphrase_2", "positive"): 4.0, ("wrong_object", "positive"): 1.0,
        ("wrong_destination_or_relation", "positive"): 0.0,
        ("wrong_verb_or_state", "positive"): -1.0,
        ("wrong_order_direction_or_feasible_alternative", "positive"): -2.0,
        ("canonical", "wrong_phase"): 2.0, ("paraphrase_1", "wrong_phase"): 1.5,
        ("paraphrase_2", "wrong_phase"): 1.0, ("canonical", "wrong_task_hard"): 0.5,
        ("paraphrase_1", "wrong_task_hard"): 0.0, ("paraphrase_2", "wrong_task_hard"): -0.5,
    }
    records = []
    for episode in range(episodes):
        pair_id = f"source:{episode}:4"
        for variant, action_variant in EXPECTED_COMBINATIONS:
            positive = variant in ("canonical", "paraphrase_1", "paraphrase_2") and action_variant == "positive"
            vl_score = 3.0 if variant in ("canonical", "paraphrase_1", "paraphrase_2") else -2.0
            action_score = 1.0 if action_variant == "positive" else -1.0
            records.append({
                "variant_instance_id": f"{pair_id}:{variant}:{action_variant}",
                "pair_id": pair_id, "variant_id": variant, "action_variant_id": action_variant,
                "label": int(positive), "logit": main_scores[(variant, action_variant)],
                "vl_probe_logit": vl_score, "action_probe_logit": action_score,
                "donor_action_logit": 0.0, "mean_action_logit": 0.0,
                "negative_type": None if positive else (variant if action_variant == "positive" else action_variant),
                "donor_type": None if action_variant == "positive" else action_variant,
                "source_dataset": "source", "episode_index": episode, "task_index": 0,
                "anchor_type": "interaction_transport",
            })
    return records


def _distributed_gather_worker(rank, init_path, output_path):
    import os
    import torch.distributed as dist
    from starVLA.training.classifier_metrics import classifier_record_report, gather_classifier_records
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        records = _strict_records(episodes=1)
        local = records[:7] if rank == 0 else records[7:] + [records[0]]
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
        pooled = masked_mean_pool(hidden, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0]]))


class LanguageActionClassifierTest(unittest.TestCase):
    def _model(self):
        return LanguageActionClassifier(8, 4, 3, hidden_dim=8, dropout=0.0, num_heads=2)

    def test_forward_shapes_joint_gradients_and_no_concat_path(self):
        model = self._model()
        hidden = torch.randn(6, 5, 8, requires_grad=True)
        actions = torch.randn(6, 4, 3, requires_grad=True)
        output = model(hidden, actions, torch.ones(6, 5, dtype=torch.bool))
        self.assertEqual(set(output), {"main_logits", "vl_probe_logits", "action_probe_logits"})
        self.assertEqual(tuple(output["main_logits"].shape), (6,))
        self.assertEqual(model.main_head[1].in_features, 8)
        labels = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32)
        roles = ["positive", "language_negative", "action_negative"] * 2
        losses = model.loss(output, labels, roles=roles)
        losses["loss"].backward()
        self.assertGreater(float(hidden.grad.abs().sum()), 0)
        self.assertGreater(float(actions.grad.abs().sum()), 0)
        self.assertGreater(float(model.vl_cross_attention.attention.in_proj_weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.action_cross_attention.attention.in_proj_weight.grad.abs().sum()), 0)

    def test_audit_heads_detach_encoder_inputs(self):
        model = self._model()
        hidden = torch.randn(2, 5, 8, requires_grad=True)
        actions = torch.randn(2, 4, 3, requires_grad=True)
        output = model(hidden, actions)
        probe_loss = output["vl_probe_logits"].sum() + output["action_probe_logits"].sum()
        probe_loss.backward()
        self.assertIsNone(hidden.grad)
        self.assertIsNone(actions.grad)
        self.assertIsNotNone(model.vl_audit_head[-1].weight.grad)
        self.assertIsNotNone(model.action_audit_head[-1].weight.grad)

    def test_rejects_wrong_action_shape_and_incomplete_triplets(self):
        model = self._model()
        with self.assertRaisesRegex(ValueError, "actions must have shape"):
            model(torch.randn(2, 5, 8), torch.randn(2, 3, 3))
        output = model(torch.randn(2, 5, 8), torch.randn(2, 4, 3))
        with self.assertRaisesRegex(ValueError, "complete"):
            model.loss(output, torch.tensor([1.0, 0.0]), roles=["positive", "language_negative"])


class ClassifierMetricsTest(unittest.TestCase):
    def test_binary_metrics(self):
        metrics = binary_classifier_metrics(torch.tensor([4.0, -3.0, 2.0, -2.0]), torch.tensor([1.0, 0.0, 1.0, 0.0]), prefix="val")
        self.assertEqual(metrics["val/accuracy"], 1.0)
        self.assertEqual(metrics["val/auroc"], 1.0)

    def test_synthetic_joint_signal_beats_single_modality_shortcuts(self):
        records = _strict_records(episodes=2)
        padded = records + [dict(records[0]), dict(records[-1])]
        deduplicated = deduplicate_classifier_records(padded)
        report = classifier_record_report(deduplicated, prefix="val", bootstrap_samples=20)
        self.assertEqual(report["protocol_version"], "2.0.0")
        self.assertEqual(report["counts"]["samples"], 26)
        self.assertEqual(report["metrics"]["val/language_paired_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["val/action_paired_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["val/vl_probe/action_paired_accuracy"], 0.5)
        self.assertEqual(report["metrics"]["val/action_probe/language_paired_accuracy"], 0.5)
        self.assertEqual(report["metrics"]["val/donor_action/auroc"], 0.5)
        for donor in ACTION_NEGATIVES:
            self.assertEqual(report["metrics"][f"val/action_paired_accuracy/{donor}"], 1.0)
            self.assertEqual(report["metrics"][f"val/action_negative_auroc/{donor}"], 1.0)
        self.assertIn("action_paired_accuracy", report["confidence_intervals_95"])

    def test_checkpoint_score_prioritizes_worst_pra_then_harmonic(self):
        def report(language, action, auroc=0.9, loss=0.2):
            return {"metrics": {
                "eval/language_paired_accuracy": language,
                "eval/action_paired_accuracy": action,
                "eval/auroc": auroc, "eval/loss": loss,
            }}
        self.assertGreater(
            classifier_checkpoint_score(report(0.75, 0.75)),
            classifier_checkpoint_score(report(0.95, 0.70, auroc=1.0, loss=0.0)),
        )
        self.assertGreater(
            classifier_checkpoint_score(report(0.75, 0.90)),
            classifier_checkpoint_score(report(0.75, 0.80, auroc=1.0, loss=0.0)),
        )

    def test_validation_threshold_freezes_for_test_and_checkpoint_reloads(self):
        records = _strict_records(1)
        validation = classifier_record_report(records, prefix="eval", bootstrap_samples=0)
        test = classifier_record_report(records, prefix="test", threshold=validation["threshold"], bootstrap_samples=0)
        self.assertEqual(test["threshold_source"], "frozen_validation")
        model = LanguageActionClassifier(8, 4, 3, hidden_dim=8, dropout=0.0, num_heads=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best-v2.pt"
            torch.save(model.state_dict(), path)
            restored = LanguageActionClassifier(8, 4, 3, hidden_dim=8, dropout=0.0, num_heads=2)
            restored.load_state_dict(torch.load(path, weights_only=True))
            for left, right in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(left, right)

    @unittest.skipUnless(torch.distributed.is_available(), "torch.distributed is unavailable")
    def test_two_rank_object_gather_matches_single_process(self):
        expected = classifier_record_report(_strict_records(1), prefix="val", bootstrap_samples=0)["metrics"]
        with tempfile.TemporaryDirectory() as directory:
            init_path, output_path = str(Path(directory) / "dist-init"), str(Path(directory) / "metrics.json")
            mp.spawn(_distributed_gather_worker, args=(init_path, output_path), nprocs=2, join=True)
            actual = json.loads(Path(output_path).read_text())
        self.assertEqual(actual, expected)

    def test_rejects_incomplete_pair(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            classifier_record_report(_strict_records(1)[:-1], prefix="val", bootstrap_samples=0)


if __name__ == "__main__":
    unittest.main()
