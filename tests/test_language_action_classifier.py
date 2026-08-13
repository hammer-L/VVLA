import unittest

import torch

from starVLA.model.modules.language_action_classifier import LanguageActionClassifier, masked_mean_pool
from starVLA.training.classifier_metrics import binary_classifier_metrics, example_group_metrics


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


if __name__ == "__main__":
    unittest.main()
