import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from deployment.model_server import policy_wrapper
from starVLA.model.classifier_checkpoint import (
    classifier_state_dict,
    load_classifier_checkpoint,
    save_classifier_checkpoint,
)
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.language_action_classifier import LanguageActionClassifier
from starVLA.model.framework.VLM4A.QwenGR00TClassifier import select_action_candidates
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


class ClassifierCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.classifier = LanguageActionClassifier(8, 4, 3, hidden_dim=8, dropout=0.0, num_heads=2)

    def test_compact_checkpoint_has_only_module_local_keys_and_restores_exactly(self):
        full = {
            "qwen_vl_interface.large.weight": torch.randn(2),
            **{f"language_classifier.{key}": value for key, value in self.classifier.state_dict().items()},
            "action_model.large.weight": torch.randn(2),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.pt"
            save_classifier_checkpoint(full, path, "pt")
            stored = torch.load(path, weights_only=True)
            self.assertEqual(set(stored), set(self.classifier.state_dict()))
            self.assertFalse(any("qwen" in key or "action_model" in key for key in stored))
            restored = LanguageActionClassifier(8, 4, 3, hidden_dim=8, dropout=0.0, num_heads=2)
            load_classifier_checkpoint(restored, path)
            for expected, actual in zip(self.classifier.parameters(), restored.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_loader_accepts_prefixed_and_legacy_full_checkpoints(self):
        local = self.classifier.state_dict()
        for prefix in ("language_classifier.", "module.language_classifier."):
            extracted = classifier_state_dict({
                "unrelated.weight": torch.randn(1),
                **{prefix + key: value for key, value in local.items()},
            })
            self.assertEqual(set(extracted), set(local))

    def test_unprefixed_checkpoint_with_foreign_keys_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save({"action_model.weight": torch.randn(1)}, path)
            with self.assertRaisesRegex(RuntimeError, "non-classifier keys"):
                load_classifier_checkpoint(self.classifier, path)

    def test_resume_search_recovers_classifier_step(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "steps_25_classifier.pt").write_bytes(b"checkpoint")
            fake_trainer = SimpleNamespace(accelerator=SimpleNamespace(print=lambda *args: None))
            path, step = TrainerUtils._get_latest_checkpoint(fake_trainer, directory)
        self.assertTrue(path.endswith("steps_25_classifier.pt"))
        self.assertEqual(step, 25)


class _IdentityActionEncoder(nn.Module):
    def forward(self, actions, timesteps):
        return actions


class _IdentityFlowModel(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class FlowGuidanceTest(unittest.TestCase):
    @staticmethod
    def _head():
        head = FlowmatchingActionHead.__new__(FlowmatchingActionHead)
        nn.Module.__init__(head)
        head.action_horizon = 2
        head.action_dim = 2
        head.num_inference_timesteps = 2
        head.num_timestep_buckets = 100
        head.config = SimpleNamespace(add_pos_embed=False)
        head.state_encoder = None
        head.action_encoder = _IdentityActionEncoder()
        head.future_tokens = nn.Embedding(1, 2)
        nn.init.zeros_(head.future_tokens.weight)
        head.model = _IdentityFlowModel()
        head.action_decoder = nn.Identity()
        return head

    def test_gradient_guidance_increases_logit_clamps_and_leaves_no_parameter_grads(self):
        head = self._head()
        vl_embs = torch.zeros(2, 1, 2)
        initial = torch.zeros(2, 2, 2)
        score = lambda actions: actions.sum(dim=(1, 2))
        baseline = head.predict_action(vl_embs, initial_actions=initial)
        guided, diagnostics = head.predict_action(
            vl_embs,
            initial_actions=initial,
            guidance_callback=score,
            guidance_scale=0.3,
            action_bounds=(torch.full((1, 1, 2), -0.1), torch.full((1, 1, 2), 0.1)),
            return_diagnostics=True,
        )
        self.assertTrue(torch.all(score(guided) > score(baseline)))
        self.assertLessEqual(float(guided.max()), 0.1 + 1e-6)
        self.assertGreaterEqual(float(guided.min()), -0.1 - 1e-6)
        self.assertEqual(len(diagnostics["guidance_logits_before_steps"]), 2)
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))

    def test_scale_zero_and_seeded_noise_match_baseline(self):
        head = self._head()
        vl_embs = torch.zeros(1, 1, 2)
        left_generator = torch.Generator().manual_seed(17)
        right_generator = torch.Generator().manual_seed(17)
        baseline = head.predict_action(vl_embs, generator=left_generator)
        scale_zero = head.predict_action(
            vl_embs,
            generator=right_generator,
            guidance_callback=lambda actions: actions.sum(dim=(1, 2)),
            guidance_scale=0.0,
        )
        torch.testing.assert_close(baseline, scale_zero)

    def test_rerank_selects_maximum_logit_for_every_batch_item(self):
        candidates = torch.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
        logits = torch.tensor([[0.1, 3.0, 2.0], [4.0, -1.0, 5.0]])
        selected, indices = select_action_candidates(candidates, logits, rerank=True)
        torch.testing.assert_close(indices, torch.tensor([1, 2]))
        torch.testing.assert_close(selected[0], candidates[0, 1])
        torch.testing.assert_close(selected[1], candidates[1, 2])

    def test_non_rerank_always_uses_first_candidate(self):
        candidates = torch.randn(2, 4, 2, 2)
        selected, indices = select_action_candidates(candidates, torch.randn(2, 4), rerank=False)
        torch.testing.assert_close(indices, torch.zeros(2, dtype=torch.long))
        torch.testing.assert_close(selected, candidates[:, 0])


class _ServerFramework(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_classifier = nn.Linear(2, 1)


class DualCheckpointServerTest(unittest.TestCase):
    def test_assisted_mode_loads_base_then_classifier_strictly(self):
        framework = _ServerFramework()
        cfg = {
            "framework": {"name": "QwenGR00T", "action_model": {"action_horizon": 8}},
            "datasets": {"vla_data": {"data_mix": "libero"}},
        }
        with (
            mock.patch.object(policy_wrapper.baseframework, "from_pretrained", return_value=framework) as base_load,
            mock.patch.object(policy_wrapper, "read_mode_config", return_value=(cfg, {"a": {}, "b": {}})),
            mock.patch.object(policy_wrapper, "load_classifier_checkpoint") as classifier_load,
        ):
            wrapper = policy_wrapper.PolicyServerWrapper(
                base_ckpt_path="/tmp/base.pt",
                classifier_ckpt_path="/tmp/classifier.pt",
                classifier_mode="rerank",
                num_candidates=4,
                device="cpu",
            )
        base_load.assert_called_once_with(
            "/tmp/base.pt",
            config_overrides=["framework.name=QwenGR00TClassifier"],
            allowed_missing_prefixes=("language_classifier.",),
        )
        classifier_load.assert_called_once_with(framework.language_classifier, "/tmp/classifier.pt")
        self.assertEqual(wrapper.metadata["base_ckpt_path"], "/tmp/base.pt")
        self.assertEqual(wrapper.metadata["classifier_ckpt_path"], "/tmp/classifier.pt")

    def test_assisted_mode_rejects_implicit_or_missing_checkpoint_pair(self):
        with self.assertRaisesRegex(ValueError, "both base_ckpt_path"):
            policy_wrapper.PolicyServerWrapper(
                ckpt_path="/tmp/base.pt", classifier_ckpt_path="/tmp/classifier.pt", classifier_mode="gradient"
            )


if __name__ == "__main__":
    unittest.main()
