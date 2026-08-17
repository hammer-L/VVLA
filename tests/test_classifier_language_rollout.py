import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py"
)
SPEC = importlib.util.spec_from_file_location("classifier_language_rollout", MODULE_PATH)
rollout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rollout)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class LanguageManifestTest(unittest.TestCase):
    def test_metadata_manifest_is_seeded_split_isolated_and_uses_two_tasks(self):
        variants = [
            {"variant_id": variant, "text": f"text {variant}"}
            for variant in rollout.POSITIVE_VARIANTS + rollout.NEGATIVE_VARIANTS
        ]
        rows = [
            {"source_dataset": suite, "task_index": task, "split": split,
             "language_group_id": f"{suite}:{task}:{split}", "variants": variants}
            for suite in ("libero_goal", "libero_object")
            for task in range(4)
            for split in ("val", "test")
        ]
        with tempfile.TemporaryDirectory() as directory:
            _write_jsonl(Path(directory) / "language_bank.jsonl", rows)
            val = rollout.build_metadata_manifest(directory, "val", seed=42)
            test = rollout.build_metadata_manifest(directory, "test", seed=42)
        self.assertEqual(len(val["entries"]), 2 * 2 * 2)
        self.assertEqual({row["variant_id"] for row in val["entries"]}, {"canonical", "paraphrase_1"})
        self.assertTrue(all(row["initial_state_index"] == 0 for row in val["entries"]))
        self.assertEqual(test["split"], "test")
        self.assertEqual(test["tasks_per_suite"], -1)
        self.assertEqual({row["task_index"] for row in test["entries"]}, set(range(4)))
        self.assertTrue(all(row["variant_id"] in rollout.POSITIVE_VARIANTS + rollout.NEGATIVE_VARIANTS for row in test["entries"]))

    def test_manifest_expands_explicit_initial_states(self):
        variants = [
            {"variant_id": variant, "text": variant}
            for variant in rollout.POSITIVE_VARIANTS + rollout.NEGATIVE_VARIANTS
        ]
        with tempfile.TemporaryDirectory() as directory:
            _write_jsonl(Path(directory) / "language_bank.jsonl", [{
                "source_dataset": "libero_goal", "task_index": 0, "split": "test",
                "language_group_id": "group", "variants": variants,
            }])
            manifest = rollout.build_metadata_manifest(
                directory, "test", initial_state_indices=(0, 2, 4)
            )
        self.assertEqual({row["initial_state_index"] for row in manifest["entries"]}, {0, 2, 4})

    def test_manifest_maps_dataset_and_task_ids_to_libero(self):
        variants = [
            {"variant_id": variant, "text": variant}
            for variant in rollout.POSITIVE_VARIANTS + rollout.NEGATIVE_VARIANTS
        ]
        with tempfile.TemporaryDirectory() as directory:
            _write_jsonl(Path(directory) / "language_bank.jsonl", [{
                "source_dataset": "custom_goal", "task_index": 11, "split": "val",
                "language_group_id": "group", "variants": variants,
            }])
            manifest = rollout.build_metadata_manifest(
                directory,
                "val",
                suite_map={"custom_goal": {"suite": "libero_goal", "task_index_map": {"11": 3}}},
            )
        self.assertEqual({row["suite"] for row in manifest["entries"]}, {"libero_goal"})
        self.assertEqual({row["task_index"] for row in manifest["entries"]}, {3})
        self.assertEqual({row["source_task_index"] for row in manifest["entries"]}, {11})

    def test_libero_plus_filter_is_strictly_language(self):
        tasks = [{"id": 1, "category": "Language"}, {"id": 2, "category": "Spatial"}]
        self.assertEqual(rollout.filter_libero_plus_language(tasks), [tasks[0]])


class ResultSelectionTest(unittest.TestCase):
    @staticmethod
    def _result(mode, k, scale, success, latency, phase="validation"):
        episodes = [{"episode_id": f"e{i}", "success": i < success, "latency_ms": latency} for i in range(2)]
        return rollout.new_rollout_result(
            phase=phase,
            mode=mode,
            base_checkpoint="base.pt",
            classifier_checkpoint=None if mode == "off" else "classifier.pt",
            seed=42,
            num_candidates=k,
            guidance_scale=scale,
            episodes=episodes,
        )

    def test_selection_uses_success_then_latency_then_k_then_scale(self):
        results = [self._result("off", 1, 0.0, 1, 4)]
        results += [self._result("rerank", 2, 0.0, 1, 9), self._result("rerank", 4, 0.0, 1, 8)]
        results += [self._result("gradient", 1, 0.03, 1, 7), self._result("gradient", 1, 0.1, 2, 12)]
        results += [
            self._result("gradient_rerank", 2, 0.03, 1, 8),
            self._result("gradient_rerank", 4, 0.03, 1, 8),
        ]
        selected = rollout.select_validation_hyperparameters(results)
        self.assertEqual(selected["rerank"]["num_candidates"], 4)
        self.assertEqual(selected["gradient"]["guidance_scale"], 0.1)
        self.assertEqual(selected["gradient_rerank"]["num_candidates"], 2)

    def test_test_results_cannot_change_frozen_parameters(self):
        frozen = {
            "off": {"num_candidates": 1, "guidance_scale": 0.0},
            "rerank": {"num_candidates": 2, "guidance_scale": 0.0},
            "gradient": {"num_candidates": 1, "guidance_scale": 0.1},
            "gradient_rerank": {"num_candidates": 2, "guidance_scale": 0.1},
        }
        results = [
            self._result(mode, hp["num_candidates"], hp["guidance_scale"], 1, 5, "metadata_test")
            for mode, hp in frozen.items()
        ]
        results[-1]["hyperparameters"]["guidance_scale"] = 0.3
        with self.assertRaisesRegex(ValueError, "differ from the frozen"):
            rollout.aggregate_test_results(results, frozen)

    def test_merge_preserves_provenance_and_recomputes_summary(self):
        left = self._result("off", 1, 0.0, 1, 5)
        right = self._result("off", 1, 0.0, 2, 7)
        for index, episode in enumerate(right["episodes"]):
            episode["episode_id"] = f"right-{index}"
        merged = rollout.merge_rollout_results([left, right])
        self.assertEqual(len(merged["episodes"]), 4)
        self.assertEqual(merged["summary"]["successes"], 3)
        self.assertEqual(merged["checkpoint"], left["checkpoint"])


if __name__ == "__main__":
    unittest.main()
