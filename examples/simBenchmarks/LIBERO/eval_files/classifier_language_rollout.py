"""Build language-rollout manifests and aggregate the four inference modes.

This module intentionally does not import LIBERO, so manifest construction,
validation/test isolation, and aggregation can be checked on login/CI nodes.
The simulator runner writes records in the schema documented by
``new_rollout_result`` below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from starVLA.model.classifier_checkpoint import load_state_dict_file, save_classifier_checkpoint


MODES = ("off", "rerank", "gradient", "gradient_rerank")
POSITIVE_VARIANTS = ("canonical", "paraphrase_1", "paraphrase_2")
NEGATIVE_VARIANTS = (
    "wrong_object",
    "wrong_destination_or_relation",
    "wrong_verb_or_state",
    "wrong_order_direction_or_feasible_alternative",
)
SEARCH_SPACE = {
    "off": ((1, 0.0),),
    "rerank": tuple((k, 0.0) for k in (2, 4, 8)),
    "gradient": tuple((1, scale) for scale in (0.03, 0.1, 0.3)),
    "gradient_rerank": tuple((k, scale) for k in (2, 4, 8) for scale in (0.03, 0.1, 0.3)),
}


def normalize_instruction(text: str) -> str:
    """Normalize task text for exact canonical-task alignment checks."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def assert_manifest_task_alignment(
    entries: Iterable[dict[str, Any]],
    suite: str,
    task_index: int,
    environment_instruction: str,
) -> str:
    """Reject a manifest task ID that points at a different simulator goal."""
    task_rows = [
        row for row in entries
        if row.get("suite") == suite and int(row.get("task_index", -1)) == int(task_index)
    ]
    canonical_instructions = {
        str(row["canonical_instruction"])
        for row in task_rows
        if row.get("canonical_instruction")
    }
    canonical_instructions.update(
        str(row["instruction"])
        for row in task_rows
        if row.get("variant_id") == "canonical" and row.get("instruction")
    )
    normalized = {normalize_instruction(text): text for text in canonical_instructions}
    if not normalized:
        raise ValueError(
            f"manifest has no canonical instruction for {suite} task {task_index}; "
            "regenerate it with metadata-manifest"
        )
    if len(normalized) != 1:
        raise ValueError(
            f"manifest maps multiple canonical instructions to {suite} task {task_index}: "
            f"{sorted(canonical_instructions)}"
        )
    expected_normalized, expected = next(iter(normalized.items()))
    if expected_normalized != normalize_instruction(environment_instruction):
        raise ValueError(
            "rollout manifest task mismatch: "
            f"{suite} task {task_index} is {environment_instruction!r} in LIBERO, "
            f"but the manifest maps it to {expected!r}. Fix suite_map.json task_index_map "
            "and regenerate/remap the manifest before running rollouts."
        )
    return expected


def _mapped_suite_and_task(
    source_suite: str,
    source_task_index: int,
    suite_map: dict[str, Any] | None,
) -> tuple[str, int]:
    mapping = (suite_map or {}).get(source_suite, {})
    if isinstance(mapping, str):
        mapping = {"suite": mapping}
    if not isinstance(mapping, dict):
        raise ValueError(f"suite map entry for {source_suite!r} must be a string or object")
    rollout_suite = str(mapping.get("suite", source_suite))
    task_map = mapping.get("task_index_map")
    if task_map is None:
        return rollout_suite, int(source_task_index)
    if not isinstance(task_map, dict):
        raise ValueError(f"task_index_map for {source_suite!r} must be an object")
    mapped = task_map.get(str(source_task_index), task_map.get(source_task_index))
    if mapped is None:
        raise ValueError(
            f"task_index_map for {source_suite!r} lacks source task {source_task_index}"
        )
    return rollout_suite, int(mapped)


def remap_metadata_manifest(
    manifest: dict[str, Any], suite_map: dict[str, Any]
) -> dict[str, Any]:
    """Apply a corrected suite/task map to an already-generated metadata manifest."""
    output = dict(manifest)
    remapped_entries = []
    for row in manifest.get("entries", []):
        mapped = dict(row)
        source_suite = str(row.get("source_dataset", row["suite"]))
        source_task_index = int(row.get("source_task_index", row["task_index"]))
        mapped["suite"], mapped["task_index"] = _mapped_suite_and_task(
            source_suite, source_task_index, suite_map
        )
        remapped_entries.append(mapped)
    output["entries"] = remapped_entries
    output["task_mapping_applied"] = True
    return output


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stable_key(seed: int, *parts: Any) -> str:
    return hashlib.sha256("\x1f".join(map(str, (seed, *parts))).encode()).hexdigest()


def build_metadata_manifest(
    meta_dir: str | Path,
    split: str,
    seed: int = 42,
    tasks_per_suite: int | None = None,
    initial_state_indices: tuple[int, ...] = (0,),
    suite_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a split-isolated rollout manifest.

    Validation defaults to two seeded tasks per suite.  Test defaults to every
    task present in the metadata split.  Initial-state indices are explicit
    because demonstration episode IDs are not LIBERO simulator state IDs.
    """
    if split not in {"val", "test"}:
        raise ValueError("rollout manifest split must be 'val' or 'test'")
    meta_dir = Path(meta_dir)
    if tasks_per_suite is None:
        tasks_per_suite = 2 if split == "val" else -1
    if tasks_per_suite == 0 or tasks_per_suite < -1:
        raise ValueError("tasks_per_suite must be -1 (all) or a positive integer")
    if not initial_state_indices or any(index < 0 for index in initial_state_indices):
        raise ValueError("initial_state_indices must contain non-negative integers")
    groups = [row for row in _jsonl(meta_dir / "language_bank.jsonl") if row["split"] == split]
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for row in groups:
        source_suite = str(row["source_dataset"])
        source_task_index = int(row["task_index"])
        rollout_suite, rollout_task_index = _mapped_suite_and_task(
            source_suite, source_task_index, suite_map
        )
        mapped_row = dict(row)
        mapped_row["_rollout_suite"] = rollout_suite
        mapped_row["_rollout_task_index"] = rollout_task_index
        by_suite.setdefault(rollout_suite, []).append(mapped_row)
    entries = []
    variants_to_use = ("canonical", "paraphrase_1") if split == "val" else POSITIVE_VARIANTS + NEGATIVE_VARIANTS
    for suite, rows in sorted(by_suite.items()):
        rows.sort(key=lambda row: _stable_key(seed, suite, row["_rollout_task_index"]))
        selected_rows = rows if tasks_per_suite == -1 else rows[:tasks_per_suite]
        for row in selected_rows:
            variants = {variant["variant_id"]: variant for variant in row["variants"]}
            for initial_state_index in initial_state_indices:
                for variant_id in variants_to_use:
                    if variant_id not in variants:
                        raise ValueError(f"language group {row.get('language_group_id')} lacks {variant_id}")
                    entries.append({
                        "suite": suite,
                        "task_index": int(row["_rollout_task_index"]),
                        "source_dataset": row["source_dataset"],
                        "source_task_index": int(row["task_index"]),
                        "initial_state_index": int(initial_state_index),
                        "variant_id": variant_id,
                        "instruction": variants[variant_id]["text"],
                        "canonical_instruction": variants["canonical"]["text"],
                        "positive_instruction": variant_id in POSITIVE_VARIANTS,
                        "language_group_id": row.get("language_group_id"),
                    })
    return {
        "schema_version": 1,
        "source": "metadata",
        "split": split,
        "seed": seed,
        "tasks_per_suite": tasks_per_suite,
        "initial_state_indices": list(initial_state_indices),
        "entries": entries,
    }


def filter_libero_plus_language(tasks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only LIBERO-plus rows whose declared category is Language."""
    filtered = []
    for row in tasks:
        category = row.get("category", row.get("type", row.get("dimension")))
        if str(category).strip().casefold() == "language":
            filtered.append(dict(row))
    return filtered


def new_rollout_result(
    *,
    phase: str,
    mode: str,
    base_checkpoint: str,
    classifier_checkpoint: str | None,
    seed: int,
    num_candidates: int,
    guidance_scale: float,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    if phase not in {"validation", "libero_original", "libero_plus_language", "metadata_test"}:
        raise ValueError(f"unknown rollout phase {phase!r}")
    if mode not in MODES:
        raise ValueError(f"unknown classifier mode {mode!r}")
    successes = [bool(row["success"]) for row in episodes if row.get("positive_instruction", True)]
    latencies = [float(row["latency_ms"]) for row in episodes if row.get("latency_ms") is not None]
    return {
        "schema_version": 1,
        "phase": phase,
        "mode": mode,
        "checkpoint": {"base": base_checkpoint, "classifier": classifier_checkpoint},
        "seed": int(seed),
        "hyperparameters": {"num_candidates": int(num_candidates), "guidance_scale": float(guidance_scale)},
        "episodes": episodes,
        "summary": {
            "successes": sum(successes),
            "episodes": len(successes),
            "success_rate": sum(successes) / max(len(successes), 1),
            "mean_latency_ms": mean(latencies) if latencies else None,
        },
    }


def select_validation_hyperparameters(results: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Freeze one configuration per mode using validation results only."""
    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for result in results:
        if result.get("phase") != "validation":
            raise ValueError("hyperparameters may only be selected from validation rollouts")
        mode = result["mode"]
        hp = result["hyperparameters"]
        pair = (int(hp["num_candidates"]), float(hp["guidance_scale"]))
        if pair not in SEARCH_SPACE[mode]:
            raise ValueError(f"configuration {pair} is outside the declared search space for {mode}")
        by_mode[mode].append(result)
    selected = {}
    for mode, candidates in by_mode.items():
        if not candidates:
            raise ValueError(f"missing validation result for mode {mode}")
        selected_result = min(candidates, key=lambda result: (
            -float(result["summary"]["success_rate"]),
            float(result["summary"].get("mean_latency_ms") or float("inf")),
            int(result["hyperparameters"]["num_candidates"]),
            float(result["hyperparameters"]["guidance_scale"]),
        ))
        selected[mode] = dict(selected_result["hyperparameters"])
    return selected


def merge_rollout_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge suite/variant shards belonging to one mode and configuration."""
    results = list(results)
    if not results:
        raise ValueError("at least one rollout result is required")
    first = results[0]
    invariant_keys = ("phase", "mode", "checkpoint", "seed", "hyperparameters")
    for result in results[1:]:
        changed = [key for key in invariant_keys if result.get(key) != first.get(key)]
        if changed:
            raise ValueError(f"cannot merge rollout shards with different {changed}")
    hp = first["hyperparameters"]
    return new_rollout_result(
        phase=first["phase"],
        mode=first["mode"],
        base_checkpoint=first["checkpoint"]["base"],
        classifier_checkpoint=first["checkpoint"].get("classifier"),
        seed=first["seed"],
        num_candidates=hp["num_candidates"],
        guidance_scale=hp["guidance_scale"],
        episodes=[episode for result in results for episode in result["episodes"]],
    )


def aggregate_test_results(
    results: Iterable[dict[str, Any]], frozen_hyperparameters: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate final tests and reject any post-hoc test hyperparameter change."""
    results = list(results)
    results_by_mode = {result["mode"]: result for result in results}
    if len(results) != len(MODES) or set(results_by_mode) != set(MODES):
        raise ValueError(f"final report requires exactly four modes: {MODES}")
    phases = {result.get("phase") for result in results}
    if len(phases) != 1:
        raise ValueError(f"final report cannot mix rollout phases: {sorted(phases)}")
    baseline = results_by_mode["off"]
    baseline_success = {
        (row["episode_id"], row.get("variant_id", "canonical")): bool(row["success"])
        for row in baseline["episodes"] if row.get("positive_instruction", True)
    }
    report_modes = {}
    for mode in MODES:
        result = results_by_mode[mode]
        if result.get("phase") == "validation":
            raise ValueError("validation rollouts cannot appear in the final test report")
        if result["hyperparameters"] != frozen_hyperparameters[mode]:
            raise ValueError(f"test hyperparameters for {mode} differ from the frozen validation choice")
        paired = []
        for episode in result["episodes"]:
            pair_key = (episode["episode_id"], episode.get("variant_id", "canonical"))
            if episode.get("positive_instruction", True) and pair_key in baseline_success:
                paired.append(int(bool(episode["success"])) - int(baseline_success[pair_key]))
        variants: dict[str, list[dict[str, Any]]] = {}
        for episode in result["episodes"]:
            variants.setdefault(episode.get("variant_id", "canonical"), []).append(episode)
        positive_variant_success = {
            variant: sum(bool(row["success"]) for row in rows) / max(len(rows), 1)
            for variant, rows in variants.items() if variant in POSITIVE_VARIANTS
        }
        canonical_by_pair = {
            row.get("pair_id", row["episode_id"]): row for row in variants.get("canonical", [])
        }
        negative_diagnostics = {}
        for variant in NEGATIVE_VARIANTS:
            suppression, divergence, scores = [], [], []
            for row in variants.get(variant, []):
                canonical = canonical_by_pair.get(row.get("pair_id", row["episode_id"]))
                if canonical is not None:
                    suppression.append(int(bool(canonical["success"])) - int(bool(row["success"])))
                    if "action_trajectory" in canonical and "action_trajectory" in row:
                        left = np.asarray(canonical["action_trajectory"], dtype=np.float32)
                        right = np.asarray(row["action_trajectory"], dtype=np.float32)
                        length = min(len(left), len(right))
                        if length:
                            divergence.append(float(np.sqrt(np.mean((left[:length] - right[:length]) ** 2))))
                if row.get("classifier_score") is not None:
                    scores.append(float(row["classifier_score"]))
            negative_diagnostics[variant] = {
                "canonical_goal_suppression": mean(suppression) if suppression else None,
                "action_trajectory_rms_divergence": mean(divergence) if divergence else None,
                "mean_classifier_score": mean(scores) if scores else None,
                "episodes": len(variants.get(variant, [])),
                "interpretation": "diagnostic_only_not_alternative_task_success",
            }
        report_modes[mode] = {
            **result["summary"],
            "paired_success_delta": mean(paired) if paired else None,
            "positive_instruction_success_rate": positive_variant_success,
            "wrong_instruction_diagnostics": negative_diagnostics,
            "checkpoint": result["checkpoint"],
            "hyperparameters": result["hyperparameters"],
        }
    return {"schema_version": 1, "frozen_hyperparameters": frozen_hyperparameters, "modes": report_modes}


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("metadata-manifest")
    manifest.add_argument("--meta-dir", required=True)
    manifest.add_argument("--split", choices=("val", "test"), required=True)
    manifest.add_argument("--seed", type=int, default=42)
    manifest.add_argument(
        "--tasks-per-suite", type=int, default=None,
        help="Default: 2 for val, all (-1) for test.",
    )
    manifest.add_argument(
        "--initial-state-indices", default="0",
        help="Comma-separated LIBERO simulator initial-state indices, e.g. 0,1,2.",
    )
    manifest.add_argument(
        "--suite-map", default=None,
        help="Optional JSON mapping source_dataset/task indices to LIBERO suite/task indices.",
    )
    remap = subparsers.add_parser("remap-manifest")
    remap.add_argument("--manifest", required=True)
    remap.add_argument("--suite-map", required=True)
    plus = subparsers.add_parser("filter-libero-plus-language")
    plus.add_argument("--tasks-json", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("results", nargs="+")
    merge = subparsers.add_parser("merge")
    merge.add_argument("results", nargs="+")
    extract = subparsers.add_parser("extract-classifier")
    extract.add_argument("--checkpoint", required=True, help="Legacy complete model checkpoint.")
    extract.add_argument("--save-format", choices=("pt", "safetensors"), default="pt")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--frozen", required=True)
    aggregate.add_argument("results", nargs=4)
    for command in (manifest, remap, plus, select, merge, extract, aggregate):
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "metadata-manifest":
        try:
            state_indices = tuple(int(item) for item in args.initial_state_indices.split(",") if item != "")
        except ValueError as exc:
            parser.error(f"--initial-state-indices must be comma-separated integers: {exc}")
        output = build_metadata_manifest(
            args.meta_dir,
            args.split,
            args.seed,
            args.tasks_per_suite,
            state_indices,
            _load_json(args.suite_map) if args.suite_map else None,
        )
    elif args.command == "remap-manifest":
        output = remap_metadata_manifest(
            _load_json(args.manifest), _load_json(args.suite_map)
        )
    elif args.command == "filter-libero-plus-language":
        payload = _load_json(args.tasks_json)
        output = filter_libero_plus_language(payload.get("tasks", payload))
    elif args.command == "select":
        output = select_validation_hyperparameters(_load_json(path) for path in args.results)
    elif args.command == "merge":
        output = merge_rollout_results(_load_json(path) for path in args.results)
    elif args.command == "extract-classifier":
        state = load_state_dict_file(args.checkpoint)
        output_path = save_classifier_checkpoint(state, args.output, args.save_format)
        compact = load_state_dict_file(output_path)
        print(
            f"Extracted {len(compact)} classifier tensors to {output_path} "
            f"({output_path.stat().st_size / 1024**2:.2f} MiB)"
        )
        return
    else:
        output = aggregate_test_results(
            (_load_json(path) for path in args.results), _load_json(args.frozen)
        )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
