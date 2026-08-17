# Classifier-guided language rollout

The base and classifier weights are separate inputs. `BASE_CKPT` must be the
complete QwenGR00T checkpoint used to train the compact classifier in
`CLASSIFIER_CKPT`.

Start one policy server at a time:

```bash
BASE_CKPT=/path/base.pt CLASSIFIER_CKPT=/path/best_classifier.pt \
  ./examples/simBenchmarks/LIBERO/eval_files/run_classifier_policy_server.sh off

BASE_CKPT=/path/base.pt CLASSIFIER_CKPT=/path/best_classifier.pt NUM_CANDIDATES=4 \
  ./examples/simBenchmarks/LIBERO/eval_files/run_classifier_policy_server.sh rerank

BASE_CKPT=/path/base.pt CLASSIFIER_CKPT=/path/best_classifier.pt GUIDANCE_SCALE=0.1 \
  ./examples/simBenchmarks/LIBERO/eval_files/run_classifier_policy_server.sh gradient

BASE_CKPT=/path/base.pt CLASSIFIER_CKPT=/path/best_classifier.pt \
  NUM_CANDIDATES=4 GUIDANCE_SCALE=0.1 \
  ./examples/simBenchmarks/LIBERO/eval_files/run_classifier_policy_server.sh gradient_rerank
```

Build the deterministic validation manifest (seed 42, two tasks per suite,
canonical plus `paraphrase_1`, initial state zero):

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  metadata-manifest --meta-dir /path/meta --split val --seed 42 \
  --output results/validation_manifest.json
```

Run `eval_libero.py` once per mode/suite/variant with
`--args.rollout-manifest`, `--args.instruction-variant`,
`--args.rollout-phase validation`, and `--args.result-json`. Search only the
declared grids: K in `{2,4,8}` and scale in `{0.03,0.1,0.3}`. Freeze choices:

Merge suite/variant shards for the same mode and hyperparameters first:

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  merge results/off_val_shards/*.json --output results/off_val.json
```

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  select results/val_*.json --output results/frozen_hyperparameters.json
```

Use those exact values for LIBERO-plus Language and metadata test. Do not tune
on either test. LIBERO-plus task exports can be filtered with the
`filter-libero-plus-language` subcommand. Build the metadata test manifest by
changing `--split test`; it includes canonical, both paraphrases, and all four
wrong-instruction variants. Finally aggregate exactly one result per mode:

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  aggregate --frozen results/frozen_hyperparameters.json \
  results/off.json results/rerank.json results/gradient.json \
  results/gradient_rerank.json --output results/final_report.json
```

The report contains per-mode task success, paired deltas from `off`, latency,
checkpoint provenance, and classifier diagnostics. Wrong instructions are
reported only as canonical-goal suppression, trajectory divergence, and
classifier score; they are not treated as alternative-task success rates.
