# Qwen2.5-VL-GR00T-LIBERO-4in1 推理操作文档

本文档用于完成两个实验：

1. 用发布的 Qwen2.5 StarVLA-GR00T 权重在四个原始 LIBERO suite 上运行
  canonical 指令；
2. 保持模型、任务和初始状态不变，在语言扩充 benchmark 的 `test` split 上
  运行 canonical、paraphrase 和错误指令。

这一步只建立原始 VLA baseline（`classifier_mode=off`），不需要 classifier
checkpoint。之后要验证 classifier 是否提升成功率，应在相同 rollout 清单上再运行
`rerank`、`gradient` 和 `gradient_rerank`。

## 1. 已下载权重的结论

本仓库中的实际 run 目录为：

```text
playground/Pretrained_models/StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1/
├── checkpoints/steps_30000_pytorch_model.pt
├── config.yaml
├── config.json
├── dataset_statistics.json
└── summary.jsonl
```

`steps_30000_pytorch_model.pt` 是完整 VLA checkpoint，不是 classifier-only
checkpoint。它包含 Qwen2.5-VL 和已经训练过的 GR00T/DiT action head。配置中的关键
契约是：

- framework：`QwenGR00T`；
- base VLM：Qwen2.5-VL-3B-Instruct；
- action head：`DiT-B`；
- action horizon：8，action dimension：7；
- 推理 flow-matching steps：4；
- 输入图像顺序：primary camera、wrist camera；
- 不输入 state；
- 图像送给客户端后 resize 到 224 × 224；
- 动作使用该 run 自己的 `dataset_statistics.json` 反归一化。

仓库 README 给出的该发布 checkpoint 参考结果为 Spatial 97.8、Object 98.2、
Goal 94.6、LIBERO-10 90.8、平均 95.4。每个 suite 是 10 个任务 × 50 个初始
状态。冒烟测试的样本很少，不能拿来与该数字比较。

## 2. 补齐两个运行依赖

### 2.1 下载 Qwen2.5-VL-3B 基座

虽然完整 VLA checkpoint 中已有训练后的 VLM tensors，当前模型构造过程仍会先从
`base_vlm` 路径创建 Qwen 模型并读取 processor/tokenizer，再加载 VLA checkpoint。
因此下面的目录必须存在：

```text
playground/Pretrained_models/Qwen2.5-VL-3B-Instruct/
```

在可联网的 starVLA 环境执行：

```bash
cd /home/happigo/vla_ws/starVLA

hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
```

如果机器上的命令仍叫 `huggingface-cli`，等价命令是：

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
```

如果基座已经放在其他位置，不必复制。启动服务时增加：

```text
--config_override framework.qwenvl.base_vlm=/绝对路径/Qwen2.5-VL-3B-Instruct
```

检查至少包含配置、processor/tokenizer 和模型权重：

```bash
test -f playground/Pretrained_models/Qwen2.5-VL-3B-Instruct/config.json
test -f playground/Pretrained_models/Qwen2.5-VL-3B-Instruct/preprocessor_config.json
find playground/Pretrained_models/Qwen2.5-VL-3B-Instruct -maxdepth 1 \
  \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print
```

### 2.2 安装 LIBERO 仿真环境

推荐服务端和仿真端使用两个独立 conda 环境。以下安装脚本会安装 MuJoCo 3.2.3、
LIBERO 及评测依赖：

```bash
cd /home/happigo/vla_ws/starVLA

LIBERO_CONDA_ENV=libero \
LIBERO_PARENT_DIR=/root/gpufree-data/liumingyu/LIBERO \
bash examples/simBenchmarks/LIBERO/eval_files/install_libero.sh
```

安装后假设源码位于 `/path/to/install/LIBERO`。验证：

```bash
conda run -n libero python -c \
  "from libero.libero import benchmark; import mujoco; print(mujoco.__version__)"
```

## 3. 启动发布模型 baseline 服务

终端 A 使用能够加载 starVLA 和 GPU 的环境：

```bash
cd ~/starVLA
conda activate starVLA

export BASE_CKPT="$PWD/playground/Pretrained_models/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt"
export PORT=10093
export CUDA_VISIBLE_DEVICES=0

python deployment/model_server/server_policy.py \
  --ckpt_path "$BASE_CKPT" \
  --classifier_mode off \
  --port "$PORT" \
  --use_bf16
```

如基座不在配置记录的位置，完整命令为：

```bash
python deployment/model_server/server_policy.py \
  --ckpt_path "$BASE_CKPT" \
  --classifier_mode off \
  --config_override framework.qwenvl.base_vlm=/绝对路径/Qwen2.5-VL-3B-Instruct \
  --port "$PORT" \
  --use_bf16
```

服务成功启动后保持终端 A 不退出。日志应显示 `action_chunk_size: 8`，且
`classifier_mode` 为 `off`。没有安装 `flash-attn` 时会自动回退到 SDPA，这不是
错误，只会影响速度和显存。

## 4. 原始 LIBERO 冒烟测试

先只跑一个 suite 中的一个任务、一个初始状态。终端 B：

```bash
cd ~/starVLA
conda activate libero
export LIBERO_HOME=/root/gpufree-data/liumingyu/LIBERO
export LIBERO_CONFIG_PATH="$LIBERO_HOME/libero"
export PYTHONPATH="$LIBERO_HOME:$PWD:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PORT=10093
export RESULT_ROOT="$PWD/results/qwen25_groot_off"
mkdir -p "$RESULT_ROOT/original_smoke/videos"

python examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.host 127.0.0.1 \
  --args.port "$PORT" \
  --args.task-suite-name libero_goal \
  --args.task-ids 8 \
  --args.max-tasks 1 \
  --args.num-trials-per-task 1 \
  --args.image-views primary,wrist \
  --args.rollout-phase libero_original \
  --args.result-json "$RESULT_ROOT/original_smoke/libero_goal.json" \
  --args.video-out-path "$RESULT_ROOT/original_smoke/videos/libero_goal"
```

成功完成的标准不是这一条 rollout 必须成功，而是：仿真能初始化、服务能连续返回
8 × 7 action chunk、视频和 JSON 都能写出，并且没有图像数、动作维数或动作
反归一化错误。

## 5. 原始 LIBERO 正式评测

正式复现运行四个 suite，每个 suite 10 个任务，每任务 50 个初始状态：

```bash
mkdir -p "$RESULT_ROOT/original/videos"

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  python examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
    --args.host 127.0.0.1 \
    --args.port "$PORT" \
    --args.task-suite-name "$SUITE" \
    --args.max-tasks -1 \
    --args.num-trials-per-task 50 \
    --args.seed 42 \
    --args.image-views primary,wrist \
    --args.rollout-phase libero_original \
    --args.result-json "$RESULT_ROOT/original/${SUITE}.json" \
    --args.video-out-path "$RESULT_ROOT/original/videos/${SUITE}"
done
```

合并四个 suite：

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  merge \
  "$RESULT_ROOT/original/libero_spatial.json" \
  "$RESULT_ROOT/original/libero_object.json" \
  "$RESULT_ROOT/original/libero_goal.json" \
  "$RESULT_ROOT/original/libero_10.json" \
  --output "$RESULT_ROOT/original_all.json"
```

查看成功率和延迟：

```bash
python - "$RESULT_ROOT/original_all.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(result["summary"], indent=2))
for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    rows = [row for row in result["episodes"] if row["suite"] == suite]
    print(suite, sum(row["success"] for row in rows) / len(rows), f"n={len(rows)}")
PY
```

## 6. 从扩充 benchmark 生成 test rollout 清单

以下流程适用于“原 LIBERO 场景和目标不变，只扩充语言指令”的 benchmark。如果扩充
数据新增了物体布局、任务 BDDL 或 simulator scene，不能只换指令文本；需要先把新任务
注册为 LIBERO benchmark，并提供对应 initial states。

假设 metadata 目录含有：

```text
benchmark.json
splits.jsonl
language_bank.jsonl
anchors.jsonl
```

设置路径：

```bash
export META_DIR=/root/gpufree-data/liumingyu/datasets/libero/benchmark/meta
export EXPANDED_DIR="$RESULT_ROOT/expanded_test"
mkdir -p "$EXPANDED_DIR/videos"
```

如果 `language_bank.jsonl` 的 `source_dataset` 是 IPEC LeRobot 长名称，必须同时
映射 suite 和 task ID。LeRobot 的 task 顺序与 LIBERO simulator 不同，不能只改
suite 名。仓库已经提供完整映射：

```bash
cp examples/simBenchmarks/LIBERO/eval_files/ipec_lerobot_suite_map.json \
  "$EXPANDED_DIR/suite_map.json"
```

如果 `source_dataset` 已经是 `libero_spatial`、`libero_object`、`libero_goal`、
`libero_10`，后面的 manifest 命令删除 `--suite-map` 参数即可。

为了与原始正式评测使用相同的 50 个 initial states，生成 test manifest：

```bash
export INITIAL_STATES="$(seq -s, 0 49)"

python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  metadata-manifest \
  --meta-dir "$META_DIR" \
  --split test \
  --seed 42 \
  --tasks-per-suite -1 \
  --initial-state-indices "$INITIAL_STATES" \
  --suite-map "$EXPANDED_DIR/suite_map.json" \
  --output "$EXPANDED_DIR/test_manifest.json"
```

生成后先检查映射范围。四个标准 suite 的 `task_index` 应在 0–9，initial state 应在
0–49：

```bash
python - "$EXPANDED_DIR/test_manifest.json" <<'PY'
import collections
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
counts = collections.Counter((r["suite"], r["variant_id"]) for r in manifest["entries"])
print("entries:", len(manifest["entries"]))
for key, value in sorted(counts.items()):
    print(key, value)
assert all(0 <= int(r["task_index"]) < 10 for r in manifest["entries"])
assert all(0 <= int(r["initial_state_index"]) < 50 for r in manifest["entries"])
PY
```

若 metadata 里的 task 编号与原 LIBERO 编号不一致，必须先在 `suite_map.json` 中增加
完整的 `task_index_map`。evaluator 默认会把 manifest 的 canonical 指令与 LIBERO
环境任务核对；不一致会在 rollout 前报错，不能关闭该检查来掩盖映射问题。

已经生成过错误 manifest 时，可以直接修复，无需重新生成语言 metadata：

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  remap-manifest \
  --manifest "$EXPANDED_DIR/test_manifest.json" \
  --suite-map "$EXPANDED_DIR/suite_map.json" \
  --output "$EXPANDED_DIR/test_manifest.corrected.json"
```

## 7. 扩充 benchmark 冒烟测试和正式评测

先选一个正指令变体做冒烟测试：

```bash
python examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.host 127.0.0.1 \
  --args.port "$PORT" \
  --args.task-suite-name libero_goal \
  --args.max-tasks 1 \
  --args.max-episodes-per-task 1 \
  --args.rollout-manifest "$EXPANDED_DIR/test_manifest.json" \
  --args.instruction-variant paraphrase_1 \
  --args.image-views primary,wrist \
  --args.rollout-phase metadata_test \
  --args.result-json "$EXPANDED_DIR/smoke_paraphrase_1.json" \
  --args.video-out-path "$EXPANDED_DIR/videos/smoke_paraphrase_1"
```

这里显式选择 simulator task 8（`put the bowl on the plate`），用于复查之前的错配；
`--max-episodes-per-task 1` 保证只跑一个 initial state。

正式运行七个语言变体：

```bash
VARIANTS=(
  canonical
  paraphrase_1
  paraphrase_2
  wrong_object
  wrong_destination_or_relation
  wrong_verb_or_state
  wrong_order_direction_or_feasible_alternative
)

for VARIANT in "${VARIANTS[@]}"; do
  for SUITE in libero_spatial libero_object libero_goal libero_10; do
    python examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
      --args.host 127.0.0.1 \
      --args.port "$PORT" \
      --args.task-suite-name "$SUITE" \
      --args.max-tasks -1 \
      --args.seed 42 \
      --args.rollout-manifest "$EXPANDED_DIR/test_manifest.json" \
      --args.instruction-variant "$VARIANT" \
      --args.image-views primary,wrist \
      --args.rollout-phase metadata_test \
      --args.result-json "$EXPANDED_DIR/${SUITE}_${VARIANT}.json" \
      --args.video-out-path "$EXPANDED_DIR/videos/${SUITE}_${VARIANT}"
  done
done
```

合并所有 shard：

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  merge "$EXPANDED_DIR"/libero_*.json \
  --output "$EXPANDED_DIR/all_variants.json"
```

成功率只解释三种正指令：`canonical`、`paraphrase_1`、`paraphrase_2`。四种错误
指令的环境目标仍是 canonical 目标，因此其 `success` 不能解释为“错误指令任务成功
率”；它们只应用于 canonical-goal suppression 和动作轨迹分歧诊断。

## 8. 在自建 benchmark 上使用发布权重

先判断自建 benchmark 属于哪一类：

1. **只扩充语言**：场景、BDDL goal 和 initial states 都来自原 LIBERO。直接使用本节
   的 manifest 流程；每条语言组的 `canonical` 必须是对应 LIBERO 环境的原始指令，
   paraphrase 放在其他 variant 中。
2. **新增场景、布局或 BDDL goal**：先把任务注册进 LIBERO benchmark，并让
   `task_suite.get_task()`、`get_task_init_states()` 和环境 success predicate 都可用；
   仅在 JSON 中增加一句指令不会创建新的仿真任务。
3. **更换机器人、相机或动作空间**：发布权重不能直接公平评测。必须提供 observation/
   action adapter，通常还需要在同一数据契约上微调。

发布 Qwen2.5-GR00T 权重的评测输入契约必须保持：两路图像按 primary、wrist 顺序，
客户端 resize 到 224 × 224，不输入 state，动作为 7 维 LIBERO delta action，并使用
checkpoint 同目录的 `dataset_statistics.json` 反归一化。先运行一任务一 initial state
的 smoke，再扩大到完整 test split。

建议保留两份 manifest：

- `smoke_manifest.json`：每任务只含 initial state 0，或运行时设置
  `--args.max-episodes-per-task 1`；
- `test_manifest.json`：正式实验固定 0–49，并对所有对比方法复用完全相同的 manifest、
  seed、checkpoint 和输入设置。

官方权重只作为 zero-shot/base VLA baseline。如果自建任务与训练分布差异明显，低成功率
不能单独说明语言模块失效；需要同时报告原始 LIBERO 复现结果，区分部署错误与任务 OOD。

## 9. 下一步：验证 classifier 是否提升

先完成上述 `off` baseline，再用 benchmark 的 `train` split 在这个完全相同的
Qwen2.5-GR00T base checkpoint 上训练 classifier。不要直接把此前基于 Qwen3-VL-4B
训练的 classifier 用于最终对比：即使隐藏维度碰巧一致，两个 VLM 的特征分布也不同，
结果不能归因于 classifier 引导。

训练新 classifier 后：

1. 在 `val` rollout 上选择 K 和 guidance scale；
2. 冻结每种模式的超参数；
3. 在本文件生成的同一 `test_manifest.json` 上运行四种模式；
4. 使用相同 seed，使各模式共享首个初始噪声；
5. 比较配对成功率差值和推理延迟。

完整的 classifier 训练、val 选参和四模式聚合命令见  
`[classifier_guided_libero_rollout_zh.md](classifier_guided_libero_rollout_zh.md)`。
