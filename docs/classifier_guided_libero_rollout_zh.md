# 在 LIBERO 上验证 Classifier 是否提升 VLA 成功率

本文档说明如何使用已经划分为 `train / val / test` 的语言 benchmark，在
LIBERO 仿真环境中比较以下四种推理方式：

- `off`：原始 VLA，不使用 classifier 修改动作；
- `rerank`：生成 K 条动作候选，用 classifier 选择得分最高者；
- `gradient`：在 flow-matching 采样过程中使用 classifier 梯度引导；
- `gradient_rerank`：生成 K 条梯度引导轨迹，然后再次重排。

这里关注的是 LIBERO 环境最终是否成功，而不是 classifier 的离线分类
accuracy、AUROC 或 diagnostic report。

## 1. 实验原则

一次公平比较必须固定：

- 相同的完整 base VLA checkpoint；
- 相同的 classifier checkpoint；
- 相同的 LIBERO suite、task 和初始状态；
- 相同的语言变体；
- 相同的 rollout seed；
- 相同的图像、state、动作反归一化配置。

`val` 只用于选择 K 和 guidance scale。选定后必须冻结参数，不能根据
`test` 成功率重新选择。

推荐的 split 用法：

| Split | 用途 |
|---|---|
| `train` | 训练 classifier，不运行最终成功率结论 |
| `val` | 小规模 LIBERO rollout，选择 K 和 guidance scale |
| `test` | 使用冻结参数运行最终成功率比较 |

如果 benchmark 只是按 demonstration episode 划分，而同一个 LIBERO task
同时出现在 train、val、test 中，那么最终结论是“在 held-out 语言/episode
metadata 上的成功率”，不能表述为“对未见任务泛化”。

## 2. 准备路径

在仓库根目录执行：

```bash
cd /home/happigo/vla_ws/starVLA
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export META_DIR=/path/to/benchmark/meta
export BASE_CKPT=/path/to/base_vla/checkpoints/steps_N_pytorch_model.pt
export CLASSIFIER_CKPT=/path/to/classifier/checkpoints/best_classifier.pt
export RESULT_ROOT=$PWD/results/classifier_libero_rollout
export PORT=10093
mkdir -p "$RESULT_ROOT"
```

`META_DIR` 至少应包含：

```text
benchmark.json
splits.jsonl
language_bank.jsonl
anchors.jsonl
```

### 2.1 只有旧版完整 classifier checkpoint 时

旧训练代码生成的 `best_classifier_pytorch_model.pt` 同时包含 VLM、action
model 和 classifier。只要该 run 使用的是当前 classifier 架构/protocol v2，
不需要重新训练：

- 旧完整 checkpoint 本身就是 `BASE_CKPT`；
- 可以从同一个文件中一次性抽取小 `CLASSIFIER_CKPT`；
- 服务启动时先加载完整 base，再严格加载抽取出的 classifier。

例如旧权重位于：

```text
/root/gpufree-data/liumingyu/starVLA/playground/Checkpoints/s1/checkpoints/best_classifier_pytorch_model.pt
```

如果当前用户不能读取 `/root`，不要只复制 checkpoint 文件，因为 loader 还
需要 run 目录下的 `config.yaml` 和 `dataset_statistics.json`。建议保持目录
结构复制到当前用户可读的位置：

```bash
export OLD_RUN=/root/gpufree-data/liumingyu/starVLA/playground/Checkpoints/s1
export LOCAL_RUN=/home/happigo/vla_ws/starVLA/playground/Checkpoints/s1_legacy

mkdir -p "$LOCAL_RUN/checkpoints"
sudo cp --reflink=auto \
  "$OLD_RUN/checkpoints/best_classifier_pytorch_model.pt" \
  "$LOCAL_RUN/checkpoints/"

for name in config.yaml config.full.yaml dataset_statistics.json; do
  if sudo test -f "$OLD_RUN/$name"; then
    sudo cp "$OLD_RUN/$name" "$LOCAL_RUN/$name"
  fi
done
sudo chown -R "$(id -un):$(id -gn)" "$LOCAL_RUN"
```

`--reflink=auto` 在文件系统支持时不会立即复制 9GB 数据；不支持时会退化为
普通复制。

确认配置协议：

```bash
grep -n "protocol_version\|name: QwenGR00TClassifier" \
  "$LOCAL_RUN/config.yaml" "$LOCAL_RUN/config.full.yaml" 2>/dev/null
```

然后抽取 classifier-only 权重。该过程会把旧完整 checkpoint 加载一次到
CPU 内存，执行时应预留约 10GB 以上可用内存：

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  extract-classifier \
  --checkpoint "$LOCAL_RUN/checkpoints/best_classifier_pytorch_model.pt" \
  --save-format pt \
  --output "$LOCAL_RUN/checkpoints/best_classifier.pt"
```

成功时会输出 classifier tensor 数量和文件大小。随后设置：

```bash
export BASE_CKPT="$LOCAL_RUN/checkpoints/best_classifier_pytorch_model.pt"
export CLASSIFIER_CKPT="$LOCAL_RUN/checkpoints/best_classifier.pt"
```

也可以临时把旧完整 checkpoint 同时传给 `BASE_CKPT` 和
`CLASSIFIER_CKPT`；当前 loader 会自动抽取 `language_classifier.*`。但服务
启动时需要读取完整 9GB 文件两次，因此只建议用于一次性验证。

如果配置明确写的是 protocol v1，或者抽取后严格加载报告 classifier key
缺失/尺寸不一致，就不能仅靠改名兼容，需要针对旧 classifier 架构写转换或
重新训练。不要把 protocol v1 配置直接伪装成 v2。

检查 metadata 协议和 split 完整性：

```bash
python - <<'PY'
import os
from starVLA.dataloader.language_overlay import validate_language_overlay_metadata

validate_language_overlay_metadata(os.environ["META_DIR"])
print("metadata validation: OK")
PY
```

检查 classifier checkpoint 确实是小权重：

```bash
python - <<'PY'
import os
from starVLA.model.classifier_checkpoint import load_state_dict_file

state = load_state_dict_file(os.environ["CLASSIFIER_CKPT"])
print("tensor keys:", len(state))
bad = [key for key in state if key.startswith("qwen_vl_interface.") or key.startswith("action_model.")]
assert not bad, bad[:10]
print("classifier checkpoint: OK")
PY
```

## 3. 确认 metadata 到 LIBERO 的映射

rollout manifest 最终必须包含：

```json
{
  "suite": "libero_goal",
  "task_index": 3,
  "initial_state_index": 0,
  "variant_id": "canonical",
  "instruction": "put the ..."
}
```

其中 `initial_state_index` 是 LIBERO simulator 的 initial-state 编号，不是
LeRobot demonstration 的 `episode_index`。

支持的标准 suite 通常是：

```text
libero_spatial
libero_object
libero_goal
libero_10
libero_90
```

如果 `language_bank.jsonl` 中的 `source_dataset` 已经是上述 suite 名，且
`task_index` 与 LIBERO task ID 一致，可以跳过映射文件。

如果数据集名称类似
`libero_goal_no_noops_1.0.0_lerobot`，suite 名和 task ID 都需要映射。复制仓库提供的
IPEC LeRobot 完整映射：

```bash
cp examples/simBenchmarks/LIBERO/eval_files/ipec_lerobot_suite_map.json \
  "$RESULT_ROOT/suite_map.json"
```

如果 task ID 也不同，可以显式映射：

```json
{
  "my_goal_dataset": {
    "suite": "libero_goal",
    "task_index_map": {
      "0": 3,
      "1": 7
    }
  }
}
```

后续命令中的映射参数记为：

```bash
export SUITE_MAP_ARG="--suite-map $RESULT_ROOT/suite_map.json"
```

如果不需要映射，则设置为空：

```bash
export SUITE_MAP_ARG=""
```

## 4. 生成 validation manifest

validation 默认按 seed 42 从每个 suite 固定选择两个 task，只使用
`canonical` 和 `paraphrase_1`，每个 task 使用 initial state 0：

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  metadata-manifest \
  --meta-dir "$META_DIR" \
  --split val \
  --seed 42 \
  --tasks-per-suite 2 \
  --initial-state-indices 0 \
  $SUITE_MAP_ARG \
  --output "$RESULT_ROOT/val_manifest.json"
```

查看将要运行的 suite、task 和 episode 数量：

```bash
python - <<'PY'
import json, os
from collections import Counter
from pathlib import Path

p = Path(os.environ["RESULT_ROOT"]) / "val_manifest.json"
m = json.loads(p.read_text())
print("entries:", len(m["entries"]))
print("suites:", sorted({x["suite"] for x in m["entries"]}))
print("tasks:", Counter(x["suite"] for x in m["entries"]))
print("variants:", sorted({x["variant_id"] for x in m["entries"]}))
PY
```

预期 variant 只有：

```text
canonical
paraphrase_1
```

## 5. 先做单 episode smoke test

### 5.1 启动原始 VLA baseline

在终端 A 中执行：

```bash
BASE_CKPT="$BASE_CKPT" \
CLASSIFIER_CKPT="$CLASSIFIER_CKPT" \
PORT="$PORT" \
NUM_CANDIDATES=1 \
GUIDANCE_SCALE=0.0 \
./examples/simBenchmarks/LIBERO/eval_files/run_classifier_policy_server.sh off
```

即使模式是 `off`，仍建议加载 classifier checkpoint。classifier 只负责输出
diagnostics，不会修改 baseline 动作。

等待日志出现 `server running`。

### 5.2 运行一个 suite

在终端 B 中执行；将 `libero_goal` 替换为 manifest 中实际存在的 suite：

```bash
python examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.host 127.0.0.1 \
  --args.port "$PORT" \
  --args.task-suite-name libero_goal \
  --args.rollout-manifest "$RESULT_ROOT/val_manifest.json" \
  --args.instruction-variant canonical \
  --args.rollout-phase validation \
  --args.seed 42 \
  --args.max-tasks 1 \
  --args.result-json "$RESULT_ROOT/smoke_off.json" \
  --args.video-out-path "$RESULT_ROOT/videos/smoke_off"
```

检查输出：

```bash
python -m json.tool "$RESULT_ROOT/smoke_off.json" | head -80
```

至少确认：

- `checkpoint.base` 是预期的 base VLA；
- `checkpoint.classifier` 是预期的 classifier；
- `mode` 为 `off`；
- `episodes` 非空；
- 每个 episode 包含 `success`、`latency_ms`、`classifier_diagnostics`；
- 如果 base VLA 训练时使用 state，则传入的 state 维度为 7 且服务端没有 shape error；
- action 反归一化 key 与 LIBERO 训练数据一致。

smoke test 成功后停止终端 A 的服务。

## 6. 在 validation 上选择参数

搜索空间固定为：

| 模式 | K | guidance scale |
|---|---|---|
| `off` | 1 | 0 |
| `rerank` | 2、4、8 | 0 |
| `gradient` | 1 | 0.03、0.1、0.3 |
| `gradient_rerank` | 2、4、8 | 0.03、0.1、0.3 |

每个配置都要执行以下步骤：

1. 启动对应模式的 server；
2. 对 manifest 中每个 suite 分别运行 `canonical` 和 `paraphrase_1`；
3. 将同一配置产生的 JSON shard 合并成一个 JSON；
4. 停止 server，再测试下一个配置。

启动命令模板：

```bash
BASE_CKPT="$BASE_CKPT" \
CLASSIFIER_CKPT="$CLASSIFIER_CKPT" \
PORT="$PORT" \
NUM_CANDIDATES=<K> \
GUIDANCE_SCALE=<SCALE> \
./examples/simBenchmarks/LIBERO/eval_files/run_classifier_policy_server.sh <MODE>
```

rollout 命令模板：

```bash
MODE=rerank
K=4
SCALE=0.0
SUITE=libero_goal
VARIANT=canonical
RUN_ID="${MODE}_k${K}_s${SCALE}"

mkdir -p "$RESULT_ROOT/val/$RUN_ID/shards"
python examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.host 127.0.0.1 \
  --args.port "$PORT" \
  --args.task-suite-name "$SUITE" \
  --args.rollout-manifest "$RESULT_ROOT/val_manifest.json" \
  --args.instruction-variant "$VARIANT" \
  --args.rollout-phase validation \
  --args.seed 42 \
  --args.max-tasks -1 \
  --args.result-json "$RESULT_ROOT/val/$RUN_ID/shards/${SUITE}_${VARIANT}.json" \
  --args.video-out-path "$RESULT_ROOT/videos/val/$RUN_ID/${SUITE}_${VARIANT}"
```

同一配置的所有 suite 和两个 variant 完成后合并：

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  merge "$RESULT_ROOT/val/$RUN_ID/shards/"*.json \
  --output "$RESULT_ROOT/val/${RUN_ID}.json"
```

所有 16 个配置完成后选择参数：

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  select "$RESULT_ROOT/val/"*.json \
  --output "$RESULT_ROOT/frozen_hyperparameters.json"

python -m json.tool "$RESULT_ROOT/frozen_hyperparameters.json"
```

选择顺序是：

1. validation 成功率更高；
2. 平局时平均延迟更低；
3. 再平局时 K 更小；
4. 再平局时 guidance scale 更小。

保存 `frozen_hyperparameters.json`。正式 test 不允许修改其中的数值。

## 7. 生成正式 test manifest

正式 test 默认遍历 metadata test split 中的全部 task。建议先用 initial state
0 调通，然后根据计算预算扩展到多个状态。

例如每个 task 使用 10 个固定初始状态：

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  metadata-manifest \
  --meta-dir "$META_DIR" \
  --split test \
  --seed 42 \
  --tasks-per-suite -1 \
  --initial-state-indices 0,1,2,3,4,5,6,7,8,9 \
  $SUITE_MAP_ARG \
  --output "$RESULT_ROOT/test_manifest.json"
```

test manifest 包含：

- `canonical`；
- `paraphrase_1`；
- `paraphrase_2`；
- 四类错误指令。

如果当前重点只是验证成功率提升，第一轮正式结果可以只运行三个正指令。四类
错误指令用于语言敏感性诊断，不应该当作替代任务成功率。

## 8. 使用冻结参数运行 test

对每个模式，从 `frozen_hyperparameters.json` 读取 K 和 scale，启动 server。
然后针对每个 suite 和以下正指令运行 rollout：

```text
canonical
paraphrase_1
paraphrase_2
```

test rollout 命令与 validation 相同，只需修改：

```text
--args.rollout-manifest  $RESULT_ROOT/test_manifest.json
--args.rollout-phase     metadata_test
--args.result-json       $RESULT_ROOT/test/<MODE>/shards/<SUITE>_<VARIANT>.json
```

每个模式完成后合并：

```bash
MODE=off
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  merge "$RESULT_ROOT/test/$MODE/shards/"*.json \
  --output "$RESULT_ROOT/test/${MODE}.json"
```

对 `rerank`、`gradient` 和 `gradient_rerank` 重复执行。所有模式必须使用同一
份 `test_manifest.json` 和 seed 42。

## 9. 聚合四种模式

```bash
python examples/simBenchmarks/LIBERO/eval_files/classifier_language_rollout.py \
  aggregate \
  --frozen "$RESULT_ROOT/frozen_hyperparameters.json" \
  "$RESULT_ROOT/test/off.json" \
  "$RESULT_ROOT/test/rerank.json" \
  "$RESULT_ROOT/test/gradient.json" \
  "$RESULT_ROOT/test/gradient_rerank.json" \
  --output "$RESULT_ROOT/final_report.json"

python -m json.tool "$RESULT_ROOT/final_report.json"
```

重点查看每种模式的：

```text
success_rate
paired_success_delta
mean_latency_ms
positive_instruction_success_rate
checkpoint
hyperparameters
```

判断 classifier 是否提升成功率时，以 `off` 为基线：

```text
absolute improvement = assisted success_rate - off success_rate
```

`paired_success_delta` 使用同一个 task、initial state 和语言变体进行配对，
比两个独立成功率更适合判断 classifier 是否真正改变了结果。

建议另外统计两类 episode：

- `off` 失败、classifier 模式成功；
- `off` 成功、classifier 模式失败。

只有第一类明显多于第二类，才能说明 classifier 带来了稳定净收益，而不是随机
改变动作。

## 10. 可选：运行错误指令诊断

对下面四个 variant 重复 test rollout：

```text
wrong_object
wrong_destination_or_relation
wrong_verb_or_state
wrong_order_direction_or_feasible_alternative
```

将这些 shard 与正指令 shard 一起 merge 后，最终报告会增加：

- `canonical_goal_suppression`；
- `action_trajectory_rms_divergence`；
- `mean_classifier_score`。

错误指令下的环境 `success` 不能解释为错误指令对应的新任务成功率，因为
LIBERO 环境仍然使用 canonical goal 判定完成条件。

## 11. 常见问题

### manifest 提示没有对应 suite/variant

检查 `source_dataset`。如果它不是标准 LIBERO suite 名，使用
`--suite-map`。同时确认 `language_bank.jsonl` 中确实存在请求的 split 和
variant。

### `task_index` 越界

metadata 的 task 编号与 LIBERO benchmark task ID 不一致。通过
`task_index_map` 显式映射，不要直接使用 demonstration episode 编号。

### `initial_state_index` 越界

减少 `--initial-state-indices`。可先使用 `0` 做 smoke test，再确认该 suite
可用 initial states 的数量。

### state shape 错误

如果 base VLA 训练时使用 state，QwenGR00TClassifier 的 LIBERO 配置要求
state 为 `[1, 7]`：末端位置 3 维、axis-angle 3 维、单个 gripper 状态 1
维；运行 evaluator 时启用 `--args.use-state`。官方 Qwen2.5 GR00T 4-in-1
配置只声明了 `obs: [image_0]`，应保持默认，不传 state。

### CUDA OOM

先降低 K。`gradient_rerank` 的显存与 K 基本同步增长。不要因为 test OOM
临时改成另一个 K；应在 val 阶段重新确定可运行的搜索空间并完整重跑选择。

### 四种模式结果无法 aggregate

聚合器会拒绝：

- 缺少任一模式；
- test 使用的 K/scale 与 frozen 参数不同；
- 混合 validation 和 test 结果；
- 不同 checkpoint、seed 或配置的 shard 被错误 merge。

### baseline 无法复现

确认四种模式使用同一 `--args.seed`，并且每次 episode 的 task、initial state
和 variant 完全一致。服务响应中的 `classifier_diagnostics.seed` 可用于审计。

## 12. 最终报告最低要求

正式实验报告至少应包含：

- base VLA checkpoint 路径或哈希；
- classifier checkpoint 路径或哈希；
- benchmark metadata 版本和 split；
- validation 选出的 K 和 scale；
- test task 数、initial-state 数和总 episode 数；
- 四种模式的总体及每语言变体成功率；
- 相对 `off` 的绝对成功率差值和配对差值；
- `失败→成功`、`成功→失败` 数量；
- 平均推理延迟；
- 明确声明 test 上没有重新选参。
