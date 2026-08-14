# Language–Action Classifier 技术报告

> 分析对象：`QwenGR00TClassifier` 的 LIBERO baseline  
> 代码工作树：`main`，基线提交 `135283e`，包含当前未提交的 classifier/overlay 实现  
> 分析日期：2026-08-14

## 1. 结论摘要

这里的 classifier 不是 action policy，也不是把 action 离散成 token 的生成模型。它是一个二分类兼容性打分器：给定当前两路相机图像、一个自然语言指令和一段候选连续动作，判断这段动作是否与视觉–语言上下文相符。

- 输入上下文：两张当前时刻图像（主视角、腕部视角）和一条语言指令，经 Qwen3-VL-4B 编码为 `[B, L, 2560]`。
- 输入动作：未来 8 步、每步 7 维的连续 action chunk，即 `[B, 8, 7]`。
- action 编码：不做 tokenization、量化或 diffusion；直接把 `8×7` 展平成 56 维，再用 `LayerNorm + Linear(56,512) + GELU + Dropout` 编成 512 维。
- 多模态融合：视觉–语言特征 `z_vl`、动作特征 `z_a` 和逐元素乘积 `z_vl ⊙ z_a` 拼接成 1536 维，再通过 MLP 输出一个 logit。
- 输出：每个样本一个未归一化 logit；`sigmoid(logit)` 才是兼容概率。
- classifier 本身：2,136,177 个参数，约 2.136M；BF16 权重约 4.07 MiB，FP32 权重约 8.15 MiB。
- 完整模型对象：约 4.601B 参数，因为继承结构仍实例化 Qwen3-VL（4.438B）和 GR00T action head（161.47M）。默认两者冻结，只有 classifier 的 2.136M 参数训练，占总参数的约 0.0464%。
- 当前 `playground/Checkpoints/libero-language-classifier/` 中没有已完成的模型或评估报告，所以本文只报告结构和协议，不报告尚不存在的 accuracy/AUROC。

## 2. 任务定义

模型学习函数：

\[
s = f(I_{primary}, I_{wrist}, q, a_{t:t+7}), \qquad
p(y=1)=\sigma(s)
\]

其中：

- `I_primary`、`I_wrist` 是 anchor step 的两路图像；
- `q` 是候选语言描述；
- `a_{t:t+7}` 是从 anchor step 开始的 8 步动作；
- `y=1` 表示语言与这段视觉证据和动作相容，`y=0` 表示不相容；
- `s` 是 logit，不是概率。

当前数据的正负样本主要通过“固定图像与动作，只替换语言”生成。因此它实际学习的重点是：同一段机器人行为是否符合某种语言描述，而不是判断动作本身是否成功，也不是直接预测下一步动作。

## 3. 端到端数据流

```text
每个样本
├── image = [primary_image, wrist_image]       两张 224×224 PIL 图像
├── lang = instruction                         字符串
├── action = a[t:t+8]                          [8, 7]，float16 打包
└── classifier_label                           0 或 1

image + lang
  → Qwen chat template / processor
  → Qwen3-VL-4B 最后一层 hidden states [B, L, 2560]
  → attention-mask mean pooling              [B, 2560]
  → VL projector                             [B, 512] = z_vl

action [B, 8, 7]
  → flatten，保持“时间优先、维度次之”的固定顺序
  → [B, 56]
  → action encoder                            [B, 512] = z_a

[z_vl, z_a, z_vl ⊙ z_a]
  → concat                                    [B, 1536]
  → classifier MLP
  → logit                                     [B]
  → sigmoid                                   [B] compatibility score
```

## 4. 输入到底是什么

### 4.1 图像输入

LIBERO dataloader 在每个 anchor step 读取：

1. `video.primary_image`；
2. `video.wrist_image`。

`_pack_sample` 将每张图缩放至 `224×224`，然后作为一个两图列表传入 Qwen processor。classifier 不直接读取像素；它读取 Qwen 对图像和语言共同编码后的 token hidden states。

这里只使用 anchor step 当前帧，不输入历史视频序列。虽然 LIBERO data config 定义了 16 步 state history，但 classifier 样本默认没有启用 `include_state`，而且 classifier 的 `forward` 也完全不读取 `state`。

### 4.2 语言输入

语言以字符串 `example["lang"]` 输入。Qwen interface 把两张图和文本组成 user message，并带 `add_generation_prompt=True` 做 chat-template tokenization。序列长度 `L` 随图像 token 数、文本长度和 padding 改变。

模型取 Qwen 最后一层 `[B,L,2560]`，用 attention mask 对所有非 padding token 做平均：

\[
\bar h_i = \frac{\sum_{j=1}^{L}m_{ij}h_{ij}}
                  {\max(\sum_{j=1}^{L}m_{ij},1)}.
\]

因此 pooled feature 混合了视觉 token、文本 token、chat-template 特殊 token和 generation prompt token；当前实现没有单独抽取“最后一个文本 token”，也没有分别池化视觉与语言。

### 4.3 Action 的物理/数据表示

每步 7 维的固定顺序为：

| 索引 | 字段 |
|---:|---|
| 0 | `action.x` |
| 1 | `action.y` |
| 2 | `action.z` |
| 3 | `action.roll` |
| 4 | `action.pitch` |
| 5 | `action.yaw` |
| 6 | `action.gripper` |

时间索引是 `[0,1,...,7]`，也就是 anchor step 到之后 7 步。overlay metadata 强制 `action_start == anchor_step` 且 `action_end - action_start == 8`；模型侧也硬性拒绝非 `[8,7]` 的 baseline 配置。

需要注意，仓库的 LIBERO `modality.json` 只声明了字段切片，并没有声明前三维/旋转维究竟是绝对目标还是增量命令。loader 当前采用默认 `action_mode="abs"`，其含义是“不额外执行 delta/relative 转换”，即使用数据文件中存储的 action。本文因此不对数据之外的控制语义作额外推断。

### 4.4 Action 归一化

前六个连续维度按每个源数据集各自的 min/max 做线性归一化：

\[
\hat x = 2\frac{x-x_{min}}{x_{max}-x_{min}}-1.
\]

若 `min == max`，实现令该维为 0。与 q99 归一化不同，当前 `min_max` 分支没有再做 clip。第 7 维 gripper 没有列入 `normalization_modes`，因此原值直接通过；已检查四个本地 LIBERO 子集的统计，其 gripper 范围均为 `[0,1]`。

最终 `_pack_sample` 把七个字段按列拼接并存为 `float16 [8,7]`。模型再将它转换成与 Qwen hidden state 相同的 dtype/device。

## 5. Action 怎么编码

当前 action encoder 是一个最小基线：

\[
x_a=\operatorname{vec}(a)\in\mathbb{R}^{56}
\]

\[
z_a=\operatorname{Dropout}\left(
\operatorname{GELU}\left(W_a\operatorname{LN}(x_a)+b_a\right)
\right)\in\mathbb{R}^{512}.
\]

“展平”仍保留固定槽位含义，例如输入第 0–6 维永远是第 0 步的 7 个动作，第 7–13 维永远是第 1 步动作；但网络没有显式的时间 embedding、1D convolution、RNN、Transformer 或跨模态 attention。它依靠 `Linear(56,512)` 为每个时间–动作槽位学习不同权重。

同样需要区分继承来的 GR00T action head：classifier forward 从不调用该 flow-matching/DiT action head。candidate action 也不会经过 GR00T 的 `ActionEncoder`；使用的是独立的 `language_classifier.action_encoder`。

## 6. Classifier 网络结构与张量形状

以当前本地 Qwen3-VL-4B 和配置 `hidden_dim=512` 为准：

| 分支 | 层 | 输入 → 输出 | 参数量 |
|---|---|---:|---:|
| VL | masked mean pool | `[B,L,2560] → [B,2560]` | 0 |
| VL | LayerNorm | `2560 → 2560` | 5,120 |
| VL | Linear + GELU + Dropout | `2560 → 512` | 1,311,232 |
| Action | flatten | `[B,8,7] → [B,56]` | 0 |
| Action | LayerNorm | `56 → 56` | 112 |
| Action | Linear + GELU + Dropout | `56 → 512` | 29,184 |
| Fusion | concat | `512 + 512 + 512 → 1536` | 0 |
| Head | LayerNorm | `1536 → 1536` | 3,072 |
| Head | Linear + GELU + Dropout | `1536 → 512` | 786,944 |
| Head | Linear | `512 → 1` | 513 |
| **合计** |  |  | **2,136,177** |

三个带参数的子模块合计为：

| 子模块 | 参数量 | classifier 内占比 |
|---|---:|---:|
| `vl_projector` | 1,316,352 | 61.62% |
| `action_encoder` | 29,296 | 1.37% |
| `classifier` MLP | 790,529 | 37.01% |
| **总计** | **2,136,177** | **100%** |

融合公式为：

\[
u=[z_{vl};z_a;z_{vl}\odot z_a]\in\mathbb{R}^{1536},
\qquad s=W_2\phi(W_1\operatorname{LN}(u)+b_1)+b_2.
\]

这里的乘积项提供了廉价的逐维交互，但没有完整的双线性层或 token/action cross-attention。

## 7. 网络到底多大

### 7.1 仅 classifier 分支

- 参数：2,136,177（2.136M）。
- BF16 参数存储：约 4.07 MiB。
- FP32 参数存储：约 8.15 MiB。
- 粗略按 FP32 参数、梯度和 Adam 一二阶状态合计 16 bytes/parameter 估算，训练状态约 32.60 MiB；实际还受 mixed precision master weights、分布式切分和框架实现影响。

### 7.2 完整 `QwenGR00TClassifier` 对象

下表来自本地 Qwen safetensors header 和按当前配置实际实例化后的参数计数：

| 部件 | 参数量 | 是否默认训练 | 是否参与 classifier forward |
|---|---:|---:|---:|
| Qwen3-VL-4B | 4,437,815,808 | 否 | 是，负责提取 `[B,L,2560]` |
| GR00T DiT-B action head | 161,472,775 | 否 | 否 |
| language classifier | 2,136,177 | 是 | 是 |
| **总计** | **4,601,424,760** | **2,136,177 可训练** |  |

Qwen checkpoint 本身是全 BF16，weight index 记录 8,875,631,616 bytes，约 8.27 GiB。训练时显存不能只按 classifier 的 4 MiB 估计：冻结的 Qwen 权重、Qwen 前向激活、图像 token 和 batch size 仍是主要显存/计算开销。冻结 action head 不参与该 forward，但它仍被实例化，并可能进入完整 state dict，造成不必要的内存和 checkpoint 体积。

YAML 中虽然写有 `qwenvl.vl_hidden_dim: 2048`，classifier 构造实际读取的是 `self.qwen_vl_interface.model.config.hidden_size`。本地 Qwen config 的真实值为 2560，所以 2.136M 是正确的当前参数量；如果真的换成 hidden size 2048，classifier 才会是 1,873,009 参数。

## 8. 标签与样本构造

每个 anchor 对应固定的七种语言变体：

| 类型 | variant | 标签 |
|---|---|---:|
| 正样本 | `canonical` | 1 |
| 正样本 | `paraphrase_1` | 1 |
| 正样本 | `paraphrase_2` | 1 |
| 负样本 | `wrong_object` | 0 |
| 负样本 | `wrong_destination_or_relation` | 0 |
| 负样本 | `wrong_verb_or_state` | 0 |
| 负样本 | `wrong_order_direction_or_feasible_alternative` | 0 |

训练使用 `balanced_train`：一个 anchor 在一个 epoch 只产生一个语言变体；按照稳定 hash rank 与 epoch 奇偶，整体在正负样本间近似 1:1，并随 epoch 轮换具体变体。验证与测试使用 `exhaustive_eval`：每个 anchor 必须完整产生 3 个正样本和 4 个负样本。

因此训练分布约平衡，但 exhaustive eval 的固有正负比是 3:4。默认 `pos_weight=None`，当前没有为这个 3:4 比例额外加权。

## 9. 训练目标和配置

损失函数为 binary cross entropy with logits：

\[
\mathcal{L}=-\frac{1}{B}\sum_i
\left[y_i\log\sigma(s_i)+(1-y_i)\log(1-\sigma(s_i))\right].
\]

当前训练配置：

| 项目 | 值 |
|---|---:|
| per-device batch size | 16 |
| 最大优化步数 | 10,000 |
| warmup | 500 steps |
| optimizer | AdamW |
| classifier learning rate | `1e-4` |
| betas | `(0.9, 0.95)` |
| weight decay | `1e-8` |
| scheduler | cosine with min LR |
| min LR | `1e-6` |
| gradient clipping | 1.0 |
| dropout | 0.1 |
| validation interval | 250 steps |
| checkpoint interval | 1,000 steps |

冻结有两层保证：framework 的 `freeze_backbones=true` 会将 Qwen 和 action model 设为 `requires_grad=False`；trainer 配置又声明 `freeze_modules: "qwen_vl_interface,action_model"`。优化器给 `language_classifier` 单独建立 `1e-4` 参数组。

## 10. 推理与判定

`predict_compatibility(examples)` 返回：

- `compatibility_logits`：原始 logit；
- `compatibility_scores`：`sigmoid(logit)` 后的 `[0,1]` 分数。

配置中的默认 threshold 是 0.5，但正式协议不是直接拿 0.5 测 test：

1. 每 250 steps 遍历完整 validation split；
2. 在 validation 上选择最大 F1 的 threshold；
3. checkpoint 优先按 validation AUROC，其次 AP，再其次更低 loss 排序；
4. 训练结束后加载选中的 checkpoint；
5. 固定 validation threshold，只评估一次 test。

这避免了在 test set 上选择 threshold。

## 11. 评估指标

实现记录了：

- BCE loss、accuracy、precision、recall、F1；
- AUROC、average precision；
- Brier score、10-bin ECE；
- 0.5 threshold 指标与 best-F1 threshold 指标；
- 正负 logit 均值、标准差和 margin；
- 同一 anchor 内正语言是否排在负语言之上的 paired accuracy/margin；
- 四类 negative 的独立 AUROC；
- anchor、episode、task 三个层级的 macro 指标；
- 以 episode 为 cluster 的 10,000 次 bootstrap 95% confidence interval。

训练时还将 batch 内 action 循环错位一次，计算 shuffled-action AUROC drop 和 score sensitivity，作为“模型是否真的使用 action”的诊断。这个 shuffled action 只用于诊断，不加入 loss。

## 12. 当前实现的重要限制与风险

1. **训练负例只有错语言，没有显式错 action。** 同一 anchor 的七个样本共享相同图像和动作，负标签来自语言扰动。模型可能主要靠图像–语言一致性完成分类，而弱化 action。batch-roll 只做诊断，不能强迫模型学习 action compatibility。

2. **VL mean pooling 过于粗糙。** 它平均所有非 padding token，视觉 token 数量可能远多于文本 token，且特殊 token 也被纳入。短指令信号可能被稀释。

3. **Action flatten 没有显式时序归纳偏置。** 固定 8 步 baseline 可以工作，但不能自然迁移到不同 horizon，也不善于表达局部速度、阶段顺序和长程依赖。

4. **冻结 action head 仍被实例化。** 约 161.5M 参数不参与 classifier forward，却占内存并可能增大保存文件。更干净的 classifier framework 可以不构建它，或者保存时只保存 Qwen 引用信息与 `language_classifier.*`。

5. **配置字段容易误导。** `qwenvl.vl_hidden_dim=2048` 不是 classifier 实际输入维度；真实维度由加载的 Qwen checkpoint 决定。本地实际是 2560。

6. **训练态 shuffled diagnostic 混入 dropout 噪声。** 第二次 score 在 `torch.no_grad()` 中执行，但模型仍处于 train mode，两个分支各自采样 dropout mask。`action_sensitivity` 不完全等同于只替换 action 的因果变化。诊断时临时切 eval mode 或复用 dropout mask 会更严谨。

7. **当前配置没有加载已训练的 GR00T action checkpoint。** 这对 classifier forward 没影响，因为 action head 根本未调用；但若把完整对象误当成可执行 policy，冻结的随机 action head 不能提供有效动作。

8. **没有现成实验结果。** 当前 checkpoint 目录只有空的 `checkpoints/` 子目录，不能据此判断泛化能力或是否出现语言 shortcut。

## 13. 建议的下一轮改进优先级

1. 把 batch 内其他 anchor 的 action、时间反转 action、片段错位 action 纳入训练负例，并明确区分 `wrong_language`、`wrong_action`、`both_wrong`。
2. 报告 action-shuffle 后的 AUROC drop；如果接近 0，应视为模型没有利用 action，而不是仅凭总体 AUROC 判定成功。
3. 将 VL pooling 改为文本 token pooling、视觉/文本分支分别 pooling，或让 action tokens 对 VL tokens 做 cross-attention。
4. 用小型 temporal encoder 替代 flatten，例如 2 层 Transformer/TCN，并增加 timestep embedding。
5. 拆掉 classifier-only 训练中未使用的 GR00T action head，减小模型驻留内存和 checkpoint。
6. 在报告中同时给出 micro、episode macro、task macro、paired accuracy 和四类 negative AUROC，避免单一 accuracy 掩盖类别或任务偏差。

## 14. 代码索引

- classifier 主体：[`starVLA/model/modules/language_action_classifier.py`](starVLA/model/modules/language_action_classifier.py)
- framework 接线：[`starVLA/model/framework/VLM4A/QwenGR00TClassifier.py`](starVLA/model/framework/VLM4A/QwenGR00TClassifier.py)
- Qwen–GR00T 基类：[`starVLA/model/framework/VLM4A/QwenGR00T.py`](starVLA/model/framework/VLM4A/QwenGR00T.py)
- LIBERO action schema/归一化：[`starVLA/dataloader/gr00t_lerobot/data_config.py`](starVLA/dataloader/gr00t_lerobot/data_config.py)
- 七变体 overlay：[`starVLA/dataloader/language_overlay.py`](starVLA/dataloader/language_overlay.py)
- 训练配置：[`examples/simBenchmarks/LIBERO/train_files/starvla_classifier_overlay.yaml`](examples/simBenchmarks/LIBERO/train_files/starvla_classifier_overlay.yaml)
- trainer 与 checkpoint 选择：[`starVLA/training/train_starvla.py`](starVLA/training/train_starvla.py)
- 指标实现：[`starVLA/training/classifier_metrics.py`](starVLA/training/classifier_metrics.py)
- 单元测试：[`tests/test_language_action_classifier.py`](tests/test_language_action_classifier.py)
