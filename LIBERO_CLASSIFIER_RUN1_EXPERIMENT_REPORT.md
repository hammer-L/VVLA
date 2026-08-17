# LIBERO 语言—动作兼容性分类器实验报告（run1）

## 摘要

本实验在 LIBERO 数据上训练语言—动作兼容性分类器，目标是判断给定视觉场景、语言指令与动作片段是否相互匹配。实验采用 protocol v2 的对比训练与穷举验证方案，并冻结原始视觉语言模型和动作模型，仅训练兼容性分类头。

在记录到的 24 次验证中，模型于 step 4000 取得最佳综合结果：语言配对准确率为 **0.9294**，动作配对准确率为 **0.8299**，AUROC 为 **0.8542**。其中，模型对错误动作阶段的识别能力较强（paired accuracy **0.9381**），对跨任务困难负例的识别相对较弱（**0.7216**），后者是当前最主要的性能瓶颈。单模态审计结果显示，视觉语言分支在看不到动作时无法区分动作正负例，而动作分支在看不到语言时也无法判断语言是否匹配，说明主模型的性能主要来自语言与动作的联合建模，而非单模态捷径。

## 1. 实验目的

实验主要验证以下问题：

1. 模型能否识别与视觉场景不一致的语言指令；
2. 模型能否区分正确动作与错误阶段、错误任务动作；
3. 模型是否真正利用了语言和动作的交互，而不是依赖单一模态完成分类；
4. 模型在不同 episode 和任务上的表现是否稳定。

## 2. 实验设置

### 2.1 模型

- 框架：`QwenGR00TClassifier`
- 分类头隐藏维度：512
- 动作输入形状：8 × 7
- Dropout：0.1
- VLM 与原始 action model：冻结
- 可训练模块：`language_classifier`
- 主分类头：依次对视觉语言 token 和动作 token 做交叉注意力

### 2.2 数据与对比协议

- 数据集：LIBERO all
- 训练模式：`contrastive_train`
- 验证模式：`exhaustive_eval`
- 每个训练对比组包含：
  - 1 个语言、动作均正确的正样本；
  - 1 个语言错误、动作正确的语言负样本；
  - 1 个语言正确、动作错误的动作负样本。
- 每个验证 anchor 包含 13 个协议样本：
  - 3 个语言正变体；
  - 4 个语言负变体；
  - 3 个 `wrong_phase` 动作负例；
  - 3 个 `wrong_task_hard` 动作负例。
- 单次验证规模：3744 个样本，其中 864 个正样本、2880 个负样本，共 288 个 anchor。

### 2.3 优化设置

| 配置项 | 数值 |
|---|---:|
| 随机种子 | 42 |
| 训练进程数 | 2 |
| 单进程 batch size（日志实际值） | 36 |
| 全局 batch size | 72 |
| 梯度累积 | 1 |
| 基础学习率 | 1e-4 |
| Warmup steps | 500 |
| 学习率调度 | cosine with min LR |
| 最小学习率 | 1e-6 |
| 验证间隔 | 250 steps |
| 日志间隔 | 10 steps |

训练目标由加权 BCE、语言排序损失、动作排序损失及两个审计 probe 损失组成：

```text
L = weighted_BCE
    + 0.5 × language_rank_loss
    + 0.5 × action_rank_loss
    + 0.1 × vl_probe_loss
    + 0.1 × action_probe_loss
```

## 3. 评价指标

本实验主要使用以下指标：

- `language_paired_accuracy`：正确语言的分数高于语言负例的比例；
- `action_paired_accuracy`：正确动作的分数高于动作负例的比例；
- `paired_margin`：正例与对应负例 logit 的平均差值；
- `AUROC`：不依赖固定阈值的整体排序性能；
- `F1`：在验证集选择的阈值下平衡 precision 与 recall；
- `ECE`、`Brier score`：模型概率的校准误差。

paired accuracy 的随机基线为 0.5。由于每个验证组包含 3 个正样本和 10 个负样本，始终预测负类即可获得 10/13，即 0.7692 的普通 accuracy，因此普通 accuracy 不作为核心结论依据。

protocol v2 按以下字典序选择最佳 checkpoint：

1. 最大化语言和动作 paired accuracy 的较小值；
2. 最大化两者的调和平均；
3. 最大化 AUROC；
4. 最小化 loss。

## 4. 实验结果

### 4.1 验证指标变化

| Step | Eval loss | AUROC | Language paired acc. | Action paired acc. | Action paired margin |
|---:|---:|---:|---:|---:|---:|
| 250 | 1.8345 | 0.6822 | 0.9444 | 0.5327 | 0.0305 |
| 500 | 1.6944 | 0.6989 | 0.9719 | 0.5367 | 0.0099 |
| 750 | 1.7846 | 0.7607 | 0.9596 | 0.6574 | 1.1445 |
| 1000 | 1.6571 | 0.7964 | 0.9518 | 0.7240 | 2.4415 |
| 1250 | **1.3230** | 0.8307 | 0.9497 | 0.7888 | 2.9651 |
| 2000 | 1.4874 | 0.8327 | 0.9256 | 0.7977 | 3.9941 |
| 2750 | 1.5074 | 0.8538 | 0.9436 | 0.8183 | 5.5460 |
| 3000 | 1.7019 | 0.8486 | 0.9349 | 0.8255 | 5.8500 |
| 3500 | 1.7451 | 0.8467 | 0.9261 | 0.8255 | 6.4321 |
| **4000** | 1.6574 | **0.8542** | **0.9294** | **0.8299** | **6.9650** |
| 4750 | 2.0279 | 0.8452 | 0.9212 | 0.8287 | 7.3630 |
| 5500 | 2.2915 | 0.8479 | 0.9190 | 0.8203 | 8.6928 |
| 6000 | 2.4882 | 0.8381 | 0.9078 | 0.8096 | 8.8686 |

语言配对能力在训练早期即达到较高水平；动作配对能力则在 step 500 后开始快速提升，并在 step 3000—4750 进入约 0.82—0.83 的平台区间。按照 protocol v2 的 checkpoint 选择规则，step 4000 为最佳点。

### 4.2 最佳 checkpoint 的总体结果

| 指标 | Step 4000 |
|---|---:|
| Validation loss | 1.6574 |
| Weighted BCE loss | 1.0210 |
| Accuracy | 0.7845 |
| Precision | 0.5204 |
| Recall | 0.8403 |
| F1 | 0.6428 |
| AUROC | 0.8542 |
| Average precision | 0.6132 |
| Brier score | 0.1623 |
| ECE | 0.1468 |
| Validation-selected threshold | 0.2200 |
| Language paired accuracy | 0.9294 |
| Language paired margin | 11.5283 |
| Action paired accuracy | 0.8299 |
| Action paired margin | 6.9650 |

验证集最优阈值为 0.2200，明显低于默认的 0.5。因此，后续测试与部署应读取 `best_classifier.json` 中保存的验证阈值，而不应直接使用 0.5。

### 4.3 动作负例分解

| 动作负例 | Paired accuracy | Paired margin | AUROC |
|---|---:|---:|---:|
| `wrong_phase` | 0.9381 | 9.8785 | 0.9121 |
| `wrong_task_hard` | 0.7216 | 4.0514 | 0.7352 |

模型对错误动作阶段具有较强辨别能力，但对来自其他相似任务的困难动作负例仍有较大提升空间。`wrong_task_hard` 的 paired accuracy 比 `wrong_phase` 低 0.2164，且 margin 低 5.8271，是动作侧性能的主要限制因素。

### 4.4 语言负例分解

| 语言负例 | AUROC |
|---|---:|
| `wrong_object` | 0.8211 |
| `wrong_destination_or_relation` | 0.9462 |
| `wrong_verb_or_state` | 0.8903 |
| `wrong_order_direction_or_feasible_alternative` | 0.9426 |

模型最容易识别错误目的地、关系以及错误顺序或方向；`wrong_object` 的 AUROC 最低，说明在物体身份相近或视觉表征相似时仍容易混淆。

### 4.5 单模态与反事实审计

| 审计项 | 关键指标 | 结果 |
|---|---|---:|
| 完整模型 | Action paired accuracy | 0.8299 |
| VL-only probe | Action paired accuracy | 0.5032 |
| Action-only probe | Language paired accuracy | 0.5000 |
| Action-only probe | Action paired accuracy | 0.5443 |
| Donor-action audit | Action paired accuracy | 0.6580 |
| Donor-action audit | AUROC | 0.7432 |
| Mean-action audit | Action paired accuracy | 0.4916 |

完整模型相对 VL-only probe 的动作配对优势为 0.3267，相对 action-only probe 的语言配对优势为 0.4294。这说明只看视觉语言或只看动作都不足以完成任务，模型的主要性能来自跨模态交互。

将动作替换为审计 donor 后，动作 paired accuracy 从 0.8299 降至 0.6580，下降 0.1719；将动作统一替换为均值动作后，动作 paired accuracy 降至接近随机的 0.4916。两项结果均表明主分类分数对真实动作输入具有实质敏感性。

### 4.6 Micro 与 macro 一致性

step 4000 的 task-macro 指标为：

- Task-macro AUROC：0.8602；
- Task-macro language paired accuracy：0.9316；
- Task-macro action paired accuracy：0.8337。

这些结果与 micro 指标非常接近，说明总体性能并非主要由少数高频任务贡献，跨任务表现具有较好一致性。

## 5. 结果分析

### 5.1 语言能力学习较快

语言 paired accuracy 在首次验证时已达到 0.9444，并在 step 500 达到 0.9719。这表明冻结 VLM 提供的视觉语言特征已经具备较强语义区分能力，分类头可以较快学习语言负例。

### 5.2 动作兼容性需要更长的学习过程

动作 paired accuracy 从 step 250 的 0.5327 提升至 step 1250 的 0.7888，并在 step 4000 达到 0.8299。与语言侧相比，动作侧需要从原始 8 × 7 动作片段中学习时序与任务语义，收敛速度更慢。

### 5.3 最佳训练区间位于 3000—4500 steps

训练集 action rank loss 持续下降，训练 AUROC 在 step 6000 达到 1.0；但验证集 AUROC、语言 paired accuracy 和动作 paired accuracy 在 step 4000 后不再同步提升。结果表明模型在约 4000 step 后开始出现训练集继续改善而验证集趋于平台或下降的现象。

因此，当前设置下推荐使用 step 4000 的最佳 checkpoint，并将后续同类实验的 early-stopping 观察区间设在 3000—4500 steps。

### 5.4 概率校准仍有提升空间

最佳 checkpoint 的 ECE 为 0.1468，验证最优阈值为 0.2200，表明分类概率尚未充分校准。当前模型更适合作为排序器或使用验证集阈值的二分类器，不宜直接把 0.5 当作统一决策阈值。

## 6. 结论

本实验验证了冻结 backbone、仅训练语言—动作兼容性分类头的方案能够有效学习跨模态兼容性。step 4000 的模型同时取得 0.9294 的语言 paired accuracy 和 0.8299 的动作 paired accuracy，且单模态与反事实审计均支持模型确实利用了动作信息，而非仅依赖语言或视觉捷径。

当前模型的主要问题不是动作 paired accuracy 接近随机，而是对 `wrong_task_hard` 困难动作负例的区分能力明显弱于 `wrong_phase`。后续改进应优先聚焦困难跨任务 donor 的构造和动作表征，而不是简单延长训练时间。

## 7. 后续建议

1. 采用 step 4000 的 `best_classifier` 作为本实验最终模型；
2. 使用 `best_classifier.json` 中的 0.2200 验证阈值进行独立测试；
3. 增加或优化 `wrong_task_hard` donor，尤其关注动作轨迹相似但任务语义不同的样本；
4. 对 `wrong_object` 语言负例进行更细粒度分析；
5. 在独立 test split 上报告 paired accuracy、95% episode-cluster bootstrap 置信区间以及 protocol-v2 acceptance checks；
6. 后续训练可在 3000—4500 steps 区间启用 early stopping，避免仅依据训练损失选择模型。

## 8. 数据来源

- 训练日志：`/home/happigo/文档/output/run1/output.log`
- W&B 汇总：`/home/happigo/文档/output/run1/wandb-summary.json`
- 分析范围：step 250—6000 的 24 次完整 validation 记录

