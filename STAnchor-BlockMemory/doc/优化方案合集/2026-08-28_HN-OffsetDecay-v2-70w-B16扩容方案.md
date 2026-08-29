# HN-OffsetDecay v2：历史约 70 万对照与当前约 96 万主线扩容方案

更新时间：2026-08-28

> **修订说明（2026-08-28）：** 后续主线容量实验改用 `hidden_dim=128、encoder_layers=4、ffn_multiplier=2、batch_size=16`，旧的 `96/4/FFN4` 配置保留为可复现的约 70 万参数对照，不覆盖其 Bank 或实验产物。新配置的代码实测参数量为 **958,704**；本次切换不改变 HN-OffsetDecay 监督、采样协议、单次前向和索引/缓存策略。

本修订同时补充 HN-OffsetDecay 的 `teacher/student K_eff` 日志统计。该统计在实际有效候选池上计算，仅用于诊断和曲线记录，不进入总损失，也不向模型参数回传梯度。

其中 \(K_{\mathrm{eff}}=1/\sum_j p_j^2\) 是分布的有效支持数：teacher 使用 future-distance 分布，student 使用加入 hard-negative 权重后的 key-logit 分布。数值越大表示权重越分散，不能直接解释为真实候选数量。

## 1. 决策摘要

本方案的目标是只增加检索编码器的有效表达容量，同时把批次大小固定为 `16`，使容量效应与批内采样规模、显存分页效应分离。原始约 70 万参数候选为：

```yaml
model:
  hidden_dim: 96
  encoder_layers: 4
  ffn_multiplier: 4
  retrieval_dim: 64
  adapter_bottleneck_dim: 96

pretrain:
  batch_size: 16
```

按当前代码和 METR-LA 的 `retrieval_context_length=288` 重新实例化统计，该候选共有 **713,744 个可训练参数**。经进一步容量评估，当前主线改用独立配置 `metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16.yaml`：`hidden_dim=128、encoder_layers=4、ffn_multiplier=2、retrieval_dim=64、adapter_bottleneck_dim=96、batch_size=16`，代码实测 **958,704 个可训练参数**。原始 `96/4/FFN4` 配置和 Bank 仅作为历史对照保留。

两种配置都不重新加入 relation projection、rank loss、event-key 改造或新的 Bank 字段；HN-OffsetDecay、masked single-view one-forward、route/index/cache、数据切分、归一化和优化器保持不变。

原始约 70 万参数候选只扩大每个时空块的 FFN 宽度（`2D -> 4D`）；当前主方案改为扩大 token hidden（`96 -> 128`）并回到 FFN2（`D -> 2D -> D`）。两者都不改变 retrieval key 的 64 维接口或批内 pairwise 候选协议。

## 2. 现象与证据边界

### 2.1 30 万版本的正式证据

`artifacts/metrla_e5_tgge_single_view_hn_offset_decay_v1_seed42/pretrain.log` 是完整的 50 轮正式训练，记录为：

| 项目 | v1 |
|---|---:|
| hidden / layers / retrieval dim | 80 / 3 / 48 |
| pretrain batch | 16 |
| 参数量 | 303,727 |
| train batches | 1,418 |
| Epoch 1 训练时间 | 6.3 min |
| Epoch 50 训练时间 | 6.1 min |

因此“30 万版本一轮约 6 分钟”是有完整日志支持的，不是印象或 smoke 结果。

### 2.2 当前扩容日志不能直接作为公平对照

已清理的 B24 原型配置使用 `hidden=96、layers=4、retrieval_dim=64、adapter_bottleneck=96、batch=24`，按当前代码为 565,520 参数；该配置只用于扩容边界诊断，不再作为可运行方案保留。

已清理的旧中断日志实际启动的是 `hidden=120、batch=16、690,655` 参数版本，只运行到第 1 轮的 batch 120/1,418，并非正式完整实验。因此其约 49--53 分钟/轮 ETA 不能归因于 565,520 参数原型。

### 2.3 受控测速结果（诊断证据，不作为模型效果结果）

在同一台 RTX 5060 Laptop（专用显存约 7.96 GiB）上，使用随机训练批次、同一 HN-OffsetDecay 目标测得：

| 配置 | 参数量 | batch | 峰值 allocated | 3/12/20 批稳态观测 |
|---|---:|---:|---:|---:|
| v1 | 303,727 | 16 | 3.98 GiB | 约 0.37 s/批（20 批） |
| 当前 v2 | 565,520 | 16 | 6.12 GiB | 约 0.54 s/批（20 批） |
| 当前 v2 | 565,520 | 24 | 9.17 GiB | 约 18--21 s/批；超过专用显存 |
| v2-FFN4 候选 | 713,744 | 16 | 6.51 GiB | 约 0.60 s/批（20 批） |

短测包含 CUDA 首次启动开销，不能替代完整 epoch；它的用途是判断显存拐点和配置相对关系。`batch=24` 的峰值已超过专用显存，Windows WDDM 会使用共享显存，搬运/分页会导致非线性变慢。

## 3. 根因分析

HN-OffsetDecay 的关系监督对每个 batch 的样本两两计算距离。核心操作在 `stanchor/losses/pretraining.py` 中把：

```text
[B, T, N, C] -> [B, B, T, N, C]
```

构造为成对的 `absolute` 和观测 mask。单个 batch 从 16 增至 24 时，pairwise 元素数为：

\[
\left(\frac{24}{16}\right)^2=2.25.
\]

同一训练集的 batch 数会从 1,418 降至约 946，所以整轮理论 pairwise 工作量仍约增加 1.5 倍；更重要的是峰值显存增加 2.25 倍，跨过显存上限后会触发共享显存分页。因而不能用“参数量增加百分比”线性推算耗时。

参数也并非只增加了一点：v1 到当前 v2 是 303,727 -> 565,520，增加约 86.2%；实际旧日志中的 hidden=120 版本则为 690,655，增加约 127.4%。

## 4. 为什么选择 hidden 扩张与 FFN2

### 4.1 保持四层 encoder，并控制 FFN 宽度

hidden 维度同时出现在 temporal attention、稀疏 graph attention、route value projection 和所有中间残差中。扩大 hidden 会扩大每层的激活和 attention 投影，直接增加显存风险，因此仍固定四层 encoder；FFN 使用 2 倍宽度控制额外计算量。新主方案需要在 16 GB 显存设备上正式验证，不能把本机 8 GB smoke 的耗时外推为正式训练耗时。

### 4.2 采用 hidden 扩张配合 FFN2

当前主方案每个 `FactorizedSTBlock` 的 FFN 为 `D=128` 的 2 倍扩展：

\[
\operatorname{Linear}(D,2D)\rightarrow\operatorname{GELU}
\rightarrow\operatorname{Linear}(2D,D).
\]

这里当前主方案取 `D=128`，即 `128 -> 256 -> 128`。

历史 FFN4 候选使用 `D=96` 的 4 倍扩展：

\[
\operatorname{Linear}(D,4D)\rightarrow\operatorname{GELU}
\rightarrow\operatorname{Linear}(4D,D).
\]

历史 FFN4 候选取 `D=96`，即 `96 -> 384 -> 96`。

FFN 位于 temporal/spatial 信息融合之后；hidden 扩张提高了所有通道投影的表示容量，FFN2 保持每层非线性变换的成本可控。新 token 形状为 `[B,P,N,128]`，但 retrieval key 接口仍为 64 维，pairwise 候选协议和 loss 形状不变。

### 4.3 参数预算

| 配置 | 参数量 | 本机判断 |
|---|---:|---|
| v1：80/3/FFN2/key48 | 303,727 | 已有正式 50 轮证据 |
| 当前 v2：96/4/FFN2/key64/adapter96 | 565,520 | 容量对照候选，batch 必须为 16 |
| 历史扩容方案：96/4/FFN4/key64/adapter96 | 713,744 | 保留作约 70 万参数对照 |
| **当前主方案：128/4/FFN2/key64/adapter96** | **958,704** | 当前主线容量实验；需在 16 GB 显存设备验证 |
| 96/5/FFN2/key64/adapter96 | 688,065 | 本机不首选，显存过于接近上限 |
| 108/4/FFN2/key64/adapter108 | 703,568 | 扩大所有 attention 激活，需更大显存后再考虑 |

当前主方案组件统计：embedding 40,576；encoder 850,372（其中 route attention 54,980）；retrieval head 66,208；reconstruction head 1,548。

## 5. 不变项与明确不做项

### 保持不变

- HN-OffsetDecay 主监督及其 E2 风格正样本/硬负样本定义。
- `masked_relation_single_view` 和单次 encoder forward 优化。
- `retrieval_context_length=288`、patch size、mask 比例和时间块大小。
- route/index/cache 优化、图稀疏注意力、节点级 key 检索。
- train/validation/test 时间切分、训练段 scaler、seed=42、AdamW、lr=0.001、weight decay=1e-4。
- relation teacher/student temperature、candidate protocol 和 downstream 接口。

### 本轮不做

- 不恢复 horizon relation projection、relation loss 或 rank loss。
- 不重新引入 event key、profile semantic key 或旧 Bank。
- 不把 `batch_size` 提到 24；有效 batch 24 若以后仍需要，应在更大显存机器上单独验证。
- 不把当前 v2 的半轮日志当作正式效果证据。

## 6. 实施步骤

### 阶段 A：静态契约检查

1. 使用新配置设置 `hidden_dim=128`、`ffn_multiplier=2`、`pretrain.batch_size=16`，并使用独立的 `runtime.run_name`，避免覆盖任何旧产物。
2. 启动前打印并核对 `hidden/layers/ffn/retrieval_dim/batch`、总参数量和各组件参数量；新主线总量必须为 958,704（旧 FFN4 对照仍应为 713,744）。
3. 检查 checkpoint 的 retrieval fingerprint；新模型不得加载旧 v1/v2 Bank 的 key 作为同一实验结果。

### 阶段 B：数值与显存验收

使用随机训练批次做 20 batch 诊断，记录：

- forward、HN-OffsetDecay loss、backward 和 optimizer step 的耗时；
- `max_memory_allocated`、`max_memory_reserved`；
- anchors、positive pairs、hard negatives、relation candidates；
- loss、梯度和 key 是否全部 finite。

通过条件：无 NaN/Inf；峰值 allocated < 7.0 GiB、reserved < 7.6 GiB；没有持续使用共享显存；稳态单 batch 不出现数量级跳变。若任一条件失败，回退到 FFN3（约 64 万）或当前 v2，不在本机继续堆宽度。

### 阶段 C：短训练对照

固定 seed、数据顺序和所有监督配置，比较：

1. v1 303,727 / batch16；
2. 原始 v2 565,520 / batch16；
3. 历史 FFN4 对照 713,744 / batch16；
4. 当前主方案 958,704 / batch16。

先跑 3 轮或固定少量 batch，仅用于检查收敛方向、峰值显存和每轮时间；短训结果不能写成最终精度结论。主方案只有在不崩溃且时间符合预算后才进入完整 50 轮。

### 阶段 D：正式训练与 Bank

正式 50 轮仍关闭 early stopping。完成后用该 checkpoint 在同一目标训练历史上重建全新 Bank，并在下游使用完全匹配的 Bank/encoder fingerprint。旧 Bank 只能作为旧版本对照，不能混用。

## 7. 评价指标与停止规则

每轮记录：

- validation total、reconstruction、HN-OffsetDecay retrieval loss；
- valid anchors、positive/hard-negative pairs、relation candidates；
- 单轮训练和验证耗时；
- 峰值 allocated/reserved 显存与是否使用共享显存；
- 下游 base-only、TGGE 的 overall MAE、RMSE、15/30/60 分钟 MAE。

建议判断：

- **保留**：无数值异常，峰值通过验收，正式轮次优选不超过约 10--12 分钟，且下游相对当前 v2 至少有 0.01 MAE 改善，或在误差持平（不退化超过 0.005）时显著改善检索关系指标并保持跨模型稳定。
- **暂缓**：预训练 loss 下降但下游没有改善；先保留 checkpoint，暂停继续加容量，检查 key 几何和 Bank 候选质量。
- **回退**：峰值超过 7.6 GiB、出现共享显存分页/NaN，或下游退化超过 0.02。优先回退到 FFN3 约 64 万，而不是改 batch 或重新加入 relation 分支。

## 8. 后续微调衔接

主方案为后续跨数据集微调保留了三层清晰边界：冻结 encoder 的 zero-shot、只更新 domain adapter、再解冻顶部 FFN/encoder 的低学习率微调。由于 retrieval key 维度仍为 64，接口不再因本轮 FFN 扩容变化；但权重已变化，任何目标域 Bank 仍必须用新 checkpoint 重新构建。

本方案的科学结论边界是：它验证“更大的、但显存可控的检索表示容量”是否有助于 future-aware retrieval；它不承诺消除时空海市蜃楼造成的 oracle gap，也不把训练阶段可见的 future teacher 当作部署信息。
