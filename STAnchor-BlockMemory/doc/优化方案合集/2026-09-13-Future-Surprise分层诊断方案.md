# Future Surprise 分层诊断方案

## 1. 观察问题与诊断目标

当前 Offset-only 检索编码器没有发生全局 context collapse，但现有证据显示其节点 Key 对 context 的末端状态 `last` 和长窗口边界差 `slope=last-first` 较敏感。本诊断只回答一个问题：当真实 future 明显偏离“当前状态将延续”的简单交通先验时，Offset-only 的检索排序和历史 future 聚合是否比 OffsetDecay 更容易退化。

本方案不修改模型、不重新预训练、不重建 Memory Bank，也不引入新的推理特征。

## 2. 术语、输入与张量

### 2.1 Future Surprise

Future Surprise 表示真实未来轨迹偏离 persistence baseline（持久性基线）的程度。持久性基线将 context 最后一个可见交通速度复制到未来所有步。

- 查询 context：`X ∈ R^[B,12,N,C]`，其中 `B` 为 batch，`N` 为节点数，`C=1` 为速度通道。
- 查询 future：`Y ∈ R^[B,12,N,C]`，仅在检索完成后的离线评估阶段使用。
- context 末端：`L ∈ R^[B,N,C]`；若最后一步缺失，使用 12 步 context 的可见均值回退。

对每个查询节点定义：

\[
S_{b,n}
=
\frac{\sum_{h,c}M_{b,h,n,c}\lvert Y_{b,h,n,c}-L_{b,n,c}\rvert}
{\sum_{h,c}M_{b,h,n,c}},
\]

其中 `M` 是 future 与 endpoint 的共同有效掩码。`S` 在数据集物理速度单位中计算，越大表示 future 越违反 persistence 先验。

### 2.2 Offset-only 与 OffsetDecay

Offset-only 使用相对 context endpoint 的 future 表示：

\[
\phi_h^{\mathrm{OO}}=Y_h-L.
\]

OffsetDecay 使用随预测步衰减的 endpoint 扣除：

\[
\phi_h^{\mathrm{OD}}=Y_h-\lambda_hL,
\qquad \lambda_h:1\rightarrow0.
\]

二者的查询 Key 均只由历史 context、calendar 和图结构生成；future 只提供预训练监督或离线评估标签。

## 3. 分层协议

在完全相同的 1024 个 validation 查询上，对所有有效 `(query,node)` 的 surprise 分数使用统一阈值分层：

- `all`：全部有效样本；
- `ordinary_80`：`S` 低于全体第 80 百分位；
- `surprise_top20`：`S` 高于或等于第 80 百分位；
- `surprise_top5`：`S` 高于或等于第 95 百分位。

阈值只由真实 future 离线计算，不参与 Key、候选生成、候选排序或 Memory 预测。Offset-only 与 OffsetDecay 必须复用同一查询索引和同一 surprise 阈值。

每个版本分别使用与自身 encoder fingerprint 严格匹配的 Bank。比较前必须验证数据集、validation split、sample id、calendar、候选协议和查询集合一致。

## 4. 指标

每个 surprise 层报告：

- `Memory MAE/RMSE/MAPE`：历史候选 future 聚合预测的物理单位误差；
- `Horizon MAE`：12 个预测步分别的 MAE；
- `Future-ranking Spearman`：Key 距离排序与该版本 future teacher 距离排序的一致性；
- `Recall@5`：Key Top-5 与 teacher Top-5 的重合候选数除以 5；只有有效候选多于 5 个的 `(query,node)` 才参与统计。

由于 Offset-only 与 OffsetDecay 的 teacher 定义不同，跨版本主要比较同一 surprise 层的 Memory 误差，以及各版本从 ordinary 到 top20/top5 的相对退化；两版 Spearman 的绝对值只作辅助说明。

## 5. Future 信息边界

检索前允许输入仅包括：查询历史、查询 calendar、图结构、因果 Bank 元数据、历史 Bank Key、历史 Bank level 和历史 Bank future。查询 future 禁止用于 Key 编码、候选池、候选排序、权重或预测。

真实查询 future 只在所有检索和聚合完成后用于：

1. 计算 Future Surprise；
2. 划分离线评估层；
3. 计算排序和预测指标。

因此该诊断不能作为部署时的路由规则，只能用于判断现有检索机制在先验失效样本上的鲁棒性。

## 6. 实现和资源约束

- 新增纯统计模块和 original-only 诊断脚本；不修改训练接口或 checkpoint schema。
- 使用 `C:\Users\31396\.conda\envs\research\python.exe`。
- 使用 `torch.inference_mode()`，目标峰值显存低于 1 GB。
- 正式结果目录只保存完整 JSON、图表和必要运行元数据；临时 smoke 输出在验证后删除。

## 7. 决策规则

- **Keep Offset-only**：其 `surprise_top20/top5` Memory MAE 不劣于 OffsetDecay，或相对 ordinary 的退化没有明显扩大。
- **Warning / 保留 OffsetDecay 对照**：Offset-only 普通层有优势，但 top20/top5 明显落后，说明 last/slope 先验形成长尾风险。
- **Stop 新增 Patch-level level**：无论结果如何，本诊断都不直接授权增加 24 组 level；只有证据明确指出当前 context 表达缺少局部状态且简单方案失败时，才讨论新结构。

## 8. 正式结果（1024 个 validation 查询）

正式运行使用 Offset-only epoch 32 checkpoint 及指纹 `94a93c...` 的 Bank、OffsetDecay epoch 41 checkpoint 及指纹 `7f34ac...` 的 Bank。两版的数据配置、模型结构、seed、查询 `sample_id`、候选事件 ID、target 与有效掩码均严格一致。共有 206,399 个有效 `(query,node)`；Future Surprise 的第 80/95 百分位阈值分别为 4.3721 和 13.2213 个物理速度单位。

### 8.1 Memory 预测

| Surprise 层 | 节点锚点数 | Offset-only MAE | OffsetDecay MAE | Offset-only - OffsetDecay |
|---|---:|---:|---:|---:|
| all | 206,399 | 3.2862 | 3.1794 | +0.1068 |
| ordinary_80 | 165,119 | 1.9301 | 1.9349 | -0.0048 |
| surprise_top20 | 41,280 | 8.7234 | 8.1692 | +0.5543 |
| surprise_top5 | 10,320 | 16.6740 | 14.6632 | +2.0109 |

负差值表示 Offset-only 更好。普通 80% 的差异可忽略，但越偏离 persistence 先验，Offset-only 的劣势越大。

为避免把 207 个空间节点视为完全独立样本，补充按 query 整块重采样的 paired bootstrap。该方法每次有放回抽取完整查询并保留查询内全部节点，统计逐锚点 `MAE_Offset-only - MAE_OffsetDecay`，重复 5,000 次。ordinary_80 的 95% 区间为 `[-0.0140, 0.0096]`，包含 0；top20 为 `[0.5031, 0.6331]`，top5 为 `[1.8592, 2.2473]`，均稳定高于 0。该区间只衡量固定 seed 与固定验证集上的 query 采样不确定性，不替代多 seed 实验。

### 8.2 Key 排序

| Surprise 层 | Offset-only Spearman | OffsetDecay Spearman | Offset-only Recall@5 | OffsetDecay Recall@5 |
|---|---:|---:|---:|---:|
| all | 0.5016 | 0.4328 | 0.4112 | 0.3746 |
| ordinary_80 | 0.5283 | 0.4647 | 0.4073 | 0.3778 |
| surprise_top20 | 0.3945 | 0.3054 | 0.4270 | 0.3618 |
| surprise_top5 | 0.3611 | 0.1293 | 0.4290 | 0.3174 |

两版 teacher 定义不同，因此不能用跨版本 Spearman 的绝对高低证明哪个 Key 更优。版本内看，Offset-only 的 Spearman 在 surprise 层下降，但 Recall@5 没有下降，因而证据不支持“Offset-only Key 在突发样本上彻底忽略 context 或发生坍缩”。

### 8.3 误差来源判断

在 surprise_top5 中，第 1 个 horizon 的 MAE 为 7.8041（Offset-only）和 7.8575（OffsetDecay），Offset-only 还略好；二者差值随后随 horizon 基本单调扩大，到第 12 个 horizon 变为 `19.2187 - 14.7246 = 4.4940`。这与两种 payload 的结构差异一致：Offset-only 在全部 horizon 保留完整 endpoint 对齐，而 OffsetDecay 逐步减弱 endpoint 影响。

因此当前最合理的结论是：**长尾风险主要指向 offset-only payload 的远期持续锚定，而不是检索 Key 的 context collapse。** 这是由分层误差和 horizon 形态支持的机制推断，还不能完全排除 Key 与 payload 的交互。若需要进一步归因，下一步应做无需训练的 `2 个 Key × 2 种 payload` 交叉前向；不应直接增加 patch-level level 模块。

## 9. 成本与产物

- 双模型正式前向总耗时：42.88 秒；
- 峰值 CUDA 显存：0.461 GB；
- 正式结果：`artifacts/diagnostic_future_surprise_offset_only_vs_offset_decay_1024_seed42/surprise_diagnostic.json`；
- 逐锚点可审计数组：同目录 `surprise_anchor_values.npz`；
- 对比图：同目录 `surprise_stratified_memory_mae.png`。
