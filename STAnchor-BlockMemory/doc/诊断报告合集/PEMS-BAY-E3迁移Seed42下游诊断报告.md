# PEMS-BAY E3 迁移 Seed 42 下游诊断报告

> **后续归因更正（2026-07-30）**：target-random 对照已经完成。历史 memory 系统的收益仍成立，但本报告第 7 节“E3 encoder-selector 在 PEMS-BAY 上构建了有效迁移表示”的表述不能再归因于 METR-LA 预训练，因为 random encoder 的检索 MAE 更低、最终 MAE 与 source-pretrained 持平。最新结论见 `PEMS-BAY-E3迁移Target-Random归因诊断报告.md`。

## 1. 本报告回答的问题

本报告只比较 PEMS-BAY validation 上的两个已冻结模式：

- `base_only`：轻量 backbone 只使用最近 12 个时间步；
- `learned_topk_horizon`：相同 backbone 加入冻结的 METR-LA E3 encoder-selector、PEMS-BAY Bank 和 12 个 horizon fusion 参数。

本报告不使用 test，不讨论多 seed 稳定性，也不能代替 target-random 预训练对照。

## 2. 最佳 checkpoint

| 模式 | 最佳 epoch | Validation MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|---:|
| base-only | 34 | 2.171461 | 5.097139 | 5.232578 |
| **E3 learned Top-K + horizon fusion** | **34** | **1.881370** | **4.107756** | **4.390352** |

相对改善为：

| 指标 | 绝对改善 | 相对改善 |
|---|---:|---:|
| MAE | 0.290092 | 13.36% |
| RMSE | 0.989383 | 19.41% |
| MAPE | 0.842226 | 16.10% |

以 MAE 为例：

\[
G_{\mathrm{mem}}
=
\frac{2.171461-1.881370}{2.171461}\times100\%
=13.36\%.
\]

该结果通过 seed 42 的第一道门槛：加入冻结 E3 memory 后，最终预测显著优于相同数据和预算下的短期 backbone。

## 3. 收益随预测距离变化

| 预测位置 | Base MAE | E3 Memory Fusion MAE | 相对改善 |
|---|---:|---:|---:|
| 15 min | 1.555374 | 1.498895 | 3.63% |
| 30 min | 2.182222 | 1.965845 | 9.92% |
| 60 min | 3.055083 | 2.359796 | 22.76% |

结论不是“所有预测步同等受益”，而是预测越远，历史模式的补充价值越明显。最近 12 步已经包含很强的短时惯性，因此 15 分钟收益较小；当预测推进到 60 分钟时，短期输入的信息衰减，而历史事件 future 提供了更稳定的形状先验。

## 4. 同一 checkpoint 的分支归因

`learned_topk_horizon` 最佳 checkpoint 在同一 validation 样本上的分支结果为：

| 分支 | MAE | 含义 |
|---|---:|---|
| base branch | 2.248062 | 该联合训练 checkpoint 中的 backbone 单独预测 |
| memory branch | 2.209415 | 冻结 E3 learned weighted Top-K 历史预测 |
| final fusion | **1.881369** | 两者按预测步融合后的最终预测 |

memory 单独只比同 checkpoint 的 base branch 改善 1.72%，但最终融合比 base branch 改善 16.31%。这是合理的，因为全局 MAE 比较的是所有位置的平均误差：即使两个分支各自的平均 MAE 接近，只要它们在不同位置犯错，逐位置凸组合仍可能同时优于两个单分支。

需要注意，联合训练 checkpoint 的 base branch MAE 2.248062 弱于独立训练的 base-only 2.171461。这说明最终系统是作为“backbone + memory + fusion”共同优化的，不能把 13.36% 全部解释为在一个完全不变的 backbone 上直接外挂 memory。严格的模块结论是：

> 在相同下游结构和训练预算下，引入冻结 E3 memory 的联合预测系统优于 base-only 系统。

## 5. Horizon Fusion 学到了什么

horizon-only 的 confidence 恒为 1，最终权重只由 12 个可训练的预测步上限决定：

\[
\widehat{\mathbf Y}_{h}
=
\widehat{\mathbf Y}^{\mathrm{base}}_{h}
+
w_h
\left(
\widehat{\mathbf Y}^{\mathrm{mem}}_{h}
-
\widehat{\mathbf Y}^{\mathrm{base}}_{h}
\right),
\qquad 0<w_h<1.
\]

其中，\(w_h\) 是第 \(h\) 个预测步共享给所有 batch 和节点的 memory 权重。学到的 12 个权重为：

```text
0.0777, 0.1590, 0.2371, 0.3084,
0.3775, 0.4426, 0.5028, 0.5570,
0.6051, 0.6456, 0.6815, 0.7134
```

关键位置：

| 预测位置 | Memory 权重 |
|---|---:|
| 15 min | 0.2371 |
| 30 min | 0.4426 |
| 60 min | 0.7134 |

权重随 horizon 单调增加，与误差改善随 horizon 增加完全一致。这提供了清晰语义：短期主要相信当前窗口，远期逐渐增加历史模式贡献。

## 6. 为什么仍需要验证 Confidence

在 19,155,216 个有效 validation 预测位置中，memory 单独优于 base 的比例为 47.31%。也就是说，memory 并非对每个节点、每个时刻都可靠。

horizon-only 只能学习全局 \(w_h\)，无法区分同一预测步中“当前节点应该使用 memory”还是“当前节点应该回退到 base”。因此下一步 confidence 的实验问题很具体：

> 节点级、预测步级 confidence 能否在保留长 horizon 收益的同时，降低 memory 对无帮助位置的干扰？

本模式的 confidence 恒为 1，所以诊断文件中 `AUROC=0.5`、`AUPRC=prevalence` 和 confidence 四分位没有机制意义；这些值不能用于评价 confidence 网络。只有 `learned_topk_confidence` checkpoint 才能评价置信度质量。

## 7. 当前能下与不能下的结论

### 可以下的结论

1. 历史 memory 系统在 PEMS-BAY 上能够构建有效检索结果，但后续 target-random 对照表明该收益不能归因于 METR-LA E3 预训练；
2. E3 memory 与短期 backbone 存在明显互补性；
3. memory 的主要预测收益集中在中长 horizon；
4. 仅 12 个 horizon fusion 参数就学到了符合机制预期的单调权重；
5. seed 42 结果支持继续运行 confidence 单变量对照。

### 不能下的结论

1. 不能用单 seed 声称迁移结果稳定；
2. 尚未证明 source pretraining 优于同架构 target-random encoder；
3. 尚未证明 confidence 在 PEMS-BAY 上有效；
4. 尚未使用 test，不能报告最终泛化结果；
5. 不能把当前结果称为跨物理量或多源基础模型能力。

## 8. 训练效率

- base-only 完整训练约 6.1 分钟；
- horizon-only 完整训练约 107.7 分钟；
- 两者均在 epoch 44 早停，最佳 checkpoint 均来自 epoch 34。

额外耗时主要来自每个 epoch 重复运行冻结的 288 步 encoder 和精确检索。因为 pretrained、Bank 和 query 都被冻结，后续多 seed 前应优先考虑缓存 calibration/validation 的检索输出。这属于等价工程优化，必须先验证缓存前后输出一致，不能借此修改检索结果或实验口径。

## 9. 下一步决策

下一步只运行 seed 42 的 `learned_topk_confidence`：

- 若其 validation MAE 低于 1.881370，并且 AUROC/AUPRC、Brier、ECE 与 confidence 分桶 gain 具有一致语义，则保留 confidence；
- 若最终 MAE 没有改善，或 confidence 仍接近常数，则 PEMS-BAY 主结果保留 horizon-only，移除 confidence；
- 只有保留模式确定后才运行 seed 2024、2025；
- 多 seed 之前还要完成 target-random 对照，隔离预训练表示贡献。

## 10. 证据文件

- base-only checkpoint：`artifacts/pemsbay_e3_base_only_seed42/downstream_best.pt`
- horizon-only checkpoint：`artifacts/pemsbay_e3_learned_topk_horizon_seed42/downstream_best.pt`
- horizon-only 分支诊断：`artifacts/pemsbay_e3_learned_topk_horizon_seed42/branch_diagnostics_val.json`
- 检索诊断：`artifacts/pemsbay_e3_transfer_diagnostics/retrieval_diagnostics_val.json`
