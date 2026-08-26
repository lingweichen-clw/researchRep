# E5 TGGE HN-OffsetDecay v1：完整检索与下游 Case Study 报告

## 1. 报告目的与证据边界

本报告评估已经完成的 HN-OffsetDecay 预训练版本是否学习到了可迁移的时空事件关系，并验证检索排序和候选聚合能否辅助下游预测。实验对象为 METR-LA 验证集，主 checkpoint 为 `pretrain_best.pt`，完整预训练 50 轮且关闭 early stopping，最佳 checkpoint 来自第 44 轮。

本报告把结果分为两类：

1. **完整验证集 Case Study**：检索诊断、排序指标、OffsetDecay/raw-future 对照、曲线和案例图均使用完整验证集，不使用 `max_batches` 或 smoke 配置。
2. **下游匹配验证**：新 checkpoint 和新 Bank 已完成严格匹配的 1 epoch 验证；新 Bank 的 50 epoch 后验训练尚未完成，因此不能将 1 epoch 数字写成正式 50 epoch 结论。

报告中的 `random` 是严格控制变量的随机 key 对照：event axis、future payload、候选协议和数据划分均与预训练 Bank 相同，仅替换 key。这样可以把收益归因到 key 学习，而不是 Bank 内容或候选数量变化。

## 2. 模型、数据和符号

METR-LA 有 `N=207` 个传感器，采样间隔为 5 分钟。历史窗口为 24 小时，即 `T=288`，预测窗口为 `H=12`，每个传感器输入一个速度变量：

\[
X^{hist}\in\mathbb{R}^{B\times T\times N\times 1},\qquad
Y^{future}\in\mathbb{R}^{B\times H\times N\times 1}.
\]

`B` 是 batch size，`T` 是历史时间步，`N` 是节点数，`H` 是预测 horizon。编码器输出每个节点的 48 维检索表示：

\[
K=f_\theta(X^{hist},G)\in\mathbb{R}^{B\times N\times48},
\]

其中 `G` 是图结构，`K_{q,n}` 是第 `q` 个查询样本、第 `n` 个节点的 key。

### 2.1 OffsetDecay

**OffsetDecay** 是对候选事件 future 进行水平偏移校正后计算的 future teacher 距离。它的作用是消除不同事件当前交通水平不同造成的尺度/基线影响，只比较未来变化形状和动态关系。令 `L_q`、`L_j` 为查询事件和候选事件历史末端的水平统计量，`Y_j(h)` 为候选 future，`\lambda_h` 为随 horizon 衰减的偏移系数，则：

\[
\widetilde Y_j(h)=Y_j(h)+\lambda_h(L_q-L_j),\qquad
\lambda_h=1-\frac{h}{H-1}.
\]

随后在相同节点和 horizon 上计算归一化的 future distance。该 teacher 只在预训练监督和离线诊断中使用真实 future；推理时 key 检索不使用 query future，因此没有未来信息泄漏。

### 2.2 HN-OffsetDecay 监督

正样本来自 OffsetDecay future 距离较近的候选，普通负样本来自其余候选，hard negative 为 context 相似但 future 距离较远的候选。损失严格采用 E2 式加权分母：

\[
\mathcal L_{HN-OD}
=\log\left(\sum_{p\in P}e^{\ell_p}
+\sum_{j\in D}e^{\ell_j}
+w_h\sum_{h\in H}e^{\ell_h}\right)
-\log\sum_{p\in P}e^{\ell_p},
\]

其中 `P` 为正样本集合，`D` 为普通负样本集合，`H` 为 hard negative 集合，`\ell_j` 为 key 相似度 logits，`w_h` 为 hard-negative 权重。该写法避免把 hard negative 同时计入普通负样本和加权负样本。

## 3. 实验协议

- 数据集/划分：METR-LA validation，完整 2,993 个 query。
- 预训练：Latent48，`masked_relation_single_view`，单次 masked encoder forward，OffsetDecay teacher，50 轮，关闭 early stopping，保留索引优化。
- Bank：预训练 Bank 和 random Bank 均含 15,876 个事件、207 个节点、48 维 key，future payload 完全对齐。
- 候选协议：`exact_calendar`，event top-R=32，node top-K=5；平均有效候选数为 7.85，范围 2–9。
- 评价对象：key–future 对齐、排序质量、raw-future 与 OffsetDecay 聚合、固定规则 strong-win/representative/failure 案例。
- 案例选择不是人工挑选：以 `random_memory_mae - pretrained_memory_mae` 为分数，90% 分位选 strong-win，50% 分位选 representative，10% 分位选 failure；并按绝对 gap、sample_id、node_id 进行 tie-break。

## 4. Key 是否学习到 future 关系

完整验证集共得到 4,748,149 个有效 query-candidate pairs。预训练 key 与 OffsetDecay future distance 的 Spearman 相关为 `0.3722`，random key 为 `0.2163`，增益 `+0.1559`。按 key 距离分箱后，预训练模型最近距离箱的 future distance 均值为 `0.8049`，最远距离箱上升到约 `1.2963`；random key 最近箱也有约 `0.8763`，但整体单调关系更弱。

| 指标 | HN-OffsetDecay | Random | 增益 |
|---|---:|---:|---:|
| pair Spearman | 0.3722 | 0.2163 | +0.1559 |
| pair Kendall | 0.2521 | 0.1444 | +0.1076 |
| Recall@1 | 0.2282 | 0.1851 | +0.0431 |
| NDCG@5 | 0.6336 | 0.5719 | +0.0617 |
| Recall@5 | 0.7357 | 0.6957 | +0.0401 |

在逐 anchor 的排序统计中，预训练 key 的 Spearman 均值为 `0.3224`，Kendall 均值为 `0.2521`，NDCG@5 为 `0.6336`。这说明模型确实学习到了“key 越近，future dynamics 通常越相近”的关系，但排序不是完美的，仍存在多模态 future、局部突发事件和候选池截断造成的误排。

## 5. OffsetDecay 曲线与时空海市蜃楼案例

### 5.1 曲线实验

图 `key_future_alignment.png` 展示 key 距离分箱与 OffsetDecay future distance 的关系；图 `ranking_metrics.png` 展示预训练 key 和 random key 的排序指标对照。曲线的目标不是证明严格线性，而是验证整体单调趋势：预训练 key 距离增加时，future distance 总体增加。该趋势在完整验证集上成立，因此可以作为“编码器学习到动态关系”的可视化证据。

### 5.2 时空海市蜃楼定义

**时空海市蜃楼** 指两个事件在历史 context 上相似，因此 key 距离较近，但未来 dynamics 明显不同，因此 OffsetDecay future distance 较大。它是检验编码器是否只记住表面 context、能否保留未来可区分性的关键样本。

本次不手工挑选单个好看的样本，而是保留固定规则选出的 strong-win、representative 和 failure，并在 `cases.json` 中记录 query、node、候选、key 距离和 future distance。论文展示时应同时保留 failure，避免把 Case Study 误写成所有样本都可分离。

图 `deterministic_top5_cases.png` 和 `top5_error_profiles.png` 分别展示固定规则案例的候选/预测误差及 horizon 误差剖面；图 `offset_decay_payload_cases.png` 展示 raw-future 聚合与 OffsetDecay 聚合的三联对照。

### 5.3 固定规则案例

| 类型 | sample/node | 预训练 memory MAE | random memory MAE | gain |
|---|---|---:|---:|---:|
| strong-win | 26443 / 60 | 2.6382 | 3.5052 | +0.8670 |
| representative | 24302 / 89 | 1.5519 | 1.5929 | +0.0411 |
| failure | 27035 / 17 | 1.8183 | 1.3451 | -0.4732 |

OffsetDecay 本身的固定规则案例为：

| 类型 | sample/node | raw-future MAE | OffsetDecay MAE | gain |
|---|---|---:|---:|---:|
| strong-win | 25018 / 127 | 4.8004 | 3.8729 | +0.9275 |
| representative | 26309 / 34 | 1.4552 | 1.4523 | +0.0029 |
| failure | 27285 / 83 | 0.4561 | 0.8750 | -0.4189 |

这些案例传达两个信息：第一，key 学习能够在一部分候选池中把更有用的 future 排到前面；第二，OffsetDecay 能明显修正水平偏移，但在某些本来就接近的 raw future 上可能过校正。failure 不是异常删除对象，而是方法边界证据。

## 6. 聚合与检索排序是否辅助预测

### 6.1 Memory 聚合对照

| 方法 | MAE | RMSE | 解释 |
|---|---:|---:|---|
| weekly mean | 4.3728 | 8.1154 | 周期均值基线 |
| raw future top-1 | 4.3126 | 8.5823 | 仅取 raw future 的第一候选 |
| raw future top-k | 3.9831 | 7.7076 | raw future 加权聚合 |
| learned key top-1 | 4.2334 | 8.3582 | 学习 key 排序后第一候选 |
| learned uniform top-k | 3.9490 | 7.6483 | 学习排序、均匀 top-k |
| learned top-k | 3.7438 | 7.3130 | 学习排序、加权 top-k |
| oracle top-1 | 2.7668 | 5.6881 | 使用真实 future 选择最佳候选，仅作上限 |

以 OffsetDecay 聚合为主结果时，预训练 key memory MAE 为 `3.5308`、RMSE 为 `6.6685`；random key 为 `3.6734`、`6.7786`。因此控制 candidate protocol 和 future payload 后，预训练 key 带来 MAE 改善 `0.1427`，RMSE 改善 `0.1101`。raw-future 聚合中预训练为 `3.8165/7.4735`，random 为 `3.9148/7.5126`，同样存在正增益。

### 6.2 结论与 oracle gap

检索排序对下游预测有实际帮助，但收益尚未达到 oracle 水平。oracle top-1 MAE 为 `2.7668`，learned top-k MAE 为 `3.5308`，差距 `0.7640`。这说明候选池中确实存在有价值的 future，主要瓶颈不是“完全检索不到”，而是：

1. key 排序仍有误排；
2. top-K 截断可能丢掉更优候选；
3. 加权聚合会平均掉多模态或突发 future；
4. 当前 memory 还没有完全转化为下游最终预测收益。

因此当前证据支持优先优化“候选排序与聚合表达”，而不是直接否定预训练检索器；同时不能声称排序已经解决了所有问题。

## 7. 下游匹配验证

新 HN-OffsetDecay checkpoint 和新 Bank 采用与 base-only 相同的 Graph WaveNet、冻结 base、相同数据划分和候选协议。已完成 1 epoch 严格匹配验证：

| 指标 | 新 Bank + residual-additive calibrator |
|---|---:|
| Val MAE | 2.8341 |
| RMSE | 5.7749 |
| 15 min MAE | 2.5254 |
| 30 min MAE | 2.8655 |
| 60 min MAE | 3.2834 |
| 可训练校正器参数量 | 200,911 |
| 单 epoch 时间 | 234.74 s |

旧 Bank 同协议 1 epoch 的 Val MAE 为 `2.8350`，新 Bank 启动阶段改善约 `0.0009`。这一数字只能说明新 Bank 与旧 Bank 的启动行为可复现且没有立即退化，不能替代完整 50 epoch 下游结论。此前尝试启动的新 Bank 50 epoch 进程没有产生有效日志，目录中的日志为 0 bytes，因此当前不报告其训练结果，也不把该进程视为完成。

## 8. 结论

1. **预训练关系学习成立但不完美**：HN-OffsetDecay 相比 random 在 Spearman、Kendall、NDCG@5、Recall@1/5 上均有增益，key 距离分箱曲线呈现正确方向。
2. **OffsetDecay 有独立价值**：在相同 key 下，OffsetDecay memory 优于 raw-future memory；预训练 key 在控制 OffsetDecay 后仍优于 random。
3. **检索排序能辅助预测，但聚合仍是主要损失来源**：oracle 与 learned top-k 之间存在 0.7640 MAE gap，说明候选排序、top-K 保留和多候选融合仍未充分利用候选池信息。
4. **时空海市蜃楼没有被完全消除**：failure 案例表明 context 相似并不必然意味着 future 相似，模型具有一定区分能力但仍会误排。
5. **下游正式结论暂缓**：新 Bank 的完整 50 epoch 下游验证尚未完成，当前只有严格匹配的 1 epoch 证据。

## 9. 后续建议

- 首先完成新 Bank 的完整 50 epoch Graph WaveNet 下游训练，并使用 best checkpoint 与 base-only 做严格匹配比较。
- 若下游增益仍小于 retrieval diagnostics 的增益，优先做单变量聚合实验：OffsetDecay residual 聚合、top-K 温度/稀疏化和 oracle gap 分解；不要同时改编码器和校正器。
- 对时空海市蜃楼进行分层统计：按 key 近但 future distance 高的候选定义 hard-mirage 子集，报告其占比、Recall@K 和下游误差。
- 继续保留 strong-win、representative、failure 三类案例和完整 aggregate 指标；Case Study 只能解释机制，不能替代多 seed、跨数据集和完整下游验证。

## 10. 复现实验产物

主目录：`artifacts/casestudy_hn_offset_decay/`

- 完整检索诊断：`retrieval_diagnostics.json`
- 可视化指标：`visualization_exact/metrics.json`
- 案例明细：`visualization_exact/cases.json`
- 对齐分箱：`visualization_exact/alignment_bins.csv`
- 排序指标：`visualization_exact/ranking_metrics.csv`
- 关键图：`key_future_alignment.png`、`ranking_metrics.png`、`deterministic_top5_cases.png`、`top5_error_profiles.png`、`offset_decay_payload_cases.png`
- 预训练 checkpoint：`artifacts/metrla_e5_tgge_single_view_hn_offset_decay_v1_seed42/pretrain_best.pt`
- 新 Bank：`artifacts/case_bank_hn_offset_decay_v1_seed42`
- 下游 1 epoch 匹配验证：`artifacts/convergence/validation_20260826_hn_offset_decay_gwn_residual_epoch1/`

本报告不引用 smoke、失败启动或未完成训练作为正式性能证据。