# 目标域 Future 关系迁移 CaseStudy 方案

## 1. 修订目标

目标域 CaseStudy 必须与 METR-LA 源域 CaseStudy 使用同一套指标。本方案废弃旧目标域脚本中的 trend-cosine Spearman 作为主结果，改为 HN-OffsetDecay v2 teacher distance。

本次工作只修正离线诊断口径，不改变检索编码器、T1 微调方法、Bank 格式、候选检索实现或下游校准器。

## 2. 实验范围

目标域为 PEMS-BAY、PEMS04 和 PEMS08。每个目标域均在完整 validation split 上运行以下两种候选协议：

- `broad_causal`：所有 future 已结束于 query context 开始前的历史事件均合法；若候选数超过 `event_top_r=96`，按固定时间顺序等距抽取 96 个事件。该抽样不依赖任何模型 key。
- `weekday_radius1_overlap`：query 星期的前一天、当天、后一天的相同时间槽候选；候选 future 只需结束于 query context 结束前，因此允许候选 context 与 query context 重叠。这是当前正式下游协议。

每个协议比较三种表示：

1. `random`：随机初始化检索编码器对应的目标域 Bank；
2. `source`：METR-LA 预训练检索编码器直接应用到目标域的 Bank；
3. `finetuned`：目标域 T1 head-adapter 微调后检索编码器的 Bank。

三个表示必须共享 query、候选事件 ID、candidate future payload 和有效性 mask。差异只能来自 key 表示。

## 3. Teacher future distance

对每个 query 或候选事件，在节点 n、预测步 h 上构造 HN-OffsetDecay v2 future signature：

\[
\phi_{h,n}=Y_{h,n}-\lambda_h e_n,
\qquad
\lambda_h=1-\frac{h}{H-1}.
\]

其中：

- \(Y_{h,n}\) 是 12 步未来值；
- \(e_n\) 是该事件 12 步 context 的最后一个有效观测；
- \(\lambda_h\) 从 1 线性衰减到 0。

在 query 和候选共同可观测的位置计算 masked MAE。随后按 query/candidate event 的局部尺度作 `symmetric_geometric_mean` 归一化，得到：

\[
d^{future}_{q,n,r}\in\mathbb{R}^{B\times N\times R}.
\]

query key 与候选 key 的一减 cosine distance 为：

\[
d^{key}_{q,n,r}=1-\cos(z^q_{q,n},z^c_{q,r,n}).
\]

未来仅用于 validation 后的 teacher metric，不进入 query key、candidate key、候选池构造或检索排序，因而没有部署期 future leakage。

## 4. 统一统计指标

对 random、source、finetuned 分别计算：

1. Global alignment Spearman：对所有有效 \((q,n,r)\) pair 展平，计算 \(d^{key}\) 和 \(d^{future}\) 的 Spearman；
2. Anchor-wise ranking Spearman：固定一个 \((q,n)\)，在候选轴 r 内计算排序相关，再对全部有效 anchor 求平均；
3. Anchor-wise Kendall、Recall@1、NDCG@5、Recall@5；
4. `future_neighbor_recall_at_5`；
5. candidate pool 的 mean、median、min、max；有效 pair 数和 eligible anchor 数。

输出 `source - random` 和 `finetuned - source`。旧的 trend-cosine 结果可保留为内部快速诊断，但不得同 HN 指标并列比较或写入论文主表。

## 5. 简单 CaseStudy 的含义

“简单”仅表示不生成复杂图件、逐样本轨迹和案例 payload；正式结果仍必须遍历完整 validation split。`--max-batches` 只能用作 smoke，不得用于最终报告。

## 6. 输出与审计

每个 JSON 输出必须包含：数据集、split、协议、`event_top_r`、`node_top_k`、三个 checkpoint 和 Bank 路径、Bank 事件轴对齐检查、候选池统计、metric definition、future-information boundary 和运行耗时。

结果位置：

```text
artifacts/cross_dataset_case_study_hn/
  pemsbay_broad_causal_hn.json
  pemsbay_weekday_radius1_overlap_hn.json
  pems04_broad_causal_hn.json
  pems04_weekday_radius1_overlap_hn.json
  pems08_broad_causal_hn.json
  pems08_weekday_radius1_overlap_hn.json
  summary.json
```

## 7. 决策规则

- 若 `finetuned` 在两个协议、三个目标域的大多数 HN 指标上均优于 `source`，支持 T1 微调改善目标域 future-relation alignment；
- 若只在某一个协议上改善，应报告协议敏感性，不能宣称稳定的普遍增益；
- 若 `source` 和 `finetuned` 均稳定优于 `random`，说明源域预训练表示在目标域仍保留有效的 future relation；
- 若三者接近或随机更优，结论只能是当前数据集/协议下未观察到可靠表示优势，后续应以微调或模型选择结果为准。
