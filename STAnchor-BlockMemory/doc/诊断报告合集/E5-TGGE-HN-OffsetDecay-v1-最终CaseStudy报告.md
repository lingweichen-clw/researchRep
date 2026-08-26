# E5 TGGE HN-OffsetDecay v1：最终 Case Study 实验报告

## 1. 实验目的与协议

本报告验证 HN-OffsetDecay 预训练是否学习到可检索的 future dynamics 关系，并验证检索排序、OffsetDecay 聚合和时空海市蜃楼区分能力。论文主 Case Study 使用 `pretrain_broad_causal`：严格因果、允许不同 weekday/slot、每个 query 最多 32 个候选事件。该协议与预训练监督语义一致；`exact_calendar` 作为部署约束下的补充实验。

验证集包含 2,993 个 query。预训练 Bank 与 random Bank 共享 event axis、future payload、候选协议和数据划分，仅替换 key。所有结果均为完整验证集结果，不使用 smoke 或 max_batches。

## 2. 模型与术语

METR-LA 包含 N=207 个传感器，5 分钟采样；检索窗口 T=288，预测窗口 H=12：

$$X^{hist}\in\mathbb{R}^{B\times288\times207\times1},\qquad Y^{future}\in\mathbb{R}^{B\times12\times207\times1}.$$ 

编码器输出 48 维节点 key：$$K=f_\theta(X^{hist},G)\in\mathbb{R}^{B\times207\times48}.$$ `K_{q,n}` 表示 query q 在节点 n 上的 key。

OffsetDecay 对候选 future 做 horizon-dependent 水平偏移校正：

$$\widetilde Y_j(h)=Y_j(h)+\lambda_h(L_q-L_j),\qquad \lambda_h=1-\frac{h}{H-1}.$$ 

其中 L_q、L_j 是 query 和候选事件的历史末端水平。该操作降低绝对水平差异，突出未来变化形状；真实 future 只用于离线 teacher、指标和作图，推理排序不使用 query future。

HN-OffsetDecay 将 future 近的候选作为正样本，将 context 相似但 future 远的候选作为 hard negative，并使用：

$$\mathcal L=\log(\sum_{p\in P}e^{\ell_p}+\sum_{j\in D}e^{\ell_j}+w_h\sum_{r\in H}e^{\ell_r})-\log\sum_{p\in P}e^{\ell_p}.$$ 

## 3. 主结果：pretrain_broad_causal

完整验证集产生 19,020,445 个有效 query-candidate pairs，平均候选数为 32。

| 指标 | HN-OffsetDecay v1 | Random | 增益 |
|---|---:|---:|---:|
| Pair Spearman | 0.6212 | 0.1365 | +0.4847 |
| Anchor Spearman | 0.5599 | 0.1420 | +0.4179 |
| Anchor Kendall | 0.4173 | 0.0990 | +0.3183 |
| Recall@1 | 0.1313 | 0.0559 | +0.0755 |
| NDCG@5 | 0.4566 | 0.2647 | +0.1919 |
| Recall@5 | 0.3997 | 0.2183 | +0.1815 |

Pair Spearman 衡量所有 query-candidate 对中 key 距离与 future distance 的整体单调关系；Anchor Spearman/Kendall 衡量每个 query 内部排序一致性；Recall@K 表示真实 future 近邻进入 key 排名前 K；NDCG@5 同时考虑前 5 名的位置质量。

预训练 key 距离分箱的 future distance 均值从最近箱 0.4880 上升至约 0.6000、0.6713、0.7411，呈清晰单调趋势。

![图 1：Key 距离与 OffsetDecay future distance 的分箱关系](../../artifacts/casestudy_hn_offset_decay/visualization_pretrain_broad_causal/key_future_alignment.png)

图 1 是关系学习的主要机制证据：预训练曲线随 key 距离增大而上升，random 曲线明显更平坦。

## 4. 检索排序与聚合对预测的作用

| 方法 | MAE | RMSE |
|---|---:|---:|
| Weekly mean | 4.3728 | 8.1154 |
| Raw future top-1 | 4.3126 | 8.5823 |
| Raw future top-k | 3.9831 | 7.7076 |
| Learned OffsetDecay top-k | 3.5060 | 6.6698 |
| Random OffsetDecay top-k | 4.8036 | 8.3532 |
| Oracle top-1 | 2.7668 | 5.6881 |

MAE 是平均绝对误差，RMSE 对大误差更敏感。HN-OffsetDecay 相比 random OffsetDecay 将 MAE 降低 1.2975、RMSE 降低 1.6834；相比 raw future top-k，MAE 降低 0.4771。Oracle 与 learned top-k 仍有 0.7392 MAE gap，说明主要瓶颈是候选误排、Top-K 截断和多模态 future 被平均。

![图 2：排序指标对照](../../artifacts/casestudy_hn_offset_decay/visualization_pretrain_broad_causal/ranking_metrics.png)

![图 3：固定规则案例](../../artifacts/casestudy_hn_offset_decay/visualization_pretrain_broad_causal/deterministic_top5_cases.png)

![图 4：不同 horizon 的误差剖面](../../artifacts/casestudy_hn_offset_decay/visualization_pretrain_broad_causal/top5_error_profiles.png)

图 2–4 将 aggregate 指标与 strong-win、representative、failure 案例对应起来；案例按固定分位数规则自动选择，不是人工挑选。

## 5. 时空海市蜃楼实验

时空海市蜃楼指历史 context 与 future dynamics 的相似性不一致。实验使用 5,000 个 Bank 事件和全部 207 个节点，context 为最近 12 步、future 为 12 步，并进行节点内 robust 标准化。

类型 A：context distance ≤ P8、future distance ≥ P92、key distance ≥ P92，表示历史几乎相同但未来明显分叉，且 key 将其推远。类型 B：context distance ≥ P92、future distance ≤ P8、key distance ≤ P8，表示历史明显不同但未来高度相似，且 key 将其聚集。阈值为 context P8=0.0464、P92=0.5232；future P8=0.1477、P92=1.7892；key P8=0.0339、P92=0.1435。

类型 A 三对样本的 context distance 为 0.0228、0.0197、0.0226；future distance 为 12.1309、12.1302、12.1218；key distance 为 0.1770、0.1778、0.1739，均超过 key P92。

![图 5：Context 高度相似、future 明显不同且 key 分散](../../artifacts/casestudy_hn_offset_decay/mirage_analysis/context_similar_future_different.png)

类型 B 三对样本的 context distance 为 3.7788、3.7649、3.3511；future distance 为 0.1006、0.1046、0.0948；key distance 为 0.0328、0.0226、0.0335，均低于 key P8。

![图 6：Context 不相似、future 高度相似且 key 集中](../../artifacts/casestudy_hn_offset_decay/mirage_analysis/context_different_future_similar.png)

图 5 和图 6 均为 context、future、key PCA 三联图。它们用于解释机制，不替代 aggregate 统计。

## 6. 与 TGGE Joint-v3 的 broad-causal 比较

| 指标 | TGGE Joint-v3 | HN-OffsetDecay v1 | 变化 |
|---|---:|---:|---:|
| Pair Spearman | 0.6826 | 0.6212 | -0.0614 |
| Anchor Spearman | 0.5740 | 0.5599 | -0.0141 |
| Anchor Kendall | 0.4313 | 0.4173 | -0.0140 |
| Recall@1 | 0.1204 | 0.1313 | +0.0109 |
| NDCG@5 | 0.4435 | 0.4566 | +0.0131 |
| Recall@5 | 0.3890 | 0.3997 | +0.0108 |
| Memory MAE | 3.5632 | 3.5060 | -0.0571 |

HN-OffsetDecay 并非在所有全局相关性指标上超过 v3：v3 的 Spearman/Kendall 略高；当前版本的 Recall@1、NDCG@5、Recall@5 和 memory MAE 更好。这与教师目标设计一致：v3 的连续关系/散度目标拟合全体候选的平滑全局结构，而 HN-OffsetDecay 只对阈值筛选出的正样本和 hard negative 施加重点约束，更集中优化有限候选集合内的 Top-K 可用性。因此，当前版本 Spearman 略低不必然表示关系学习退化；本实验中它体现为 Top-K 有用候选命中和候选聚合更强。

Exact-calendar 作为部署补充：当前版本 Spearman=0.3722、NDCG@5=0.6336、Memory MAE=3.5308；v3 分别为 0.3124、0.6292、3.5389。

![图 7：Exact-calendar 部署约束下的排序对照](../../artifacts/casestudy_hn_offset_decay/visualization_exact/ranking_metrics.png)

## 7. 结论与证据边界

HN-OffsetDecay 已证明 48 维 key 学到了可检索的 future dynamics；OffsetDecay 降低了水平偏移误差；hard-negative 监督提升了 Top-K 有用候选命中；模型对两类时空海市蜃楼具有一定区分和聚集能力。当前 learned 与 oracle 之间仍有明显差距，候选聚合会抹平多模态 future。新 Bank 的完整 50 epoch 下游训练尚未形成有效日志，因此本报告不声称最终下游预测增益已经完成验证。

## 8. 复现实验产物

- broad-causal 指标：`artifacts/casestudy_hn_offset_decay/visualization_pretrain_broad_causal/metrics.json`
- broad-causal 图表：同目录下的 `key_future_alignment.png`、`ranking_metrics.png`、`deterministic_top5_cases.png`、`top5_error_profiles.png`
- exact-calendar 指标：`artifacts/casestudy_hn_offset_decay/visualization_exact/metrics.json`
- mirage 案例与图表：`artifacts/casestudy_hn_offset_decay/mirage_analysis/`
