# 统一 OffsetDecay 候选载荷的检索选择器消融方案

## 一、问题与最终决策

前期 GWN 对照同时改变了候选排序方法与候选 future 载荷，因而不能把下游 MAE 差异单独归因于检索选择器。当前研究已确定保留 OffsetDecay，不再扩展或展示 raw-future 载荷实验。

最终只比较三种候选选择器，并强制使用完全相同的 OffsetDecay future 载荷：

1. `learned_key`：HN-OffsetDecay v2 的 64 维节点 key 距离排序；
2. `raw_l1`：同一合法事件池内的 288 步节点历史 L1 距离排序；
3. `matched_random`：架构、Bank 事件轴和初始化 seed 匹配的随机编码器 key 排序。

该设计只回答一个问题：在候选协议、Top-K、载荷和 Router 全部一致时，学习得到的 key 是否提供优于非学习历史距离与随机表示的候选顺序。

## 二、统一实验口径

| 项目 | 固定设置 |
|---|---|
| 数据集 | METR-LA |
| 候选协议 | `weekday_radius1_overlap` |
| 事件候选上限 | 32 |
| 节点 Top-K | 12 |
| 候选载荷 | OffsetDecay future |
| Router | Retrieval-Aware MHA Residual Router |
| 损失空间 | physical MAE |
| batch size | 32 |
| 学习率 | 0.0005 |
| epoch | 50 |
| 缓存 | `frozen_path_cache=true` |

Raw-L1 仅替换候选排序距离，不改变合法事件池、候选 future、OffsetDecay、Router 输入字段、训练目标或缓存协议。

## 三、OffsetDecay 定义与信息边界

对 query 节点历史末端水平 `L_q`、候选历史末端水平 `L_j` 和候选未来 `Y_j(h)`，载荷为：

$$
\widetilde{Y}_j(h)=Y_j(h)+\lambda_h(L_q-L_j),
\qquad
\lambda_h=1-\frac{h}{H-1}.
$$

其中 `h=0,...,H-1`，`H=12`。近端 horizon 更强地对齐 query 当前水平，远端逐渐恢复候选自身演变。候选 future 均来自历史 Bank；query 真实 future 只用于训练标签和离线诊断指标，不参与候选排序、OffsetDecay 或部署推理。

## 四、Case Study 指标

在 `pretrain_broad_causal` 与 `weekday_radius1_overlap` 两个协议上，对三个选择器使用同一合法候选轴、同一有效 mask 和同一 OffsetDecay 载荷，报告：

- Pair Spearman：汇总所有合法 query-node-candidate 对，衡量选择器距离与 teacher future 距离的全局单调一致性；
- Anchor Spearman / Kendall：先在每个 `(query,node)` 的候选池内计算排序一致性，再跨 anchor 平均；
- Recall@1、NDCG@5、Recall@5：衡量选择器能否找回 teacher 定义的 future 近邻；
- OffsetDecay Memory MAE / RMSE / MAPE：三种选择器的 Top-12 候选均经 OffsetDecay 后等权聚合，再与 query future 比较。

排序指标评估 selector；Memory 指标同时反映 selector 与统一载荷形成的直接历史先例质量。OffsetDecay 不改变候选排名，因此不把排序指标表述为 OffsetDecay 的收益。

## 五、实现范围

- `stanchor/diagnostics/retrieval_visualization.py`：加入 Raw-L1 全候选排序统计；三种选择器统一调用 `offset_decay_aggregation`；移除新产物中的 raw-future Memory 与载荷案例图。
- `configs/ablation_rawl1_offset_decay_router_gwn.yaml`：GWN 的 Raw-L1 + OffsetDecay 下游对照。
- `configs/ablation_rawl1_offset_decay_router_argcn.yaml`：ARGCN 的 Raw-L1 + OffsetDecay 下游对照。
- `doc/诊断报告合集/E5-TGGE-HN-OffsetDecay-v2-hidden128-最终CaseStudy报告.md`：更新为三选择器、统一 OffsetDecay 的统计口径。

底层 `candidate_payload=raw_future` 仅为读取历史配置和检查点保留兼容性；不创建新的 raw-future 训练配置，不把旧 raw-future 指标写入最终 Case Study。

## 六、判定规则

- 若 learned selector 的排名指标和 OffsetDecay Memory 指标总体优于 matched-random，则支持“预训练学到了 future-relevant key geometry”。
- 若 learned 与 Raw-L1 的下游单 seed MAE 差异绝对值不超过 `0.01`，按工程预设视为近似持平，不宣称稳定优胜。
- 若 Raw-L1 在个别 backbone/seed 更优，保留该结果并将其解释为 Router 对候选载荷利用方式的差异，不能据此否定已由直接排序指标验证的编码器质量。
- 完成 GWN 与 ARGCN 的既定补充后停止扩展，不更改已定稿的编码器、Router、Bank、候选协议或缓存机制。
