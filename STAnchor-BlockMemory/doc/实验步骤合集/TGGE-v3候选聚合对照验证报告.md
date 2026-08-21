# TGGE v3 候选聚合对照验证报告

## 1. 验证目的

本实验判断当前 `Y_memory` 的固定加权聚合是否是下游平台期的主要原因。实验不训练任何新参数，只固定已经训练好的 base、TGGE 编码器、Memory Bank、候选集合和检索权重，仅替换候选 future 的聚合规则。

因此，本实验回答的是：

> 在相同候选池和相同检索权重下，简单改变聚合公式，是否能显著改善 Memory 预测？

如果 Top-1、Top-3、trimmed mean 或 sign-cluster 明显优于 weighted mean，说明聚合公式是主要瓶颈；如果都没有改善，则需要改为 horizon-specific candidate weighting 或重新审查候选排序和 payload。

## 2. 固定实验条件

- 数据集：METR-LA；
- 验证划分：validation；
- 下游 base：STGCN 已训练 checkpoint；
- TGGE encoder：single-view v3 reconstruction2；
- Memory Bank：同一 Bank；
- event Top-R：32；
- node Top-K：5；
- candidate protocol：`exact_calendar`；
- level weight：0；
- seed：42；
- 当前 learned gate checkpoint：v7 STGCN checkpoint；
- 真实 future：只用于验证指标，不用于候选选择或聚合权重。

## 3. 候选与聚合定义

候选 future 保留为：

$$
Y^{cand}\in\mathbb{R}^{B\times H\times N\times K\times C}.
$$

其中 B 是 batch 大小，H=12 是预测 horizon，N=207 是节点数，K=5 是候选数，C=1 是通道数。

当前 weighted mean 为：

$$
Y^{mean}_{q,h,n,c}=\frac{\sum_k w_{q,n,k}m_{q,h,n,k,c}Y^{cand}_{q,h,n,k,c}}{\sum_k w_{q,n,k}m_{q,h,n,k,c}}.
$$

对照方法为：

1. `memory_weighted_mean`：当前检索权重加权平均；
2. `memory_top1`：直接使用检索分数最高的候选；
3. `memory_top3`：只对检索分数最高的 3 个候选重新归一化加权平均；
4. `memory_trimmed`：去掉最低权重候选后，对剩余候选加权平均；
5. `memory_sign_cluster`：按候选相对 base 的修正方向分为正向和负向簇，选择总权重较大的方向簇进行聚合。

这些方法都不使用当前 query 的真实 future，因此属于部署可用的固定聚合对照。

## 4. 结果

| 方法 | MAE | RMSE | Helpful rate |
|---|---:|---:|---:|
| Base | 3.037813 | 6.235178 | — |
| Current learned gate | 2.959441 | 5.968418 | — |
| Weighted mean Memory | 3.554837 | 6.636025 | 43.80% |
| Top-1 Memory | 4.088585 | 7.880428 | 38.09% |
| Top-3 Memory | 3.645953 | 6.856247 | 43.05% |
| Trimmed mean Memory | 3.584059 | 6.709984 | 43.59% |
| Sign-cluster Memory | 4.016904 | 7.321302 | 37.39% |

这里的 Memory MAE 是直接把聚合结果作为预测值与真实 future 比较；current learned gate 是当前冻结 base 加校正器后的最终输出，两者不是同一个方法层级，但可以用于判断聚合结果的质量和校正器是否能弥补聚合误差。

## 5. 结果解释

### 5.1 简单聚合变体都没有改善

当前 weighted mean 的 Memory MAE 为 3.554837。Top-3、trimmed mean 和 sign-cluster 分别为 3.645953、3.584059 和 4.016904，均差于 weighted mean；Top-1 最差，为 4.088585。

因此不能得出“只要把 weighted mean 改成 Top-1、Top-3 或 sign-cluster 就能解决平台期”的结论。

### 5.2 检索排序并不等于 future 质量排序

Top-1 比 weighted mean 更差，说明最高 latent 相似度候选不一定拥有最准确的 future payload。相似度 key 更接近历史状态语义，而不是直接等价于未来误差最小。

这也解释了 oracle candidate top-1 约 1.66 与部署可用 top-1 约 4.09 之间的巨大差距：oracle top-1 使用真实 future 选择候选，部署 top-1 只使用检索分数，两者不能混淆。

### 5.3 方向聚类不能直接作为最终聚合

sign-cluster 的 helpful rate 只有 37.39%，且 MAE 为 4.016904，说明简单按相对 base 的正负方向聚类会丢失 level、幅度和候选质量信息。方向符号可以作为诊断或重打分特征，但不适合直接决定唯一 Memory 输出。

### 5.4 当前主要瓶颈不是“均值公式”本身

weighted mean、Top-3、trimmed mean 的结果都处于 3.55–3.65，简单替换聚合规则没有带来收益；而 oracle candidate 仍约 1.64–1.66。这说明候选池中存在有效 future，但部署时可获得的检索分数不足以选择正确候选，且固定聚合无法根据 horizon 区分候选质量。

## 6. 当前结论

本实验不支持直接采用 Top-1、Top-3、trimmed mean 或 sign-cluster 替换当前 weighted mean。

更准确的结论是：

> 聚合阶段确实存在结构性瓶颈，但问题不是简单的“平均公式不对”，而是当前候选权重 `w_{q,n,k}` 与 horizon 无关，也没有根据候选 future 的部署可见形状进行重新打分。

因此下一步应优先验证：

$$
 w_{q,n,k}\rightarrow w_{q,h,n,k},
$$

即 horizon-specific candidate weighting。可以使用一个很小的候选重打分器，根据 retrieval similarity、candidate offset、offset magnitude、candidate trend、payload dispersion、base risk 和 horizon position 生成 horizon-specific bias，再重新 softmax 聚合。

## 7. 是否需要训练

本次固定聚合对照不需要训练，因为它的目的就是隔离聚合公式本身。下一阶段 horizon-specific weighting 才需要训练一个小型 candidate reranker；该模块应先在固定 base 和固定 Bank 上训练，不重新训练 TGGE encoder，不重新训练 Graph WaveNet 或 STGCN backbone。

## 8. 产物与未来信息边界

正式对照结果保存在：

`artifacts/convergence/aggregation_comparison_v1/stgcn_val.json`

该 JSON 只包含验证集诊断结果，真实 future 仅用于评价聚合结果，不参与候选检索、候选权重或部署前向。临时输出已清理。

## 9. Keep/Remove/Stop

- Remove：不采用直接 Top-1、Top-3、trimmed mean、sign-cluster 替换 weighted mean；
- Keep：保留当前 weighted mean 作为聚合基线；
- Next：验证 horizon-specific candidate reranking；
- Stop：如果 horizon-specific reranking 仍不能明显改善，则停止继续堆叠聚合器，转向检索 key 与 future payload 对齐问题。
