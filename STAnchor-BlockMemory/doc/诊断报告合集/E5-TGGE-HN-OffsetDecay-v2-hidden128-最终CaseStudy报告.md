# E5 TGGE HN-OffsetDecay v2（hidden128 + FFN2）：最终 Case Study 实验报告

## 1. 报告范围与结论摘要

本报告评估 `HN-OffsetDecay v2` 预训练编码器是否学习到可检索的 future dynamics 关系，并验证这种关系在候选排序、OffsetDecay 聚合和时空海市蜃楼案例中的表现。模型配置为 `hidden_dim=128`、4 层时空编码器、`FFN multiplier=2`、`retrieval_dim=64`，总参数量为 `958,704`。训练使用 `masked_relation_single_view`：同一个 masked history 前向同时计算 future-relation 损失和 masked reconstruction 损失。

主要结论如下：

1. 预训练完整运行 50 轮，无跳过 batch、NaN 或中途退出；验证总损失在第 41 轮达到最佳 `1.971686`，第 50 轮为 `1.976423`，说明训练已收敛。
2. 在与预训练监督语义一致的 `pretrain_broad_causal` 协议下，trained key 的 Pair Spearman 为 `0.6693`，matched-random 为 `0.0611`；OffsetDecay memory MAE 为 `3.2033`，random 为 `4.0984`。这说明 64 维 key 学到了与 future dynamics 相关的可检索结构。
3. 在当前部署侧 `weekday_radius1_overlap` 协议下，候选事件池平均为 `23.98` 个（范围 `19--27`），trained key 的 Pair Spearman 为 `0.4399`，matched-random 为 `0.1337`；OffsetDecay memory MAE 为 `3.2215`，random 为 `3.5674`。候选池相较旧的单 weekday 协议明显扩大，但排序增益仍保持。
4. 在两个协议中都加入独立的 `Raw-L1 Top-12` 对照。该基线使用同一合法事件池、masked 288-step context、节点级原始 L1 排序和原始 future 等权聚合，不使用 learned key、learned weight 或 OffsetDecay；其 memory MAE 分别为 broad-causal `5.2695`、weekday-radius `4.3296`，明显高于 learned retrieval。
5. OffsetDecay 在 learned 候选上继续降低绝对 level mismatch：broad-causal 的 raw-future MAE `3.5344` 降至 `3.2033`，weekday-radius 的 raw-future MAE `3.4351` 降至 `3.2215`。两类时空海市蜃楼案例均由固定分位数规则自动选择；整体有效性由全量候选统计、分箱曲线、Raw-L1 和 matched-random 对照承担。

## 2. 数据、张量与信息边界

METR-LA 包含 `N=207` 个传感器，采样间隔为 5 分钟。检索历史窗口为 288 步（24 小时），未来窗口为 12 步（60 分钟）：

$$
X^{hist}\in\mathbb{R}^{B\times288\times207\times1},\qquad
Y^{future}\in\mathbb{R}^{B\times12\times207\times1}.
$$

其中 `B` 是 batch size，最后一维是速度通道。编码器输出节点 key：

$$
K=f_\theta(X^{hist},G)\in\mathbb{R}^{B\times207\times64},
$$

`K[q,n]` 表示 query `q` 在节点 `n` 上的 64 维检索表示。

本报告中使用三个距离：

- **context distance**：两个事件完整 288 步（24 小时）观测历史的节点内 robust 标准化 RMS 距离。绘图时按小时聚合为 24 个点以便阅读，但距离计算使用完整 288 步且只在两事件共同有效的观测上计算。
- **future distance**：两个事件 12 步（60 分钟）future trend signature 的 RMS 距离。signature 从各自 future 的首个有效水平开始中心化，并按自身时间标准差归一化，因此比较的是演变趋势而不是绝对速度水平；缺失位置不参与距离。
- **key distance**：原始 64 维 node key 的欧氏距离；PCA 只用于可视化，所有指标仍在原始 64 维空间计算。

训练阶段允许使用真实 future 构造 relation teacher；验证阶段真实 future 只用于离线指标、分位数案例选择和绘图。检索排序、候选筛选和 Bank key 构建只使用 query history、calendar metadata、causal metadata、历史 Bank keys 和历史 level。部署阶段不能访问 query future。

## 3. 模型与训练配置

| 项目 | 设置 |
|---|---:|
| dataset / split | METR-LA / train-val-test |
| train/val/test samples | 22,681 / 2,993 / 6,025 |
| retrieval context / forecast context | 288 / 12 steps |
| horizon / sampling | 12 steps / 5 minutes |
| hidden / retrieval dimension | 128 / 64 |
| encoder layers / heads | 4 / 4 |
| FFN multiplier | 2 |
| route | enabled, top-k=6, local quota=0 |
| objective | `masked_relation_single_view` |
| retrieval loss | `hard_negative_offset_decay` |
| reconstruction/retrieval weight | 2.0 / 1.0 |
| batch size / epochs | 16 / 50 |
| trainable parameters | 958,704 |

### 3.1 目标函数

联合目标为：

$$
\mathcal{L}=\mathcal{L}_{retrieval}+2.0\,\mathcal{L}_{reconstruction}.
$$

`L_retrieval` 使用 OffsetDecay future relation teacher，同时提高 context 相似但 future 差异大的 hard negative 权重。`L_reconstruction` 只在被 mask 且原始观测有效的位置计算，用于防止 masked encoder 依赖局部可见值的简单捷径；它不是最终 forecasting loss。

### 3.2 OffsetDecay

对候选事件 `j` 在 horizon `h` 的 future 值做水平对齐：

$$
\widetilde{Y}_j(h)=Y_j(h)+\lambda_h(L_q-L_j),
\qquad
\lambda_h=1-\frac{h}{H-1}.
$$

`L_q` 和 `L_j` 是 query 与候选历史窗口末端的节点 level。近端 horizon 的 level 校正较强，远端 horizon 逐渐衰减为零。真实 query future 不参与该推理时的校正。

## 4. 训练收敛与候选支持

![图 1：v2 预训练收敛](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/report_figures/training_convergence.png)

图 1 左侧显示 relation train loss 从约 `1.77` 降至 `1.45`，validation relation loss 从 `1.68` 降至约 `1.52`；中间两图显示 validation reconstruction 从 `0.318` 降至约 `0.23`。最佳 validation total 出现在 epoch 41，而不是最后一轮，符合固定 50 轮训练下的收敛波动。

Teacher effective support 是 teacher 分布的有效候选数，v2 验证阶段约为 `3.787`；student effective support 从第 1 轮约 `7.66` 降至第 50 轮 `5.54`。student support 高于 teacher support，表示模型逐步集中到更有用的候选，但没有塌缩为单一候选。

## 5. Broad-causal：与预训练语义一致的主结果

`pretrain_broad_causal` 要求候选严格早于 query，但不强制相同 weekday 或 slot；每个 query 从合法历史事件中按时间分位抽样，最多保留 96 个候选。完整验证集包含 2,993 个 query、56,694,054 个有效 query-candidate pairs，事件候选池平均为 96.0。

| 指标 | HN-OffsetDecay v2 | Matched random | 差值 |
|---|---:|---:|---:|
| Pair Spearman | 0.6693 | 0.0611 | +0.6083 |
| Anchor Spearman | 0.5989 | 0.0926 | +0.5063 |
| Anchor Kendall | 0.4443 | 0.0625 | +0.3817 |
| Recall@1 | 0.0661 | 0.0289 | +0.0373 |
| NDCG@5 | 0.3408 | 0.2013 | +0.1396 |
| Recall@5 | 0.2131 | 0.1052 | +0.1080 |
| OffsetDecay memory MAE | 3.2033 | 4.0984 | -0.8952 |
| OffsetDecay memory RMSE | 6.2482 | 7.4150 | -1.1669 |
| Raw-L1 Top-12 memory MAE | 5.2695 | — | — |

Pair Spearman 汇总所有合法 query-candidate 对的单调关系；Anchor Spearman/Kendall 先在每个 query 内计算再平均；Recall@K 衡量 future 近邻是否进入 key 排名前 K；NDCG@5 同时考虑 Top-5 候选的位置质量。trained key 的排名指标和 memory 误差均明显优于同一事件轴上的 random key。

![图 2：broad-causal 全量 key-future 分箱关系](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/visualization_pretrain_broad_causal/key_future_alignment.png)

图 2 显示 broad-causal 中 key-distance decile 从近到远增加时，trained future distance 从 `0.488` 上升到约 `1.690`，random 曲线仅在约 `0.784--0.996` 间波动。这个分箱曲线是“key 学到 future dynamics 关系”的主要总体证据；它不要求关系严格线性，但要求整体单调趋势可观察。

![图 3：broad-causal 全量 ranking 对照与绝对增益](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/visualization_pretrain_broad_causal/ranking_metrics.png)

图 3 同时给出 trained/random 的原始分数和绝对增益，避免只展示一侧曲线造成视觉误读。

![图 4：broad-causal Top-12 候选误差 profile](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/visualization_pretrain_broad_causal/top5_error_profiles.png)

图 4 展示固定 strong-win、representative 和 failure 三类 anchor 中，12 个 retrieved rank 的候选误差与聚合误差。它同时给出 Raw-L1、learned 和 matched-random 曲线，避免把单一 rank 或单一案例误认为总体结论。

## 6. Weekday-radius：当前部署侧候选协议

`weekday_radius1_overlap` 要求候选与 query 具有相同日内 slot、weekday 差不超过 1，并满足候选事件的 context end 不晚于 query context end；允许 288-step context 窗口重叠，不再额外去重。完整验证集事件候选池平均为 `23.984`，范围为 `19--27`；节点级有效候选数平均为 `23.494`。该协议正是“query 前后相邻一天的同一时段”扩展，Raw-L1 与 learned/random 使用完全相同的事件轴。

| 指标 | HN-OffsetDecay v2 | Matched random | 差值 |
|---|---:|---:|---:|
| Pair Spearman | 0.4399 | 0.1337 | +0.3063 |
| Anchor Spearman | 0.3994 | 0.1167 | +0.2828 |
| Anchor Kendall | 0.2951 | 0.0828 | +0.2123 |
| Recall@1 | 0.1061 | 0.0754 | +0.0307 |
| NDCG@5 | 0.3947 | 0.3062 | +0.0884 |
| Recall@5 | 0.3660 | 0.2771 | +0.0890 |
| OffsetDecay memory MAE | 3.2215 | 3.5674 | -0.3459 |
| OffsetDecay memory RMSE | 6.1163 | 6.5171 | -0.4009 |
| Raw-L1 Top-12 memory MAE | 4.3296 | — | — |

![图 5：weekday-radius 的 key-future 分箱关系](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/visualization_weekday_radius1_overlap/key_future_alignment.png)

图 5 与图 2 共用布局，展示当前部署候选约束下的结果。trained 曲线在近 key-distance 区间给出更低的 future distance，并随 key distance 增长呈明显上升趋势；相较 broad-causal，绝对相关性收缩是候选范围和 weekday 约束共同作用的结果，但 trained/random 方向保持一致。

![图 6：weekday-radius ranking 对照](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/visualization_weekday_radius1_overlap/ranking_metrics.png)

![图 7：weekday-radius Top-12 候选误差 profile](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/visualization_weekday_radius1_overlap/top5_error_profiles.png)

## 7. Raw future、Raw-L1 与 OffsetDecay 聚合

在相同 trained/random 候选和权重下，直接聚合绝对 future 会把事件 level 差异带入 memory；OffsetDecay 先对齐历史末端 level，再聚合动态变化。

| 协议 | learned raw-future MAE | learned OffsetDecay MAE | OffsetDecay 降低 | Raw-L1 Top-12 MAE | matched-random raw MAE |
|---|---:|---:|---:|---:|---:|
| pretrain_broad_causal | 3.5344 | 3.2033 | 0.3312 | 5.2695 | 4.5092 |
| weekday_radius1_overlap | 3.4351 | 3.2215 | 0.2136 | 4.3296 | 3.7843 |

这里的 `Raw-L1 Top-12` 是独立的非学习检索基线：它只在共享合法事件池中按节点 raw context L1 排序，并对原始 Bank future 等权平均；它不使用 learned key、learned attention、OffsetDecay 或真实 query future。表中的 learned raw-future 与 OffsetDecay 只改变 payload alignment，候选事件轴和 learned key 排序保持不变。

该对照只改变 payload alignment，不改变 encoder、候选事件轴或 key 排序，因此可将误差降低归因于 OffsetDecay 的水平校正。对应的案例图保存在两组 visualization 目录下的 `offset_decay_payload_cases.png`，案例选择遵循 strong-win/representative/failure 三个固定增益分位数。

## 8. 时空海市蜃楼案例

**时空海市蜃楼**指 context 相似性与 future 相似性不一致的事件对。案例候选从 Bank 的 5,000 个事件和全部 207 个节点中构造，context/future 在节点内进行 robust 标准化。A 类满足 context distance `<=P8`、future distance `>=P92`、key distance `>=P92`；B 类满足 context distance `>=P92`、future distance `<=P8`、key distance `<=P8`。阈值为：

| 距离 | P8 | P92 |
|---|---:|---:|
| context | 0.3240 | 4.4118 |
| future trend | 0.3978 | 3.0270 |
| key | 0.0452 | 0.1285 |

### 8.1 A 类：context 相似、future 不相似、key 分散

选择规则不是手工挑图：先用 `context <= P8`、`future trend >= P92`、`key >= P92` 定义 A 类，再在合格候选中按三项距离相对该类中位数的标准化距离排序，优先选择类中心附近的样本；随后施加不重复事件和同一节点最多 2 对的约束。本次 3 对分别位于 node 120、191、155，context distance 为 `0.279/0.285/0.258`，future trend distance 为 `3.487/3.467/3.580`，key distance 为 `0.140/0.137/0.143`。因此图中的样本仍满足 A 类的“context 相似、future 趋势不同、key 分散”条件，同时避免极端值造成曲线和 PCA 连线过度拉伸。

![图 8：A 类 mirage 的 3 对 context/future 曲线与局部 key](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/mirage_analysis/context_similar_future_different_cluster.png)

这些样本的完整 24 小时 context 曲线在节点内接近，而 60 分钟 future 轨迹明显分叉；对应 key 点通过连线呈现为相对分离。它们说明只用历史观测相似性会产生潜在错误先例，HN-OffsetDecay 的 key 在部分情况下保留了 future-relevant 的区分信号。

### 8.2 B 类：context 不相似、future 相似、key 集中

B 类先用 `context >= P92`、`future trend <= P8`、`key <= P8` 定义候选，再在固定 B 类内部按 key distance 从小到大排序，施加每个节点最多一对的多样性约束。本次 3 对分别位于 node 67、119、9，context distance 为 `4.442/6.817/6.682`，future trend distance 为 `0.334/0.249/0.290`，key distance 为 `0.0089/0.0127/0.0129`；因此三对在原始 cosine-distance 定义下都更紧凑，且仍满足 B 类的 future 相似条件。

![图 9：B 类 mirage 的 3 对 context/future 曲线与局部 key](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/mirage_analysis/context_different_future_similar_cluster.png)

这些样本的完整 24 小时 context 明显不同，但 60 分钟 future 演变趋势接近；key 空间中的点保持集中，说明编码器并非简单复制 context 距离，而是保留了一部分 future-relevant 的等价关系。future 距离计算使用有效观测 mask，三对的有效 future 重叠均为 1.0。

### 8.3 Population key PCA：总体背景与案例高亮

只画 12 个样本会被质疑为选择偏差，因此图 10 在 5,000 个 Bank 事件的有限 event-node keys 中确定性抽取 24,000 点拟合 PCA，左侧保留全量背景分布，右侧给出彩色核心区域的局部放大视图。future trajectory 先在原始 future 空间划分为 32 个趋势簇；随后在每个趋势簇内搜索 PCA 空间的紧凑真实邻域，要求显示点的 pairwise future-trend cosine 不低于 `0.80`，并以圆形边界约束区域之间不重叠。最终保留 6 个满足条件的核心区域，每个区域显示 96 个真实 event-node key 点，而不是显示簇中心或人为移动点。A/B 六对案例的 12 个 key 通过独立案例图展示，PCA 的二维坐标只承担视觉解释，不能替代原始 64 维指标。

![图 10：全体 key 背景与 future-trend cluster 的代表点](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/mirage_analysis/key_pca_population_clusters.png)

图 10 的作用是同时显示总体分布和局部核心结构，而不是把少量案例点人为拉开。右侧 6 个核心区域分别来自 future-trend cluster `4/31/23/19/6/3`，每个区域均显示 96 个真实点；圆形边界半径为该区域内点到中心的最大 PCA 距离，并由贪心约束保证任意两个边界不相交。6 个区域的 within-future cosine 为 `0.982/0.959/0.931/0.899/0.874/0.868`，按显示点数加权的总体相似度为 `0.9188`，最低区域仍为 `0.8682`。左侧背景和右侧核心均使用原始 PCA 坐标，右侧只是坐标范围放大，不进行平移或类别强制分离；未入选的 32 簇及其完整统计保存在 `trend_cluster_summary.csv` 中。

需要区分展示统计与全量统计：对全部 32 个 future-trend 簇按簇内样本对数加权后，within-future cosine 为 `0.6059`（989,457 个有效 event-node 点）。因此 `0.9188` 只表示论文图中 6 个高一致性核心的总体相似度，不能外推为整个 Bank 的聚类相似度；全量值和 6 个核心值同时保存在 `mirage_cases.json` 中。

## 9. 案例数量与统计证据边界

六对、合计 12 个样本适合作为主文机制图的规模：每类 3 对足以展示重复出现的模式，且图面仍能读清单条曲线和 key 连线。它们不适合独立证明总体有效性；future-trend cluster 的全量统计也不依赖图中显示的几十个点。因此本报告采用三层证据：

1. **全量统计**：56,694,054 个 broad-causal 有效 pairs、14,205,552 个 weekday-radius 有效 pairs，以及完整验证集的 Spearman/Kendall、Recall@K、NDCG@5、Memory MAE/RMSE 和 Raw-L1 MAE。
2. **分箱趋势**：全体 key-distance decile 的 future distance 均值，trained 与 random 使用同一坐标轴和同一事件轴。
3. **代表性案例**：按预先固定的 P8/P92 规则、不重复事件约束和类中心优先的确定性排序选择，不按“看起来最好”手工删除失败区域。案例用于机制说明，失败和边界通过 aggregate 指标及固定 strong-win/representative/failure 案例保留。

因此，论文中可以把图 8--10 表述为“representative qualitative evidence”，而把图 2--7 和表格作为“quantitative evidence”。

## 10. 结论、局限与可复现产物

HN-OffsetDecay v2 在 958,704 参数规模下学习到了稳定的 future-relevant key geometry：在 broad-causal 和当前 weekday-radius 部署协议中，整体相关性、局部 Top-K 排序和 OffsetDecay memory 误差均优于 matched-random；与 Raw-L1 非学习基线相比，learned retrieval 也明显更低。OffsetDecay 进一步降低了绝对 level mismatch，使候选聚合更接近 query future。

结论不应扩展为“部署阶段可以完美识别所有时空海市蜃楼”。A/B 案例和失败分位数显示，未来仍受突发事件、候选截断和多模态演化影响；当前证据支持“学习到可用但不完美的 future dynamics 检索结构”。本报告当前仅使用 seed 42，未将单 seed 的结果表述为统计显著性结论。

主要产物：

- 预训练日志：`artifacts/metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42/pretrain.log`
- 最优 checkpoint：`artifacts/metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42/pretrain_best.pt`（epoch 41）
- trained/random Bank：`artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42/`、`artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42/`
- broad-causal 指标：`artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/visualization_pretrain_broad_causal/metrics.json`
- weekday-radius 指标：`artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/visualization_weekday_radius1_overlap/metrics.json`
- 报告图表：`artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/report_figures/`
- mirage 案例、future-trend cluster 统计、PCA 核心区域与总体 future 相似度：`artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/mirage_analysis/`

所有上述诊断均为完整验证集运行，未使用 `--max-batches`；mirage 使用 5,000 个 Bank 事件，PCA 拟合背景为 24,000 点，展示核心为 6 个互不重叠区域、每区 96 个真实点。核心区域只是可读性的确定性子集，不影响原始检索指标或完整 cluster 规模计算；整体 future 相似度及区域半径记录在 `mirage_cases.json` 和 `key_pca_core_regions.csv` 中。真实 future 只用于离线案例选择、聚类统计与图表评估，不参与部署阶段检索。
