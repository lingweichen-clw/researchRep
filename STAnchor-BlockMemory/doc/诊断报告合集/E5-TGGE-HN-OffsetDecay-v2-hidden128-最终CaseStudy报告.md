# E5 TGGE HN-OffsetDecay v2（hidden128 + FFN2）：最终 Case Study 实验报告

## 1. 报告范围与结论摘要

本报告评估 `HN-OffsetDecay v2` 预训练编码器是否学习到可检索的 future dynamics 关系，并验证这种关系在候选排序、OffsetDecay 聚合和时空海市蜃楼案例中的表现。模型配置为 `hidden_dim=128`、4 层时空编码器、`FFN multiplier=2`、`retrieval_dim=64`，总参数量为 `958,704`。训练使用 `masked_relation_single_view`：同一个 masked history 前向同时计算 future-relation 损失和 masked reconstruction 损失。

主要结论如下：

1. 预训练完整运行 50 轮，无跳过 batch、NaN 或中途退出；验证总损失在第 41 轮达到最佳 `1.971686`，第 50 轮为 `1.976423`，说明训练已收敛。
2. 在与预训练监督语义一致的 `pretrain_broad_causal` 协议下，trained key 的 Pair Spearman 为 `0.6419`，matched-random 为 `0.0814`；OffsetDecay memory MAE 为 `3.4713`，random 为 `4.7592`。这说明 64 维 key 学到了与 future dynamics 相关的可检索结构。
3. 在部署约束更强的 `exact_calendar` 协议下，trained key 仍优于 random：Spearman `0.3291` 对 `0.1957`，Recall@5 `0.7382` 对 `0.6977`，OffsetDecay memory MAE `3.5106` 对 `3.6641`。绝对增益收缩是候选池变小、随机 Top-5 覆盖率自然升高的预期结果。
4. OffsetDecay 在 broad-causal 中将 trained raw-future memory MAE 从 `3.8413` 降至 `3.4713`；在 exact-calendar 中从 `3.8129` 降至 `3.5106`，说明水平偏移校正能减少候选事件的 level mismatch。
5. 两类时空海市蜃楼案例均由固定分位数规则自动选择。案例图只用于解释机制；整体有效性由全量候选统计、分箱曲线和 matched-random 对照承担。PCA 在全体有限 Bank key 的确定性子样本上拟合，彩色 cluster 点仅显示每类距中心最近的最多 80 个代表点；图例仍报告每类的真实统计规模，不把显示点数冒充总体证据。

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

`pretrain_broad_causal` 要求候选严格早于 query，但不强制相同 weekday 或 slot；每个 query 通过时间分位抽样最多保留 32 个候选。完整验证集包含 2,993 个 query、19,020,445 个有效 query-candidate pairs，平均候选数为 32.0。

| 指标 | HN-OffsetDecay v2 | Matched random | 差值 |
|---|---:|---:|---:|
| Pair Spearman | 0.6419 | 0.0814 | +0.5605 |
| Anchor Spearman | 0.5804 | 0.1083 | +0.4721 |
| Anchor Kendall | 0.4355 | 0.0752 | +0.3603 |
| Recall@1 | 0.1351 | 0.0547 | +0.0804 |
| NDCG@5 | 0.4650 | 0.2647 | +0.2004 |
| Recall@5 | 0.4094 | 0.2224 | +0.1870 |
| OffsetDecay memory MAE | 3.4713 | 4.7592 | -1.2879 |
| OffsetDecay memory RMSE | 6.6101 | 8.2870 | -1.6769 |

Pair Spearman 汇总所有合法 query-candidate 对的单调关系；Anchor Spearman/Kendall 先在每个 query 内计算再平均；Recall@K 衡量 future 近邻是否进入 key 排名前 K；NDCG@5 同时考虑 Top-5 候选的位置质量。trained key 的排名指标和 memory 误差均明显优于同一事件轴上的 random key。

![图 2：全量 key-future 分箱关系](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/report_figures/aggregate_rank_profile.png)

图 2 左侧显示 broad-causal 中 key-distance decile 从近到远增加时，trained future distance 从 `0.478` 上升到约 `1.59`，random 曲线仅在约 `0.76--1.03` 间小幅波动。这个分箱曲线是“key 学到 future dynamics 关系”的主要总体证据；它不要求关系严格线性，但要求整体单调趋势可观察。

![图 3：全量 ranking 对照与绝对增益](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/report_figures/full_validation_ranking_gain.png)

图 3 同时给出 trained/random 的原始分数和绝对增益，避免只展示一侧曲线造成视觉误读。

![图 4：不同预测 horizon 的 memory 误差与增益](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/report_figures/horizon_wise_gain.png)

图 4 显示 trained memory 在 5 分钟到 60 分钟全部 horizon 上低于 random；随着 horizon 变长，绝对误差整体上升，但 trained 相对 random 的优势仍保持。

## 6. Exact-calendar：部署侧候选约束复核

`exact_calendar` 同时要求候选与 query 具有相同 weekday、相同 slot，并满足严格因果约束。完整验证集平均候选数为 `8.014`，范围为 `5--9`。候选池较小会使 random Recall@5 的自然期望升高，因此 exact-calendar 只在同一协议内比较 trained 与 random，不能与 broad-causal 的绝对数值直接比较。

| 指标 | HN-OffsetDecay v2 | Matched random | 差值 |
|---|---:|---:|---:|
| Pair Spearman | 0.3725 | 0.2126 | +0.1599 |
| Anchor Spearman | 0.3291 | 0.1957 | +0.1334 |
| Anchor Kendall | 0.2578 | 0.1509 | +0.1069 |
| Recall@1 | 0.2298 | 0.1897 | +0.0400 |
| NDCG@5 | 0.6354 | 0.5765 | +0.0590 |
| Recall@5 | 0.7382 | 0.6977 | +0.0404 |
| OffsetDecay memory MAE | 3.5106 | 3.6641 | -0.1535 |
| OffsetDecay memory RMSE | 6.6333 | 6.7853 | -0.1520 |

![图 5：exact-calendar 的 key-future 分箱关系](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/report_figures/aggregate_rank_profile.png)

图 5 与图 2 共用布局，右侧是部署约束下的 exact-calendar 结果。trained 曲线在近 key-distance 区间给出更低的 future distance，并随 key distance 增长呈更明显的上升趋势；差距比 broad-causal 小，说明候选过滤是部署侧主要限制因素之一。

## 7. 与 HN-OffsetDecay v1 的同协议对照

v1 和 v2 都使用 METR-LA、`pretrain_broad_causal`、32 候选和 OffsetDecay memory 定义。v1 的 hidden/retrieval dimension 为 `80/48`，v2 为 `128/64`；两者的比较用于描述扩容后的关系学习变化，不替代多 seed 显著性检验。

| 指标（broad-causal） | v1 | v2 | v2 - v1 |
|---|---:|---:|---:|
| Pair Spearman | 0.6212 | 0.6419 | +0.0207 |
| Anchor Spearman | 0.5599 | 0.5804 | +0.0205 |
| Anchor Kendall | 0.4173 | 0.4355 | +0.0182 |
| NDCG@5 | 0.4566 | 0.4650 | +0.0084 |
| Recall@5 | 0.3997 | 0.4094 | +0.0097 |
| OffsetDecay memory MAE | 3.5060 | 3.4713 | -0.0347 |

v2 在该匹配协议下的全局和局部排序指标均有小幅提高，memory MAE 也降低 `0.0347`。这说明扩容后的表示容量至少没有破坏 v1 已验证的机制，并在当前单 seed 上取得有限增益；由于训练配置、随机性和数据划分仍需多 seed 复核，报告不把这些差值表述为统计显著提升。

## 8. Raw future 与 OffsetDecay 聚合

在相同 trained/random 候选和权重下，直接聚合绝对 future 会把事件 level 差异带入 memory；OffsetDecay 先对齐历史末端 level，再聚合动态变化。

| 协议 | trained raw MAE | trained OffsetDecay MAE | 降低 |
|---|---:|---:|---:|
| broad-causal | 3.8413 | 3.4713 | 0.3699 |
| exact-calendar | 3.8129 | 3.5106 | 0.3023 |

该对照只改变 payload alignment，不改变 encoder、候选事件轴或 key 排序，因此可将误差降低归因于 OffsetDecay 的水平校正。对应的案例图保存在两组 visualization 目录下的 `offset_decay_payload_cases.png`，案例选择遵循 strong-win/representative/failure 三个固定增益分位数。

## 9. 时空海市蜃楼案例

**时空海市蜃楼**指 context 相似性与 future 相似性不一致的事件对。案例候选从 Bank 的 5,000 个事件和全部 207 个节点中构造，context/future 在节点内进行 robust 标准化。A 类满足 context distance `<=P8`、future distance `>=P92`、key distance `>=P92`；B 类满足 context distance `>=P92`、future distance `<=P8`、key distance `<=P8`。阈值为：

| 距离 | P8 | P92 |
|---|---:|---:|
| context | 0.3246 | 4.3998 |
| future trend | 0.3976 | 3.0130 |
| key | 0.0452 | 0.1285 |

### 9.1 A 类：context 相似、future 不相似、key 分散

选择规则不是手工挑图：先用 `context <= P8`、`future trend >= P92`、`key >= P92` 定义 A 类，再在合格候选中按三项距离相对该类中位数的标准化距离排序，优先选择类中心附近的样本；随后施加不重复事件和同一节点最多 2 对的约束。本次 3 对分别位于 node 187、144、91，context distance 为 `0.246/0.275/0.279`，future trend distance 为 `3.559/3.622/3.523`，key distance 为 `0.139/0.138/0.134`。因此图中的样本仍满足 A 类的“context 相似、future 趋势不同、key 分散”条件，同时避免极端值造成曲线和 PCA 连线过度拉伸。

![图 6：A 类 mirage 的 3 对 context/future 曲线与局部 key](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/mirage_analysis/context_similar_future_different_cluster.png)

这些样本的完整 24 小时 context 曲线在节点内接近，而 60 分钟 future 轨迹明显分叉；对应 key 点通过连线呈现为相对分离。它们说明只用历史观测相似性会产生潜在错误先例，HN-OffsetDecay 的 key 在部分情况下保留了 future-relevant 的区分信号。

### 9.2 B 类：context 不相似、future 相似、key 集中

B 类也采用相同的类中心优先规则：先用 `context >= P92`、`future trend <= P8`、`key <= P8` 定义候选，再按三项距离到类别中位数的标准化距离排序并施加多样性约束。本次 3 对分别位于 node 176、145、64，context distance 为 `5.709/5.844/5.593`，future trend distance 为 `0.298/0.274/0.320`，key distance 为 `0.0313/0.0311/0.0315`。

![图 7：B 类 mirage 的 3 对 context/future 曲线与局部 key](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/mirage_analysis/context_different_future_similar_cluster.png)

这些样本的完整 24 小时 context 明显不同，但 60 分钟 future 演变趋势接近；key 空间中的点保持集中，说明编码器并非简单复制 context 距离，而是保留了一部分 future-relevant 的等价关系。future 距离计算使用有效观测 mask，三对的有效 future 重叠均为 1.0。

### 9.3 Population key PCA：总体背景与案例高亮

只画 12 个样本会被质疑为选择偏差，因此图 8 在 5,000 个 Bank 事件的有限 event-node keys 中确定性抽取 24,000 点拟合 PCA，背景点使用低透明度浅灰色；此外对 future-trend 聚类的 6 个大类各显示距类中心最近的最多 80 个点，并在图例中标出真实的 `n`。A/B 六对案例的 12 个 key 通过独立案例图展示，PCA 的二维坐标只承担视觉解释，不能替代原始 64 维指标。

![图 8：全体 key 背景与 future-trend cluster 的代表点](../../artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/mirage_analysis/key_pca_population_clusters.png)

图 8 的作用是显示 future-trend 类别在总体 key 空间中的分布，而不是把少量案例点人为拉开。每个 cluster 的统计规模分别为 `164,384`、`130,383`、`184,154`、`225,828`、`152,250` 和 `132,458` 个 event-node 点；彩色点最多显示 80 个/类，仅为可读性的确定性中心邻域抽样。由于时空系统存在不可观测的突发因素，cluster 之间仍有重叠区域，这种重叠被保留而没有被视觉筛除。

## 10. 案例数量与统计证据边界

六对、合计 12 个样本适合作为主文机制图的规模：每类 3 对足以展示重复出现的模式，且图面仍能读清单条曲线和 key 连线。它们不适合独立证明总体有效性；future-trend cluster 的全量统计也不依赖图中显示的几十个点。因此本报告采用三层证据：

1. **全量统计**：19,020,445 个 broad-causal 有效 pairs、4,748,149 个 exact-calendar 有效 pairs，以及完整验证集的 Spearman/Kendall、Recall@K、NDCG@5、Memory MAE/RMSE。
2. **分箱趋势**：全体 key-distance decile 的 future distance 均值，trained 与 random 使用同一坐标轴和同一事件轴。
3. **代表性案例**：按预先固定的 P8/P92 规则、不重复事件约束和类中心优先的确定性排序选择，不按“看起来最好”手工删除失败区域。案例用于机制说明，失败和边界通过 aggregate 指标及固定 strong-win/representative/failure 案例保留。

因此，论文中可以把图 6--8 表述为“representative qualitative evidence”，而把图 2--5 和表格作为“quantitative evidence”。

## 11. 结论、局限与可复现产物

HN-OffsetDecay v2 在 958,704 参数规模下学习到了稳定的 future-relevant key geometry：broad-causal 的整体相关性、局部 Top-K 排序和 OffsetDecay memory 误差均显著优于 matched-random；在 exact-calendar 的部署候选池约束下，优势收缩但方向保持。OffsetDecay 进一步降低了绝对 level mismatch，使候选聚合更接近 query future。

结论不应扩展为“部署阶段可以完美识别所有时空海市蜃楼”。A/B 案例和失败分位数显示，未来仍受突发事件、候选截断和多模态演化影响；当前证据支持“学习到可用但不完美的 future dynamics 检索结构”。本报告当前仅使用 seed 42，未将单 seed 的结果表述为统计显著性结论。

主要产物：

- 预训练日志：`artifacts/metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42/pretrain.log`
- 最优 checkpoint：`artifacts/metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42/pretrain_best.pt`（epoch 41）
- trained/random Bank：`artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42/`、`artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42/`
- broad-causal 指标：`artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/visualization_pretrain_broad_causal/metrics.json`
- exact-calendar 指标：`artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/visualization_exact_calendar/metrics.json`
- 报告图表：`artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/report_figures/`
- mirage 案例、future-trend cluster 统计与 PCA：`artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/mirage_analysis/`

所有上述诊断均为完整验证集运行，未使用 `--max-batches`；mirage 使用 5,000 个 Bank 事件，PCA 背景图的 24,000 点和每类最多 80 个彩色代表点仅为可读性确定性抽样，不影响任何原始指标或 cluster 规模计算。真实 future 只用于离线案例选择、聚类统计与图表评估，不参与部署阶段检索。
