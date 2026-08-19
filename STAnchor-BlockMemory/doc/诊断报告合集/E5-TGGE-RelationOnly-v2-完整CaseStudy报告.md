# E5 TGGE Relation-only v2：完整检索 Case Study 报告

## 1. 报告范围与结论

本报告独立说明 `TGGE Relation-only v2` 在 METR-LA 验证集上的检索能力。这里的 **key** 是编码器将 288 个历史时间步压缩为每个节点 48 维向量的结果；它只用于在历史 Bank 中检索相似事件，不直接进入下游预测器。本报告同时给出一个 matched-random 对照：随机初始化编码器与训练模型使用相同的事件轴、未来 payload、候选规则和聚合代码，因此差异归因于 key 学到的排序信号，而不是 Bank 内容或候选数量。

结论先行：

1. 在与预训练语义一致的 `pretrain_broad_causal` 协议下，Relation-only 明显优于 matched-random：Anchor-wise Spearman `0.5677` 对 `0.1412`，Kendall `0.4267` 对 `0.0984`，NDCG@5 `0.4376` 对 `0.2644`，OffsetDecay Memory MAE `3.5752` 对 `4.8050`。
2. 在更接近部署的 `exact_calendar` 协议下，优势收缩但方向保持一致：Spearman `0.3046` 对 `0.1877`，Kendall `0.2376` 对 `0.1444`，NDCG@5 `0.6257` 对 `0.5720`，Memory MAE `3.5357` 对 `3.6735`。
3. Relation-only 是有效的、低假设的检索基线，但不能凭一次 seed 宣称最终优于 v3。当前 v3 在两种协议的排序指标都略高，Relation-only 仅在 exact-calendar 的 Memory MAE 上低 `0.0032`；两者路由配额不同，跨模型差异属于受控程度有限的比较。
4. 失败案例被按固定规则保留。图表采用局部纵轴、gain 图和 horizon-wise 图增强可读性，但没有修改候选池或删除不利样本。

## 2. 实验对象、数据和张量契约

### 2.1 数据与切分

实验使用 METR-LA，节点数 `N=207`，采样间隔 5 分钟。每个样本包含：

- 检索历史 `x^hist: [B, 288, 207, 1]`，即过去 24 小时；
- 预测未来 `y^future: [B, 12, 207, 1]`，即未来 60 分钟；
- 固定图 `G=(V,E)`，日志中一阶图边数为 1,722；
- 训练/验证/测试时间步为 `22,681/2,993/6,025`。

`B` 是 batch size，`N` 是节点数，`T=288` 是历史长度，`H=12` 是预测 horizon，最后一维是速度通道。归一化统计量只由当前历史窗口的可观测值计算。训练和 Bank 构建遵守时间切分，验证 future 不会进入 key 或候选构造。

### 2.2 Relation-only 模型

编码器计算：

\[
K=f_\theta(x^{hist},G)\in\mathbb{R}^{B\times N\times48}.
\]

`K[b,n]` 是第 `b` 个历史窗口中节点 `n` 的检索 key。训练目标只有 future-relation loss：同一 batch 内，根据真实 future 的 OffsetDecay teacher 距离形成软排序，优化 key 距离与 teacher 排序的一致性。没有 reconstruction 分支，因而 `reconstruction_weight=0`。

正式配置为 `configs/metrla_e5_tgge_latent48_relation_only_v2.yaml`：

| 项目 | 设置 |
|---|---:|
| objective | `relation_only` |
| retrieval dimension | 48 |
| hidden dimension | 80 |
| encoder layers / heads | 3 / 4 |
| route top-k | 10 |
| route local quota | 4 |
| retrieval weight | 1.0 |
| reconstruction weight | 0 |
| batch size / maximum epochs | 16 / 50 |
| seed | 42 |
| total/trainable parameters | 303,727 |

该版本的 route 配额是 `4 local + 6 remote`：路由分支对一阶邻居最多取 4 个，对非一阶候选取 6 个。它是历史主线 v2 的实现，不应与 v3 的 `route_top_k=6, local_quota=0` 当成完全相同的图模块。

### 2.3 训练收敛

训练在 epoch 34 触发 patience=10 的 early stopping；按 relation 验证损失保存的 checkpoint 为 epoch 14，最佳 `val_retrieval=2.189446`。最终 epoch 的日志为 `val_retrieval=2.197626`，说明后期已进入窄幅波动区间。验证间隔为 2 个 epoch，所以训练曲线中的空缺是“该 epoch 按配置未执行验证”，不是把验证结果错误地连成一条线。

![Relation-only training convergence](../../artifacts/convergence/visualization/tgge_relation_only_v2/report_figures/training_convergence.png)

图 1 展示三件事：训练 relation loss 从约 2.47 降到约 2.31；验证 relation loss 在 epoch 14 达到最低后围绕 2.19--2.20 波动；student effective support 从约 8.5 降至约 7.8，但仍高于 teacher support 约 3.8。这里的 effective support 是温度 softmax 后候选权重的有效支撑数，不是实际候选数量；它反映排序分布是否过于平均。

## 3. 预训练目标与 OffsetDecay teacher

### 3.1 Future-relation loss

对一个 anchor `(q,n)`，`q` 是 query 历史窗口，`n` 是目标节点。Bank 候选 `j` 的 key 距离为

\[
d^{key}_{qjn}=\lVert K_q(n)-K_j(n)\rVert_2.
\]

真实 future 只在训练标签阶段计算 teacher 距离 `d^{teacher}_{qjn}`，再以温度 `\tau=0.1` 转为软分布。relation loss 约束 student 的 key 距离排序逼近 teacher 排序。推理时，`d^{teacher}` 不可用，也不会参与候选筛选。

### 3.2 OffsetDecay

**OffsetDecay** 是把候选历史窗口的未来速度先校正到 query 历史末端水平，再按 horizon 逐步衰减校正量的零参数坐标。对候选未来 `Y_j(h)`、候选历史末端水平 `L_j`、query 历史末端水平 `L_q`，定义：

\[
\widetilde Y_j(h)=Y_j(h)+\lambda_h(L_q-L_j),\qquad
\lambda_h=1-\frac{h}{H-1},
\]

其中 `h=0,...,H-1`。因此第一步完全使用 level offset，末步不再施加 offset。实际实现还要求 query/candidate 的观测 mask 有效，并对候选权重归一化后聚合：

\[
\widehat Y_q(h)=
\frac{\sum_{j\in C_q}w_{qj}\widetilde Y_j(h)\,m_{qjh}}
{\sum_{j\in C_q}w_{qj}m_{qjh}}.
\]

OffsetDecay 是训练 teacher 和离线 Memory 评估中的未来语义坐标；它不意味着部署时读取 query future。

## 4. 候选协议与未来信息边界

一个 **candidate protocol** 规定 Bank 候选如何生成。两种协议使用同一验证 query、同一事件轴和同一随机 Bank，只改变候选过滤规则。

### 4.1 Broad-causal（主协议）

`pretrain_broad_causal` 复现预训练时的宽因果候选：严格要求候选事件早于 query（strict causal），不强制 weekday 相同，按时间分位抽样，最多保留 `event_top_r=32` 个候选。2,993 个 query 的平均候选数为 32.0，范围 32--32。这个协议最能回答“key 是否学到了 future relation”。

### 4.2 Exact-calendar（部署侧复核）

`exact_calendar` 同时要求相同 weekday、相同 slot 和严格因果。平均候选数为 8.014，范围 5--9。候选池更小，随机 Recall@1/Recall@5 自然更高，因此 exact-calendar 的绝对分数不能与 broad-causal 直接比较，只能在同一协议内比较 trained 与 random。

### 4.3 Future-information boundary

候选构造和 key 排序只读取：query history、query calendar、因果 Bank metadata、历史 Bank keys 和历史 levels。query future 只在排序完成后用于：

1. 生成离线 teacher 排序；
2. 计算 NDCG、Recall 和 Memory MAE；
3. 按预先声明的规则选择 strong-win、representative、failure 案例。

因此这是一项离线诊断，不是把未来标签泄漏到部署检索。

## 5. 指标定义

- **Anchor-wise Spearman**：先在每个 `(q,n)` 候选集合内部计算 key 排名与 teacher 排名的 Spearman，再对所有 anchor 平均；评价整体名次差异。
- **Anchor-wise Kendall**：在同一个候选集合内统计候选两两顺序的一致性；评价相对顺序是否正确。
- **Recall@1**：key 排名第一名是否等于 teacher 第一名。broad-causal 的随机期望约 `1/32=0.0318`；不能只看绝对百分比。
- **NDCG@5**：用 teacher future distance 产生 graded relevance，评价 Top-5 内部是否把更相似候选放在更前面。
- **Recall@5**：Top-5 是否覆盖 teacher 最优候选集合，作为辅助指标，不作为唯一结论。
- **Memory MAE/RMSE**：将 Top-k 候选按 OffsetDecay 聚合后，在真实速度单位上与 query future 比较；这是 payload 质量指标，不等于 ranking 指标。

## 6. Broad-causal 结果：预训练对齐的主证据

| 指标 | Relation-only | Matched random | 绝对增益/降低 |
|---|---:|---:|---:|
| Anchor-wise Spearman | 0.5677 | 0.1412 | +0.4265 |
| Anchor-wise Kendall | 0.4267 | 0.0984 | +0.3283 |
| Recall@1 | 0.1156 | 0.0558 | +0.0598 |
| NDCG@5 | 0.4376 | 0.2644 | +0.1732 |
| Recall@5（辅助） | 0.3845 | 0.2181 | +0.1664 |
| Memory MAE | 3.5752 | 4.8050 | -1.2299 |
| Memory RMSE | 6.7349 | 8.3570 | -1.6221 |

![Relation-only aggregate rank profile](../../artifacts/convergence/visualization/tgge_relation_only_v2/report_figures/aggregate_rank_profile.png)

![Relation-only full-validation ranking gain](../../artifacts/convergence/visualization/tgge_relation_only_v2/report_figures/full_validation_ranking_gain.png)

图 2 的 rank profile 中，Relation-only 的近 key-distance 分位对应更低的 teacher future distance，并随距离单调上升；random 曲线较平，说明随机 key 没有同等的 future 几何结构。图 3 将每个指标和增益拆开显示：Spearman/Kendall 的绝对提升最大，NDCG@5 也有清晰提升；Recall@1 的绝对值不高，但相对于 broad-causal 随机期望 `0.0318` 仍有约 3.63 倍。

![Relation-only horizon-wise gain](../../artifacts/convergence/visualization/tgge_relation_only_v2/report_figures/horizon_wise_gain.png)

图 4 的 broad-causal horizon 面板显示 Memory MAE 在 5--60 分钟均低于 random，且 horizon 越长绝对 MAE reduction 越大：从约 `0.41` 增至约 `2.11`。这说明排序优势不仅停留在离线 rank 指标，也传递到了 OffsetDecay payload 聚合。

## 7. Exact-calendar 结果：部署侧复核

| 指标 | Relation-only | Matched random | 绝对增益/降低 |
|---|---:|---:|---:|
| Anchor-wise Spearman | 0.3046 | 0.1877 | +0.1169 |
| Anchor-wise Kendall | 0.2376 | 0.1444 | +0.0932 |
| Recall@1 | 0.2221 | 0.1854 | +0.0367 |
| NDCG@5 | 0.6257 | 0.5720 | +0.0537 |
| Recall@5（辅助） | 0.7304 | 0.6957 | +0.0347 |
| Memory MAE | 3.5357 | 3.6735 | -0.1378 |
| Memory RMSE | 6.5914 | 6.7787 | -0.1873 |

![Relation-only exact-calendar rank profile](../../artifacts/convergence/visualization/tgge_relation_only_v2/report_figures/aggregate_rank_profile.png)

右侧 exact-calendar 曲线的差异比 broad-causal 小，这是候选池缩小、随机排序变强的预期结果，而不是训练信号消失。图 3 右列的局部纵轴将 `+0.0367` 的 Recall@1 和 `+0.0537` 的 NDCG@5 如实放大；该视觉缩放只改变坐标显示，没有改变数据。

## 8. 固定规则案例分析

案例选择完全由 `random_memory_mae - pretrained_memory_mae` 在完整验证集上的分位数决定：90% 分位为 strong-win，50% 为 representative，10% 为 failure；平分时按绝对差、sample id、node id 排序。没有手工挑选。

### 8.1 Broad-causal 案例

| 类型 | sample/node | Relation-only MAE | random MAE | gain |
|---|---|---:|---:|---:|
| strong-win | 27402 / 18 | 4.9955 | 9.5739 | +4.5784 |
| representative | 24373 / 171 | 1.8255 | 2.2428 | +0.4172 |
| failure | 26313 / 110 | 16.5173 | 15.8555 | -0.6618 |

![Relation-only broad-causal deterministic cases](../../artifacts/convergence/visualization/tgge_relation_only_v2/pretrain_broad_causal/deterministic_top5_cases.png)

![Relation-only broad-causal Top-5 error profiles](../../artifacts/convergence/visualization/tgge_relation_only_v2/pretrain_broad_causal/top5_error_profiles.png)

strong-win 显示 learned key 让权重分散到多个更接近 teacher 的候选，避免 random 的单一错误候选主导聚合；representative 体现典型但幅度有限的改善；failure 表明在突发变化或历史模式稀疏时，排序仍可能把相似历史误认为相似未来。failure 不是异常值删除对象，应作为方法边界报告。

### 8.2 Exact-calendar 案例

| 类型 | sample/node | Relation-only MAE | random MAE | gain |
|---|---|---:|---:|---:|
| strong-win | 26258 / 31 | 3.6364 | 4.4017 | +0.7654 |
| representative | 25756 / 173 | 1.6201 | 1.6526 | +0.0325 |
| failure | 25886 / 177 | 1.2959 | 0.8930 | -0.4029 |

Exact-calendar 案例中，strong-win 的提升仍存在，但 representative 接近，failure 仍保留。由于 exact-calendar 平均只有约 8 个候选，Top-5 指标天然接近饱和，论文主文应优先展示 aggregate gain 和 horizon gain，而不是只展示一条 Top-5 轨迹。

## 9. 与 TGGE v3 的交叉比较

下表只做当前单 seed、不同路由配置下的诊断比较，不能当作严格消融结论。

| 协议/指标 | Relation-only | TGGE v3 | v3 - Relation-only |
|---|---:|---:|---:|
| broad Spearman | 0.5677 | 0.5740 | +0.0063 |
| broad Kendall | 0.4267 | 0.4313 | +0.0046 |
| broad Recall@1 | 0.1156 | 0.1204 | +0.0048 |
| broad NDCG@5 | 0.4376 | 0.4435 | +0.0059 |
| broad Memory MAE | 3.5752 | 3.5632 | -0.0120 |
| exact Spearman | 0.3046 | 0.3124 | +0.0078 |
| exact Kendall | 0.2376 | 0.2440 | +0.0064 |
| exact Recall@1 | 0.2221 | 0.2239 | +0.0018 |
| exact NDCG@5 | 0.6257 | 0.6292 | +0.0035 |
| exact Memory MAE | 3.5357 | 3.5389 | +0.0032 |

v3 的 masked reconstruction 辅助在当前结果上带来小幅、方向一致的 rank 增益，但 Relation-only 仍然是更简单的有效基线。由于 v3 使用 `6 remote`、Relation-only 使用 `4 local + 6 remote`，应在固定 route 配额并增加 seed 后再决定是否把 reconstruction 写成主线贡献。

## 10. 复现产物与文件索引

- 配置：`configs/metrla_e5_tgge_latent48_relation_only_v2.yaml`
- checkpoint：`artifacts/metrla_e5_tgge_latent48_relation_only_v2_seed42/pretrain_best_relation.pt`
- trained Bank：`artifacts/convergence/tgge_relation_only_v2/bank/`
- matched-random Bank：`artifacts/convergence/tgge_relation_only_v2/random_bank/`
- broad-causal 数值：`artifacts/convergence/visualization/tgge_relation_only_v2/pretrain_broad_causal/metrics.json`
- exact-calendar 数值：`artifacts/convergence/visualization/tgge_relation_only_v2/exact_calendar/metrics.json`
- 论文图：`artifacts/convergence/visualization/tgge_relation_only_v2/report_figures/`

完整验证使用 `2,993` 个 query，未使用 `max_batches` 截断。`metrics.json`、`cases.json`、CSV 和图像共同构成可审计证据。

## 11. 局限与决策

**局限。** 当前只有 seed 42；Relation-only 与 v3 的 route allocation 不同；case study 评价的是检索排序和 Memory payload，还没有把两者接入同一 STGCN 下游协议完成迁移比较。因此不能报告统计显著性，也不能把 0.004--0.008 的跨模型差异称为决定性提升。

**保留。** 保留 Relation-only 作为 future-relation 纯目标基线、完整 Bank 和两种协议的所有数值/失败案例。

**下一步。** 固定相同 route 配额和相同 downstream 训练协议，至少增加一个 seed；若 v3 的排序与下游泛化都保持优势，再保留 reconstruction 辅助。若差异消失，则采用更简单的 Relation-only 主线。
