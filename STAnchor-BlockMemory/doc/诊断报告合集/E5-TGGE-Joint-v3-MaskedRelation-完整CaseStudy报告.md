# E5 TGGE Joint v3 masked-relation：完整检索 Case Study 报告

## 1. 报告范围与核心结论

本报告独立说明 `TGGE v3 masked_relation_single_view` 在 METR-LA 验证集上的检索诊断。v3 在同一个 masked history 前向中同时优化 future-relation 和 masked reconstruction：前者学习可检索的 future relation，后者只在人工遮挡且原始观测有效的位置恢复历史值。推理和 Bank 构建使用 clean history，不带 mask。

核心结论：

1. v3 在与预训练目标一致的 broad-causal 协议下，Spearman `0.5740`、Kendall `0.4313`、NDCG@5 `0.4435`、Memory MAE `3.5632`，均优于 matched-random；其中 Spearman 增益 `+0.4320`，Kendall 增益 `+0.3323`。
2. exact-calendar 部署侧复核仍保持正增益：Spearman `+0.1248`、Kendall `+0.1000`、NDCG@5 `+0.0572`、Memory MAE 降低 `0.1346`。
3. 与 Relation-only 的当前单 seed 比较中，v3 在两种协议的所有 ranking 指标均略高，broad-causal Memory MAE 低 `0.0120`；exact-calendar Memory MAE 则高 `0.0032`。因为 route 配额不同，这说明“v3 有潜力”，还不能证明 reconstruction 本身造成了全部增益。
4. 图表使用完整验证集、固定候选池、局部坐标轴和 gain 视图；strong-win、representative、failure 都按预先固定规则保留。

## 2. 数据、张量与模型定义

### 2.1 数据契约

METR-LA 有 `N=207` 个节点，5 分钟采样。历史检索窗口和未来标签分别为：

\[
x^{hist}\in\mathbb{R}^{B\times288\times207\times1},\qquad
y^{future}\in\mathbb{R}^{B\times12\times207\times1}.
\]

`B` 为 batch size，`T=288` 为 24 小时历史，`H=12` 为 60 分钟未来，通道为速度。训练/验证/测试时间步为 `22,681/2,993/6,025`。所有 key、候选和聚合操作都遵守时间切分；验证 future 只在离线 teacher 和误差计算阶段使用。

### 2.2 单视图 masked-relation 数据流

v3 的 objective 名为 `masked_relation_single_view`。mask sampler 生成：

- patch mask `[B,24,207]`，其中 patch size 为 12；
- value mask `[B,288,207,1]`，用于 reconstruction loss。

编码器只接收一次 masked history：

```text
masked x^hist [B,288,207,1]
  -> normalization using visible observations
  -> patch embedding [B,24,207,80]
  -> FactorizedSTEncoder
  -> retrieval key K [B,207,48]
  -> reconstruction y_hat [B,288,207,1]
```

其中 `K=f_theta(x_masked,G)` 是后续检索 key。训练时 masked normalization 只使用可见值；推理时 `encode_clean` 使用完整可观测历史，不携带训练 mask。

正式配置 `configs/metrla_e5_tgge_single_view_masked_relation_v3.yaml`：

| 项目 | 设置 |
|---|---:|
| objective | `masked_relation_single_view` |
| reconstruction/retrieval weight | 2.0 / 1.0 |
| hidden/retrieval dimension | 80 / 48 |
| encoder layers / heads | 3 / 4 |
| route top-k / local quota | 6 / 0 |
| batch size / maximum epochs | 16 / 50 |
| seed | 42 |
| total/trainable parameters | 303,727 |
| reconstruction head | 972 parameters |

v3 的图分支保留固定一阶图注意力；route 分支只从非一阶候选补充最多 6 个远端节点。与 Relation-only 的 `4 local + 6 remote` 不同，这是后续公平比较需要固定的混杂因素。

## 3. 目标函数、OffsetDecay 与信息边界

### 3.1 联合目标

v3 优化：

\[
\mathcal L_{v3}
=1.0\,\mathcal L_{relation}
+2.0\,\mathcal L_{reconstruction}.
\]

`L_relation` 使用 future relation teacher 约束 key 排序；`L_reconstruction` 只在 `value_mask & observed` 的位置计算：

\[
\mathcal L_{reconstruction}
=\frac{1}{|M|}\sum_{(t,n)\in M}
\left|\hat x_{t,n}-x_{t,n}\right|,
\]

其中 `M` 是被遮挡且原始观测有效的位置。它是表示鲁棒性正则，不是最终 traffic forecasting loss。

### 3.2 Future-relation 与 OffsetDecay

对 query-node anchor `(q,n)`，key 距离为

\[
d^{key}_{qjn}=\lVert K_q(n)-K_j(n)\rVert_2.
\]

训练标签使用真实 future 构造 teacher 排序。**OffsetDecay** 将候选未来 `Y_j(h)` 对齐到 query 历史末端水平：

\[
\widetilde Y_j(h)=Y_j(h)+\lambda_h(L_q-L_j),
\qquad \lambda_h=1-\frac{h}{H-1}.
\]

再按 key softmax 权重聚合：

\[
\widehat Y_q(h)=
\frac{\sum_j w_{qj}\widetilde Y_j(h)m_{qjh}}
{\sum_jw_{qj}m_{qjh}}.
\]

query future 仅作为训练 relation 标签和离线评估 teacher；它从不进入 masked key、Bank 候选筛选或部署时的 aggregation。

## 4. 候选协议、指标与公平性

**Broad-causal** 是预训练对齐协议：strict causal、不过滤 weekday、最多 32 个按时间分位抽样的历史事件。2,993 个 query 的候选均值为 32.0。

**Exact-calendar** 是部署侧协议：same weekday、same slot、strict causal；候选均值 8.014，范围 5--9。候选数量变少后 random 的 Recall@5 会自然升高，所以只比较同协议的 trained/random。

对每个 `(q,n)` 先在候选集合内部计算，再跨 anchor 平均：

- Anchor-wise Spearman：整体名次差异；
- Anchor-wise Kendall：候选两两顺序一致性；
- Recall@1：第一名命中；broad random 期望约 `0.0318`；
- NDCG@5：Top-5 内部 graded ranking；
- Recall@5：辅助覆盖率；
- Memory MAE/RMSE：OffsetDecay payload 聚合的物理速度误差。

matched-random 与 v3 共享同一事件轴、future payload、query 和候选池，只有 encoder/checkpoint fingerprint 不同。未来信息边界由 `metrics.json` 记录：ranking inputs 仅为 history、calendar、causal metadata、historical keys/levels；query future 仅用于 post-ranking metrics、案例选择和图。

## 5. 训练收敛与参数证据

v3 训练到 epoch 29 后触发 patience=10 early stopping。按 relation 验证损失保存的 checkpoint 是 epoch 19，`val_retrieval=2.196112`；按联合总损失保存的 `pretrain_best.pt` 在结束时 `best_val=2.653692`。模型共 303,727 个可训练参数，其中 embedding 25,360、encoder 260,467、route attention 25,827、retrieval head 16,928、reconstruction head 972。

![v3 training convergence](../../artifacts/convergence/visualization/tgge_single_view_v3_reconstruction2/report_figures/training_convergence.png)

图 1 显示 relation 验证损失从约 2.29 降至 2.20 附近，epoch 19 达到最低；reconstruction 验证损失从约 0.31 降至约 0.224。student effective support 从约 9.5 降至约 8.0，teacher support 约 3.8，说明模型逐步集中权重但没有退化为单一候选。

## 6. Broad-causal：与预训练目标对齐的主结果

| 指标 | v3 trained | Matched random | 绝对增益/降低 |
|---|---:|---:|---:|
| Anchor-wise Spearman | 0.5740 | 0.1420 | +0.4320 |
| Anchor-wise Kendall | 0.4313 | 0.0990 | +0.3323 |
| Recall@1 | 0.1204 | 0.0559 | +0.0646 |
| NDCG@5 | 0.4435 | 0.2647 | +0.1788 |
| Recall@5（辅助） | 0.3890 | 0.2183 | +0.1707 |
| Memory MAE | 3.5632 | 4.8035 | -1.2404 |
| Memory RMSE | 6.7158 | 8.3532 | -1.6374 |

![v3 aggregate rank profile](../../artifacts/convergence/visualization/tgge_single_view_v3_reconstruction2/report_figures/aggregate_rank_profile.png)

![v3 full-validation ranking gain](../../artifacts/convergence/visualization/tgge_single_view_v3_reconstruction2/report_figures/full_validation_ranking_gain.png)

图 2 中 v3 的 key-distance decile 与 teacher future distance 呈更清晰的单调关系，random 的曲线相对平坦。图 3 的 broad gain 显示主要提升来自完整局部排序（Spearman/Kendall），NDCG@5 也有 `+0.1788` 的绝对提升；Recall@1 绝对值为 0.1204，但相对 broad random 期望 0.0318 仍明显高出。

![v3 horizon-wise gain](../../artifacts/convergence/visualization/tgge_single_view_v3_reconstruction2/report_figures/horizon_wise_gain.png)

图 4 左下的 MAE reduction 从约 `0.43` 增至约 `2.13`，覆盖 5--60 分钟全部 horizon；右下 exact-calendar 的增益则从约 `0.16` 逐步降到约 `0.10`，符合长 horizon 和小候选池下的保守收益形态。

## 7. Exact-calendar：部署侧结果

| 指标 | v3 trained | Matched random | 绝对增益/降低 |
|---|---:|---:|---:|
| Anchor-wise Spearman | 0.3124 | 0.1877 | +0.1248 |
| Anchor-wise Kendall | 0.2440 | 0.1444 | +0.0996 |
| Recall@1 | 0.2239 | 0.1851 | +0.0388 |
| NDCG@5 | 0.6292 | 0.5719 | +0.0572 |
| Recall@5（辅助） | 0.7330 | 0.6957 | +0.0373 |
| Memory MAE | 3.5389 | 3.6735 | -0.1346 |
| Memory RMSE | 6.6026 | 6.7788 | -0.1762 |

exact-calendar 的随机 Recall@1 期望约 `0.1294`，Recall@5 期望约 `0.6424`。因此 `0.7330` 的 Recall@5 只能说明有正增益，不能被解读为 broad-causal 级别的强排序。局部纵轴的 gain 面板是为了解释小差异，不是选择性放大结果。

## 8. 固定规则案例

案例选择依据完整验证集上的 `random_memory_mae - v3_memory_mae`：90% 分位 strong-win、50% 分位 representative、10% 分位 failure；不手工挑选。

### 8.1 Broad-causal

| 类型 | sample/node | v3 MAE | random MAE | gain |
|---|---|---:|---:|---:|
| strong-win | 24922 / 160 | 3.8626 | 8.4406 | +4.5780 |
| representative | 25614 / 33 | 2.6422 | 3.0627 | +0.4204 |
| failure | 26206 / 8 | 3.4555 | 2.8013 | -0.6543 |

![v3 broad-causal deterministic cases](../../artifacts/convergence/visualization/tgge_single_view_v3_reconstruction2/pretrain_broad_causal/deterministic_top5_cases.png)

![v3 broad-causal Top-5 error profiles](../../artifacts/convergence/visualization/tgge_single_view_v3_reconstruction2/pretrain_broad_causal/top5_error_profiles.png)

strong-win 说明 masked-relation key 能把权重分配给更接近 teacher 的候选；representative 说明平均收益并非只由极端样本贡献；failure 说明突发交通变化、历史观测稀疏或 future 多模态时，历史相似性仍可能失效。failure 保留在报告中，作为适用边界。

### 8.2 Exact-calendar

| 类型 | sample/node | v3 MAE | random MAE | gain |
|---|---|---:|---:|---:|
| strong-win | 25387 / 54 | 2.6835 | 3.4377 | +0.7542 |
| representative | 25460 / 206 | 1.5457 | 1.5789 | +0.0331 |
| failure | 25570 / 62 | 2.5350 | 2.1364 | -0.3986 |

exact-calendar 的 representative 接近零增益、failure 为负，说明部署过滤条件下 v3 不是对每个 anchor 都有效；论文叙述应以完整 aggregate 为主，案例图用来解释机制与边界，而不能替代总体统计。

## 9. 与 Relation-only 的交叉比较

| 协议/指标 | v3 | Relation-only | v3 - Relation-only |
|---|---:|---:|---:|
| broad Spearman | 0.5740 | 0.5677 | +0.0063 |
| broad Kendall | 0.4313 | 0.4267 | +0.0046 |
| broad Recall@1 | 0.1204 | 0.1156 | +0.0048 |
| broad NDCG@5 | 0.4435 | 0.4376 | +0.0059 |
| broad Memory MAE | 3.5632 | 3.5752 | -0.0120 |
| exact Spearman | 0.3124 | 0.3046 | +0.0078 |
| exact Kendall | 0.2440 | 0.2376 | +0.0064 |
| exact Recall@1 | 0.2239 | 0.2221 | +0.0018 |
| exact NDCG@5 | 0.6292 | 0.6257 | +0.0035 |
| exact Memory MAE | 3.5389 | 3.5357 | +0.0032 |

v3 的每个 ranking 指标都略高，但差值很小；broad Memory MAE 低 0.0120，exact Memory MAE 反而高 0.0032。更重要的是两种模型的 route 配置不同：v3 `6 remote`，Relation-only `4 local + 6 remote`。因此当前证据支持“masked reconstruction 是可行的辅助正则”，不支持“它已经被严格证明优于 relation-only”。

## 10. 复现产物

- 配置：`configs/metrla_e5_tgge_single_view_masked_relation_v3.yaml`
- relation checkpoint：`artifacts/convergence/tgge_single_view_v3_higher_order_reconstruction2/pretrain/pretrain_best_relation.pt`
- 联合总损失 checkpoint：`artifacts/convergence/tgge_single_view_v3_higher_order_reconstruction2/pretrain/pretrain_best.pt`
- trained/random Bank：对应 `artifacts/convergence/visualization/tgge_single_view_v3_reconstruction2/` 中 `metrics.json` 的路径
- broad-causal 数值和案例：`.../pretrain_broad_causal/metrics.json`、`cases.json`
- exact-calendar 数值和案例：`.../exact_calendar/metrics.json`、`cases.json`
- 报告图和 CSV：`artifacts/convergence/visualization/tgge_single_view_v3_reconstruction2/report_figures/`

所有 case study 都是完整验证集运行，未使用 `--max-batches`。`ranking_gain.csv` 和 `horizon_gain.csv` 可直接用于重新绘图或论文表格。

## 11. 局限、保留和下一步

**局限。** 目前只有 seed 42；v3 与 Relation-only 的图路由配额不同；尚未完成同一 STGCN 下游模型和跨数据集迁移的对照；因此没有均值/标准差和显著性结论。

**保留。** 保留 v3 的 single-view masked-relation 设计、relation checkpoint、联合 checkpoint、两种候选协议的完整数值、全部案例和失败图。

**下一步。** 先固定 route allocation、Bank 事件轴和 downstream protocol，补一个以上 seed；若 v3 的 ranking gain 与下游泛化同时保持，再把 reconstruction 作为主线辅助目标。若下游不提升，则停止继续堆叠辅助头，回退到更简单的 Relation-only 主线。这个决策遵循“一个实验关闭一个不确定性”的原则。
