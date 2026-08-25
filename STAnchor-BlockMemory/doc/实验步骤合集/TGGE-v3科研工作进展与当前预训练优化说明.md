# TGGE v3 科研工作进展与当前预训练优化说明

## 1. 项目目标与研究动机

TGGE（时空动态检索增强模块）旨在作为一个可插拔模块接入不同的时空下游预测模型，并在不同数据集之间迁移。给定下游模型的基础预测，TGGE 从历史事件 Bank 中检索未来动态相似的事件，将其 future payload 形成 Memory，再由诊断校正器判断 Memory 是否可靠并决定校正幅度。

本项目针对的核心问题是时空海市蜃楼（spatiotemporal mirage）：两个历史窗口在观测空间中相似，但由于交通传播、周期错位、突发事件或拓扑影响，其未来演化可能不同。因此，单纯历史相似度不能保证 future 相似度。TGGE 的研究目标不是寻找一个只在 METR-LA 上有效的更大预测器，而是学习一种能够跨下游模型、跨数据集复用的未来相关事件表示和校正机制。

## 2. 总体数据流与模块职责

整体流程为：

$$
X \rightarrow E_\theta(X) \rightarrow z \rightarrow \mathrm{Retrieve}(z,\mathcal B)
\rightarrow Y^{cand} \rightarrow Y^{memory}
\rightarrow G_\phi(X,Y^{base},Y^{memory})
\rightarrow \widehat Y.
$$

其中：

- $X$：历史输入窗口；
- $E_\theta$：TGGE 时空预训练编码器；
- $z$：节点级连续 latent key；当前 retrieval dimension 为 48，profile 分支已废弃；
- $\mathcal B$：Memory Bank，保存历史事件的 key、时间索引和 future payload；
- $Y^{cand}$：检索得到的候选 future；
- $Y^{base}$：下游模型产生的基础预测；
- $Y^{memory}$：候选 future 的聚合结果；
- $G_\phi$：诊断校正器；
- $\widehat Y$：最终预测。

### 2.1 预训练编码器

编码器由时间 patch embedding、因子化时空编码器、图路由注意力和 retrieval head 构成。它同时承担两项任务：从历史窗口提取时空动态表示，并使 key 相似度与 future 动态关系一致。

当前编码器结构约 30 万参数，retrieval key 为连续的 48 维 latent，不再拼接 profile 表示。

### 2.2 OffsetDecay future 教师

当前 relation supervision 不是直接比较 METR-LA 原始速度值。对事件 $q$ 的历史上下文末端水平 $e_{q,n,c}$，构造：

$$
S_{q,h,n,c}=Y_{q,h,n,c}-\lambda_h e_{q,n,c},
\qquad
\lambda_h=1-\frac{h-1}{H-1}.
$$

$S$ 是 OffsetDecay future signature：近端 horizon 更强地消除当前水平，远端 horizon 逐渐保留未来绝对变化。随后计算节点级 pairwise distance：

$$
d^{OD}_{q,j,n}=\operatorname{MAE}_{h,c}(S_{q,h,n,c},S_{j,h,n,c}).
$$

代码进一步使用 symmetric geometric mean normalization：

$$
\widetilde d_{q,j,n}=
\frac{d^{OD}_{q,j,n}}
{\sqrt{(\bar d_{q,n}+\epsilon)(\bar d_{j,n}+\epsilon)}}.
$$

因此教师关系主要表达无量纲的相对动态差异，降低不同节点和数据集数值尺度差异的影响。

### 2.3 连续 relation loss

编码器输出节点 key $z_{q,n}$，学生相似度为：

$$
s_{q,j,n}=\cos(z_{q,n},z_{j,n}).
$$

OffsetDecay distance 生成教师分布：

$$
p^{OD}_{q,j,n}=\operatorname{softmax}_j(-\widetilde d_{q,j,n}/\tau_t),
$$

key similarity 生成学生分布：

$$
p^{key}_{q,j,n}=\operatorname{softmax}_j(s_{q,j,n}/\tau_s).
$$

relation loss 使两个分布匹配。它保留连续 latent 几何，有利于跨数据集泛化，但对局部候选排序交换的惩罚可能不够强。

### 2.4 当前新增 Hard-Mirage Rank Loss

本版不把 latent 离散化，也不引入原始 METR-LA future 距离。它直接复用已经计算好的 $\widetilde d$ 和 $s$。

对每个 anchor $(q,n)$：

1. 从候选中选 future distance 最小的 Top-2 作为正样本集合 $\mathcal P_{q,n}$；
2. 只保留比正样本最差距离至少大 $\delta=0.05$ 的候选；
3. 从这些 future 较差候选中选 key 相似度最高的 Top-2 作为 hard-mirage negatives $\mathcal R_{q,n}$。

对每个正负组合使用：

$$
\mathcal L_{rank}=
\frac{1}{|\mathcal A|}
\sum_{(q,n)\in\mathcal A}
\frac{1}{|\mathcal P_{q,n}||\mathcal R_{q,n}|}
\sum_{p,r}
\operatorname{softplus}
\left(
\frac{m-(s_{q,p,n}-s_{q,r,n})}{\tau_r}
\right).
$$

$m=0.05$ 为 cosine margin，$\tau_r=0.1$ 为 rank 温度。最终关系项为：

$$
\mathcal L_{retrieval}=\mathcal L_{relation}+0.05\mathcal L_{rank}.
$$

该设计每个 anchor 最多产生 4 个 pair，不增加 encoder forward、不重新查询 Bank、不重新计算 OffsetDecay distance，因此参数量不变，额外开销主要是 Top-K、gather 和 softplus。

### 2.5 Memory Bank 与候选聚合

Bank 保存历史事件 key 和 future payload。当前候选协议为 `exact_calendar`，profile/key 旧分支不再作为主线。候选 future 保留为：

$$
Y^{cand}\in\mathbb R^{B\times H\times N\times K\times C}.
$$

当前 weighted mean 使用节点级检索权重 $w_{q,n,k}$ 聚合候选。已有对照表明，Top-1、Top-3、trimmed mean 和 sign-cluster 均未超过当前 weighted mean，说明瓶颈不是简单平均公式，而是 key 排序与候选质量未充分对齐。

### 2.6 诊断校正器

诊断校正器输入 base risk、retrieval similarity、candidate dispersion、signed direction、horizon position 和 memory disagreement 等特征，输出 gate 或校正量。当前已增加 horizon-aware 的 signed direction、delta mean 和 delta std，特征维度为 12，校正器参数量约 224,817。

它的职责是判断 Memory 是否可靠以及修正幅度，而不是替代编码器重新学习候选排序。

## 3. 已完成的主要实验与指标含义

### 3.1 下游迁移验证

已接入 STGCN 和 Graph WaveNet，采用冻结的同配置 base，仅训练诊断校正器，验证 TGGE 的可插拔性。主要指标：

- MAE：平均绝对误差，越低越好；
- RMSE：均方根误差，对大误差更敏感，越低越好；
- helpful rate：Memory 或最终校正相对于 base 降低误差的位置比例；
- risk Spearman：诊断器风险排序与真实误差排序的秩相关，越高说明排序方向越可靠；
- AUROC：随机选一个 helpful 和一个 harmful 位置时，模型把 helpful 排在前面的概率；0.5 近似随机；
- AUPRC：不平衡 helpful 标签下，高风险/高 helpful 分数区域的精确率-召回表现；应结合 helpful prevalence 解读；
- 单 epoch 时间和参数量：评估工程成本与模块复杂度。

### 3.2 Oracle gating 与聚合对照

Oracle 使用验证阶段真实 future 计算可达到的理想候选选择或融合上界，不能用于部署。已有结果显示 oracle candidate Top-1 MAE 约 1.66，而部署可用 Top-1 约 4.09；这说明候选池中存在高价值 future，但当前 key 排序不可靠。

固定聚合结果为：Base MAE 约 3.04，weighted mean Memory 约 3.55，Top-1 约 4.09，Top-3 约 3.65，trimmed mean 约 3.58，sign-cluster 约 4.02。该结果不支持简单替换 weighted mean。

### 3.3 诊断器可预测性实验

helpfulness AUROC 约 0.58，AUPRC 约 0.48，risk Spearman 约 0.36，risk $R^2$ 约 0.07；高 confidence 区域 helpful rate 约 49%。这些结果说明诊断特征含有弱信号，但还不能准确恢复 oracle gate，不能把全部排序偏差转交给诊断器。

## 4. 当前阶段的科研判断

当前证据支持以下结论：

1. 候选池有价值，问题不是 Bank 中完全没有相似 future；
2. key 相似度与 future usefulness 存在排序错位，时空海市蜃楼是主要解释之一；
3. 当前连续 relation loss 保留了跨数据集所需的连续表示，但局部 rank 约束不足；
4. 诊断器存在信息瓶颈，因为候选在聚合后部分丢失身份与 horizon-specific 差异；
5. 继续单纯扩大校正器参数量缺少证据支持；应先修复 encoder 的候选排序监督。

## 5. 当前正式结果汇总

本节只汇总当前已有的正式训练结果，不包含 smoke/debug 运行，也不包含 case study。所有下游数字均来自 METR-LA validation split；case study 另有独立文件，本报告不重复展开。

### 5.1 TGGE v3 预训练结果

当前采用的 TGGE v3 预训练运行是 `tgge_single_view_v3_higher_order_reconstruction2`，使用 48 维连续 latent key、`masked_relation_single_view` 目标、`OffsetDecay` teacher 和 `symmetric_geometric_mean` distance normalization。该历史正式运行计划 50 轮，但因 validation retrieval 指标连续未改善，在第 29 轮触发 early stopping。

第 29 轮记录如下：

| 指标 | 训练集 | 验证集 |
|---|---:|---:|
| Total loss | 2.813315 | 2.653692 |
| Reconstruction loss | 0.227194 | 0.224239 |
| Retrieval relation loss | 2.358928 | 2.205214 |
| Teacher effective support | 4.239 | 3.787 |
| Student effective support | 9.532 | 8.017 |
| 编码器总参数量 | 303,727 | 303,727 |

该结果说明当前编码器已经能够学习连续的 future relation 几何，但 `student effective support` 高于 teacher，表示 key 相似度分布相对更平滑，精确候选排序仍不够集中。这是本版新增 Hard-Mirage Top-2/Top-2 rank loss 的直接动机。注意：上表是 rank loss 改动前的正式预训练基线；新增 rank loss 的正式长训练结果尚待实验机运行，不能把本机 smoke 指标当作正式结果。

### 5.2 下游 base-only 结果

`base_only` 表示只训练下游预测模型，不启用 TGGE Memory 和诊断校正器。它用于回答“下游 backbone 本身能达到什么水平”，也是接入 TGGE 后增益归因的匹配基线。

| 下游模型 | 训练轮数 | 最佳验证 MAE | 对应验证 RMSE | 最佳 checkpoint |
|---|---:|---:|---:|---|
| STGCN | 50 | 2.901381 | 6.018566 | `downstream_tgge_v3_stgcn_base_only_fulltrain_seed42/downstream_best.pt` |
| Graph WaveNet | 50 | 2.865798 | 5.8560 | `downstream_tgge_v3_graphwavenet_base_only_fulltrain_seed42/downstream_best.pt` |

其中，Graph WaveNet 的最佳 checkpoint 出现在第 47 轮，MAE 2.865798、RMSE 5.8560；第 50 轮验证记录为 MAE 2.868271、RMSE 5.883618。STGCN 的最佳 checkpoint 出现在第 23 轮，MAE 2.901381、RMSE 6.018566；第 50 轮验证记录为 MAE 3.014767。因此正式比较应使用 best checkpoint，而不能只看最后一轮。

### 5.3 接入当前 TGGE v3 模块后的结果

该实验采用冻结的已训练 base/backbone，只训练 `StructuredErrorCorrector`，因此它检验的是 TGGE 检索、Memory 和诊断校正器在不重新训练 backbone 的条件下能否改善基础预测。需要区分代码版本：下面匹配队列的正式结果是在 12 维诊断特征改动之前产生的，日志记录的校正器参数量为 224,142；当前代码的 12 维特征校正器参数量为 224,817，但尚未用当前 rank-loss/12-feature 代码重新跑出正式下游结果。冻结预训练编码器在两者中均为 303,727 参数。

| 下游模型 | 训练阶段 | 最佳验证 MAE | 第 50 轮验证 MAE | 第 50 轮 RMSE | 相对匹配 base-only 最佳 MAE |
|---|---|---:|---:|---:|---:|
| STGCN + TGGE | matched fulltrain，50 轮 | 2.837542 | 2.837542 对应最佳点 | 5.841685 | -0.063839 |
| Graph WaveNet + TGGE | matched fulltrain，当前已记录至第 47 轮 | 2.859315 | 当前最好值 2.859315 | 5.747124 | -0.006483 |

这里“相对匹配 base-only 最佳 MAE”定义为：

$$
\Delta MAE=MAE_{TGGE}-MAE_{base\_only}.
$$

因此当前匹配实验的 MAE 差值为：STGCN 为 $2.837542-2.901381=-0.063839$，Graph WaveNet 为 $2.859315-2.865798=-0.006483$。负值表示接入 TGGE 后 MAE 下降，即有改善；正值才表示变差。当前结果显示 STGCN 已获得约 0.0638 MAE 的验证集改善，Graph WaveNet 的改善约 0.0065，幅度较小，仍需以完整训练结束后的 best checkpoint 复核。

Graph WaveNet 的 TGGE matched-fulltrain 日志当前已记录到第 47 轮，尚未看到最终完成标记，因此 2.859315 应标为“当前最好值”，不能写成最终收敛结果。STGCN 已完成 50 轮，最佳 MAE 2.837542。此前 detached v7 运行的 STGCN 2.928708 和 Graph WaveNet 3.057256 属于另一轮队列记录，不作为本表的主比较结果。后续正式比较统一使用相同 split、seed、base checkpoint、Bank/candidate protocol，并报告 best checkpoint。

### 5.4 结果传达的科研信号

1. **预训练层面**：连续 OffsetDecay relation 已学到可用的动态关系，但排序分布过平滑，说明需要显式强化 hard-mirage 局部顺序。
2. **下游层面**：在匹配的冻结 base、Bank、候选协议和 posthoc 校正协议下，旧版 9 特征校正器的 STGCN + TGGE 最佳验证 MAE 从 2.901381 降至 2.837542，Graph WaveNet 当前最好值从 2.865798 降至 2.859315，说明接入 TGGE 后确实取得了改善；其中 STGCN 改善更明显，Graph WaveNet 改善幅度较小。但这组结果不能直接等同于当前 12 特征 + rank-loss 版本，后者需要重新完成正式下游验证。
3. **归因层面**：已有 oracle candidate 约 1.66 MAE 的上界证明候选池有价值；当前真实增益仍受 key 排序、候选聚合和诊断决策的信息损失限制，因此需要继续通过 rank loss 和 horizon-aware 机制提升增益稳定性。
4. **实验边界**：当前 rank-loss 版本尚未完成正式长训练，不能提前把它写成已验证提升；其角色是下一轮单变量预训练对照。

## 6. 本版训练命令与验证协议

本版配置已将 `rank_loss_weight=0.05`、`rank_positive_count=2`、`rank_negative_count=2`、`rank_future_gap=0.05`、`rank_margin=0.05`、`rank_temperature=0.1` 写入：

`configs/metrla_e5_tgge_single_view_masked_relation_v3.yaml`

实验机使用 `research` 环境运行：

```powershell
conda activate research
cd D:\projects\researchProjects\TrafficRobustST\STAnchor-BlockMemory
python scripts\pretrain.py `
  --config configs\metrla_e5_tgge_single_view_masked_relation_v3.yaml `
  --run-name metrla_e5_tgge_single_view_masked_relation_rank_top2_seed42
```

正式训练需记录：epoch wall-clock time、`val_retrieval`、`val_total`、rank loss、有效 rank pairs、key Top-1 future MAE、Recall@5、NDCG@5 和 downstream learned-gate MAE。单轮时间验收上限为 10 分钟；本地实现不增加编码器参数和编码器前向次数。

## 7. 后续需要补的主要实验

1. **Rank loss 与原版 relation loss 对照**：固定数据、seed、Bank 和训练轮数，只比较 `rank_loss_weight=0` 与 `0.05`，判断 Top-2/Top-2 排序约束是否改善 key Top-1、NDCG@5 和 downstream MAE。
2. **跨下游模型验证**：STGCN 与 Graph WaveNet 使用完全匹配的冻结 base、Bank、候选协议和 batch/optimizer，报告 MAE、RMSE、helpful rate、risk Spearman、单轮时间和校正器参数量。
3. **跨数据集验证**：在 PEMS-BAY 或其他目标数据集固定迁移协议，检验 OffsetDecay+rank supervision 是否仍优于无 rank 版本。
4. **最终多 seed**：在主结论稳定后补充有限 seed，报告均值和标准差，不在当前赶进度阶段优先展开大规模组合实验。

## 8. Keep / Remove / Stop

### Keep

- 连续 48 维 latent key；
- OffsetDecay + symmetric geometric mean relation teacher；
- 冻结 base 的下游验证协议；
- 候选级、signed、horizon-aware 诊断特征；
- Hard-Mirage Top-2/Top-2 rank loss 作为低成本单变量优化。

### Remove

- 主线中的 profile key/profile weight 分支；
- 将原始 METR-LA future 数值距离直接作为跨数据集 rank teacher；
- 只按 sign-cluster 决定唯一 Memory 的简单聚合规则。

### Stop

- 在 rank 对照未验证前继续扩大诊断器容量；
- 同时修改 encoder、Bank、聚合器和校正器；
- 将 oracle 指标当作部署性能；
- 将 smoke/debug 训练产物当作正式科研证据。
