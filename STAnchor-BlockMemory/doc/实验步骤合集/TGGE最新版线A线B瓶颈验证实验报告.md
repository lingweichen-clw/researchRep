# TGGE 最新版线 A / 线 B 瓶颈验证实验报告

## 1. 实验背景与验证问题

本报告只讨论当前最新 TGGE 主线：固定 TGGE 预训练编码器、同一 METR-LA Memory Bank、`exact_calendar` 因果候选协议，以及两个下游模型 Graph WaveNet 和 STGCN。其他历史 robust 目录不作为本报告证据。

当前现象是两个下游模型最终验证 MAE 均稳定在约 2.84–2.86，继续增加校正器训练轮数没有明显收益。本组实验不直接增加新模块，而是依次回答两个问题：

1. **线 A：Memory 与候选池到底有没有更高的理论上限？** 如果 oracle 也只能达到约 2.82，则应停止优化诊断器，转向检索器和 payload；如果 oracle 明显更好，则说明 2.84 不是 Memory 的固有上限。
2. **线 B：在不使用推理时真实 future 的前提下，现有部署特征能否学会“何时修正、修正多少”？** 如果一个直接的监督策略也学不好，则瓶颈主要是可见证据不足，而不只是当前神经网络容量不足。

## 2. 固定实验条件

- 数据集：METR-LA。
- 下游输入：过去 12 个时间步，即 60 分钟。
- 预测范围：未来 12 个时间步，即 60 分钟。
- 检索上下文：过去 288 个时间步，即 24 小时。
- 预训练检查点：`pretrain_best_relation.pt`。
- Memory Bank：最新 TGGE single-view v3 reconstruction2 Bank。
- 候选协议：`exact_calendar`，并满足候选 `future_end < query.context_start`。
- 节点候选数：Top-K = 5。
- 节点 level rerank 权重：0，避免把 level 启发式混入 TGGE key 归因。
- 下游：Graph WaveNet、STGCN。
- 随机种子：42。
- 运行环境：Conda `research`。

## 3. 未来信息边界

### 3.1 线 A

线 A 的 oracle 在验证阶段使用真实 future，因此只能用于诊断上界，不能部署，也不能作为测试集正式方法结果。

对查询位置 \((q,h,n)\)，定义 base、聚合 Memory 与真实值分别为

\[
Y^{base}_{q,h,n},\qquad Y^{memory}_{q,h,n},\qquad Y_{q,h,n}.
\]

真实 future 只用于计算 oracle gate、oracle 步长和候选 oracle，不参与检索 key、候选筛选、Bank 构建或正式推理。

### 3.2 线 B

线 B 使用训练段真实 future 生成监督标签；验证段真实 future 只用于评价。策略输入全部来自推理时可获得的信息：历史输入、base 输出、检索分数、候选权重、候选历史 future payload 以及由这些量计算的统计特征。

## 4. 线 A：反事实上界诊断

### 4.1 方法定义

比较以下方法：

1. `base`：仅下游模型预测。
2. `current_learned_gate`：当前 `StructuredErrorCorrector` 输出。
3. `fixed_alpha`：统一使用固定融合权重 \(\alpha\)。
4. `oracle_binary_gate`：真实 future 判断 Memory 是否优于 base；有帮助时完全使用 Memory，否则使用 base。
5. `oracle_continuous_alpha`：对每个 \((q,h,n)\) 在 \([0,1]\) 网格选择误差最小的融合步长：

\[
\hat Y(\alpha)=Y^{base}+\alpha\left(Y^{memory}-Y^{base}\right).
\]

6. `oracle_candidate_top1`：在合法 Top-K 候选中，用真实 future 选择误差最小的单个候选。
7. `oracle_candidate_error_weighted`：用真实 future 误差对 Top-K 候选进行理想加权。

Memory helpful prevalence 定义为

\[
\Pr\left(
|Y^{memory}-Y| < |Y^{base}-Y|
\right),
\]

表示 Memory 单独预测优于 base 的位置比例。

### 4.2 完整验证集结果

| 下游模型 | Base MAE | 当前 Gate MAE | Oracle Binary | Oracle Continuous | Oracle Candidate Top-1 | Oracle Candidate Weighted | Memory Helpful Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Graph WaveNet | 2.8658 | 2.8593 | 2.1923 | 2.0299 | 1.6604 | 1.6410 | 41.30% |
| STGCN | 2.9014 | 2.8375 | 2.1868 | 1.9903 | 1.6578 | 1.6384 | 41.80% |

固定权重的最佳结果为：

- Graph WaveNet：\(\alpha=0.05\)，MAE 2.8569，仅比 base 改善 0.0089。
- STGCN：\(\alpha=0.10\)，MAE 2.8728，仅比 base 改善 0.0286。

Oracle continuous 的平均最优 \(\alpha\) 随 horizon 略微增大：

- Graph WaveNet：从第 1 步约 0.393 增至第 12 步约 0.438。
- STGCN：从第 1 步约 0.416 增至第 12 步约 0.437。

### 4.3 线 A 结果解释

第一，**2.84–2.86 不是 Memory 的固有上限**。两个下游的 oracle continuous 均可达到约 2.0，候选 oracle 约 1.64，远低于当前结果。

第二，**不能简单增大 Memory 权重**。Memory helpful rate 只有约 41%–42%，且固定 \(\alpha\) 很快恶化；Graph WaveNet 在 \(\alpha=0.5\) 时 MAE 达到 3.0381。因此关键不是“让 Memory 更强地参与”，而是准确判断使用位置和步长。

第三，存在两级明显损失：

- `oracle candidate` 到 `oracle continuous` 的差距说明当前 Top-K 聚合没有充分利用候选池，候选选择/聚合仍是一个瓶颈。
- `oracle continuous` 到 `current learned gate` 的巨大差距说明诊断与融合策略同样是主要瓶颈。

因此不能把问题单独归因于检索器，也不能单独归因于诊断器；更准确的结论是：**候选池有高价值信息，但当前 selector/aggregation 与 deployable gate 都未能识别和利用这些信息。**

## 5. 线 B：直接 Gain / Helpfulness 策略

### 5.1 验证目的

线 B 不再增加深层神经网络，而是使用梯度提升树直接预测：

1. `helpfulness`：Memory 是否比 base 更好；
2. `alpha`：沿 \(Y^{memory}-Y^{base}\) 方向的最优凸融合步长。

该实验的作用是区分两种解释：

- 若简单监督策略明显优于当前校正器，说明当前 `StructuredErrorCorrector` 的结构或训练目标是主瓶颈；
- 若简单策略也只能弱预测，说明部署可见特征与 oracle 决策之间存在信息缺口。

### 5.2 新增候选级 horizon-specific 特征

保留候选 future

\[
Y^{cand}\in\mathbb{R}^{B\times H\times N\times K\times C},
\]

并定义候选相对 base 的修正量

\[
\Delta^{cand}_{q,h,n,k,c}
=Y^{cand}_{q,h,n,k,c}-Y^{base}_{q,h,n,c}.
\]

由候选权重计算五类逐 horizon 特征：加权修正均值、修正标准差、signed direction agreement、正方向权重质量、负方向权重质量。这里保留 signed 值，不再只取绝对值，因此“候选整体向上”和“候选整体向下”可以被区分。

### 5.3 指标定义

- **AUROC**：随机抽取一个 helpful 和一个 harmful 位置，模型把 helpful 排得更高的概率。0.5 约等于随机排序。
- **AUPRC**：在类别不平衡情况下，衡量高分位置是否真正集中 helpful 样本；应与 helpful prevalence 对照。
- **Alpha Spearman**：预测 \(\alpha\) 与 oracle \(\alpha\) 的秩相关，衡量是否至少能正确排序“应多修正”和“应少修正”的位置。
- **Alpha \(R^2\)**：预测对 oracle \(\alpha\) 方差的解释比例。小于 0 表示还不如始终预测训练均值。
- **Alpha MAE**：预测步长与 oracle 步长之间的平均绝对误差。

### 5.4 抽样验证结果

训练和验证各抽样 200,000 个 Memory 有效且方向非退化的位置。由于这是条件子集上的位置级 MAE，其绝对值不能与完整验证集 2.84 直接横向比较，只能比较同一子集内的策略差值。

| 下游模型 | Base 子集 MAE | Memory 子集 MAE | Direct Alpha | Helpfulness × Alpha | AUROC | AUPRC | Prevalence | Alpha Spearman | Alpha R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Graph WaveNet | 3.9422 | 4.4727 | 3.8891 | 3.9039 | 0.5607 | 0.4736 | 0.4315 | 0.0596 | -0.2691 |
| STGCN | 4.1446 | 4.4679 | 3.9259 | 3.9916 | 0.5976 | 0.5515 | 0.4514 | 0.1216 | -0.2992 |

### 5.5 线 B 结果解释

第一，signed/horizon-specific 特征带来了一定 helpfulness 排序信号，但强度仍弱。Graph WaveNet AUROC 0.5607，STGCN 0.5976；AUPRC 仅略高于各自 prevalence。

第二，最优步长几乎不可预测。两个模型的 Alpha Spearman 都很低，且 \(R^2<0\)。这说明当前部署特征不足以恢复 oracle continuous 所使用的逐位置信息。

第三，直接 Alpha 在相同有效子集内确实优于 base，但远没有接近线 A oracle 上界。因此当前 224K 参数校正器的瓶颈不只是“网络不够大”，主要是：

- 候选聚合提前压缩了候选身份与差异；
- 检索分数与未来实际 helpfulness 对齐较弱；
- 现有历史/base 风险特征难以判断未来误差方向；
- oracle 决策依赖的部分信息在当前部署特征中不可辨识。

## 6. 综合瓶颈判断

按证据强度排序：

1. **候选选择和聚合是明确瓶颈。** 合法 Top-K 内存在约 1.64 MAE 的 oracle 上界，而当前聚合 Memory 单独 MAE 约 3.54。
2. **诊断 gate 是明确瓶颈。** 在固定当前聚合 Memory 时，oracle continuous 约 2.0，当前 gate 约 2.84–2.86。
3. **当前部署特征存在信息瓶颈。** 加入 signed/horizon-specific 候选统计后，直接监督模型仍无法准确预测 oracle \(\alpha\)。
4. **不是单纯模型容量瓶颈。** 继续扩大 `StructuredErrorCorrector` 参数量缺少证据支持。

## 7. Keep / Remove / Stop 决策

### Keep

- 保留 TGGE 预训练编码器和因果 Memory Bank；候选池被 oracle 证明包含高价值信息。
- 保留多个下游模型验证路线；两个下游给出一致瓶颈结论。
- 保留 candidate-level、signed、horizon-specific 统计作为后续 selector 诊断输入。

### Remove / Avoid

- 不再把 `abs(direction_agreement)` 作为唯一方向特征。
- 不再优先扩大 224K 校正器容量。
- 不把固定大 \(\alpha\) 或无条件强化 Memory 作为优化方向。

### Stop

- 停止继续训练当前 Graph WaveNet 校正器；其曲线在约 2.86 附近平台化，且线 A/B 已证明继续训练不能解决信息与目标错配。
- 暂停多 seed；先确定结构性改动是否有效，再对最终版本补 seed。

## 8. 下一步最小实验

下一步只做一个单变量实验：**候选级 selector/mixture，不先聚合成单一 Memory**。

输入保留 \([B,H,N,K,C]\) 候选 payload，输出每个 \((h,n,k)\) 的可部署候选权重和一个 base fallback 权重。先固定 retriever、backbone 和 Bank，只训练这个小 selector。其判定规则为：

- 若完整验证集 MAE 明显优于当前 gate，并缩小 `oracle candidate` 差距，则保留候选级 selector；
- 若 helpfulness AUROC 和 Alpha Spearman 仍接近线 B，则停止继续设计复杂 gate，转向改造 TGGE key 的监督，使检索分数直接对齐 downstream conditional gain。

## 9. 产物与复现

- 线 A 脚本：`scripts/diagnose_counterfactual.py`。
- 线 B 脚本：`scripts/diagnose_direct_gain_policy.py`。
- 候选级特征：`stanchor/diagnostics/direct_gain.py`。
- 单元测试：`tests/test_direct_gain_policy.py`。
- 正式结果目录：`artifacts/convergence/line_ab_validation/`。
- 测试结果：31 个直接相关测试通过；此前主线 41 个测试通过。

本报告中的线 A 使用完整验证集；线 B 是 200,000 训练位置和 200,000 验证位置的机制验证。线 B 的子集 MAE不是正式完整验证集指标，论文主表不得直接引用该绝对值。


## 8. 面向 5 分钟单轮约束的新优化方案（2026-08-21）

### 8.1 决策依据

线 A 已显示：Memory helpful rate 约 41%–42%，oracle continuous 约 2.0，而当前 learned gate 仍在 2.84–2.86；因此当前优先瓶颈是候选证据如何按 horizon 被诊断和融合，不是继续扩大 backbone 或重新训练 TGGE 编码器。方案不改变检索候选数量、Bank、下游 backbone 和数据协议。

### 8.2 实施改动

1. horizon-aware 候选诊断：保留候选 future 张量 Y_cand [B,H,N,K,C]，在候选维 K 上计算加权偏移均值、偏移方差和方向统计。
2. 有符号方向特征：保留 signed_direction，不再只使用其绝对值。正值表示候选整体位于 base 预测之上，负值表示整体位于 base 预测之下，绝对值仍表示方向一致程度。
3. 新增两个低成本特征：delta_mean_abs 和 delta_std。
4. 移除主检索路径的 profile 分支：当前 TGGE v3 的 profile_dim=0、key 为 latent 表示，TwoStageRetriever 改为直接使用完整 latent key 排序；NodeCandidates 不再暴露 profile_scores 和 latent_scores。

### 8.3 复杂度与训练成本

新增特征只涉及 [B,H,N,K,C] 的逐元素减法、加权求和和方差，时间复杂度为 O(BHNKC)，不增加 backbone 参数，也不增加检索次数。StructuredErrorCorrector 默认参数量为 224,817，仍与 TGGE 编码器约 300k 处于同量级预算。正式训练建议先跑 3–5 epoch smoke，确认单 epoch 不超过 5 分钟后再跑完整验证。

### 8.4 最小验证矩阵与决策

- base_only：确认 backbone 基线不变。
- current_error_aware：旧特征对照。
- signed+horizon_error_aware：本次改动。
- 固定训练轮数 5，早停 patience 2；Graph WaveNet 和 STGCN 各一组，保持 seed、Bank、候选协议、batch 和优化器一致。
- 记录 MAE、RMSE、helpful rate、risk Spearman、epoch time、corrector 参数量。

若新方案在两个下游平均 MAE 至少改善 0.02 且单轮不超过 5 分钟，则保留；若只改善校准指标、不改善 MAE，则保留为诊断增强但不宣称预测增益；若无改善或超时，则删除新增 delta_std，只保留 signed direction 和 delta mean。

## 9. 当前代码验证记录

- python -m py_compile stanchor/retrieval/retriever.py stanchor/models/downstream.py：通过。
- conda run -n research python 合成前向：特征形状 [2,12,5,12]，memory validity [2,12,5,1]，通过。
- conda run -n research python：StructuredErrorCorrector 参数量 224817。
- 尚未运行正式 Graph WaveNet/STGCN 训练，不能提前声称 MAE 已改善；下一步执行 5-epoch matched smoke。

## 10. 环境约定

本项目所有训练、验证和指标复现实验统一使用 Conda 环境 research。不要使用系统默认 Python 直接判断训练代码是否可运行。
