# PEMS-BAY E3 迁移 Seed 42 Confidence 诊断报告

## 1. 诊断问题与边界

本报告只回答一个问题：

> 在相同 PEMS-BAY 数据划分、E3 检索 Bank、下游 backbone、训练预算和 seed 42 下，节点级、预测步级 confidence 是否比 horizon-only fusion 提供了额外收益，并且是否真的学会区分“memory 有帮助”和“memory 有害”的位置？

本报告仅使用 validation。PEMS-BAY test 尚未读取，因此这里的结果不是最终测试结果，也不能证明三随机种子稳定性。

## 2. 实验一致性

两种待比较模式共享以下条件：

- 源 checkpoint：`artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt`；
- 目标 Bank：`artifacts/pemsbay_bank_from_metrla_e3_relation`；
- 下游配置：`configs/pemsbay_e3_transfer_v1.yaml`；
- 随机种子：42；
- 最佳 checkpoint 选择标准：validation MAE；
- 最佳 epoch：两种模式均为 epoch 34。

唯一的主要结构差异是：`learned_topk_confidence` 在 horizon fusion 上增加一个 257 参数的 confidence head。其下游可训练参数总数为 6,041，其中 backbone 为 5,772，fusion 为 12，confidence head 为 257；冻结的 E3 模块为 391,836 参数。

## 3. 最终预测结果

| 模式 | Validation MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| base-only | 2.171461 | 5.097139 | 5.232578 |
| learned Top-K + horizon-only | 1.881370 | 4.107756 | 4.390352 |
| **learned Top-K + confidence** | **1.804609** | **4.037364** | **4.192822** |

confidence 相对 horizon-only 的改善为：

| 指标 | 绝对改善 | 相对改善 |
|---|---:|---:|
| MAE | 0.076761 | **4.08%** |
| RMSE | 0.070392 | **1.71%** |
| MAPE | 0.197530 | **4.50%** |

以 MAE 为例，设 $M_{\mathrm{hor}}$ 和 $M_{\mathrm{conf}}$ 分别为 horizon-only 与 confidence 模式的 validation MAE，则

\[
G_{\mathrm{conf}}
=
\frac{M_{\mathrm{hor}}-M_{\mathrm{conf}}}{M_{\mathrm{hor}}}
\times 100\%
=
\frac{1.881370-1.804609}{1.881370}
\times 100\%
=4.08\%.
\]

confidence 最终系统相对独立训练的 base-only 系统降低 MAE 16.89%。不过这是两个分别联合训练的系统之间的比较，不能把全部差值都归因于 confidence head 本身；判断 confidence 独立贡献时，应以 4.08% 的 horizon-only 对照为主。

## 4. 不同预测距离的收益

| 预测位置 | Horizon-only MAE | Confidence MAE | 相对改善 |
|---|---:|---:|---:|
| 15 min | 1.498895 | 1.470302 | 1.91% |
| 30 min | 1.965845 | 1.880763 | 4.33% |
| 60 min | 2.359796 | 2.245789 | 4.83% |

confidence 在三个关键预测距离上均有改善，而且中长预测步收益更明显。这与当前机制动机一致：短期预测主要依赖最近 12 步，随着 horizon 增长，history memory 更有价值，但也更需要按节点识别其可靠性。

## 5. Confidence 的训练语义

对任意有效预测位置 $i=(b,h,n)$，定义：

- $b\in\{1,\ldots,B\}$：batch 中的样本索引；
- $h\in\{1,\ldots,H\}$：预测步索引，本实验 $H=12$；
- $n\in\{1,\ldots,N\}$：节点索引，PEMS-BAY 中 $N=325$；
- $c\in\{1,\ldots,C\}$：预测通道索引，本实验 $C=1$；
- $\widehat{\mathbf Y}^{\mathrm{base}},\widehat{\mathbf Y}^{\mathrm{mem}},\mathbf Y\in\mathbb R^{B\times H\times N\times C}$：backbone 预测、memory 预测和真实未来；
- $q_{b,h,n}\in[0,1]$：confidence head 输出，形状为 $B\times H\times N\times1$。

base 与 memory 在位置 $i$ 的平均绝对误差为

\[
e_i^{\mathrm{base}}
=
\frac{1}{C}\sum_{c=1}^{C}
\left|
\widehat Y^{\mathrm{base}}_{b,h,n,c}-Y_{b,h,n,c}
\right|,
\]

\[
e_i^{\mathrm{mem}}
=
\frac{1}{C}\sum_{c=1}^{C}
\left|
\widehat Y^{\mathrm{mem}}_{b,h,n,c}-Y_{b,h,n,c}
\right|.
\]

训练时的 soft confidence target 为

\[
q_i^*
=
\sigma\!\left(
\frac{e_i^{\mathrm{base}}-e_i^{\mathrm{mem}}-m}{\tau}
\right),
\]

其中 $m$ 是“memory 必须领先多少才算有帮助”的 margin，本实验为 0；$\tau$ 是平滑温度，本实验为 0.1；$\sigma(\cdot)$ 是 Sigmoid 函数。memory 误差越小，$q_i^*$ 越接近 1。

confidence 损失为有效位置上的均方误差：

\[
\mathcal L_{\mathrm{conf}}
=
\frac{1}{|\Omega|}
\sum_{i\in\Omega}
(q_i-q_i^*)^2,
\]

其中 $\Omega$ 是 memory 有效且真实未来可观测的位置集合。最终融合为

\[
\widehat{\mathbf Y}_{b,h,n,:}
=
\widehat{\mathbf Y}^{\mathrm{base}}_{b,h,n,:}
+a_hq_{b,h,n}
\left(
\widehat{\mathbf Y}^{\mathrm{mem}}_{b,h,n,:}
-\widehat{\mathbf Y}^{\mathrm{base}}_{b,h,n,:}
\right),
\]

其中 $a_h\in(0,1)$ 是第 $h$ 个预测步共享的 memory 权重上限，实际 fusion weight 为 $w_{b,h,n}=a_hq_{b,h,n}$。

## 6. Confidence 质量指标

validation 共评估 19,155,216 个有效节点-预测步位置。诊断时使用硬标签

\[
d_i=\mathbb I\!\left(e_i^{\mathrm{mem}}<e_i^{\mathrm{base}}\right),
\]

其中 $d_i=1$ 表示 memory 在该位置比 base 更准确。正样本比例为 46.74%。注意：训练使用平滑的 $q_i^*$，诊断使用二值的 $d_i$，所以以下指标用于判断排序和校准趋势，不应把 $q_i$ 机械解释成严格的真实概率。

| 指标 | 结果 | 基准或含义 | 判断 |
|---|---:|---|---|
| AUROC | 0.5977 | 随机排序为 0.5 | 有区分力，但强度中等 |
| AUPRC | 0.5844 | 随机基准为正样本率 0.4674 | 明显高于基准 |
| Brier score | 0.2396 | 常数预测基准为 0.2489 | 降低 3.76%，校准有效但幅度不大 |
| ECE | 0.0252 | 越接近 0 越好 | 10 个等宽分箱下平均偏差约 2.52 个百分点 |

AUROC 可写成

\[
\operatorname{AUROC}
=
\Pr(q^+>q^-)
+\frac12\Pr(q^+=q^-),
\]

其中 $q^+$ 来自 memory 有帮助的位置，$q^-$ 来自 memory 无帮助的位置。AUPRC 是 precision-recall 曲线下面积，衡量高 confidence 位置中有用 memory 的集中程度。

Brier score 为

\[
\operatorname{Brier}
=
\frac{1}{M}\sum_{i=1}^{M}(q_i-d_i)^2,
\]

其中 $M=|\Omega|$。ECE 将 confidence 划入 10 个等宽区间 $B_j$：

\[
\operatorname{ECE}
=
\sum_{j=1}^{10}
\frac{|B_j|}{M}
\left|
\operatorname{acc}(B_j)-\operatorname{conf}(B_j)
\right|.
\]

这里 $\operatorname{acc}(B_j)$ 是分箱内 $d_i=1$ 的比例，$\operatorname{conf}(B_j)$ 是分箱内 $q_i$ 的均值。低 ECE 只表示数值校准较好，不等价于排序能力很强，因此必须与 AUROC/AUPRC 一起判断。

## 7. 最关键的分桶证据

将所有有效位置按 confidence 从低到高等量分成四组，定义 memory gain 为

\[
\operatorname{Gain}(Q_r)
=
\frac{1}{|Q_r|}\sum_{i\in Q_r}e_i^{\mathrm{base}}
-
\frac{1}{|Q_r|}\sum_{i\in Q_r}e_i^{\mathrm{mem}}.
\]

正值表示 memory 更好，负值表示 memory 更差。

| Confidence 分组 | Confidence 均值 | Memory helpful rate | Base MAE | Memory MAE | Memory gain |
|---|---:|---:|---:|---:|---:|
| Q1（最低） | 0.3736 | 37.15% | 2.8218 | 3.7766 | -0.9548 |
| Q2 | 0.4706 | 44.36% | 1.4910 | 1.6642 | -0.1733 |
| Q3 | 0.5049 | 46.70% | 1.1762 | 1.2668 | -0.0905 |
| Q4（最高） | 0.5899 | 58.74% | 3.3902 | 2.1300 | **+1.2601** |

这是本次最重要的机制证据：随着 confidence 升高，memory helpful rate 单调上升，memory gain 也从强负值单调上升到显著正值。尤其 Q4 中 memory 将 MAE 从 3.3902 降到 2.1300，改善 37.17%。这说明 confidence 不只是一个接近 0.5 的常数，而是学到了“哪些位置值得依赖历史模式”的方向性语义。

与此同时，Q1 至 Q3 的原始 memory 平均仍弱于 base，说明 confidence 的任务确实必要；它不是证明所有检索结果都可靠，而是在帮助系统回避时空海市蜃楼。

## 8. Confidence 与融合权重分布

| 统计量 | Confidence $q$ | 实际 fusion weight $w=a_hq$ |
|---|---:|---:|
| mean | 0.4848 | 0.4073 |
| q10 | 0.3902 | 0.1748 |
| median | 0.4890 | 0.4371 |
| q90 | 0.5531 | 0.5305 |
| min / max | 0.0018 / 0.9982 | 0.0009 / 0.9858 |

confidence 的中间 80% 仍集中在 0.39 至 0.55，说明其区分强度尚不高；但标准差为 0.1033，且分桶结果具有单调语义，因此不能判定为 0.5 坍缩。

平均 fusion weight 随预测步从 0.1329 上升到 0.4917；关键位置为：

| 预测位置 | 平均 fusion weight |
|---|---:|
| 15 min | 0.3231 |
| 30 min | 0.4627 |
| 60 min | 0.4917 |

相比 horizon-only 在 60 分钟使用统一权重 0.7134，confidence 模式将远期平均权重降到 0.4917，同时取得更低误差。这支持“选择性使用 memory”比“对所有节点统一增加 memory”更合理。

## 9. 分支归因与限制

同一 confidence checkpoint 的分支结果为：

| 分支 | Validation MAE |
|---|---:|
| base branch | 2.219782 |
| memory branch | 2.209414 |
| final confidence fusion | **1.804609** |

memory 单独只比同 checkpoint 的 base branch 改善 0.47%，但最终融合改善 18.70%。这说明主要价值来自两个误差互补分支的选择性融合，而不是 memory 在所有位置都更准确。

仍需保留三个限制：

1. 当前只有 seed 42，不能声称结果稳定；
2. 尚未完成同架构 target-random 对照，不能把迁移收益确定归因于 METR-LA 预训练；
3. confidence 的 AUROC 只有 0.5977，属于有效但中等的区分能力，不应包装成高精度分类器。

## 10. 结论与决策

本轮结论为：

> **保留 confidence 作为当前候选最终模块，不再修改其结构；seed 42 validation 已同时通过预测收益和机制语义两道门槛。**

依据是：

1. 相对 horizon-only，MAE、RMSE、MAPE 全部改善，MAE 降低 4.08%；
2. 15、30、60 分钟 MAE 全部改善；
3. AUROC/AUPRC 高于随机基准，Brier 优于常数基准；
4. confidence 四分位中的 helpful rate 和 memory gain 均呈单调改善；
5. 只增加 257 个 confidence 参数，收益并非依赖大幅扩张模型。

下一步不增加模块、不调 confidence 结构，只做同架构 target-random encoder 对照，回答“当前迁移收益究竟来自预训练，还是随机投影加历史检索也能获得”。

## 11. 证据文件

- confidence 最佳 checkpoint：`artifacts/pemsbay_e3_learned_topk_confidence_seed42/downstream_best.pt`；
- 完整训练日志：`artifacts/pemsbay_e3_learned_topk_confidence_seed42/downstream.log`；
- 每轮指标：`artifacts/pemsbay_e3_learned_topk_confidence_seed42/target_metrics.jsonl`；
- validation 分支与 confidence 诊断：`artifacts/pemsbay_e3_learned_topk_confidence_seed42/branch_diagnostics_val.json`；
- horizon-only 对照：`artifacts/pemsbay_e3_learned_topk_horizon_seed42/downstream_best.pt`；
- base-only 对照：`artifacts/pemsbay_e3_base_only_seed42/downstream_best.pt`。
