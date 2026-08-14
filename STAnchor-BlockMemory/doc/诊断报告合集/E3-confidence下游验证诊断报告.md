# E3 Confidence 下游验证诊断报告

## 1. 诊断目标与实验边界

本实验只回答一个问题：在 E3 Future-Relation 编码器、检索库、Top-K 候选和下游 backbone 全部不变时，节点级 confidence 是否比 horizon-only fusion 更能抑制有害历史记忆并提高预测精度。

严格固定的条件包括：

- 数据集与划分：METR-LA validation；
- 预训练 checkpoint：`artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt`；
- Bank：`artifacts/metrla_bank_e3_relation_relation`；
- 下游随机种子：42；
- `event_top_r=32`、`node_top_k=5`；
- batch size、学习率、backbone、早停指标和最大训练轮数均不变；
- 未使用 test 数据。

对照组是 `learned_topk_horizon`，实验组是 `learned_topk_confidence`。唯一机制变化是实验组启用 257 参数的 confidence head，并让它参与融合。

第一次前台运行在第 32 轮被外层命令超时终止，不能作为完整正式实验。随后使用相同配置从头进行干净重跑，完整训练 50 轮，最终 best checkpoint 位于第 48 轮。本报告只使用干净重跑结果。

## 2. 两种融合方式

设：

- \(\widehat{\mathbf Y}^{\mathrm{base}}\in\mathbb R^{B\times H\times N\times C}\)：当前输入窗口经下游 backbone 得到的预测；
- \(\widehat{\mathbf Y}^{\mathrm{mem}}\in\mathbb R^{B\times H\times N\times C}\)：检索到的历史 future 经 Top-K 聚合后的 memory 预测；
- \(a_h\in(0,1)\)：第 \(h\) 个预测步的可学习 horizon 权重上限；
- \(q_{b,h,n}\in[0,1]\)：confidence head 对样本 \(b\)、预测步 \(h\)、节点 \(n\) 输出的 memory helpful 程度；
- \(B,H,N,C\)：batch 大小、预测步数、节点数和输出通道数，本实验中 \(H=12\)、\(N=207\)、\(C=1\)。

### 2.1 Horizon-only

horizon-only 不进行样本级置信度判断，相当于所有 memory 有效位置都有 \(q_{b,h,n}=1\)：

\[
\widehat{Y}^{\mathrm{hor}}_{b,h,n,c}
=
\widehat{Y}^{\mathrm{base}}_{b,h,n,c}
+a_h
\left(
\widehat{Y}^{\mathrm{mem}}_{b,h,n,c}
-\widehat{Y}^{\mathrm{base}}_{b,h,n,c}
\right).
\]

### 2.2 Confidence fusion

confidence 模式使用节点和预测步相关的 \(q_{b,h,n}\) 调节 memory：

\[
w_{b,h,n}=a_h q_{b,h,n},
\]

\[
\widehat{Y}^{\mathrm{conf}}_{b,h,n,c}
=
\widehat{Y}^{\mathrm{base}}_{b,h,n,c}
+w_{b,h,n}
\left(
\widehat{Y}^{\mathrm{mem}}_{b,h,n,c}
-\widehat{Y}^{\mathrm{base}}_{b,h,n,c}
\right).
\]

因此，confidence 的任务不是重新生成历史预测，而是决定每个节点在每个预测步应该信任多少已有 memory。

## 3. 预测结果

### 3.1 Overall 指标

| 指标 | E3 horizon-only | E3 confidence | confidence - horizon | 相对改善 |
|---|---:|---:|---:|---:|
| MAE | 3.219274 | **3.134965** | **-0.084309** | **2.62%** |
| RMSE | 6.333624 | **6.300571** | **-0.033053** | **0.52%** |
| MAPE (%) | 9.179310 | **8.872265** | **-0.307045** | **3.35%** |

三类 overall 误差均改善。相比 E2 到 E3 horizon-only 约 0.09% 的 MAE 改善，这次 confidence 带来的 2.62% 单 seed MAE 改善明显更大。

### 3.2 15/30/60 分钟指标

| 预测位置 | 指标 | E3 horizon-only | E3 confidence | 差值 |
|---|---|---:|---:|---:|
| 15 min | MAE | 2.783356 | **2.750996** | -0.032360 |
| 15 min | RMSE | 5.392874 | **5.372244** | -0.020630 |
| 15 min | MAPE (%) | 7.465772 | **7.369814** | -0.095958 |
| 30 min | MAE | 3.251901 | **3.191434** | -0.060468 |
| 30 min | RMSE | 6.412666 | **6.402961** | -0.009704 |
| 30 min | MAPE (%) | 9.220407 | **9.014286** | -0.206121 |
| 60 min | MAE | 3.822934 | **3.650078** | -0.172856 |
| 60 min | RMSE | 7.411068 | **7.349329** | -0.061739 |
| 60 min | MAPE (%) | 11.624013 | **10.953967** | -0.670046 |

九项分预测步指标全部改善，且 MAE 收益随预测距离增加。confidence 对中长程预测的帮助最明显。

## 4. Confidence 指标定义

设 \(i\) 表示一个 memory 和真实目标均有效的“样本-预测步-节点”位置。base 和 memory 的绝对误差分别为：

\[
e_i^{\mathrm{base}}=
\frac{1}{C}\sum_{c=1}^{C}
\left|\widehat{Y}^{\mathrm{base}}_{i,c}-Y_{i,c}\right|,
\]

\[
e_i^{\mathrm{mem}}=
\frac{1}{C}\sum_{c=1}^{C}
\left|\widehat{Y}^{\mathrm{mem}}_{i,c}-Y_{i,c}\right|.
\]

二值 helpful 标签为：

\[
u_i=\mathbb I\!\left(e_i^{\mathrm{mem}}<e_i^{\mathrm{base}}\right),
\qquad u_i\in\{0,1\}.
\]

\(u_i=1\) 表示该位置 memory 比 base 更准确。\(q_i\in[0,1]\) 是 confidence head 的输出。真实 future 只在训练监督和离线诊断中用于构造标签，推理时不可访问。

### 4.1 AUROC

\[
\operatorname{AUROC}
=
\Pr(q_i>q_j\mid u_i=1,u_j=0),
\]

平分时计 0.5。AUROC 衡量 helpful 位置是否通常被赋予更高 confidence；0.5 约等于随机排序，越高越好。

### 4.2 AUPRC 与 prevalence

helpful-memory prevalence 是正样本比例：

\[
\pi=\frac{1}{M}\sum_{i=1}^{M}u_i,
\]

其中 \(M\) 是有效位置总数。AUPRC 是 Precision-Recall 曲线下面积。随机或常数排序的 AUPRC 基线约为 \(\pi\)，因此必须比较 \(\operatorname{AUPRC}>\pi\)。

### 4.3 Brier score

\[
\operatorname{Brier}
=
\frac{1}{M}\sum_{i=1}^{M}(q_i-u_i)^2.
\]

Brier 越低越好。固定输出正样本比例 \(q_i=\pi\) 的常数基线为：

\[
\operatorname{Brier}_{\mathrm{const}}=\pi(1-\pi).
\]

### 4.4 Confidence 四分位 memory gain

按 \(q_i\) 从低到高将位置划分为四组 \(\mathcal Q_1,\ldots,\mathcal Q_4\)。第 \(r\) 组的 memory gain 为：

\[
G_r^{\mathrm{mem}}
=
\frac{1}{|\mathcal Q_r|}
\sum_{i\in\mathcal Q_r}
\left(e_i^{\mathrm{base}}-e_i^{\mathrm{mem}}\right).
\]

正值表示 memory 有帮助，负值表示 memory 有害。若 confidence 有语义，\(G_r^{\mathrm{mem}}\) 应随置信度总体上升，且最高置信度组应为正。

## 5. Confidence 诊断结果

| 指标 | 结果 | 基线或门槛 | 判断 |
|---|---:|---:|---|
| 有效位置数 | 7,189,753 | - | 覆盖充分 |
| helpful prevalence | 0.459140 | - | 45.91% 的位置 memory 更好 |
| AUROC | **0.548021** | 0.5 | 通过，但排序能力偏弱 |
| AUPRC | **0.496208** | 0.459140 | 通过，高于 prevalence 0.037068 |
| Brier | **0.247578** | constant 0.248330 | 仅小幅通过，改善 0.000752 |

confidence 的分布为：

| mean | q10 | median | q90 | min | max |
|---:|---:|---:|---:|---:|---:|
| 0.490970 | 0.418030 | 0.497298 | 0.544723 | 0.005666 | 0.989079 |

它没有坍缩为单一常数，但主要质量仍集中在 0.42 至 0.54 附近。因此，confidence 已学习到可用排序信息，但不是一个强分离、强校准的概率估计器。

四分位结果为：

| 四分位 | confidence 均值 | BaseMAE | MemoryMAE | memory gain | helpful rate |
|---|---:|---:|---:|---:|---:|
| Q1 最低 | 0.407172 | 4.246403 | 5.768372 | **-1.521968** | 0.387097 |
| Q2 | 0.486576 | 2.434274 | 2.641176 | **-0.206902** | 0.466302 |
| Q3 | 0.503624 | 1.951606 | 2.027928 | **-0.076323** | 0.493021 |
| Q4 最高 | 0.566509 | 4.891359 | 4.779488 | **+0.111871** | 0.490140 |

memory gain 从 Q1 到 Q4 严格单调上升，并且 Q4 为正。这是本次实验中 confidence 语义最直接的证据：低置信度组的 memory 明显有害，高置信度组的 memory 平均有帮助。

不过，Q4 的二值 helpful rate 略低于 Q3，而 AUROC 只有 0.548。这表明 confidence 更接近“memory 收益幅度的软排序”，尚不能精确区分每一个 helpful/harmful 位置。该结论与当前使用连续 soft target 训练是一致的。

## 6. 分支诊断

| 指标 | Horizon-only | Confidence | 解释 |
|---|---:|---:|---|
| BaseMAE | 3.373584 | 3.381510 | confidence 训练出的 base 分支略差 |
| MemoryMAE | 3.804241 | 3.804241 | checkpoint 与 Bank 固定，符合预期 |
| FinalMAE | 3.219274 | **3.134964** | confidence 最终预测明显更好 |
| FinalGain | 0.154310 | **0.246546** | 对各自 base 的融合收益增加 |
| mean fusion weight | 0.400085 | 0.423479 | confidence 不是简单全局减小 memory 权重 |

raw memory 的 MAE 仍高于 base，因此“历史记忆本身已经足够准确”仍不成立。confidence 的价值在于根据输入特征重新分配 memory 权重：它对明显有害的 Q1 位置降低信任，同时允许 Q4 使用更多 memory。

## 7. 参数与时间成本

| 项目 | Horizon-only | Confidence |
|---|---:|---:|
| 可训练参数 | 5,784 | 6,041 |
| 新增可训练参数 | 0 | 257（约 +4.44%） |
| best epoch | 10 | 48 |
| 实际训练 epoch | 20 | 50 |
| 训练时间 | 约 17 min 23 s | 约 50 min 27 s |

confidence head 本身很小，但由于最佳 MAE 持续后移，完整训练时间约为 horizon-only 的 2.9 倍。当前配置的主要代价是优化轮数，不是模型参数规模。

## 8. 最终判断

### 8.1 Seed=42 门槛结论

预设五项门槛全部通过：

1. final MAE 低于 E3 horizon-only；
2. AUROC 大于 0.5；
3. AUPRC 大于 helpful prevalence；
4. Brier 低于常数 Brier；
5. confidence 四分位 memory gain 总体上升且 Q4 为正。

因此，**保留 confidence 作为候选主模块，不退回纯 horizon-only。**

### 8.2 结论边界

当前可以宣称：confidence 在 METR-LA validation seed=42 上改善了预测，并学习到与 memory 收益方向一致的节点级排序信息。

当前不能宣称：

- confidence 已经彻底解决时空海市蜃楼；
- confidence 是高质量校准概率；
- 2.62% MAE 收益已跨随机种子稳定；
- 当前结果可直接代表 test 或其他数据集。

AUROC 仅为 0.548，Brier 对常数基线的改善也很小，说明机制有效但仍有明显上限。此时不应继续加深 confidence 网络；应先验证现有简单 head 的下游随机种子稳定性。

## 9. 后续验证状态

E3 confidence 的下游 seed 2024 和 2025 已经补齐，并分别与 E3 horizon-only 同 seed 结果完成配对比较。

保留规则：

- confidence 三 seed 平均 overall MAE 低于 horizon-only；
- 至少两个 seed 的 overall MAE 改善；
- 15/30/60 分钟平均 MAE 不出现超过 0.01 的系统性退化；
- AUROC/AUPRC/Brier 和四分位 gain 的主要方向可复现。

三 seed 保留规则全部满足，confidence 不再是单 seed 现象。完整结果见：

```text
doc/诊断报告合集/E3-confidence三随机种子诊断报告.md
```

当前模型、损失、Bank、Top-K、confidence 配置和训练预算冻结，不再继续调参或增加模块。下一步只允许按照冻结协议执行最终 test。

## 10. 正式证据文件

- 完整正式 run：`artifacts/metrla_e3_relation_topk_confidence_seed42_retry1`；
- checkpoint：`downstream_best.pt`，epoch 48；
- 训练日志：`downstream.log`；
- 独立 validation：`val_evaluation.json`；
- 分支与 confidence 诊断：`branch_diagnostics_val.json`；
- 本报告未使用第一次被外层超时中断的结果，也未使用 test 数据。
