# E3 Confidence 三随机种子诊断报告

## 1. 诊断问题

本报告检验 E3 confidence 在 seed 42、2024、2025 三种下游初始化下是否稳定优于对应的 E3 horizon-only，并判断 confidence 的排序与校准语义能否跨 seed 保持。

三组实验固定使用：

- METR-LA validation；
- 同一个 E3 Future-Relation 预训练 checkpoint；
- 同一个 E3 检索 Bank；
- 相同的 backbone、Top-K、batch size、学习率、最大 50 轮与早停设置；
- 相同 seed 下的 horizon-only 作为配对对照；
- 不使用 test 数据。

因此，本实验验证的是**下游随机初始化稳定性**。三组实验共用一个预训练 checkpoint 和 Bank，不能写成三个独立预训练 seed。

## 2. 统计口径

设随机种子数为 \(S=3\)，方法 \(m\in\{\mathrm{hor},\mathrm{conf}\}\)，指标为 \(x_{m,s}\)。均值与样本标准差分别为：

\[
\bar{x}_{m}=\frac{1}{S}\sum_{s=1}^{S}x_{m,s},
\]

\[
\operatorname{Std}(x_m)
=
\sqrt{
\frac{1}{S-1}
\sum_{s=1}^{S}\left(x_{m,s}-\bar{x}_m\right)^2
}.
\]

其中，\(x_{m,s}\) 表示方法 \(m\) 在第 \(s\) 个下游 seed 上的 validation 指标。

同 seed 配对差定义为：

\[
\Delta_s(x)=x_{\mathrm{conf},s}-x_{\mathrm{hor},s}.
\]

对于 MAE、RMSE、MAPE，\(\Delta_s(x)<0\) 表示 confidence 更好。MAE 相对改善率为：

\[
G_s^{\mathrm{MAE}}
=
\frac{
\operatorname{MAE}_{\mathrm{hor},s}
-\operatorname{MAE}_{\mathrm{conf},s}
}{
\operatorname{MAE}_{\mathrm{hor},s}
}
\times100\%.
\]

## 3. Overall 三 seed 结果

| 指标 | Horizon-only | Confidence | 配对差 conf - horizon |
|---|---:|---:|---:|
| MAE | 3.218329 ± 0.001669 | **3.138844 ± 0.003416** | **-0.079485 ± 0.004664** |
| RMSE | 6.374059 ± 0.047382 | **6.297428 ± 0.003789** | **-0.076631 ± 0.051169** |
| MAPE (%) | 9.172200 ± 0.023557 | **8.906897 ± 0.031961** | **-0.265303 ± 0.049477** |

平均 MAE 相对改善率为：

\[
\overline{G^{\mathrm{MAE}}}
=2.4697\%\pm0.1439\%.
\]

三类 overall 误差的平均值均改善。MAE 配对差的标准差只有 0.004664，说明收益幅度在三个下游初始化之间较稳定。

## 4. 逐 seed 配对结果

| seed | Horizon MAE | Confidence MAE | MAE 差 | MAE 相对改善 | RMSE 差 | MAPE 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 3.219274 | **3.134965** | -0.084309 | 2.6189% | -0.033053 | -0.307045 |
| 2024 | 3.216402 | **3.141402** | -0.075000 | 2.3318% | -0.132974 | -0.210650 |
| 2025 | 3.219312 | **3.140166** | -0.079147 | 2.4585% | -0.063866 | -0.278215 |

confidence 在三个 seed 上都同时改善 MAE、RMSE 和 MAPE，不存在依赖某一个 seed 才成立的方向反转。

## 5. 15/30/60 分钟结果

### 5.1 MAE

| 预测位置 | Horizon-only | Confidence | 配对差 conf - horizon |
|---|---:|---:|---:|
| 15 min | 2.777951 ± 0.007825 | **2.750005 ± 0.001698** | **-0.027947 ± 0.008628** |
| 30 min | 3.244463 ± 0.009087 | **3.192250 ± 0.001133** | **-0.052213 ± 0.009034** |
| 60 min | 3.836467 ± 0.017553 | **3.663236 ± 0.011400** | **-0.173231 ± 0.013385** |

三个预测位置的 MAE 在每个 seed 上都改善，并且平均改善随预测距离增加。confidence 对中长程预测帮助最明显。

### 5.2 RMSE 与 MAPE

| 预测位置 | RMSE 差 | MAPE 差 |
|---|---:|---:|
| 15 min | -0.029517 ± 0.009813 | -0.093182 ± 0.037614 |
| 30 min | -0.049713 ± 0.044476 | -0.171178 ± 0.051090 |
| 60 min | -0.138767 ± 0.095092 | -0.598344 ± 0.063648 |

RMSE 和 MAPE 的平均差在三个目标预测位置上也全部为负。confidence 没有用改善 MAE 换取相对误差退化。

## 6. Confidence 机制指标

设位置 \(i\) 的 base 与 memory 绝对误差分别为 \(e_i^{\mathrm{base}}\) 和 \(e_i^{\mathrm{mem}}\)，helpful 标签与 confidence 输出为：

\[
u_i=\mathbb I\!\left(e_i^{\mathrm{mem}}<e_i^{\mathrm{base}}\right),
\qquad q_i\in[0,1].
\]

### 6.1 排序与校准定义

AUROC 表示随机抽取一个 helpful 和一个 harmful 位置时，helpful 位置 confidence 更高的概率：

\[
\operatorname{AUROC}
=
\Pr(q_i>q_j\mid u_i=1,u_j=0),
\]

相同分数按 0.5 计。0.5 约等于随机排序。

helpful prevalence 为：

\[
\pi=\frac{1}{M}\sum_{i=1}^{M}u_i.
\]

AUPRC 必须高于 \(\pi\)，才优于随机或常数排序基线。

Brier score 与常数基线为：

\[
\operatorname{Brier}
=
\frac{1}{M}\sum_{i=1}^{M}(q_i-u_i)^2,
\qquad
\operatorname{Brier}_{\mathrm{const}}=\pi(1-\pi).
\]

Brier 越低越好。

### 6.2 三 seed 结果

| seed | AUROC | AUPRC | Prevalence | Brier | Constant Brier |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.548021 | 0.496208 | 0.459140 | **0.247578** | 0.248330 |
| 2024 | 0.550994 | 0.497833 | 0.458551 | **0.247277** | 0.248282 |
| 2025 | 0.550643 | 0.497740 | 0.458222 | **0.247218** | 0.248255 |

三个 seed 均满足：

- AUROC 大于 0.5；
- AUPRC 大于 prevalence；
- Brier 低于 constant Brier。

但 AUROC 只在 0.548 至 0.551，Brier 相比常数基线也只改善约 0.00075 至 0.00104。结论应是 confidence 学到了稳定但偏弱的排序与校准信息，而不是高精度 helpful 概率。

## 7. Confidence 四分位检验

按 \(q_i\) 从低到高划分四分位 \(\mathcal Q_1,\ldots,\mathcal Q_4\)，第 \(r\) 组 memory gain 为：

\[
G_r^{\mathrm{mem}}
=
\frac{1}{|\mathcal Q_r|}
\sum_{i\in\mathcal Q_r}
\left(e_i^{\mathrm{base}}-e_i^{\mathrm{mem}}\right).
\]

| seed | Q1 最低 | Q2 | Q3 | Q4 最高 |
|---:|---:|---:|---:|---:|
| 42 | -1.521968 | -0.206902 | -0.076323 | **+0.111871** |
| 2024 | -1.578773 | -0.228223 | -0.070988 | **+0.180620** |
| 2025 | -1.540137 | -0.256217 | -0.094479 | **+0.184475** |

每个 seed 都满足：

\[
G_1^{\mathrm{mem}}
<G_2^{\mathrm{mem}}
<G_3^{\mathrm{mem}}
<G_4^{\mathrm{mem}},
\qquad G_4^{\mathrm{mem}}>0.
\]

这是 confidence 语义跨 seed 稳定性的最强证据：它能够把 memory 严重有害的位置排到低置信度组，并把平均有帮助的位置集中到最高置信度组。

## 8. 分支归因

| 指标 | Horizon-only | Confidence | 配对差 conf - horizon |
|---|---:|---:|---:|
| BaseMAE | **3.371564 ± 0.002475** | 3.380087 ± 0.001669 | +0.008522 ± 0.002923 |
| MemoryMAE | 3.804241 ± 0.000000 | 3.804241 ± 0.000000 | 约 0 |
| FinalGain | 0.153235 ± 0.000977 | **0.241243 ± 0.004620** | +0.088008 ± 0.003749 |

confidence 模式的 base 分支反而略差，memory 分支因为 checkpoint 和 Bank 固定而保持不变，最终预测却稳定更好。因此收益不能归因于 backbone 偶然增强，而来自样本级融合权重更有效地使用了同一份 memory。

raw MemoryMAE 仍高于 BaseMAE，说明历史记忆不能独立替代当前窗口。confidence 的贡献是控制 memory 的使用条件，而不是把检索结果变成普遍可靠的直接预测。

## 9. 训练成本

| seed | Best epoch | Stop epoch | 训练时间 |
|---:|---:|---:|---:|
| 42 | 48 | 50 | 50 min 27 s |
| 2024 | 45 | 50 | 49 min 22 s |
| 2025 | 47 | 50 | 51 min 23 s |

confidence 三 seed 均跑满 50 轮，平均训练时间约 50 min 24 s；horizon-only 平均约 16 min 51 s。confidence head 只新增 257 个可训练参数，但最佳 checkpoint 明显后移，因此当前主要代价是约 3 倍训练时间，而不是参数规模。

## 10. 最终决策

### 10.1 保留结论

预设多 seed 保留规则全部通过：

1. confidence 三 seed 平均 overall MAE 低于 horizon-only；
2. 三个 seed 的 overall MAE 均改善；
3. 15/30/60 分钟平均 MAE 全部改善，没有系统性退化；
4. AUROC、AUPRC、Brier 和四分位 gain 的主要方向在三个 seed 上复现。

因此，**正式保留 E3 confidence 作为当前最终下游主方法，E3 horizon-only 保留为无 confidence 消融。**

### 10.2 可以宣称的内容

在 METR-LA validation 和固定 E3 预训练 checkpoint/Bank 下，简单的 257 参数 confidence head：

- 跨三个下游 seed 稳定改善预测；
- 平均降低 MAE 约 2.47%；
- 学习到与 memory 收益方向一致的节点级置信排序；
- 对 30/60 分钟中长程预测帮助更明显。

### 10.3 不能宣称的内容

当前仍不能宣称：

- confidence 已完全解决时空海市蜃楼；
- confidence 是强校准概率模型；
- 结果已经跨预训练 seed、数据集或交通变量类型泛化；
- validation 结果等同于 test 结果。

## 11. 最终 Test 状态

现有结构、损失、Bank、Top-K、confidence 配置和最大训练轮数已经冻结。三个 confidence checkpoint 的唯一一次最终 test 已完成，完整结果见：

```text
doc/诊断报告合集/E3-confidence最终Test报告.md
```

test 三 seed overall 结果为 `MAE 3.373908 ± 0.004599`、`RMSE 6.653555 ± 0.003634`、`MAPE 9.836541% ± 0.035791%`。此后禁止根据 test 更换 seed、checkpoint、超参数或模型结构。

## 12. 正式证据

- 机器可读汇总：`artifacts/metrla_e3_confidence_multiseed_val_summary.json`；
- confidence 正式运行：`artifacts/metrla_e3_relation_topk_confidence_seed42_retry1`、`seed2024`、`seed2025`；
- horizon-only 对照：`artifacts/metrla_e3_relation_topk_horizon_seed42`、`seed2024`、`seed2025`；
- 每个正式目录均包含训练日志、best checkpoint、validation 评估和分支诊断；
- 本报告没有使用 test，也没有使用 smoke/debug 指标。
