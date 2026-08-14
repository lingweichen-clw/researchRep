# E3 Confidence 最终 Test 报告

## 1. Test 使用协议

本报告是 E3 confidence 在 METR-LA 上的最终 test 结果。执行 test 前已经在 validation 上完成：

- E2 与 E3 horizon-only 三 seed 对照；
- E3 confidence seed 42 机制门槛诊断；
- E3 confidence seed 42、2024、2025 稳定性验证；
- 模型结构、预训练 checkpoint、Bank、Top-K、confidence、融合方式、训练预算和三个 downstream checkpoint 冻结。

三个冻结 checkpoint 各执行一次 `evaluate.py --split test`。test 结果不用于选择 seed、checkpoint、超参数或模型结构，也没有在 test 上运行额外机制诊断。

## 2. 统计口径

设 test 随机种子数为 \(S=3\)，第 \(s\) 个 seed 的指标为 \(x_s\)。均值和样本标准差为：

\[
\bar{x}=\frac{1}{S}\sum_{s=1}^{S}x_s,
\]

\[
\operatorname{Std}(x)
=
\sqrt{
\frac{1}{S-1}
\sum_{s=1}^{S}(x_s-\bar{x})^2
}.
\]

其中，\(x_s\) 是固定方法在第 \(s\) 个 downstream seed checkpoint 上的 test 指标。本文所有“均值 ± 标准差”均采用样本标准差。

## 3. Overall Test 结果

| seed | MAE | RMSE | MAPE (%) |
|---:|---:|---:|---:|
| 42 | 3.368660 | 6.653607 | 9.795437 |
| 2024 | 3.377238 | 6.649896 | 9.860817 |
| 2025 | 3.375825 | 6.657163 | 9.853368 |
| **均值 ± 标准差** | **3.373908 ± 0.004599** | **6.653555 ± 0.003634** | **9.836541 ± 0.035791** |

三个 checkpoint 的 overall MAE 极差为 0.008578，样本标准差仅 0.004599。最终方法对 downstream 初始化不敏感，validation 上观察到的稳定性延续到了 test。

## 4. 15/30/60 分钟 Test 结果

### 4.1 MAE

| seed | 15 min | 30 min | 60 min |
|---:|---:|---:|---:|
| 42 | 2.976904 | 3.416611 | 3.923903 |
| 2024 | 2.976210 | 3.422564 | 3.943968 |
| 2025 | 2.975535 | 3.422046 | 3.942492 |
| **均值 ± 标准差** | **2.976216 ± 0.000684** | **3.420407 ± 0.003297** | **3.936788 ± 0.011183** |

### 4.2 RMSE

| 预测位置 | 均值 ± 标准差 |
|---|---:|
| 15 min | 5.732800 ± 0.007773 |
| 30 min | 6.739613 ± 0.009339 |
| 60 min | 7.753726 ± 0.006948 |

### 4.3 MAPE

| 预测位置 | 均值 ± 标准差 |
|---|---:|
| 15 min | 8.230374 ± 0.010768 |
| 30 min | 9.955682 ± 0.016909 |
| 60 min | 12.158223 ± 0.099382 |

随着预测距离增加，三类误差均上升，符合多步交通预测误差累积规律。三个 seed 在各预测位置上的标准差仍然较小。

## 5. Validation 与 Test 差异

| 指标 | Validation 三 seed 均值 | Test 三 seed 均值 | Test - Validation |
|---|---:|---:|---:|
| MAE | 3.138844 | 3.373908 | +0.235064 |
| RMSE | 6.297428 | 6.653555 | +0.356127 |
| MAPE (%) | 8.906897 | 9.836541 | +0.929643 |

test 三类误差均高于 validation，说明数据时间后段更难或存在分布变化。这个差异本身不能证明 confidence 过拟合，因为：

- validation 和 test 对应不同时间区间；
- 本报告没有在 test 上重新运行 horizon-only 对照；
- test 结果没有参与 checkpoint 选择；
- 所有 seed 都出现相近的 test 水平，随机初始化不是主要原因。

准确结论是：最终方法在 test 上保持了很低的 seed 方差，但 validation 到 test 存在明显性能下降。后续若研究分布漂移，应另立预先定义的时间漂移诊断实验，不能根据当前 test 结果回头修改本模型。

## 6. 最终结论

METR-LA 当前阶段已经闭环：

1. E3 Future-Relation 替代 E2 离散阈值关系后，horizon-only 获得稳定但较小的 validation 改善；
2. 257 参数 confidence head 在 validation 三 seed 上平均降低 MAE 约 2.47%，且置信度四分位语义稳定；
3. 最终 confidence 方法在 test 上得到 `MAE 3.373908 ± 0.004599`、`RMSE 6.653555 ± 0.003634`、`MAPE 9.836541% ± 0.035791%`；
4. 最终 test 结果对 downstream seed 稳定，但存在 validation 到 test 的时间分布差异；
5. test 已经使用完毕，不能再通过更换 seed、checkpoint 或超参数追逐 test 指标。

当前应停止 METR-LA 上的结构扩张和调参。下一阶段若继续验证论文主张，应转向预先确定的第二个同类型数据集或迁移设置，而不是继续挖掘当前 test。

## 7. 正式证据文件

- 机器可读 test 汇总：`artifacts/metrla_e3_confidence_multiseed_test_summary.json`；
- seed 42：`artifacts/metrla_e3_relation_topk_confidence_seed42_retry1/test_evaluation.json`；
- seed 2024：`artifacts/metrla_e3_relation_topk_confidence_seed2024/test_evaluation.json`；
- seed 2025：`artifacts/metrla_e3_relation_topk_confidence_seed2025/test_evaluation.json`；
- 每个 checkpoint 只评估一次 test；本报告未产生 test 机制诊断或新的模型选择。
