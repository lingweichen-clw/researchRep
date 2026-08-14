# E5A-SymNorm 零训练诊断报告

## 1. 实验动机

E5A 使用 `AnchorMean`（按锚点平均距离归一化）构造 future-relation teacher。对事件节点对 \((i,j,n)\)，原始 OffsetDecay future 距离为 \(d_{ij,n}^{OD}\)，AnchorMean 定义为：

\[
\mu_{i,n}=\frac{1}{|\mathcal C_{i,n}|}\sum_{k\in\mathcal C_{i,n}}d_{ik,n}^{OD},
\qquad
\widetilde d_{ij,n}^{AM}=\frac{d_{ij,n}^{OD}}{\mu_{i,n}+\epsilon}.
\]

其中，\(i\) 是发起匹配的事件，\(j\) 是候选事件，\(n\) 是交通传感器节点，\(\mathcal C_{i,n}\) 是该事件节点的有效候选集合，\(\epsilon\) 是防止除零的小常数。它的作用是消除不同锚点的距离量纲差异，输出仍是事件两两关系距离。

问题在于，虽然原始距离满足 \(d_{ij,n}^{OD}=d_{ji,n}^{OD}\)，AnchorMean 通常满足 \(\mu_{i,n}\ne\mu_{j,n}\)，所以 \(\widetilde d_{ij,n}^{AM}\ne\widetilde d_{ji,n}^{AM}\)。teacher 要求 key 学习不对称关系，而余弦 key logit 天生对称。这可能形成无法消除的训练目标冲突。

本实验测试 `E5A-SymNorm`。该名称表示“E5A 的对称几何均值距离归一化版本”，机制正式名称为 `SymmetricGeometricMeanNormalization`：

\[
\widetilde d_{ij,n}^{SYM}
=
\frac{d_{ij,n}^{OD}}
{\sqrt{(\mu_{i,n}+\epsilon)(\mu_{j,n}+\epsilon)}}.
\]

它同时使用事件 \(i\) 和事件 \(j\) 的距离尺度，目的不是增加模型容量，而是让 teacher 距离与对称的 key 相似度具有一致的数学结构。输入是 OD 两两距离及有效 mask，输出是同形状的无量纲距离；它没有参数，也不增加部署推理步骤。

## 2. 实验协议与 future 信息边界

- 数据：METR-LA validation，共 2,993 个 query、94 个部署候选 batch。
- 固定项：E5A seed 42 checkpoint、对应历史 Bank、RelaxedCalendar 候选协议、teacher temperature 和 OffsetDecay payload。
- `RelaxedCalendar` 表示仅从 query 时刻之前的历史 Bank 中，选取同星期且时间槽位在 query 槽位正负 1 范围内的事件，最多保留 32 个候选。本次平均每个 query 有 23.98 个候选。
- Bank 候选 SymNorm：每个 query 与其历史候选组成一个共同事件集，在该集合上计算事件两两 OD-MAE，从而同时得到 query 尺度 \(\mu_i\) 和每个候选尺度 \(\mu_j\)。不能只计算 query 均值，否则不是真正的对称归一化。
- 不训练模型，不修改 checkpoint、key、Bank、OffsetDecay payload 或下游模型。
- query future 只用于离线 teacher 排序和 Oracle 指标，部署检索仍只能读取 query history、日历和历史 Bank。因此本实验不存在把 query future 接入部署预测的情况。

## 3. 指标定义

- `distance/logit asymmetry`：有效事件对上 \(|a_{ij}-a_{ji}|\) 的平均值。0 表示严格对称。
- `teacher probability asymmetry`：对 teacher softmax 概率计算同样的不对称性。即使 logit 对称，不同锚点的 softmax 分母仍可能不同，因此该值不必为 0。
- `teacher effective support`：\(K_{eff}=1/\sum_j p_{ij}^2\)。它表示 teacher 实际向多少个候选分配了有效概率；越接近 1 越像 one-hot，越大越平缓。
- `Spearman`：固定 E5A key 距离与离线 future teacher 距离的秩相关系数，越大表示排序趋势越一致。
- `Recall@5`：固定 key 最近 5 个候选与 future teacher 最近 5 个候选的交集数量除以 5。
- `Top-5 Jaccard`：异常扰动前后 teacher Top-5 集合的交并比。1 表示候选集合完全不变，0 表示完全不同。
- `Total Variation (TV)`：\(\tfrac12\sum_j|p_j-p'_j|\)，衡量异常扰动前后 teacher 概率分布变化；越小越稳定。
- `Oracle Top-K memory error`：使用 query future 产生的 teacher 距离选出 Top-K 历史事件，再对其 OffsetDecay payload 做均匀聚合后计算 MAE、RMSE、MAPE。它是不可部署的离线关系质量上限，不是下游模型预测成绩。MAE 是平均绝对误差，RMSE 对大误差更敏感，MAPE 是平均绝对百分比误差。

## 4. 完整结果

### 4.1 关系结构与固定 key 对齐

| 指标 | E5A AnchorMean | E5A-SymNorm | 变化 |
|---|---:|---:|---:|
| source-batch distance asymmetry | 0.519776 | 0.000000 | 完全消除 |
| source-batch teacher logit asymmetry | 5.197763 | 0.000000 | 完全消除 |
| source-batch teacher probability asymmetry | 0.081264 | 0.069097 | -14.97% |
| source-batch teacher effective support | 4.011471 | 3.784203 | -5.67% |
| Bank teacher effective support | 5.888836 | 5.661888 | -3.85% |
| 固定 E5A key Spearman | 0.328968 | 0.376597 | +0.047629 |
| 固定 E5A key Recall@5 | 0.330803 | 0.334352 | +0.003550 |
| 异常扰动 Top-5 Jaccard | 0.858822 | 0.822925 | -0.035897 |
| 异常扰动 TV | 0.223704 | 0.231694 | +0.007990 |

SymNorm 达到了首要目标：teacher distance 和 teacher logit 与学生 key logit 一样严格对称。teacher support 仅下降约 4% 至 6%，没有坍缩为 one-hot。固定旧 E5A key 的 Spearman 明显提高，Recall@5 小幅提高；但是异常扰动后的候选集合和概率分布都略不稳定。

### 4.2 Oracle memory 整体误差

| 排序与聚合 | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| AnchorMean Oracle Top-1 | 2.342932 | 4.506333 | 5.932764% |
| SymNorm Oracle Top-1 | 2.390636 | 4.625352 | 6.051611% |
| AnchorMean Oracle Top-5 | 2.315316 | 4.647731 | 6.158727% |
| SymNorm Oracle Top-5 | **2.282147** | **4.626195** | **6.076437%** |

SymNorm 的 Top-1 三项误差约变差 2%，说明它不适合把 teacher 进一步推向 HardTop1。另一方面，当前部署配置 `node_top_k=5`，SymNorm 的 Top-5 MAE、RMSE、MAPE 分别改善 1.43%、0.46%、1.34%，因此 Top-5 证据与实际检索候选数更一致。

### 4.3 Oracle Top-5 分预测步长误差

| 预测距离 | Anchor MAE | Sym MAE | Anchor RMSE | Sym RMSE | Anchor MAPE | Sym MAPE |
|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 1.9673 | **1.9461** | 3.4595 | **3.4479** | 4.5297% | **4.5144%** |
| 10 min | 2.1204 | **2.1017** | **3.9843** | 3.9970 | 5.1528% | **5.1515%** |
| 15 min | 2.2133 | **2.1919** | **4.2524** | 4.2758 | 5.5734% | **5.5513%** |
| 20 min | 2.2660 | **2.2368** | **4.4033** | 4.4225 | 5.8776% | **5.8184%** |
| 25 min | 2.3055 | **2.2700** | **4.5134** | 4.5181 | 6.1047% | **6.0170%** |
| 30 min | 2.3293 | **2.2930** | 4.5956 | **4.5939** | 6.2558% | **6.1630%** |
| 35 min | 2.3512 | **2.3134** | 4.6774 | **4.6677** | 6.3857% | **6.2846%** |
| 40 min | 2.3658 | **2.3245** | 4.7614 | **4.7394** | 6.4876% | **6.3760%** |
| 45 min | 2.3793 | **2.3395** | 4.8519 | **4.8205** | 6.5719% | **6.4553%** |
| 50 min | 2.4072 | **2.3673** | 5.0034 | **4.9512** | 6.6947% | **6.5749%** |
| 55 min | 2.4684 | **2.4277** | 5.2458 | **5.1715** | 6.9182% | **6.7857%** |
| 60 min | 2.6158 | **2.5792** | 5.6529 | **5.5700** | 7.3805% | **7.2514%** |

SymNorm 的 Top-5 MAE 和 MAPE 在 12 个预测步长上全部改善；RMSE 在 10 至 25 分钟小幅变差，其余步长改善。它主要改善多候选聚合，不是提升单一最佳事件的精确命中。

## 5. 结论与下一步决定

本实验支持训练一个 seed 42 的 `E5A-SymNorm`，但不支持现在就替换原 E5A。

支持继续的原因：

1. 它以零参数方式消除了 teacher logit 与 student logit 的结构冲突。
2. 旧 E5A key 对 SymNorm 的 Spearman 和 Recall@5 没有下降，反而提高。
3. 与部署 `Top-5` 更一致的 Oracle MAE、RMSE、MAPE整体改善，且 MAE、MAPE 在所有预测步长上改善。

需要保留的风险：

1. Top-1 Oracle 全面变差，不能把 SymNorm 与 HardTop1 或更尖锐温度同时使用。
2. 异常扰动稳定性下降，正式预训练后必须复查 Jaccard 和 TV。
3. 当前 key 是按 AnchorMean 训练的，零训练对齐结果不能证明新的 SymNorm teacher 一定能训练出更好的 key。

下一步只改变预训练 relation distance normalization，训练一个 seed 42 的 E5A-SymNorm；其余配置全部固定。若训练后不能同时保持严格 logit 对称、提高 Recall@5，并在 pretrained-vs-random 或固定下游结果中产生可复现收益，则删除该分支，不再增加多特征 teacher。

## 6. 复现实验命令

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/diagnose_teacher_metrics.py --config configs/metrla_e5_offset_decay_relation_level0_v1.yaml --checkpoint artifacts/metrla_e5a_offset_decay_seed42/pretrain_best_relation.pt --bank artifacts/metrla_bank_e5a_offset_decay_relation_seed42 --split val --output-dir artifacts/convergence/teacher_metric_diagnostic/e5a_symnorm_zero_train --candidate-protocol relaxed_calendar
```

完整机器可读结果见：`artifacts/convergence/teacher_metric_diagnostic/e5a_symnorm_zero_train/metrics.json`。
