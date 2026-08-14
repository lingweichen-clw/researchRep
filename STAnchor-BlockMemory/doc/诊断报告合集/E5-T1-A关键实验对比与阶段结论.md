# E5 T1-A 关键实验对比与阶段诊断报告

> 日期：2026-08-06  
> 结论级别：validation 阶段诊断，seed 42，不是最终 test 或多随机种子结果  
> 核心结论：E5A 在 METR-LA 和 PEMS-BAY 的检索层均稳定优于 E3；接入无 confidence 下游后，E5A 在 METR-LA 上继续优于 E3，但总体 MAE 收益缩小为 `0.62%`。这说明 E5A 学到了更适合 OffsetDecay 的检索关系，但其下游增益目前仍属于“小而一致”的阶段信号，尚不足以形成最终论文结论。

## 1. 为什么要做 E5

E3 已经实现“用训练期 future 关系监督历史检索器”，但它仍有两个明显问题：

1. E3 根据 raw future 的相似关系训练 selector，部署时却直接搬运历史 Bank 中的 raw future。历史事件与当前 query 即使未来变化形状相似，绝对速度水平也可能不同。
2. 先前下游使用 confidence 修正，难以区分收益来自检索器、future payload，还是 confidence 网络本身。

E5 的目的不是增加更复杂的 backbone，而是修正这两个对齐问题：

- 先找到一种不增加神经网络的 future payload，使历史 future 更适合当前 query；
- 再让训练期 teacher 学习与该 payload 一致的 future 关系；
- 最后关闭 confidence，只检验检索 memory 是否能直接改善预测。

因此，E5 的核心问题是：

> 如果 selector 真正学到了“哪些历史在未来动力学上可复用”，那么使用与当前 query 对齐的 future memory 后，它是否能在不依赖 confidence 的情况下稳定改善预测？

## 2. 关键术语与未来信息边界

### 2.1 E3 selector

`E3 selector` 是 E3 阶段预训练得到的历史检索器。训练时，teacher 使用 source-train 中不同样本的 raw future 距离定义关系；encoder 的输入始终只有历史 context。部署时不读取当前 query future。

### 2.2 E5A selector

`E5A selector` 是 E5 的 T1-A 分支，只修改训练期 future teacher，不增加推理网络。它使用与 OffsetDecay 部署形式一致的 future signature 监督历史 key。E5A 不包含 FutureIncrement，也不包含趋势分解分支。

对 source-train 中的训练事件 \(i\)，E5A 定义 Deployment-Aligned OffsetDecay Signature，简称 `ODSignature`：

\[
S^{\mathrm{OD}}_{i,h,n,c}
=
Y_{i,h,n,c}-\lambda_h\alpha_{i,n,c}.
\]

其中，\(Y_i\) 是事件 \(i\) 的训练 future，\(\alpha_i\) 是事件 \(i\) 最近 12 个可见历史步的平均 level，\(\lambda_h\) 是第 2.4 节定义的 horizon 衰减权重。Teacher 比较同一 anchor 与不同历史候选的 \(S^{\mathrm{OD}}\) 距离，再让只读取历史 context 的 key 学习该关系。

为避免原始速度量纲使 teacher 分布过尖，每个 anchor 的候选距离都会除以该 anchor 的平均候选距离。该归一化只改变距离尺度，不改变同一 anchor 内的候选排序，也不增加推理步骤。训练时 ODSignature 在 `torch.no_grad()` 下构造；推理时不构造当前 query 的 ODSignature，也不读取 query future。

### 2.3 历史 Bank

`历史 Bank` 是由训练时间段中已经完整发生的事件构成的因果记忆库。每个事件保存历史 key、节点 key、事件结束时的 level 和该事件随后已经发生的 future。对当前 query 检索时，候选必须满足：

\[
t^{\mathrm{candidate}}_{\mathrm{future\_end}}
<
t^{\mathrm{query}}_{\mathrm{context\_start}}.
\]

因此 Bank 中虽然保存历史 future，但这些 future 在当前 query 时刻已经属于过去，不构成未来泄漏。

### 2.4 OffsetDecay

`OffsetDecay` 是随预测距离逐步减弱的 level 对齐机制，不是模型，也没有可学习参数。

E3/E5 selector 为 query \(q\) 的节点 \(n\) 选出历史事件集合 \(K_{q,n}\)，并给出权重 \(w_{qjn}\)。raw future memory 为：

\[
\widehat Y^{\mathrm{raw}}_{q,h,n,c}
=
\sum_{j\in K_{q,n}} w_{qjn}Y_{j,h,n,c}.
\]

其中，\(Y_{j,h,n,c}\) 是历史候选 \(j\) 在预测步 \(h\)、节点 \(n\)、通道 \(c\) 的 future；权重满足 \(\sum_jw_{qjn}=1\)。

设 \(\alpha_{q,n,c}\) 和 \(\alpha_{j,n,c}\) 分别为 query 与历史候选最近 12 个可见时间步的平均 level，则完整 level 对齐为：

\[
\widehat Y^{\mathrm{offset}}_{q,h,n,c}
=
\alpha_{q,n,c}
+
\sum_{j\in K_{q,n}}w_{qjn}
\left(Y_{j,h,n,c}-\alpha_{j,n,c}\right).
\]

OffsetDecay 在近端采用 level 对齐，在远端逐步回到 raw future：

\[
\widehat Y^{\mathrm{OD}}_{q,h,n,c}
=
\widehat Y^{\mathrm{raw}}_{q,h,n,c}
+
\lambda_h
\left(
\widehat Y^{\mathrm{offset}}_{q,h,n,c}
-
\widehat Y^{\mathrm{raw}}_{q,h,n,c}
\right),
\qquad
\lambda_h=1-\frac{h-1}{H-1}.
\]

本实验预测 \(H=12\) 步，每步 5 分钟。第 1 步 \(\lambda_1=1\)，完整采用 level 对齐；第 12 步 \(\lambda_{12}=0\)，完全回到 raw future。

### 2.5 base-only

`base-only` 只使用最近 12 个时间步，也就是最近 60 分钟，训练轻量预测 backbone；它不检索 Bank，不使用 E3/E5A selector，也不读取 memory。此前口头所说的“E5+base”实际应称为 `base-only`，因为 E5 预训练模块不参与该模式的预测。

### 2.6 两类指标不能混用

- **Memory 指标**：直接比较检索聚合出的 future memory 与真实 future，用来判断 selector 和 payload 本身是否有效。
- **Downstream 指标**：比较最终预测与真实 future，预测同时使用 backbone，并在 memory 模式下加入 12 个 horizon fusion 参数。

MAE 和 RMSE 使用反归一化后的原始速度单位；MAPE 使用百分比。Memory 指标更好不等于下游一定获得同等比例的收益，因此两部分必须分开报告。

### 2.7 未来信息边界

- E5A 预训练 teacher 只在 source-train 中读取训练样本 future，用于定义样本关系；梯度只更新历史 encoder 和 retrieval head。
- Bank 只写入训练时间段内已经发生的历史事件。
- validation query 的 future 只用于离线计算 MAE、RMSE、MAPE，不参与检索和预测。
- 当前所有主结果都关闭 confidence，`confidence_head=0`，`confidence_loss=0`。

## 3. 三组实验分别回答什么问题

| 实验 | 动机 | 固定条件 | 唯一改变 | 回答的问题 |
|---|---|---|---|---|
| T0 payload 诊断 | 判断 raw future 应如何对齐当前 query | E3 selector、候选、Top-K、权重和 Bank 相同 | future payload | 收益来自 level、趋势还是其他修正？ |
| T1-A 检索诊断 | 判断 E5A teacher 是否比 E3 teacher 更匹配 OffsetDecay | Level-0、候选协议、OffsetDecay payload、无 confidence | E3 selector 换为 E5A selector | E5A 是否真的学到更好的检索关系？ |
| 无 confidence 下游 | 判断检索改善能否转化为最终预测收益 | 同一 backbone、数据划分、训练配置和 seed 42 | 不用 memory、E3 memory、E5A memory | 收益是否能脱离 confidence，并进入最终预测？ |

数据与预测设置：

- METR-LA validation：`2993` 个 query，`207` 个节点；
- PEMS-BAY validation：`4912` 个 query；
- 短期 backbone 输入：12 步，即 60 分钟；
- 检索 encoder 输入：288 步，即 1 天历史；
- 输出：未来 12 步，即 5 到 60 分钟；
- Bank 检索先用 event key 粗筛 32 个事件，即 event Top-R=32；再为每个节点保留 5 个历史 future，即 node Top-K=5；
- selector 诊断固定 `level_weight=0`，该设置称为 `Level-0`，表示排序只使用预训练 key，不额外混入 endpoint level 距离，从而避免手工 level 特征掩盖 key 的作用。

## 4. 实验一：T0 为什么保留 OffsetDecay、删除趋势残差

### 4.1 实验动机

T0 不重新训练模型，只固定 E3 selector 的候选、Top-K 和权重，然后替换 future payload。这样可以把“检索排序是否正确”和“检索出的 future 应如何使用”分开。

被比较的四种主要 payload：

- `Raw`：直接加权历史 raw future；
- `Offset`：在全部 12 个预测步上完整搬移 query 与 candidate 的 level 差；
- `OffsetDecay`：近端采用 level 对齐，远端逐步回到 raw future；
- `Trend residual`：估计局部线性趋势后搬运趋势残差。

### 4.2 总体 memory 结果

| Dataset | Payload | MAE | RMSE | MAPE (%) | MAE 相对 Raw |
|---|---|---:|---:|---:|---:|
| METR-LA | Raw | 3.803 | 7.372 | 10.519 | 0.00% |
| METR-LA | Offset | 3.894 | 7.090 | 9.870 | -2.40% |
| METR-LA | **OffsetDecay** | **3.623** | **6.735** | **9.564** | **+4.72%** |
| METR-LA | Trend residual | 5.374 | 10.266 | 13.386 | -41.33% |
| PEMS-BAY | Raw | 2.203 | 4.577 | 5.120 | 0.00% |
| PEMS-BAY | Offset | 1.978 | 4.138 | **4.362** | +10.19% |
| PEMS-BAY | **OffsetDecay** | **1.931** | **4.027** | 4.390 | **+12.36%** |
| PEMS-BAY | Trend residual | 2.790 | 5.843 | 6.054 | -26.67% |

正值表示 MAE 下降，负值表示恶化。

### 4.3 T0 传达的信号

1. **Raw future 不是最合适的 payload。** 相同 E3 排序下，OffsetDecay 在两个数据集都降低 MAE 和 RMSE。
2. **完整 Offset 只适合近端。** 它在 METR-LA 总体 MAE 反而恶化，说明当前 level 不能无衰减地平移到整个 60 分钟。
3. **直接趋势残差不成立。** Trend residual 在两个数据集的三项指标均明显恶化，因此没有证据引入更复杂的趋势分解网络。
4. **OffsetDecay 是更稳健的折中。** PEMS-BAY 的完整 Offset MAPE 比 OffsetDecay 低 `0.028` 个百分点，但 OffsetDecay 的 MAE、RMSE 更好，并且是唯一在两个数据集上都稳定降低 MAE 的 level 机制。

阶段决策：保留零参数 OffsetDecay，删除线性趋势残差和 local scale transfer，不新增趋势分解分支。

## 5. 实验二：E5A 是否学到比 E3 更好的检索关系

### 5.1 实验动机

T0 只证明 OffsetDecay payload 有效，并没有证明需要重新预训练 selector。T1-A 因此在相同 Level-0 候选协议和相同 OffsetDecay payload 下，只把 E3 selector 替换为 E5A selector。

如果 E5A 更好，说明“让 teacher 学习与部署 payload 一致的 future relation”产生了独立作用；如果两者没有差别，则 E5A 预训练没有保留价值。

### 5.2 总体 memory 结果

| Dataset | Selector | MAE | RMSE | MAPE (%) | E5A 相对 E3 |
|---|---|---:|---:|---:|---:|
| METR-LA | E3 | 3.646 | 6.698 | 9.600 | 基线 |
| METR-LA | **E5A** | **3.530** | **6.605** | **9.334** | MAE +3.20%，RMSE +1.39%，MAPE +2.77% |
| PEMS-BAY | E3 | 1.972 | 4.049 | 4.479 | 基线 |
| PEMS-BAY | **E5A** | **1.911** | **4.008** | **4.369** | MAE +3.08%，RMSE +1.00%，MAPE +2.45% |

METR-LA common evaluation coverage 为 E3 `98.988%`、E5A `98.989%`；PEMS-BAY 两者均为 `99.557%`。改善不是通过减少困难样本的覆盖率获得的。

### 5.3 METR-LA 各预测步 memory 指标

| 预测距离 | E3 MAE | E5A MAE | E3 RMSE | E5A RMSE | E3 MAPE (%) | E5A MAPE (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 2.917 | **2.662** | 4.629 | **4.342** | 6.570 | **6.112** |
| 10 min | 3.199 | **2.965** | 5.392 | **5.146** | 7.627 | **7.201** |
| 15 min | 3.365 | **3.163** | 5.850 | **5.645** | 8.317 | **7.931** |
| 20 min | 3.490 | **3.317** | 6.187 | **6.018** | 8.877 | **8.529** |
| 25 min | 3.585 | **3.447** | 6.450 | **6.326** | 9.335 | **9.046** |
| 30 min | 3.665 | **3.553** | 6.679 | **6.583** | 9.711 | **9.460** |
| 35 min | 3.740 | **3.658** | 6.892 | **6.826** | 10.056 | **9.851** |
| 40 min | 3.810 | **3.748** | 7.096 | **7.050** | 10.373 | **10.193** |
| 45 min | 3.876 | **3.835** | 7.294 | **7.267** | 10.663 | **10.499** |
| 50 min | 3.951 | **3.920** | 7.498 | **7.483** | 10.947 | **10.798** |
| 55 min | 4.035 | **4.006** | 7.708 | **7.702** | 11.237 | **11.083** |
| 60 min | 4.135 | **4.098** | 7.932 | **7.924** | 11.549 | **11.371** |

### 5.4 PEMS-BAY 各预测步 memory 指标

| 预测距离 | E3 MAE | E5A MAE | E3 RMSE | E5A RMSE | E3 MAPE (%) | E5A MAPE (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 1.078 | **1.032** | 1.848 | **1.798** | 2.115 | **2.036** |
| 10 min | 1.413 | **1.361** | 2.621 | **2.574** | 2.927 | **2.833** |
| 15 min | 1.631 | **1.579** | 3.169 | **3.126** | 3.508 | **3.410** |
| 20 min | 1.792 | **1.740** | 3.568 | **3.530** | 3.958 | **3.859** |
| 25 min | 1.917 | **1.865** | 3.868 | **3.834** | 4.321 | **4.219** |
| 30 min | 2.021 | **1.969** | 4.103 | **4.074** | 4.622 | **4.521** |
| 35 min | 2.111 | **2.057** | 4.297 | **4.269** | 4.881 | **4.777** |
| 40 min | 2.190 | **2.132** | 4.464 | **4.433** | 5.101 | **4.995** |
| 45 min | 2.265 | **2.201** | 4.616 | **4.579** | 5.301 | **5.186** |
| 50 min | 2.337 | **2.267** | 4.760 | **4.716** | 5.488 | **5.364** |
| 55 min | 2.412 | **2.331** | 4.903 | **4.848** | 5.671 | **5.533** |
| 60 min | 2.492 | **2.398** | 5.048 | **4.982** | 5.860 | **5.701** |

### 5.5 检索层结论

E5A 在两个数据集、12/12 个预测步、MAE/RMSE/MAPE 三项指标上全部优于 E3。该结果说明：

- E5A 的改善不是只集中在某个特殊 horizon；
- 同域 METR-LA 和跨数据集 PEMS-BAY 都出现相同方向的改善；
- E5A teacher 与 OffsetDecay payload 的对齐确实改变了 selector，而不只是降低了预训练代理损失。

`relation-best checkpoint` 按 validation relation loss 保存，`total-best checkpoint` 按遮挡重建与 relation 的加权总损失保存。METR-LA 上，relation-best checkpoint 的 OffsetDecay memory 为 `3.530 / 6.605 / 9.334%`，total-best checkpoint 为 `3.536 / 6.615 / 9.366%`。三项指标均支持使用 relation-best checkpoint 构建正式 Bank。

## 6. 实验三：无 confidence 下游是否真正改善最终预测

### 6.1 实验动机

检索 memory 更准仍不等于最终预测更准。下游实验固定同一 METR-LA 数据划分、轻量 backbone、训练超参数和 seed 42，比较：

- `base-only`：只使用最近 12 步历史；
- `E3 + OffsetDecay`：E3 selector、E3 Bank 和 OffsetDecay memory；
- `E5A + OffsetDecay`：E5A selector、E5A Bank 和 OffsetDecay memory。

三组均关闭 confidence。E3 与 E5A 的 downstream 结构完全相同，均只比 base-only 多 12 个 horizon fusion 参数：

| 模式 | Backbone 参数 | Horizon fusion 参数 | Confidence head 参数 | 下游可训练参数 |
|---|---:|---:|---:|---:|
| base-only | 5,772 | 0 | 0 | 5,772 |
| E3 + OffsetDecay | 5,772 | 12 | 0 | 5,784 |
| E5A + OffsetDecay | 5,772 | 12 | 0 | 5,784 |

因此 E5A 与 E3 的差异只来自预训练 selector 和对应 Bank，不来自下游参数量。

### 6.2 最佳 validation 总体结果

模型按照 validation MAE 选择 checkpoint。base-only 最佳轮为 epoch 15，E3 和 E5A 最佳轮均为 epoch 11。

| 模式 | MAE | RMSE | MAPE (%) | MAE 相对 base-only | MAE 相对 E3 |
|---|---:|---:|---:|---:|---:|
| base-only | 3.350 | 6.965 | 9.684 | 基线 | - |
| E3 + OffsetDecay | 3.219 | 6.346 | 9.086 | +3.91% | 基线 |
| **E5A + OffsetDecay** | **3.199** | **6.317** | **9.021** | **+4.51%** | **+0.62%** |

E5A 相对 E3 的总体改善为：

- MAE：`0.62%`；
- RMSE：`0.45%`；
- MAPE：`0.71%`。

E5A 相对 base-only 的总体改善为：

- MAE：`4.51%`；
- RMSE：`9.30%`；
- MAPE：`6.85%`。

### 6.3 各预测步 MAE

| 预测距离 | base-only | E3 + OffsetDecay | E5A + OffsetDecay |
|---:|---:|---:|---:|
| 5 min | 2.261 | 2.260 | **2.249** |
| 10 min | 2.562 | 2.554 | **2.539** |
| 15 min | 2.791 | 2.771 | **2.753** |
| 20 min | 2.993 | 2.960 | **2.935** |
| 25 min | 3.175 | 3.119 | **3.092** |
| 30 min | 3.342 | 3.257 | **3.228** |
| 35 min | 3.500 | 3.377 | **3.351** |
| 40 min | 3.650 | 3.486 | **3.462** |
| 45 min | 3.789 | 3.580 | **3.561** |
| 50 min | 3.922 | 3.670 | **3.653** |
| 55 min | 4.044 | 3.753 | **3.738** |
| 60 min | 4.165 | 3.834 | **3.821** |

E5A 的 MAE 在 12/12 个预测步上均低于 E3 和 base-only。

### 6.4 各预测步 RMSE

| 预测距离 | base-only | E3 + OffsetDecay | E5A + OffsetDecay |
|---:|---:|---:|---:|
| 5 min | 4.103 | 4.076 | **4.041** |
| 10 min | 5.002 | 4.927 | **4.878** |
| 15 min | 5.641 | 5.481 | **5.420** |
| 20 min | 6.155 | 5.886 | **5.827** |
| 25 min | 6.583 | 6.199 | **6.148** |
| 30 min | 6.957 | 6.442 | **6.400** |
| 35 min | 7.286 | 6.651 | **6.620** |
| 40 min | 7.588 | 6.830 | **6.807** |
| 45 min | 7.849 | 6.987 | **6.972** |
| 50 min | 8.082 | 7.128 | **7.120** |
| 55 min | 8.302 | 7.258 | **7.257** |
| 60 min | 8.506 | **7.387** | 7.389 |

E5A 的 RMSE 在 11/12 个预测步上低于 E3；60 分钟位置比 E3 高 `0.002`，相对恶化约 `0.03%`，但仍明显优于 base-only。

### 6.5 各预测步 MAPE

| 预测距离 | base-only | E3 + OffsetDecay | E5A + OffsetDecay |
|---:|---:|---:|---:|
| 5 min | 5.521% | 5.482% | **5.458%** |
| 10 min | 6.575% | 6.535% | **6.515%** |
| 15 min | 7.410% | 7.324% | **7.288%** |
| 20 min | 8.173% | 8.030% | **7.970%** |
| 25 min | 8.891% | 8.629% | **8.560%** |
| 30 min | 9.536% | 9.161% | **9.079%** |
| 35 min | 10.192% | 9.649% | **9.574%** |
| 40 min | 10.831% | 10.101% | **10.011%** |
| 45 min | 11.442% | 10.496% | **10.420%** |
| 50 min | 12.025% | 10.873% | **10.794%** |
| 55 min | 12.548% | 11.212% | **11.129%** |
| 60 min | 13.068% | 11.541% | **11.457%** |

E5A 的 MAPE 在 12/12 个预测步上均低于 E3 和 base-only。

## 7. 三组实验连起来说明了什么

### 信号一：future payload 的处理不是次要细节

T0 在不改变 selector 的情况下，仅将 Raw 换为 OffsetDecay，就使 memory MAE 在 METR-LA 和 PEMS-BAY 分别改善 `4.72%` 和 `12.36%`。因此，历史 future 是否与当前 query 对齐，会直接决定检索模块能否产生有效信号。

### 信号二：E5A 预训练不是只让代理损失更好看

在相同 OffsetDecay payload 下，E5A 相对 E3 的 memory MAE 在两个数据集都改善约 `3%`，而且 12 个预测步的 MAE、RMSE、MAPE 全部更好。这说明 E5A selector 学到的排序关系确实更适合实际检索对象。

### 信号三：检索收益可以脱离 confidence

METR-LA 下游没有 confidence head，E3 和 E5A 仍分别比 base-only 降低 `3.91%` 和 `4.51%` 的 MAE。当前收益不是由 confidence 网络制造的，这比此前“检索 future 后再依靠 confidence 修正”的证据更干净。

### 信号四：检索层的改善进入下游后被明显压缩

E5A 相对 E3 的 memory MAE 改善为 `3.20%`，最终下游 MAE 改善只有 `0.62%`。这说明：

- backbone 已经解释了大部分可预测信号；
- memory 与 backbone 的信息存在重叠；
- horizon fusion 会限制 memory 对最终输出的影响；
- 后续重点应是验证该小幅收益是否稳定，而不是立即增加更复杂的趋势或时空分支。

### 信号五：当前不支持趋势分解主线

直接趋势残差在 METR-LA 和 PEMS-BAY 的三项 memory 指标都明显恶化。当前证据支持“level 对齐随 horizon 衰减”，不支持把完整趋势分解作为 E5 主线。若未来考虑趋势信息，必须先指出 OffsetDecay 无法解决的具体失败样本，并用单变量实验验证，而不是直接堆叠新模块。

## 8. 目前能说什么、不能说什么

### 8.1 当前可以向导师报告的结论

1. OffsetDecay 是一个跨 METR-LA 和 PEMS-BAY 都有效的零参数 future payload 修正。
2. E5A 的 deployment-aligned teacher 在两个数据集的检索层稳定优于 E3 raw-future teacher。
3. E5A 的检索改善已经在 METR-LA 无 confidence 下游中转化为小幅但一致的预测改善。
4. E5A 不增加推理网络深度；相对 E3 只改变预训练 teacher、selector 参数和对应 Bank，E3/E5A 下游参数量相同。

### 8.2 当前不能写成最终论文结论的内容

1. **不能声称多随机种子稳定。** 当前 E5A 下游只有 seed 42。
2. **不能声称跨域下游有效。** PEMS-BAY 只完成了检索层诊断，尚未完成对应下游训练。
3. **不能声称预训练一定优于随机初始化。** random OffsetDecay 下游对照尚未完成。
4. **不能报告最终 test 性能。** 当前模型仍处于 validation 选择阶段，本报告没有读取 E5 test 结果。
5. **不能把 `0.62%` 包装成显著提升。** 它是方向一致的正信号，但需要多 seed 的均值、标准差和显著性判断。

## 9. 阶段决策

### Conditional Keep

- 保留 E5A OffsetDecay teacher；
- 保留 relation-best checkpoint 和对应 Bank；
- 保留无 confidence 的 OffsetDecay downstream；
- 把“future-guided selector + deployment-aligned future payload”作为当前核心创新候选。

### Remove

- 删除直接线性趋势残差主线；
- 不引入 STWave+/TimeMixer 式复杂趋势分解；
- 不恢复 confidence 来放大当前小收益；
- 不把 E5A total-best checkpoint 作为正式 Bank 来源。

### 下一步只关闭两个不确定性

1. 完成 random selector + OffsetDecay 的相同 METR-LA 下游对照，确认 E5A 的收益来自预训练而非随机检索结构。
2. 完成 PEMS-BAY 的 base-only、E3 OffsetDecay、E5A OffsetDecay 和 random OffsetDecay 下游对照，确认检索层跨数据集改善能否转化为下游改善。

在这两个问题关闭前，不运行 T1-B FutureIncrement，不新增趋势分解、backbone 或 confidence 模块。

## 10. 证据文件

### T0 payload 诊断

- `artifacts/e5_t0_metrla_val.json`
- `artifacts/e5_t0_pemsbay_val.json`

### T1-A E3 固定基线

- `artifacts/e5_t1_baselines/metrla_e3_level0_offset_decay_val.json`
- `artifacts/e5_t1_baselines/pemsbay_e3_level0_offset_decay_val.json`

### T1-A E5A 检索诊断

- `artifacts/e5_t1a/metrla_e5a_relation_level0_val.json`
- `artifacts/e5_t1a/metrla_e5a_total_level0_val.json`
- `artifacts/e5_t1a/pemsbay_e5a_level0_val.json`

### METR-LA 无 confidence 下游

- `artifacts/metrla_e5a_base_only_seed42/target_metrics.jsonl`
- `artifacts/metrla_e3_offset_decay_horizon_seed42/target_metrics.jsonl`
- `artifacts/metrla_e5a_offset_decay_horizon_seed42/target_metrics.jsonl`

对应最佳 checkpoint：

- `artifacts/metrla_e5a_base_only_seed42/downstream_best.pt`
- `artifacts/metrla_e3_offset_decay_horizon_seed42/downstream_best.pt`
- `artifacts/metrla_e5a_offset_decay_horizon_seed42/downstream_best.pt`
