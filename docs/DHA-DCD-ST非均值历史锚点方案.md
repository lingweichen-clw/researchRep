# DHA-DCD-ST：非均值历史锚点的偏差校准预测方案

更新时间：2026-07-01

## 1. 新问题：历史均值锚点信息太少

当前 TrafficRobustST 的历史锚点与 ST-SSDL 思路一致：对训练集内相同 `weekday-slot` 的历史值取均值，作为每个时间点的第三通道 `x_his`。

代码依据：

```text
src/preprocessing.py:51  build_history_anchor()
src/preprocessing.py:56  Build train-only weekday-slot historical means.
src/preprocessing.py:116 history_anchor = build_history_anchor(...)
src/preprocessing.py:117 data = np.stack([values, tod, history_anchor], axis=-1)
src/data.py:32        x: (B,T,N,3) with value, time-in-day, history anchor.
src/data.py:37        x_his = x[..., 2:3]
```

这个设计的优点是简单、稳定、无泄漏；但缺点也非常明显：

```text
mean anchor 只保留了“同一星期几、同一时间槽、同一节点”的平均水平。
```

它丢掉了至少五类重要信息：

| 被均值压掉的信息 | 对交通预测的意义 |
|---|---|
| 历史分布宽度 | 同一时间槽是否稳定，还是经常大幅波动 |
| 多模态模式 | 同一时间可能存在畅通/拥堵两种典型历史状态 |
| 最近历史趋势 | 近期道路状态变化是否已经偏离长期平均 |
| 极端历史样本 | 突发拥堵、节假日、事故等长尾场景 |
| 锚点可靠性 | 历史候选是否一致，是否可能形成 spatiotemporal mirage |

因此，当前 `Xc - Xa_mean` 的偏差不一定是真实偏差，它可能只是：

```text
当前状态与一个被过度平均的“虚假历史模式”之间的差异。
```

这会直接影响 DCD-ST：

1. `R_raw = Xc - Xa` 的物理语义被削弱。
2. `R_trend / R_residual / D_t / D_s` 都建立在弱锚点上。
3. `Delta_H` 学到的校正可能只是对均值误差做补偿。
4. gate 难以学出清晰策略，因为输入锚点本身缺少分布信息。

所以，下一阶段的核心问题应升级为：

```text
如何从历史分布中选择可靠上下文，而不是把历史压缩成一个均值锚点？
```

## 2. 从前人论文得到的启发

### 2.1 ST-SSDL：历史锚点有价值，但 historical average 太粗

ST-SSDL 的核心启发是：

```text
当前输入应与历史状态对比，current-history deviation 是有预测价值的信号。
```

但 ST-SSDL 使用的是 historical average anchor，然后再在 latent space 中通过 learnable prototypes 组织偏差。我们的复现实验暴露了两个问题：

1. historical average 本身信息太少。
2. learnable prototypes 出现 assignment collapse，导致所谓 scientific information space 解释力不足。

因此，我们不是否定 ST-SSDL 的历史锚点思想，而是把问题前移：

```text
在进入 latent prototype 或 deviation gate 之前，历史锚点本身就不应该只是均值。
```

### 2.2 STD-MAE / STEP：短窗口预测容易遇到 spatiotemporal mirage

STD-MAE 指出，短输入窗口可能导致 spatiotemporal mirage：相似历史输入后面跟着不同未来，或者不同历史输入后面反而跟着相似未来。

这对我们非常关键：

```text
简单使用 historical mean anchor 可能制造更严重的 mirage。
```

原因是 mean anchor 会把多个不同历史模式平均在一起，使模型看到一个“看起来平滑、但现实中不存在”的上下文。相比之下，非均值历史锚点应该保留候选历史模式，让模型知道：

```text
这个时间槽过去到底稳定不稳定？
过去有哪些典型模式？
当前更像其中哪一个？
候选模式之间是否分歧很大？
```

### 2.3 MegaCRN：记忆库与模式匹配

MegaCRN 使用 Meta-Node Bank 进行 memory reading 和 node-level prototype matching，用记忆项表示不同道路、不同时间下的典型交通模式。

我们可以借鉴“记忆读取”的思想，但做两个减法：

1. 不使用完全 learnable latent memory，避免再次出现难解释的 prototype collapse。
2. 使用物理历史候选窗口作为 anchor memory，候选可以追溯到真实历史时间。

也就是说，我们的 memory 不是：

```text
learnable prototypes P in latent space
```

而是：

```text
physical historical anchor candidates A_k from train-only same-slot history
```

这样既保留 memory/pattern matching 的能力，又比 ST-SSDL/MegaCRN 的 latent prototype 更可解释。

### 2.4 STID / STAEformer：简单结构需要时空身份上下文

STID 和 STAEformer 都说明：很多复杂 STGNN 的收益来自时空身份信息。对于我们的非均值锚点，这一点意味着：

```text
历史候选锚点的选择不能只看数值相似度，还要受 node identity 与 time identity 条件化。
```

同一个偏差大小，在不同节点、不同时间段可能含义不同：

| 场景 | 同样偏差的含义 |
|---|---|
| 通勤高峰节点 | 可能是正常拥堵波动 |
| 夜间低流量节点 | 可能是异常或事故 |
| 高速主干节点 | 邻域一致性更重要 |
| 局部支路节点 | 节点自身历史更重要 |

因此，非均值锚点选择应加入：

```text
node embedding
time-of-day embedding
day/week periodic identity
```

### 2.5 ST-Norm / STDN：不要只看 raw residual，要拆分分量

ST-Norm 把复杂影响分解为 temporal high-frequency 和 spatial local component；STDN 把交通序列拆成 trend-cyclical 与 seasonal component。

这支持我们的设计：

```text
历史锚点不应只提供一个平均曲线，而应提供趋势、波动、稳定性和多模态候选。
```

对于每个历史候选锚点 `A_k`，我们不只计算：

```text
Xc - A_k
```

还要计算：

```text
trend residual
short residual
temporal deviation
spatial deviation
candidate spread
```

这样才能判断这个候选锚点是可靠上下文，还是噪声来源。

### 2.6 ST-TTC：测试阶段可以利用流式记忆

ST-TTC 的启发是：真实部署时，测试流中已经发生的近期误差可以作为轻量记忆，用于校正未来预测。

对于非均值锚点，这意味着第二阶段可以加入：

```text
streaming anchor memory
```

即不仅使用训练集历史候选，也允许在测试阶段用已观测到的近期标签更新锚点可靠性或 residual memory。但这必须严格避免数据泄漏：

```text
只能用当前预测窗口之前已经真实观测到的 label 更新 memory。
```

## 3. 核心方案：Distributional Historical Anchor

建议将下一阶段模块命名为：

```text
DHA: Distributional Historical Anchor
```

结合当前 DCD-ST，可称为：

```text
DHA-DCD-ST: Distributional Historical Anchor enhanced Deviation-Calibrated Decomposition for Spatio-Temporal Forecasting
```

中文名：

```text
分布式历史锚点增强的偏差校准分解时空预测
```

一句话概括：

```text
用训练集内真实历史候选锚点替代单一历史均值，通过候选选择、分布可靠性估计和偏差校正，使简单下游预测器获得更丰富且更可靠的历史上下文。
```

## 4. 总体结构

原 ST-SSDL / 当前 DCD-ST 使用：

```text
Xc, Xa_mean
  -> Hc, Ha
  -> deviation = Xc - Xa_mean
  -> gate / Delta_H
  -> decoder
```

DHA-DCD-ST 使用：

```text
Xc
  -> historical candidate anchors A = {A_1, ..., A_K}
  -> anchor selector: w_k = select(Xc, A_k, node_id, time_id)
  -> Xa_ctx = sum_k w_k * A_k
  -> anchor reliability = reliability(w, spread(A), residual(A))
  -> deviation decomposition on Xc - Xa_ctx
  -> need / reliability gated correction
  -> decoder
```

核心变化：

```text
Xa_mean 变成 Xa_ctx；
单一均值锚点变成分布式候选锚点；
偏差强度变成 need；
锚点可信度变成 reliability；
校正强度由 need * reliability 控制。
```

## 5. 历史候选锚点设计

### 5.1 第一版候选集合

第一版不要直接上复杂检索，先做可控、可解释、低风险的候选集合：

```text
A_mean    同 weekday-slot 历史均值
A_median  同 weekday-slot 历史中位数
A_q25     同 weekday-slot 历史 25% 分位
A_q75     同 weekday-slot 历史 75% 分位
A_recent  训练段内最近一次同 weekday-slot 历史值
```

候选数：

```text
K = 5
```

张量：

```text
A: (B,T,N,K)
```

其中每个 `A_k` 都与 `Xc` 时间对齐：

```text
Xc: (B,T,N,1)
A_k: (B,T,N,1)
```

### 5.2 为什么第一版不用全部历史样本

全部历史样本虽然信息最完整，但有三个问题：

1. 内存和 I/O 成本高。
2. 需要复杂 top-k retrieval，第一版归因不清楚。
3. 历史样本多不等于可靠，可能引入更多 mirage。

所以第一版采用统计候选锚点，先验证：

```text
“不只用均值”是否能带来收益。
```

如果统计候选有效，再进入第二版 top-k historical retrieval。

### 5.3 第二版候选集合

第二版加入真实历史样本检索：

```text
A_top1, ..., A_topK
```

候选来自：

```text
训练段内相同 weekday-slot 附近的历史窗口
```

相似度可以先用低成本物理距离：

```text
score_k = -mean(|Xc - A_k|)
```

再升级为 representation similarity：

```text
score_k = cosine(Enc(Xc), Enc(A_k))
```

但第一版不建议直接用 representation similarity，因为会和主模型训练耦合，难以判断收益来源。

## 6. Anchor Selector

给定候选锚点：

```text
A = {A_1, ..., A_K}
```

先计算每个候选和当前窗口的残差特征：

```text
R_k = Xc - A_k                    -> (B,T,N,1)
z_k = Pool_T([R_k, |R_k|])         -> (B,N,F)
```

再加入身份上下文：

```text
z_id = concat(node_embedding, time_embedding_last)
```

候选打分：

```text
score_k = MLP_select([z_k, z_id])  -> (B,N,1)
w = softmax(score / tau, dim=K)    -> (B,N,K)
```

得到上下文锚点：

```text
Xa_ctx = sum_k w_k * A_k           -> (B,T,N,1)
```

### 6.1 为什么用 soft selection

不用 hard top-1，原因是：

1. 交通历史模式可能混合，不一定只有一个候选正确。
2. soft 权重可微，方便端到端训练。
3. 权重熵可以自然变成 reliability 指标。

### 6.2 温度参数

```text
tau = 0.5 或 1.0
```

如果 `tau` 大，选择更平滑；如果 `tau` 小，选择更接近 top-1。后续可以做消融：

```text
tau in {0.25, 0.5, 1.0, 2.0}
```

## 7. Anchor Reliability

非均值锚点的关键不是“候选越多越好”，而是：

```text
候选是否一致，当前是否真的能匹配到可靠历史模式。
```

建议第一版使用三类 reliability proxy：

### 7.1 权重熵可靠性

```text
rel_entropy = 1 - H(w) / log(K)
```

含义：

| 权重分布 | 解释 |
|---|---|
| 一个候选明显胜出 | 历史模式清晰，可靠性高 |
| 多个候选权重接近 | 历史模式分歧大，可靠性低 |

### 7.2 候选分布离散度

```text
spread = std_k(A_k)
rel_spread = exp(-spread / scale)
```

含义：

```text
同一时间槽历史候选差异越大，单个锚点越不稳定。
```

### 7.3 当前-候选残差一致性

```text
residual_min = min_k mean_T(|Xc - A_k|)
residual_gap = top2_residual - top1_residual
```

含义：

```text
如果最相似候选明显优于其他候选，说明当前确实匹配到某个历史模式；
如果 top1/top2 很接近，说明历史模式选择不确定。
```

最终可靠性：

```text
rel = sigmoid(MLP_rel([rel_entropy, rel_spread, residual_min, residual_gap, z_id]))
```

第一版也可以先用无参数版本：

```text
rel = rel_entropy * rel_spread
```

这样更符合“做减法”的实验哲学。

## 8. Need-Reliability 融合

在 DHA 中，高偏差不再直接等于强校正。

先定义：

```text
need = f(|Xc - Xa_ctx|, D_t, D_s, R_trend, R_residual)
rel  = reliability(A, w, Xc)
```

然后：

```text
g_anchor = need * rel
```

最终融合建议为双校正分支：

```text
Delta_anchor = MLP([Hc - Ha_ctx, z_dev, z_anchor])
Delta_self   = MLP([Hc, z_dev])

H_de = Hc + need * (rel * Delta_anchor + (1 - rel) * Delta_self)
```

解释：

| 情况 | 行为 |
|---|---|
| `need` 低 | 当前与历史一致，保持主干预测 |
| `need` 高，`rel` 高 | 当前偏离明显，但能找到可靠历史模式，使用 anchor correction |
| `need` 高，`rel` 低 | 当前偏离明显，但历史候选混乱，转向 self correction，抑制锚点噪声 |

这正好回应 spatiotemporal mirage：

```text
偏离越强，只代表越需要处理；
是否使用历史锚点，由 reliability 决定。
```

## 9. 训练与数据实现方案

### 9.1 最小可行实现：替换单锚点

先不改模型，只改预处理，跑以下锚点版本：

```text
mean anchor
median anchor
q25 anchor
q75 anchor
recent anchor
```

每次仍输出 `(B,T,N,3)`，第三通道还是 `x_his`，但内容换成不同 anchor。

优点：

```text
几乎不改模型，最快验证“均值是不是太弱”。
```

建议命令形式：

```text
python src/preprocessing.py --anchor-mode mean   --output-dir data/METRLA_mean
python src/preprocessing.py --anchor-mode median --output-dir data/METRLA_median
python src/preprocessing.py --anchor-mode recent --output-dir data/METRLA_recent
```

然后用同一训练命令，只替换：

```text
--processed-dir data/METRLA_median
```

### 9.2 第一版 DHA：多统计锚点通道

扩展 `x` 通道：

```text
x[...,0] value
x[...,1] time_of_day
x[...,2] anchor_mean
x[...,3] anchor_median
x[...,4] anchor_q25
x[...,5] anchor_q75
x[...,6] anchor_recent
```

张量：

```text
Xc: (B,T,N,1)
A:  (B,T,N,5)
```

需要修改：

```text
src/data.py
DCD-ST/dcd_st.py
src/preprocessing.py
```

但为了兼容 baseline，建议新增数据准备函数，而不是破坏 `prepare_x_y()`：

```text
prepare_x_y_dha()
```

### 9.3 第二版 DHA：真实历史 top-k 检索

生成一个训练集候选表：

```text
anchor_bank.npz
```

包含：

```text
slot_id -> historical windows
```

模型或数据集根据当前样本的 slot_id 取候选：

```text
A_candidates: (B,T,N,K)
```

该版本更强，但实现更重，建议在统计多锚点有效后再做。

## 10. 实验计划

### 10.1 第一轮：非均值锚点是否有效

| 实验 | processed_dir | 目的 |
|---|---|---|
| `DCD-mean-anchor` | `data/METRLA_mean` | 当前基线 |
| `DCD-median-anchor` | `data/METRLA_median` | 中位数是否比均值稳健 |
| `DCD-q25-anchor` | `data/METRLA_q25` | 低分位模式是否有用 |
| `DCD-q75-anchor` | `data/METRLA_q75` | 高分位/拥堵模式是否有用 |
| `DCD-recent-anchor` | `data/METRLA_recent` | 最近同槽是否比长期均值更有用 |

判定：

```text
如果 median/recent/q75 任意一个稳定优于 mean，说明 historical mean anchor 确实不是最优锚点。
```

### 10.2 第二轮：多锚点选择是否有效

| 实验 | 结构 | 目的 |
|---|---|---|
| `DHA-uniform` | 多锚点平均 | 判断多锚点信息是否本身有用 |
| `DHA-soft-select` | softmax selector | 判断动态选择是否有用 |
| `DHA-soft-select-rel` | selector + reliability | 判断可靠性是否抑制 mirage |
| `DHA-hard-top1` | top-1 选择 | 判断是否需要 soft 混合 |

判定：

```text
soft-select > uniform：说明选择机制有用；
soft-select-rel 在 high-deviation/noisy-anchor 下更稳：说明 reliability 有用；
hard-top1 若波动大，说明交通历史模式需要 soft 混合。
```

### 10.3 第三轮：鲁棒性

| 场景 | 目的 |
|---|---|
| current missing | 当前输入缺失时，历史候选是否提供上下文 |
| anchor noise | 历史锚点被污染时，reliability 是否能抑制噪声 |
| high-deviation split | 高偏差样本是否获益 |
| long horizon 60min | 长预测步是否更需要历史分布上下文 |

### 10.4 第四轮：简单下游结构

为了支撑“丰富上下文 + 简单下游”的动机，需要跑：

```text
rnn_units = 64
cheb_k = 1
```

对比：

```text
light DCD-mean
light DHA-soft-select-rel
```

如果后者明显更好，则说明 DHA 不是靠堆大模型，而是确实提供了更有效上下文。

## 11. 可视化设计

DHA 的可视化比当前 DCD gate 更有说服力。

建议图表：

| 图 | 内容 | 说明 |
|---|---|---|
| Anchor distribution | mean/median/q25/q75/recent 曲线 | 证明均值隐藏了历史多样性 |
| Candidate weights | `w_k` 热力图 | 当前样本选择了哪类历史模式 |
| Reliability map | `rel` 节点热力图 | 哪些节点锚点可信 |
| Mirage case | high need / low rel 案例 | 证明模型能抑制误导锚点 |
| Missing case | current missing 下预测 | 证明可靠锚点可补上下文 |
| Noisy anchor case | anchor 被污染下预测 | 证明模型不会盲目信历史 |

特别重要的一张图：

```text
同一节点同一时间槽的历史候选曲线 + 当前窗口 + 真实未来 + 模型选择权重。
```

这张图可以直接展示：

```text
均值锚点处在多个历史模式中间，可能并不是任何真实历史；
DHA 能选择更接近当前状态的候选锚点。
```

## 12. 相对 ST-SSDL 的新贡献表述

可以将论文贡献改写为：

1. We revisit the historical average anchor used in ST-SSDL-style deviation learning and show that mean anchors may erase multimodal historical patterns and induce spatiotemporal mirage.
2. We propose Distributional Historical Anchor (DHA), which replaces single mean anchors with train-only physical historical anchor candidates, including quantile, median, recent, and retrieved same-slot patterns.
3. We design an anchor selection and reliability estimation mechanism that extracts useful historical context while suppressing unreliable anchors under high-deviation or noisy-anchor scenarios.
4. We integrate DHA with a lightweight deviation-calibrated downstream predictor, showing that reliable historical context can improve forecasting without adding a heavy backbone or unstable latent prototypes.

中文贡献：

1. 发现 ST-SSDL 风格历史均值锚点存在信息塌缩风险，会掩盖多模态历史模式并诱发时空海市蜃楼。
2. 提出分布式历史锚点 DHA，用训练段真实历史候选替代单一均值锚点，保留中位数、分位数、最近同槽和相似历史模式。
3. 设计锚点选择与可靠性估计机制，在锚点可信时利用历史上下文，在锚点不可信时抑制历史噪声。
4. 将 DHA 接入轻量偏差校准预测器，使简单下游结构也能利用更丰富且更可靠的历史上下文。

## 13. 推荐执行顺序

建议立刻按这个顺序推进：

```text
Step 1: 实现 anchor-mode 预处理：mean / median / q25 / q75 / recent
Step 2: 用现有 DCD-ST 训练五个单锚点版本，验证非均值是否有效
Step 3: 实现多锚点通道与 soft selector
Step 4: 加入 reliability，并做 high-deviation / noisy-anchor 诊断
Step 5: 再决定是否实现真实 top-k historical retrieval
```

这条路线的好处是：

```text
从最小改动开始；
每一步都有明确实验结论；
如果非均值锚点无效，可以及时止损；
如果有效，它会比单纯重构 gate 更适合作为核心创新点。
```

## 14. 当前最值得先做的实验

第一条命令不是训练模型，而是改预处理：

```text
给 src/preprocessing.py 增加 --anchor-mode。
```

先支持：

```text
mean
median
q25
q75
recent
```

然后生成五套数据：

```text
data/METRLA_mean
data/METRLA_median
data/METRLA_q25
data/METRLA_q75
data/METRLA_recent
```

随后复用当前 DCD-ST 训练命令，比较：

```text
metrla_dcd_anchor_mean
metrla_dcd_anchor_median
metrla_dcd_anchor_q25
metrla_dcd_anchor_q75
metrla_dcd_anchor_recent
```

如果某个非均值锚点明显更好，我们就有了非常强的证据：

```text
问题不只在 gate，而在 ST-SSDL 的 mean historical anchor 本身。
```
