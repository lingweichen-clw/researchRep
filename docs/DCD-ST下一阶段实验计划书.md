# DCD-ST 下一阶段实验计划书

更新时间：2026-07-01

## 1. 当前阶段结论

当前已经完成两组关键实验：

| 实验 | 结论 |
|---|---|
| `DCD-ST-v1 learned gate` | `g_dev` 长期接近 0.5，整体预测性能可用，但动态门控选择性不足 |
| `DCD-ST fixed gate=0.5` | 与 learned gate 的 test MAE 几乎一致，说明当前门控没有形成独立贡献 |

因此，下一阶段不能继续把普通 `sigmoid(MLP)` 门控当作核心贡献。更合理的判断是：

```text
DCD-ST 当前真正可能有效的是 current-anchor residual decomposition 与 Delta_H 校正分支；
当前 learned gate 更像一个固定缩放因子，不足以支撑“动态门控学到了有效策略”的论文叙事。
```

下一阶段目标不是盲目加复杂模块，而是把问题拆清楚：

```text
1. Delta_H 是否真的有价值？
2. gate 是否可以被更简单的 alpha 或 no-gate 结构替代？
3. 历史锚点什么时候是上下文，什么时候是噪声？
4. 如何在不增加重型 backbone 的前提下，让模型更稳健地利用锚点信息？
```

## 2. 修正后的核心动机

原始动机可以保留，但需要更严谨地表述。

不建议表述为：

```text
当前窗口越偏离历史锚点，就越应该相信历史锚点，校正越强。
```

这句话容易被反驳，因为在 spatiotemporal mirage 场景下，历史锚点可能恰恰是错误参照。过去同一时刻的交通模式不一定适用于当前状态，尤其在事故、节假日、施工、突发天气或传感器异常下，历史锚点越多，可能带来的误导越强。

建议表述为：

```text
我们希望利用历史锚点提供更丰富的周期上下文，但不盲目信任锚点。
当当前窗口显著偏离历史锚点时，模型需要启动偏差感知模块；
该模块首先判断锚点是否可靠，再决定是利用锚点校正、抑制锚点噪声，还是退回当前窗口主导的预测。
```

也就是说：

```text
偏离越强，表示“需要更认真处理偏差”，不表示“需要更强地相信历史”。
```

这能把我们的动机从单纯的历史增强，升级为：

```text
Anchor-aware but mirage-resistant deviation modeling.
```

中文可以表述为：

```text
锚点感知但抗海市蜃楼的偏差建模。
```

## 3. 为什么不能简单认为“偏离越强，校正越强”

### 3.1 偏离强有两种完全不同的来源

设当前窗口为 `Xc`，历史锚点为 `Xa`：

```text
R = Xc - Xa
```

当 `|R|` 很大时，至少有两种解释：

| 情况 | 含义 | 应对方式 |
|---|---|---|
| 当前状态发生真实变化 | 拥堵形成、速度突降、需求突增 | 需要利用 residual correction 调整预测 |
| 历史锚点不再可靠 | 节假日、事故、特殊事件、周期错位 | 需要抑制锚点影响，避免把噪声注入模型 |

所以高偏离只说明：

```text
当前短窗口和历史周期上下文不一致。
```

它不直接说明：

```text
历史锚点一定有用。
```

### 3.2 时空海市蜃楼的风险

spatiotemporal mirage 可以理解为：

```text
模型从历史相似时间、相似空间或相似模式中看到一个“看似合理”的上下文，
但这个上下文对当前预测是误导性的。
```

在交通预测中，典型表现包括：

1. 同一时间段历史上通常拥堵，但今天道路畅通。
2. 历史锚点显示正常，但当前发生事故。
3. 邻近节点历史相关性强，但当前只在局部节点出现异常。
4. 多个历史周期给出冲突信号，简单平均会制造噪声。

因此，如果我们只是简单地增加锚点信息，确实可能适得其反。

下一阶段必须证明：

```text
我们的模块不是无条件吃进更多历史锚点；
而是从历史锚点中提取可用上下文，同时识别并过滤不可靠锚点。
```

## 4. 下一版机制：Need-Reliability 解耦

下一阶段建议把当前 `g_dev` 拆成两个语义明确的因子：

```text
correction_need: 当前样本是否需要偏差校正
anchor_reliability: 历史锚点是否值得信任
```

### 4.1 correction_need

`correction_need` 表示校正需求强度：

```text
need = f(|R_trend|, |R_residual|, |D_t|, |D_s|)
```

语义：

```text
当前窗口与历史周期存在明显不一致，普通短窗口预测可能不够，需要偏差模块介入。
```

注意，这里的介入不是一定使用历史锚点，而是启动偏差建模。

### 4.2 anchor_reliability

`anchor_reliability` 表示锚点可信度：

```text
rel = f(anchor_consistency, spatial_consistency, temporal_smoothness, node_id, time_id)
```

可用的无监督代理特征：

| 特征 | 计算依据 | 语义 |
|---|---|---|
| `trend_consistency` | `R_trend` 是否平滑 | 持续漂移比随机噪声更可信 |
| `residual_ratio` | `|R_residual| / (|R_trend| + eps)` | 突发噪声占比越高，锚点越不稳定 |
| `spatial_agreement` | 当前节点 residual 是否与邻居一致 | 局部孤立异常可能是噪声或传感器异常 |
| `temporal_identity` | time-of-day / day pattern | 某些时段周期性更强，锚点更有价值 |
| `node_identity` | sensor-specific embedding | 不同传感器的周期可靠性不同 |

### 4.3 推荐融合形式

第一版重构不建议直接上复杂 mixture-of-experts。建议从最小改动开始：

```text
need = sigmoid(MLP_need(z_dev))        -> (B,N,1)
rel  = sigmoid(MLP_rel(z_dev,z_id))    -> (B,N,1)
g_anchor = need * rel                  -> (B,N,1)
H_de = Hc + g_anchor * Delta_H
```

其中：

```text
Delta_H = MLP([Hc-Ha, z_dev]) -> (B,N,R)
```

解释：

1. `need` 高：说明当前样本确实偏离明显。
2. `rel` 高：说明历史锚点仍然可信。
3. `need * rel` 高：才真正使用锚点偏差信息做强校正。
4. `need` 高但 `rel` 低：说明出现可能的 spatiotemporal mirage，应抑制锚点校正。

如果这一版仍然不足，再考虑双分支校正：

```text
Delta_anchor = MLP([Hc-Ha, z_dev])
Delta_self   = MLP([Hc, z_dev])

H_de = Hc + need * (rel * Delta_anchor + (1-rel) * Delta_self)
```

这表示：

```text
高偏离且锚点可靠：使用 anchor correction；
高偏离但锚点不可靠：使用 current-window self correction；
低偏离：尽量保持 Hc，不额外扰动主干。
```

该结构比普通 `sigmoid(MLP)` 更容易解释，也更贴合“用锚点，但防止锚点噪声”的动机。

## 5. 下一阶段实验矩阵

### 5.1 第一组：确认 Delta_H 是否有价值

目的：判断当前 DCD-ST 的收益到底来自 `Delta_H`，还是来自训练噪声。

| 实验名 | 结构 | 目的 |
|---|---|---|
| `DCD-learned-gate` | 当前 learned `g_dev` | 已完成，作为参照 |
| `DCD-fixed-gate-05` | `g_dev=0.5` | 已完成，证明 learned gate 贡献弱 |
| `DCD-scalar-alpha` | `H_de=Hc+alpha*Delta_H` | 判断节点级门控是否必要 |
| `DCD-no-gate` | `H_de=Hc+Delta_H` | 判断直接校正是否足够 |
| `DCD-no-delta` | `H_de=Hc` | 判断偏差校正分支是否有必要 |

预期判定：

```text
如果 scalar-alpha/no-gate 与 learned-gate 持平，删除复杂门控；
如果 no-delta 明显变差，说明 Delta_H 是有效核心；
如果 no-delta 不变，说明当前 DCD-ST 主体创新还需要重新设计。
```

### 5.2 第二组：锚点可靠性诊断

目的：证明历史锚点不是越多越好，必须判断可靠性。

先不训练新模型，直接对现有 test split 做分组分析：

| 分组维度 | 计算方式 | 要回答的问题 |
|---|---|---|
| low/mid/high deviation | `mean(|Xc-Xa|)` 分位数 | 偏离越强是否越难预测 |
| trend-dominant/residual-dominant | `|R_trend|` 与 `|R_residual|` 比例 | 持续漂移和突发扰动哪个更影响误差 |
| spatial-consistent/spatial-isolated | residual 是否与邻居一致 | 局部孤立异常是否更像噪声 |
| anchor-helpful/anchor-harmful proxy | 固定 gate counterfactual 差异 | 锚点校正在哪些样本有帮助 |

诊断输出建议：

```text
summary_by_deviation.csv
summary_by_reliability_proxy.csv
deviation_reliability_quadrant.csv
```

关键图：

```text
x-axis: deviation strength
y-axis: prediction error
color: anchor reliability proxy
```

如果发现：

```text
high deviation + high reliability -> 校正有效；
high deviation + low reliability  -> 校正无效或有害；
```

那就能支撑 Need-Reliability 解耦机制。

### 5.3 第三组：Need-Reliability Gate 重构

目的：验证语义门控是否优于普通 MLP gate。

| 实验名 | 结构 | 目的 |
|---|---|---|
| `DCD-alpha` | 全局 alpha | 最强减法 baseline |
| `DCD-need-only` | 只用 deviation need | 看“偏离强就校正”是否足够 |
| `DCD-need-rel` | `need * reliability` | 验证抗海市蜃楼门控 |
| `DCD-need-rel-self` | anchor/self 双校正 | 验证低可靠锚点下是否需要 current self correction |

判定标准：

```text
DCD-need-rel 在 high-deviation/high-reliability 组优于 DCD-alpha；
DCD-need-rel 在 high-deviation/low-reliability 组不劣化；
DCD-need-only 如果在 low-reliability 组劣化，就说明“偏离越强直接校正”确实有风险。
```

### 5.4 第四组：missing 与 noisy anchor 鲁棒性

目的：直接验证锚点噪声是否会适得其反，以及新 gate 能否抑制噪声。

建议构造三类扰动：

| 扰动 | 位置 | 意义 |
|---|---|---|
| current missing | `Xc` | 当前输入缺失，历史锚点可能提供上下文 |
| anchor missing/noise | `Xa` | 历史锚点不可靠，模型应抑制锚点 |
| mixed missing | `Xc` 与 `Xa` | 最接近真实复杂缺失 |

missing ratio：

```text
0.1, 0.2, 0.3, 0.5
```

对比模型：

```text
ST-SSDL-full
ST-SSDL-no-ssdl
DCD-alpha
DCD-need-only
DCD-need-rel
```

关键结论目标：

```text
当 Xc 缺失时，可靠锚点应提升预测；
当 Xa 被污染时，Need-Reliability Gate 不应被锚点拖垮；
这能证明我们不是简单堆历史信息，而是做可靠性筛选。
```

### 5.5 第五组：简单下游结构验证

你的动机是对的：我们希望创新结构提供更丰富、更可靠的上下文，使简单下游结构也能高质量预测。

因此需要增加一组轻量主干实验：

| 设置 | 改动 | 目的 |
|---|---|---|
| full backbone | 当前 `rnn_units=128, cheb_k=3` | 主结果 |
| light backbone | `rnn_units=64` | 看 DCD 是否能补偿容量下降 |
| very-light graph | `cheb_k=1` | 看复杂图卷积是否必须 |
| no adaptive embedding | `adaptive_embedding_dim=0` 已是当前设置 | 保持减法路线 |

判定逻辑：

```text
如果 light DCD 仍明显优于 light baseline，
说明我们的偏差分解/可靠锚点机制确实提供了更丰富上下文，
而不是依赖重型下游结构硬拟合。
```

## 6. 指标体系

下一阶段不能只看 overall MAE。

必须报告：

```text
overall MAE/RMSE/MAPE
15/30/60min MAE/RMSE/MAPE
low/mid/high deviation MAE
high deviation + high reliability MAE
high deviation + low reliability MAE
missing ratio robustness
anchor corruption robustness
parameter count
training time per epoch
```

对于 gate/校正模块，还要报告：

```text
need.mean/std/q10/q90
reliability.mean/std/q10/q90
correction_to_hc_mean
delta_to_hc_mean
need_vs_deviation_corr
reliability_vs_anchor_helpfulness_corr
```

## 7. 可视化计划

下一阶段可视化要从“gate 是否高”升级为“为什么高、是否该高”。

建议输出：

| 图 | 内容 | 论文解释 |
|---|---|---|
| Figure 1 | `Xc/Xa/y/pred` 低中高偏差案例 | 历史锚点提供周期上下文，但存在偏离 |
| Figure 2 | `R_trend/R_residual` 热力图 | 区分持续漂移与突发扰动 |
| Figure 3 | deviation vs error 散点 | 高偏差样本更难预测 |
| Figure 4 | deviation-reliability 四象限 | 区分可用锚点与海市蜃楼锚点 |
| Figure 5 | need/reliability/correction heatmap | 解释模型何时校正、何时抑制锚点 |
| Figure 6 | missing/noisy anchor 下的预测对比 | 证明可靠性门控避免锚点噪声 |

四象限解释：

| 象限 | 语义 | 期望行为 |
|---|---|---|
| low deviation / high reliability | 正常周期 | 小幅或无需校正 |
| high deviation / high reliability | 可解释漂移 | 强 anchor-aware correction |
| high deviation / low reliability | 时空海市蜃楼风险 | 抑制锚点或转向 self correction |
| low deviation / low reliability | 锚点信息弱但影响小 | 保持主干预测 |

## 8. 实现优先级

### 阶段 A：先做减法消融

优先实现训练参数：

```text
--dcd-fusion-mode learned_gate
--dcd-fusion-mode fixed_gate
--dcd-fusion-mode scalar_alpha
--dcd-fusion-mode no_gate
--dcd-fusion-mode no_delta
```

先跑：

```text
metrla_dcd_scalar_alpha
metrla_dcd_no_gate
metrla_dcd_no_delta
```

这是最高优先级，因为它决定我们是否保留 gate。

### 阶段 B：做可靠性诊断脚本

新增或扩展：

```text
DCD-ST/diagnose_dcd.py
```

增加：

```text
deviation group metrics
trend/residual group metrics
spatial consistency metrics
anchor reliability proxy
```

### 阶段 C：实现 Need-Reliability Gate

新增模型参数：

```text
--dcd-gate-type mlp
--dcd-gate-type need
--dcd-gate-type need_reliability
--dcd-gate-type need_reliability_self
```

先实现 `need_reliability`，如果低可靠高偏差样本仍然不稳，再实现 self correction。

### 阶段 D：missing/noisy anchor 鲁棒性

在诊断或评估脚本中加入：

```text
--missing-target current
--missing-target anchor
--missing-ratio
--noise-target anchor
--noise-std
```

注意：这些是评估扰动，不应污染训练集。

### 阶段 E：轻量下游结构

在主结果稳定后，再跑轻量结构：

```text
rnn_units=64
cheb_k=1
```

这一步用于支撑“我们的创新结构让简单下游也能获得高质量预测”。

## 9. 推荐实验顺序

第一轮，只跑必要实验：

| 顺序 | 实验 | 目的 |
|---:|---|---|
| 1 | `DCD-no-delta` | 判断偏差校正是否有用 |
| 2 | `DCD-scalar-alpha` | 判断复杂 gate 是否必要 |
| 3 | `DCD-no-gate` | 判断直接校正是否过强 |
| 4 | 扩展诊断脚本 | 找 high-deviation / mirage 样本 |
| 5 | `DCD-need-only` | 验证“偏离越强直接校正”的风险 |
| 6 | `DCD-need-rel` | 验证可靠性门控是否解决风险 |

第二轮，再跑鲁棒性：

| 顺序 | 实验 | 目的 |
|---:|---|---|
| 1 | current missing | 当前输入缺失时是否利用锚点 |
| 2 | anchor noisy/missing | 锚点不可靠时是否抑制噪声 |
| 3 | mixed missing | 复杂扰动鲁棒性 |
| 4 | light backbone | 验证简单下游结构 |

## 10. 论文叙事调整

下一版论文主线建议改为：

```text
ST-SSDL 证明了历史锚点与 deviation learning 的价值，
但其 prototype 空间存在 assignment collapse，且直接使用历史锚点可能受到 spatiotemporal mirage 影响。

我们提出 DCD-ST：删除不稳定 prototype，将 current-anchor residual 分解为趋势、残差、时间偏差与空间偏差；
进一步将偏差校正拆成 correction need 与 anchor reliability，
从而在锚点可靠时利用历史上下文，在锚点不可靠时抑制历史噪声。

该机制为简单下游预测结构提供更丰富但经过筛选的时空上下文，
在 high-deviation、missing 和 noisy-anchor 场景下实现更稳健预测。
```

贡献点可以写成：

1. 发现 ST-SSDL prototype deviation 在复现实验中存在坍缩风险，并进一步指出历史锚点本身可能引入 spatiotemporal mirage。
2. 提出连续偏差分解，把 current-anchor residual 拆为趋势、短期扰动、时间偏差和空间偏差，删除 prototype 与辅助对比损失。
3. 提出 Need-Reliability 解耦门控，区分“是否需要校正”和“锚点是否可信”，避免高偏差样本中盲目注入历史噪声。
4. 通过 high-deviation、missing、noisy-anchor 和轻量下游实验，验证该结构能让简单预测器利用更可靠的历史上下文。

## 11. 成功与失败判定

### 11.1 可以继续推进的结果

满足以下任意两条，就值得继续做 DCD-ST-v2：

```text
no-delta 明显差于 alpha/no-gate，说明 Delta_H 有用；
need_reliability 优于 need_only，说明可靠性判断有用；
anchor noisy 场景下 need_reliability 不明显劣化，说明能抑制锚点噪声；
light DCD 优于 light baseline，说明创新结构补充了上下文。
```

### 11.2 需要收缩工作的结果

如果出现：

```text
no-delta 与所有 DCD 版本几乎一致；
missing/noisy anchor 下 DCD 没有鲁棒性优势；
high-deviation 分组没有任何收益；
```

则说明当前历史锚点路线在这个 baseline 上贡献不足，应把工作收缩为：

```text
ST-SSDL prototype collapse diagnostic + simplified residual decomposition baseline
```

不要继续堆门控模块。

## 12. 对当前问题的明确回答

你的理解基本正确：

```text
我们的动机是更好地利用过去锚点信息，让模型吃到比短窗口更丰富的上下文；
再用相对简单的下游结构实现高质量预测。
```

但要补充一个关键限制：

```text
过去锚点不是天然有益的上下文，它必须先经过可靠性判断。
```

因此，下一阶段的核心不是“更多历史信息”，而是：

```text
更可靠的历史上下文选择。
```

最终目标应表述为：

```text
用轻量偏差分解和锚点可靠性判断，从历史周期中提取可用上下文、过滤误导性上下文，
从而让简单预测结构在常规、高偏差和缺失扰动场景下都保持稳定预测能力。
```
