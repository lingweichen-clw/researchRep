# STAnchor：锚点选择实验分析与时空预训练模块工作计划

更新时间：2026-07-14

## 1. 先给结论

五种固定锚点实验已经回答了第一阶段问题：

```text
历史锚点的定义会影响结果，但不存在一个在所有指标上稳定占优的固定锚点。
当前最值得继续的方向不是把 mean 永久替换成 q25，
而是把 mean / median / q25 / q75 / recent 同时保留，学习样本级锚点选择和可靠性。
```

目前最关键的证据有四条：

1. `q25` 的测试 MAE 最好，但相对 `mean` 只改善 `0.0071`，约 `0.24%`。
2. `mean` 的 RMSE 最好，`median` 的 MAPE 最好，说明没有统一赢家。
3. 五个实验都只有 `seed=999`，固定锚点之间的差距很可能与随机种子波动处于同一量级。
4. 一个不训练参数的“当前窗口相似度选择器”，已经把未来历史锚点的直接 MAE 从最佳固定锚点的 `3.9594` 降到 `3.7027`；说明多锚点的可选择空间真实存在。

因此下一步应分成两层：

```text
近期：先证明多锚点 selector + reliability 确实优于所有固定锚点。
中期：再把 selector 预训练成轻量、可微调、可接不同预测骨干的 STAnchor 模块。
```

不建议现在直接宣称“时空基础模型”。更严谨的定位是：

```text
foundation-model-compatible spatio-temporal memory adapter
面向时空基础模型和普通预测骨干的可迁移历史记忆插件
```

只有完成跨城市、跨骨干、零样本或少样本迁移后，才有证据进一步使用 `foundation module` 的表述。

---

## 2. 本轮锚点实验是否公平

### 2.1 五组实验

当前完成的实验为：

```text
log/metrla_dha_anchor_mean
log/metrla_dha_anchor_median
log/metrla_dha_anchor_q25
log/metrla_dha_anchor_q75
log/metrla_dha_anchor_recent
```

五组实验均使用：

```text
dataset       = METR-LA
seed          = 999
epochs        = 100
patience      = 20
batch_size    = 64
lr            = 0.001
lr scheduler  = MultiStep(40, 70)
model         = DHADCDST
fusion        = learned_gate
```

### 2.2 数据通道核验

对五套 `testhis.npz` 做逐元素核验后：

```text
x[..., 0] traffic value 完全相同
x[..., 1] time-in-day 完全相同
y[..., 0] target value 完全相同
y[..., 1] future time-in-day 完全相同
```

只有第三通道锚点不同。相对 `mean` 锚点，测试输入锚点的平均绝对差为：

| 锚点 | 与 mean 锚点的平均绝对差 |
|---|---:|
| median | 1.4943 |
| q25 | 2.3548 |
| q75 | 2.9487 |
| recent | 3.5621 |

`DHA-DCD-ST/dha_dcd_st.py` 中的 `DHADCDST` 当前直接继承 `DCDST`，没有增加额外结构。因此，这一轮属于较干净的单变量消融：

```text
同一模型、同一训练协议，只替换历史锚点统计量。
```

另一个交叉检查是：`DHA-mean` 与原 `DCD-ST-v1` 的最终指标完全相同，均为 `test MAE=2.9341`，说明新入口没有悄悄改变原模型行为。

---

## 3. 五种固定锚点的训练结果

### 3.1 最佳检查点和测试指标

| Anchor | Best epoch | Val MAE | Test MAE | Test RMSE | Test MAPE | 15 min MAE | 30 min MAE | 60 min MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | 48 | 2.7434 | 2.9341 | **6.0307** | 8.13% | 2.6148 | 2.9724 | 3.3957 |
| median | 54 | **2.7255** | 2.9393 | 6.0896 | **7.92%** | 2.6200 | 2.9776 | 3.3980 |
| q25 | 51 | 2.7362 | **2.9270** | 6.0353 | 8.04% | **2.6084** | **2.9673** | 3.3797 |
| q75 | 50 | 2.7381 | 2.9272 | 6.0557 | 8.03% | 2.6115 | 2.9683 | **3.3773** |
| recent | 48 | 2.7320 | 2.9365 | 6.0317 | 7.98% | 2.6186 | 2.9754 | 3.3929 |

### 3.2 相对 mean 的变化

| Anchor | Test MAE 变化 | 相对变化 | 判断 |
|---|---:|---:|---|
| q25 | -0.0071 | -0.24% | 当前 MAE 最好，但幅度很小 |
| q75 | -0.0069 | -0.24% | 与 q25 基本相同 |
| recent | +0.0024 | +0.08% | 与 mean 近似持平 |
| median | +0.0052 | +0.18% | MAE 略差，但 MAPE 最好 |

### 3.3 不能简单宣布 q25 胜出

原因如下：

1. 所有实验只有一个随机种子，尚无方差和置信区间。
2. `q25` 与 `q75` 的 Test MAE 只差 `0.0002`，没有实际区分度。
3. 验证集最好的 `median` 在测试 MAE、RMSE 上反而最差，验证/测试排序不一致。
4. `mean` 的 RMSE 最好，说明它对少数大误差样本可能更稳。
5. `median` 的 MAPE 最好，说明它在相对误差意义下可能更有优势。

当前合理结论是：

```text
固定统计锚点之间存在轻微差异，但差异尚不足以支撑“某一个统计量普遍最优”。
```

### 3.4 q25 / q75 的物理语义需要纠正

METR-LA 记录的是交通速度，不是交通流量。因此：

```text
q25 = 同 weekday-slot 的较低速度状态，更接近拥堵模式
q75 = 同 weekday-slot 的较高速度状态，更接近畅通模式
```

后续文档和论文中不能把 `q25` 解释为“低流量/畅通”，否则物理语义会写反。

---

## 4. 当前门控是否利用了锚点差异

日志中的 `gate_sparse` 在当前权重为 0 时，本质上记录的是 `gate.mean()`。五个最佳检查点附近的均值为：

| Anchor | Gate mean |
|---|---:|
| mean | 0.5014 |
| median | 0.4958 |
| q25 | 0.4980 |
| q75 | 0.4974 |
| recent | 0.4995 |

这说明更换锚点之后，现有门控仍停留在约 `0.5`：

```text
固定锚点实验改变了输入参照，但没有解决原 DCD-ST 门控缺少选择性的问题。
```

需要强调的是，当前模型每次只能看到一个锚点。它无法回答：

```text
当前样本到底应该选 q25、q75、median，还是完全不用历史锚点？
```

因此，不能要求现有单锚点 gate 自动承担候选比较功能。下一版必须显式把候选维 `K` 暴露给 selector。

---

## 5. 锚点本身包含多少信息

下面的诊断不经过神经网络，而是直接比较测试集原始速度与训练段构造的历史锚点。

### 5.1 每个固定锚点的直接质量

| Anchor | 输入窗口 Anchor MAE | 输入相关系数 | Future Anchor MAE | Future Anchor RMSE | Future Anchor MAPE |
|---|---:|---:|---:|---:|---:|
| mean | 4.1874 | **0.8186** | 4.1864 | **7.8517** | 13.03% |
| median | **3.9604** | 0.7943 | **3.9594** | 8.4403 | 12.58% |
| q25 | 4.4536 | 0.8037 | 4.4522 | 8.6495 | **11.60%** |
| q75 | 4.5528 | 0.7537 | 4.5517 | 9.6085 | 16.09% |
| recent | 5.0045 | 0.7313 | 5.0030 | 9.8989 | 14.02% |

由此可见：

1. `median` 是最好的固定点值历史先验，但它训练出的 DCD-ST 测试 MAE 并不是最好。
2. `mean` 与当前速度的整体相关性最高、直接 RMSE 最低。
3. `recent` 的单独质量最弱，说明“最近一次同槽位”受具体历史周状态影响较大。
4. 锚点自身质量与最终模型排序不一致，说明当前 DCD-ST 没有稳定地把锚点信息转化为预测收益。

### 5.2 无参数相似度选择器

对每个测试样本和节点，计算：

```text
d_k = mean_t |X_current(t) - A_k(t)|,  k in {mean, median, q25, q75, recent}
k*  = argmin_k d_k
```

然后使用相同 `k*` 的 future-aligned train-only anchor 作为未来历史先验。这个过程只使用当前/过去输入和训练段锚点表，不读取测试标签。

选择比例为：

| Anchor | 被选择比例 |
|---|---:|
| mean | 26.70% |
| median | 24.32% |
| q25 | 16.60% |
| q75 | 25.10% |
| recent | 7.28% |

结果：

```text
最佳固定 future anchor MAE（median） = 3.9594
当前相似度动态选择 MAE             = 3.7027
相对下降                           = 6.48%
```

这条结果比固定锚点模型之间 `0.2%` 左右的差异更重要，因为它说明：

```text
同一个节点在不同窗口需要的历史状态不同；
多锚点的价值主要来自条件化选择，而不是寻找一个全局最优统计量。
```

### 5.3 Oracle 上界

如果违规使用真实未来，在每个未来时间、节点上选择误差最小的锚点，则：

```text
oracle future-anchor MAE = 1.9880
```

各锚点成为 oracle 最优的比例为：

| Anchor | Oracle 最优比例 |
|---|---:|
| mean | 13.52% |
| median | 13.77% |
| q25 | 26.34% |
| q75 | 30.82% |
| recent | 15.55% |

Oracle 不能用于部署，也不能作为最终模型结果；它只用于测量候选集合的理论可选择空间。所有锚点都在一部分场景中成为最佳，进一步否定了“只保留 q25”这一做法。

---

## 6. 当前实现还存在的结构限制

### 6.1 单锚点通道限制

当前数据契约是：

```text
x: (B,T,N,3)
channel 0 = value
channel 1 = time-in-day
channel 2 = one fixed anchor
```

候选锚点没有形成显式的 `K` 维，因此模型只能学习“如何使用给定锚点”，不能学习“选择哪个锚点”。

### 6.2 future anchor 已生成但没有被使用

预处理后的 `y[...,2]` 已经包含 future-aligned train-only anchor，但 `src/data.py::prepare_x_y()` 只返回：

```text
y0    = y[...,0:1]
y_cov = y[...,1:2]
```

未来锚点通道目前被丢弃。它不是未来标签，而是根据未来已知 calendar slot 查询训练段统计表得到的先验，因此在严格使用训练段构表时可以合法使用。

这为下一版提供了一个重要接口：

```text
过去候选锚点用于判断当前属于哪种历史状态；
未来对齐候选锚点用于生成未来历史先验分布。
```

### 6.3 当前训练锚点不满足最严格的样本时刻因果约束

当前实现对整个训练段一次性聚合锚点表。它没有验证集/测试集泄漏，但对训练段内较早的样本而言，锚点表可能包含该样本之后、但仍处于训练段内的观测。

这在普通离线训练中可接受，但如果论文声称“在线因果记忆”或使用检索未来作为预训练教师，必须升级为：

```text
leave-one-window-out，或
只允许候选历史窗口结束时间早于当前 query 的开始时间。
```

### 6.4 节点 ID 不利于跨城市迁移

当前 DCD-ST 使用固定节点 embedding。不同城市的节点数、拓扑和语义不同，直接迁移固定节点 ID embedding 不合理。

预训练核心中应优先使用：

```text
节点统计描述
局部图结构描述
时间/日历身份
目标城市少量可训练 prompt
```

固定 node embedding 只能作为目标数据集 adapter 的可选参数，不能成为通用核心。

---

## 7. 下一步：先完成多锚点选择的最小验证

在投入完整预训练之前，必须先证明 selector 本身成立。

### 7.1 数据契约

不再保存五套重复的完整 NPZ，建议改为：

```text
X_current    : (B,T,N,1)
X_time       : (B,T,N,Ct)
A_past       : (B,T,N,K)
A_future     : (B,H,N,K)
Y            : (B,H,N,1)
K            = 5
```

候选顺序固定为：

```text
[mean, median, q25, q75, recent]
```

### 7.2 第一版 selector 不要做重

建议先实现：

```text
R_k       = X_current - A_past[..., k]
f_k       = Pool_T([R_k, |R_k|, trend(R_k), residual(R_k)])
score_k   = MLP_shared([f_k, time_feature, node_descriptor])
w         = softmax(score / tau, dim=K)
A_ctx     = sum_k w_k * A_past_k
A_prior   = sum_k w_k * A_future_k
```

可靠性单独输出：

```text
rel = sigmoid(MLP_rel(candidate_spread, spatial_agreement, selector_margin, residual_features))
```

这里的 `rel` 不再等于偏差大小。它表示：

```text
当前选出的历史先验对未来预测有帮助的概率或强度。
```

### 7.3 必做对照组

| 实验 | 含义 |
|---|---|
| no-anchor | 完全不用历史锚点 |
| mean | 当前固定 mean 基线 |
| q25 / q75 | 当前最好固定锚点 |
| uniform-mixture | 五锚点均匀融合 |
| hard-similarity | 当前无参数相似度选择器 |
| learned-selector | 学习 soft selection，不使用 reliability |
| selector+reliability | 完整最小模型 |
| oracle | 只作为上界诊断，禁止进入主结果 |

### 7.4 统计要求

至少运行：

```text
seed = 42, 2024, 999
```

正式论文建议五个种子。报告：

```text
mean ± std
相对 mean baseline 的改进百分比
按连续时间块执行 paired block bootstrap 的 95% CI
```

交通序列不是独立同分布样本，因此不能把所有时间点直接当独立样本做普通 t-test。

### 7.5 selector 必须输出的诊断

```text
candidate_usage
effective_candidate_number = exp(entropy(mean_usage))
sample_weight_entropy
top1-top2 margin
reliability histogram
reliability calibration / Brier score
performance by deviation quantile
performance by anchor spread quantile
performance by missing ratio
```

这能避免再次出现 ST-SSDL prototype collapse 或 DCD gate 固定在 0.5，却只看最终 MAE 的问题。

### 7.6 第一阶段 Go / No-Go 标准

满足以下条件再进入大规模预训练：

```text
1. 三个及以上随机种子上，selector+reliability 相对 mean 的平均 MAE 改善 >= 0.5%；
2. 95% block-bootstrap CI 不跨 0；
3. high-deviation 或 missing 场景改善 >= 1%；
4. effective candidate number >= 2，且没有重新坍缩到单一候选；
5. reliability 与“锚点是否真正改善预测”之间存在可测的校准关系。
```

如果只改善 `0.1%~0.2%`，或者不同种子反复换赢家，就不应马上扩展成预训练大模型，而应先修正监督目标。

---

## 8. 中期目标：STAnchor 预训练模块

建议暂定名称：

```text
STAnchor
Retrieval-Distilled Spatio-Temporal Anchor Adapter
检索蒸馏的时空历史锚点适配器
```

一句话定义：

```text
利用训练阶段的长历史检索构造分布式锚点教师，
将“选择什么历史、什么时候相信历史”蒸馏到轻量时空模块中，
在推理阶段通过紧凑锚点表和一次前向传播增强冻结或可训练的任意预测骨干。
```

### 8.1 不做完整大模型，而做大模型友好的插件

当前项目的数据规模不足以直接支撑一个可信的通用时空基础模型。与其堆叠大参数量，更合理的论文定位是：

```text
大模型负责通用短期预测；
STAnchor 负责目标城市的历史记忆、周期状态和可靠性校准。
```

这个定位有三个现实优势：

1. 可以接当前 DCD-ST、STID、Graph WaveNet、DCRNN、STAEformer。
2. 后续也可以接 OpenCity、UrbanFM 或时间序列基础模型。
3. 参数和训练成本可控，更符合当前项目“在 baseline 上做减法”的要求。

### 8.2 模块输入输出协议

建议核心模块接口固定为：

```python
output = stanchor(
    x_context,          # (B,T,N,C)
    past_anchors,       # (B,T,N,K,C)
    future_anchors,     # (B,H,N,K,C)
    time_features,      # (B,T+H,N,Ct) or broadcastable
    graph=None,         # optional
    node_descriptors=None,
)
```

标准输出：

```text
anchor_prior      : (B,H,N,Q)     未来锚点分位数或点先验
anchor_context    : (B,N,D)       可选的上下文 token
reliability       : (B,H,N,1)     动态可信度
candidate_weights : (B,N,K) 或 (B,H,N,K)
uncertainty       : (B,H,N,1)
```

不应把内部节点 embedding、某个特定 decoder 或 DCD-ST 的隐层维度写死在预训练核心中。

### 8.3 两种接入方式

#### 接口 A：输出级适配，作为主接口

```text
Y_base  = Backbone(X)
Y_prior = STAnchor(...).anchor_prior_median
r       = STAnchor(...).reliability
Y_final = Y_base + r * Project(Y_prior - Y_base)
```

优点：

```text
不需要访问 backbone 内部隐状态；
最接近真正的即插即用；
可以冻结 backbone，只微调 Project 和 reliability calibrator。
```

#### 接口 B：隐状态上下文增强，作为性能上界

```text
H_final = H_base + r * Project(anchor_context)
```

它通常更灵活，但需要知道骨干隐层维度，因此不应成为唯一接口。

### 8.4 通用核心与目标城市适配器解耦

参考时空预训练中的 factorized 思路，将模块拆为：

```text
Universal Temporal Anchor Encoder
  学习跨节点、跨城市共享的周期、趋势和候选匹配规律。

Target Spatial Adapter / Prompt
  用少量参数吸收目标城市的图结构、节点统计和局部空间规律。
```

微调时默认：

```text
冻结 universal encoder；
训练 target prompt + selector calibration + output projector；
必要时再开放最后一层 encoder 或使用 LoRA。
```

---

## 9. 预训练教师：只在离线阶段使用长历史检索

### 9.1 紧凑物理锚点库

推理阶段保留一个小型、可解释的目标域锚点表：

```text
AnchorTable[calendar_slot, node, candidate]
candidate = mean / median / q25 / q75 / recent
```

它是 O(1) 表查询，不需要在测试阶段扫描完整历史数据库。

### 9.2 离线 privileged teacher

预训练阶段可以访问训练段内的真实历史窗口及其未来：

```text
KnowledgeBase = {(X_i, Y_i)} from train only
```

对 query：

1. 先按 weekday、time slot 或相邻 calendar bucket 限制候选。
2. 使用上下文距离或冻结编码器相似度检索 top-k。
3. 对齐候选与 query 的局部水平和尺度。
4. 聚合候选未来，形成点预测或分位数教师。
5. 根据候选集中度、未来分歧和空间一致性计算 teacher confidence。

检索教师只用于产生预训练监督：

```text
推理阶段不保留大规模向量库，不做 top-k 搜索。
```

### 9.3 必须采用严格防泄漏协议

```text
所有候选 context 和 future 必须完整位于 train split；
排除 query 自身及重叠 future；
因果实验中，候选结束时间必须早于 query 开始时间；
归一化、calendar statistics 和图统计全部只用训练段；
跨城市测试时，测试城市标签只能来自明确声明的 few-shot 微调段。
```

---

## 10. 预训练任务与损失函数

### 10.1 长上下文掩码重建

借鉴 STEP、STD-MAE、GPT-ST：

```text
L_mask = masked reconstruction loss
```

使用长历史 patch，分别进行时间 patch mask 和节点 mask，让编码器学习周期上下文和空间异质性。

### 10.2 候选效用排序

训练标签不再是“哪个锚点离当前最近”，而是：

```text
utility_k = - MAE(A_future_k, Y)
```

或使用教师检索分布定义候选软标签：

```text
p_teacher(k) = softmax(utility_k / tau_teacher)
L_rank = KL(p_teacher || w_selector)
```

这一步直接监督 selector 学习“哪个锚点对未来有用”，避免当前 gate 只停在 0.5。

### 10.3 分布式未来先验蒸馏

由检索到的多个历史未来构造分位数：

```text
Q_teacher = {q10, q50, q90}
L_quantile = PinballLoss(Q_anchor, Y) + Huber(Q_anchor, Q_teacher)
```

分位数宽度可以直接表示历史分歧和时空海市蜃楼风险。

### 10.4 可靠性校准

预训练阶段使用 base-free 标签：

```text
r_pre = I[teacher_prior 比 persistence / fixed-mean 改善超过 margin]
```

接入具体骨干后，用少量目标域数据重新校准：

```text
r_ft = I[anchor_prior 比 frozen backbone 改善超过 margin]
```

损失可使用 BCE 或 Brier loss：

```text
L_rel = Brier(reliability, r_target)
```

这里与原普通 sigmoid gate 的本质区别是：

```text
reliability 有明确、可验证的监督语义，而不是只靠最终 MAE 间接学习。
```

### 10.5 总损失

第一版建议控制复杂度：

```text
L_pre = L_mask
      + lambda_rank * L_rank
      + lambda_q    * L_quantile
      + lambda_rel  * L_rel
```

空间一致性先作为 reliability 输入特征，不急于再增加独立损失。只有诊断证明需要时，再加入空间正则。

微调阶段：

```text
L_ft = L_forecast + beta_rel * L_rel_backbone
```

---

## 11. 与前人工作的关系

| 工作 | 可借鉴思想 | STAnchor 的使用方式与区别 |
|---|---|---|
| [STEP, KDD 2022](https://doi.org/10.1145/3534678.3539396) | 长历史 patch masked pre-training；冻结 TSFormer；投影后增强下游隐状态 | 借鉴长上下文编码和冻结迁移，但学习目标改为历史锚点选择、未来分布和可靠性 |
| [STD-MAE](https://arxiv.org/abs/2312.00516) | 空间/时间解耦掩码；可接多种下游结构 | 借鉴解耦 mask；STAnchor 额外学习候选历史的未来效用，而不只学习通用表征 |
| [GPT-ST](https://arxiv.org/abs/2311.04245) | MAE 预训练；节点/时间定制；预训练表示与原输入融合 | 借鉴身份条件化和下游融合；避免重型超图结构，保持模块轻量 |
| USTC, IJCAI 2025 | 跨城市预训练；冻结 encoder；目标城市 prompt；时间/频率分解 | 借鉴 universal encoder + target prompt；第一版不同时维护四套重型编码器 |
| [STRAP, NeurIPS 2025](https://arxiv.org/abs/2505.19547) | 显式时空 pattern library；OOD 检索；历史与当前信息融合 | 借鉴显式历史记忆；STAnchor 将大规模检索限制在训练阶段，推理只使用紧凑锚点表 |
| [TS-Memory, KDD 2026](https://arxiv.org/abs/2602.11550) | train-only kNN teacher；未来分位数；confidence-gated distillation；plug-and-play | 最接近的工作。STAnchor 必须突出图结构、物理 calendar anchors、动态节点/预测步可靠性和紧凑锚点表，否则创新性不足 |
| [UniST](https://arxiv.org/abs/2402.11838) | 多场景预训练；knowledge-guided prompt；zero/few-shot | 借鉴 prompt 适配；我们的核心不是统一完整预测模型，而是可接不同骨干的历史记忆模块 |
| [OpenCity](https://arxiv.org/abs/2408.10269) | 异构交通数据大规模预训练；零样本跨城市 | 作为后期 foundation backbone 和跨城市基线，不在第一版复刻完整模型 |
| [FactoST-v2, 2026 preprint](https://arxiv.org/abs/2601.12083) | 通用时间学习与目标域空间适配解耦 | 直接支持 STAnchor 的 universal temporal core + spatial prompt 设计 |
| [UrbanFM, 2026 preprint](https://arxiv.org/abs/2602.20677) | 多城市数据规模化；统一 grid/sensor 单元；零样本泛化 | 用于规划后期 scaling 实验；当前项目不应在数据规模不足时模仿其“大模型”标签 |
| [ImPreSTDG, Scientific Reports 2025](https://doi.org/10.1038/s41598-025-11375-2) | mask recovery；冻结预训练参数；替换预测头 | 可作为缺失场景和冻结迁移参考，DDPM/Mamba 不是本项目第一版必要组件 |

### 11.1 最需要警惕的创新重叠

TS-Memory 已经提出：

```text
离线检索教师 -> 分位数监督 -> confidence-gated distillation -> 推理阶段无检索插件
```

因此我们不能只把同一流程换到交通数据上。论文创新必须落在以下时空专属部分：

1. `calendar-slot × node × regime` 的可追溯物理锚点库。
2. 过去候选选择与 future-aligned anchor prior 的联合建模。
3. 图邻域一致性驱动的、每节点/每预测步动态可靠性。
4. universal temporal core 与 target spatial prompt 的跨城市迁移。
5. 面向 STGNN 和 ST foundation model 的统一输出级适配协议。

少于其中前三项，工作很容易被评价为 TS-Memory 的交通版本。

---

## 12. 与 ST-SSDL 的核心区别

| ST-SSDL | STAnchor |
|---|---|
| 单一 historical average | 多个物理统计锚点和检索历史候选 |
| latent learnable prototypes | 可追溯到真实节点、calendar slot 和历史窗口的锚点 |
| prototype assignment 出现 collapse | 显式候选使用率、熵和有效候选数诊断 |
| deviation loss 间接约束距离 | 直接用未来效用监督候选排序 |
| 不判断历史是否有害 | 显式输出动态 reliability |
| 与当前预测模型耦合 | 输出级接口可接冻结或可训练骨干 |

最合适的论文动机不是：

```text
我们设计了一个更复杂的 gate。
```

而是：

```text
短窗口预测器缺少长历史上下文，但直接注入历史会受到时空海市蜃楼影响。
我们将可解释的历史检索压缩为分布式锚点记忆，并学习何时选择、何时拒绝历史，
从而用一个小型预训练插件增强不同复杂度的时空预测器。
```

---

## 13. 完整实验路线

### 阶段 A：多锚点选择可行性

目标：证明动态选择比任何固定统计量更好。

数据：

```text
METR-LA
```

骨干：

```text
DCD-ST / DHA-DCD-ST
```

产出：

```text
multi-anchor data contract
hard selector baseline
learned selector
selector + supervised reliability
3~5 seeds and diagnostics
```

### 阶段 B：STAnchor 离线教师与预训练

优先使用相同物理量的速度数据：

```text
METR-LA + PEMS-BAY
```

原因：两个数据集都是速度，跨城市时不会先被速度/流量单位差异干扰。

任务：

```text
masked reconstruction
candidate utility ranking
retrieval quantile distillation
reliability calibration
```

### 阶段 C：即插即用验证

至少选择三类骨干：

```text
简单骨干：STID 或 Linear/MLP predictor
经典骨干：DCRNN 或 Graph WaveNet
强骨干：STAEformer
当前工作：DCD-ST
```

对每个骨干比较：

```text
backbone only
backbone + random initialized adapter
backbone + pretrained STAnchor frozen
backbone + pretrained STAnchor few-shot tuning
full fine-tuning upper bound
```

### 阶段 D：跨城市和少样本

协议：

```text
METR-LA -> PEMS-BAY
PEMS-BAY -> METR-LA
```

微调预算：

```text
zero-shot
1 day
3 days
7 days
full data
```

第二轮再加入 PEMS03/04/07/08 流量数据。此时必须使用 instance normalization、RevIN 或显式变量类型 embedding 处理不同量纲。

### 阶段 E：鲁棒性与 OOD

必须评估：

```text
随机缺失 10% / 30% / 50%
连续块缺失
高 deviation 样本
高 anchor-spread 样本
工作日 -> 周末或时间段分布漂移
节点缺失 / 未见节点（条件允许时）
```

核心问题不是平均 MAE 是否下降一点，而是：

```text
当历史与当前冲突时，reliability 是否真的会降低；
当历史模式稳定时，模块是否能提供更明显收益。
```

### 阶段 F：效率与大模型适配

报告：

```text
预训练参数量
目标域可训练参数量
FLOPs
单 batch 延迟
锚点表大小
是否需要测试时检索
```

最后再选择 OpenCity 或其他可获得的时空基础模型，验证输出级 STAnchor 是否仍可工作。这个实验用于支撑“foundation-model-compatible”，不应成为第一阶段阻塞项。

---

## 14. 推荐代码组织

下一版建议单独建立：

```text
STAnchor/
  stanchor_module.py       # 对外唯一主模块与 AnchorOutput
  candidate_builder.py     # 紧凑锚点表和 past/future candidate lookup
  retrieval_teacher.py     # train-only 离线教师，不进入部署路径
  pretrain.py              # 预训练入口
  finetune.py              # 目标域 prompt / fusion 微调
  evaluate.py              # in-domain、cross-city、missing、OOD
  configs/
  docs/
```

`stanchor_module.py` 的公共 API 应独立于当前 `src.train` 和 DCD-ST，之后通过 wrapper 接入现有训练框架。检查点至少保存：

```text
model_state
normalization specification
candidate order
feature schema
pretraining datasets
time resolution
module version
```

否则很难真正做到跨项目即插即用。

---

## 15. 近期执行顺序

接下来按以下顺序推进：

1. 把五套锚点整理为同一样本中的 `K=5` 候选张量，先不修改预测骨干。
2. 实现可复现的 hard-similarity selector 作为无参数基线，并把本报告中的 `3.7027` 结果固化成诊断脚本。
3. 实现最小 learned selector，只训练候选权重，暂时不加预训练。
4. 加入有明确标签的 reliability head，验证它能否识别 anchor-helpful / anchor-harmful 样本。
5. 对 mean、q25、hard selector、learned selector、selector+reliability 跑 3 个种子。
6. 达到 Go 标准后，再实现 retrieval teacher 和 masked pre-training。
7. 先做 METR-LA / PEMS-BAY 跨城市，再扩展到流量数据和基础模型。

当前最重要的一步不是继续堆模块，而是完成以下因果链的第一环：

```text
多锚点确实有互补信息
-> selector 能从过去上下文预测未来有用锚点
-> reliability 能拒绝误导历史
-> 预训练让这种能力跨数据集、跨骨干迁移
```

只有四步都被实验支持，这个方向才能形成一篇逻辑完整、贴合时空预训练与模型适配热点的工作。
