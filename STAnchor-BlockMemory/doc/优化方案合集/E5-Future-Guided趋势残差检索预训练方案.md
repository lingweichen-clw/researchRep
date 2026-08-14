# E5-Future-Guided 趋势残差检索预训练方案

> 状态（2026-08-03）：T0 已完成。原始 `trend + local scale` 假设已被 validation 证伪；T1-A/T1-B 的工程接口、配置和测试已接入，当前下一步是执行 T1-A 的正式 seed 42 实验。只有 T1-A 同时通过 METR-LA、PEMS-BAY 和无 confidence 下游门槛，才执行 T1-B 的 FutureIncrement relation。仍不修改 GCRU/ST-SSDL 下游骨干。
>
> 本方案的核心判断：**future-guided 是 STAnchor-BlockMemory 的主创新方向；OffsetDecay 和 FutureIncrement 都必须服务于“让历史 key 学到可部署的未来关系”这一主线。线性趋势外推、local scale transfer 和提前接入 confidence 均不进入当前模型。**
>
> 完整定义、代码改动清单、逐阶段命令、指标和 Keep/Remove/Stop 门槛见 `doc/实验步骤合集/E5-OffsetDecay与FutureIncrement两阶段预训练实验方案.md`。若本文旧的原始假设记录与该执行方案冲突，以 2026-08-03 的两阶段执行方案为准。

## 当前正式方案摘要（2026-08-03）

### 实验代号与未来信息边界

- `E3`：当前 relation 预训练基线。训练期用按历史 mean/std 归一化的真实 future 定义 teacher relation；推理 selector 只读取历史。
- `T0`：不训练新模型，冻结 E3 selector、候选、权重和历史 Bank，只比较不同 future payload 的诊断阶段。
- `T1-A`：把 E3 teacher 改成与 OffsetDecay 推理严格对齐的 future signature，并保持 encoder、student distribution、loss weight 和训练预算不变。
- `T1-B`：只有 T1-A 通过后，才在同一 teacher 中加入真实 FutureIncrement relation；它不增加 future encoder，也不改变推理 payload。
- `Confidence`：根据检索特征学习的下游乘法门控。当前阶段完全禁用，有效 memory 位置固定为 1、无效位置为 0，只训练 horizon fusion，以免校准收益掩盖 selector 失败。

真实 future 只允许出现在 source-train teacher、已经发生的 training-history Bank payload 和离线指标中。query inference、candidate filtering 和 query level 估计都禁止读取 query future。

### T1-A：部署对齐的 OffsetDecay relation

令 \(\alpha_{i,n,c}\) 为事件 \(i\) 最近 12 步可见历史的末端 level，预测长度为 \(H=12\)，固定衰减为：

\[
\lambda_h=1-\frac{h-1}{H-1},\qquad h=1,\ldots,H.
\]

`OffsetDecay` 表示近端充分对齐 query level、远端逐步退回历史 RawFuture。其推理 memory 为：

\[
\widehat Y^{\mathrm{OD}}_{q,h,n,c}
=\lambda_h\alpha_{q,n,c}
+\sum_jw_{qjn}\left(Y_{j,h,n,c}-\lambda_h\alpha_{j,n,c}\right).
\]

因此 T1-A teacher 直接使用与被加权检索对象一致的 signature：

\[
S^{\mathrm{OD}}_{i,h,n,c}
=Y_{i,h,n,c}-\lambda_h\alpha_{i,n,c}.
\]

先计算 \(S^{\mathrm{OD}}\) 的节点级 pairwise masked MAE \(d^{\mathrm{OD}}_{ij,n}\)。由于它处在原始速度单位，而 E3 teacher temperature 面向归一化 future，必须使用无参数的 `AnchorMeanDistanceNormalization`：对每个样本-节点 anchor \((i,n)\)，将候选距离除以该 anchor 的有效候选平均距离：

\[
\widetilde d^{\mathrm{OD}}_{ij,n}
=\frac{d^{\mathrm{OD}}_{ij,n}}
{\frac{1}{|\mathcal C_{i,n}|}\sum_{k\in\mathcal C_{i,n}}d^{\mathrm{OD}}_{ik,n}+\epsilon}.
\]

它只校准 teacher 数值尺度，保持同一 anchor 内候选排序，不增加参数，也不进入推理。teacher distribution 为 \(\operatorname{Softmax}_j(-\widetilde d^{\mathrm{OD}}_{ij,n}/\tau_T)\)。

### T1-B：真实 FutureIncrement relation

`FutureIncrement` 表示真实 future 的逐步变化，不是从历史拟合斜率：

\[
G_{i,1,n,c}=Y_{i,1,n,c}-\alpha_{i,n,c},
\]

\[
G_{i,h,n,c}=Y_{i,h,n,c}-Y_{i,h-1,n,c},qquad h=2,\ldots,H.
\]

对其 pairwise masked MAE \(d^{\mathrm{inc}}\) 独立执行相同的 anchor-mean normalization，得到 \(\widetilde d^{\mathrm{inc}}\)。T1-B 固定使用：

\[
\widetilde d^{\mathrm{OD+Inc}}
=0.5\widetilde d^{\mathrm{OD}}
+0.5\widetilde d^{\mathrm{inc}}.
\]

`0.5` 不通过 validation 搜索。FutureIncrement 只在训练 teacher 中使用；推理仍然只用 OffsetDecay memory。

### 两阶段决策

1. 先重新建立 `level_weight=0` 的 E3 检索对照，隔离 learned key 的贡献。
2. T1-A 必须在 METR-LA 与 PEMS-BAY validation 的 OffsetDecay retrieval MAE 都优于 E3，且在两个数据集的无 confidence 下游同时优于 E3、random 和 base-only，才允许保留。
3. T1-A 任一数据集失败就停止 E5，不接 confidence，不增加 future encoder。
4. T1-B 必须在两个数据集 retrieval MAE 都相对 T1-A 再改善至少 `0.5%`，并在两个无 confidence 下游 final MAE 都相对 T1-A 改善至少 `0.3%`，否则删除 FutureIncrement、保留 T1-A。
5. 只有最终单 seed 版本通过后才运行预训练 seed 与下游 seed 稳定性；全部选择冻结前禁止读取 test。

## 0. T0 实证修订（2026-08-02）

完整诊断见 `doc/诊断报告合集/E5趋势残差T0诊断报告.md`。相同 learned E3 排序和权重下：

| Dataset | Raw MAE | Offset MAE | OffsetDecay MAE | Trend MAE |
|---|---:|---:|---:|---:|
| METR-LA | 3.8027 | 3.8940 | **3.6231** | 5.3743 |
| PEMS-BAY | 2.2028 | 1.9782 | **1.9305** | 2.7904 |

相对 raw，OffsetDecay 分别改善 `4.72%` 和 `12.36%`，且不使用 confidence。Trend 分别恶化 `41.33%` 和 `26.67%`；local scale transfer 曾因最大 `query/candidate scale ratio` 约为 `4151` 而产生极端值。

因此本文档后续出现的 trend/scale 公式保留为**被检验的原始假设和失败消融定义**，不再是实现目标。当前只先实现 T1-A 的 \(S^{\mathrm{OD}}=Y-\lambda\alpha\) relation；T1-A 过门后才允许实现 T1-B 的 FutureIncrement relation。

## 1. 研究定位

### 1.1 主创新与辅助机制

当前工作的真正可发表主线不应表述为“加入趋势分解”，而应表述为：

> 从未来轨迹中学习检索关系，并将历史 Bank 的 future 通过节点/实例级、随预测距离衰减的 query-level 对齐映射到当前 query，使预训练关系直接服务于跨域历史检索和未来预测修正。

因此各机制的职责如下：

| 机制 | 科学职责 | 是否独立创新 |
|---|---|---|
| Future-guided relation | 用未来轨迹定义哪些历史事件应该相互接近，是预训练的监督来源 | 是，主创新候选 |
| Horizon-decayed level residual | 近端对齐 query level，远端逐步退回 raw future | 否，服务于主创新；T0 已获得双数据集证据 |
| Linear trend / local scale | 原计划去除趋势和尺度差异 | 否；T0 已证伪，作为失败消融保留 |
| Memory retrieval | 将历史事件对应的未来残差作为可复用预测证据 | 是主创新的应用机制，不单独声称新颖 |
| Confidence fusion | 判断 memory correction 是否可信，避免错误检索伤害下游 | 校准机制，不是主创新 |

残差机制只有在下列证据成立时才保留为论文贡献的一部分：

1. 在同一候选集合、同一随机种子和同一融合器下，新的 future relation 比当前 E3 的原始 future relation 更能改善检索质量；
2. 该改善在 METR-LA 同域和 METR-LA 到 PEMS-BAY 跨域都出现，不能只在一个数据集上有效；
3. 改善不能只来自 confidence，而要先在无 confidence 的 memory 质量和最终预测上成立。

T0 已满足 OffsetDecay payload 的第 2、3 条，但尚未验证新的预训练 teacher，因此不能提前声称 E5 relation 优于 E3。Trend 和 local scale 已删除。

### 1.2 当前 E3 失败的可检验解释

已观察到的现象是：

- METR-LA 同域中，E3 source encoder 对 random 有小幅优势；
- PEMS-BAY 中，source-frozen encoder 与 target-random 基本持平甚至略差；
- confidence 能够从错误 memory 中挑出相对有用的样本，但这不能证明 encoder 学到了可迁移的未来动力学；
- 当前 encoder 主要影响有限的同日候选排序，event Top-R 候选的作用很弱；
- memory 最终提供的是少量 future 样本，若样本本身与当前 base error 方向无关，confidence 只能拒绝它，不能创造有效修正方向。

因此，本方案检验的不是“是否再加一个趋势网络”，而是下面这个更窄的问题：

> **当前 E3 是否因为用 level-sensitive 的未来表示定义关系，导致检索到的是数值水平相似而不是动态变化相似的事件？**

## 2. 原始假设：固定统计量 + Future-Guided 趋势残差关系

第一版不引入 FFT、DWT、新 Transformer、动态图或第二个预测分支。只改变 future relation 的目标表示和 memory payload，保留现有 E2/E3 encoder 作为可比基线。

### 2.1 输入与符号

对一个训练事件，令：

- 历史检索上下文为
  \[
  X_i\in\mathbb{R}^{T_r\times N\times C},
  \]
  其中 \(T_r=288\) 表示一天历史窗口，\(N\) 是节点数，\(C\) 是物理变量通道数；
- 预测未来为
  \[
  Y_i\in\mathbb{R}^{H\times N\times C},
  \]
  其中当前实验 \(H=12\)；
- \(x_{i,T_r,n,c}\) 是事件 \(i\) 在历史窗口末端的 level；
- \(\mathcal{O}_{i,t,n,c}\in\{0,1\}\) 是有效观测掩码；缺失值不能参与统计量和 teacher 计算；
- 未来 \(Y_i\) 只允许在预训练 teacher 和历史 bank 构造时使用，推理 query 不可访问未来。

所有公式默认逐节点、逐通道计算。空间聚合继续由现有 STAnchor encoder 完成，不在本方案第一阶段另造图模块。

### 2.2 局部 level、趋势和尺度

取历史末端最近 \(T_t\) 个时间点估计局部趋势，建议初始值 \(T_t=12\)，与预测长度一致。

令局部时间坐标为 \(\tau=0,1,\ldots,T_t-1\)，有效历史序列为 \(z_{i,\tau,n,c}\)。使用带掩码的最小二乘直线估计趋势斜率：

\[
\beta_{i,n,c}
=
\frac{\sum_{\tau}\mathcal{O}_{i,\tau,n,c}(\tau-\bar\tau_i)
 (z_{i,\tau,n,c}-\bar z_{i,n,c})}
 {\sum_{\tau}\mathcal{O}_{i,\tau,n,c}(\tau-\bar\tau_i)^2+\epsilon}.
\]

其中 \(\bar z_{i,n,c}\) 和 \(\bar\tau_i\) 是有效点的均值。令末端 level 为

\[
\alpha_{i,n,c}=x_{i,T_r,n,c},
\]

令局部变化尺度为一阶差分的 RMS：

\[
s_{i,n,c}
=
\sqrt{\frac{1}{M_i-1}
\sum_{t}\mathcal{O}_{i,t,n,c}\mathcal{O}_{i,t-1,n,c}
\left(x_{i,t,n,c}-x_{i,t-1,n,c}\right)^2+\epsilon}.
\]

第一版使用 RMS 而不是可学习尺度，避免把域偏移重新隐藏进一个复杂归一化网络。若异常值明显，再以 MAD 作为单变量替代，不同时加入多种 robust scale。

### 2.3 历史残差坐标

历史序列在局部趋势坐标中的残差为：

\[
r^X_{i,t,n,c}
=
\frac{x_{i,t,n,c}-\left[\alpha_{i,n,c}+(t-T_r)\beta_{i,n,c}\right]}
{s_{i,n,c}+\epsilon}.
\]

它表达“相对于当前事件局部 level 和趋势的偏离”，而不是物理变量的绝对值。该表示只用于 query/key 的动态关系学习；原始历史输入仍按现有 E3 流程进入 encoder，第一阶段不改变下游输入分布。

### 2.4 未来残差轨迹

未来第 \(h\) 个时间点的 future residual 为：

\[
u^Y_{i,h,n,c}
=
\frac{y_{i,h,n,c}-\left[\alpha_{i,n,c}+(h+1)\beta_{i,n,c}\right]}
{s_{i,n,c}+\epsilon},
\quad h=0,\ldots,H-1.
\]

这里的 \(u^Y_i\in\mathbb{R}^{H\times N\times C}\) 是 future-guided teacher 的核心未来对象：

- \(\alpha\) 去除局部 level 差异；
- \(\beta\) 去除短期线性趋势；
- \(s\) 去除局部变化幅度差异；
- 剩余部分保留突变、拥堵释放、传播延迟等动态形状。

如果历史末端趋势估计不稳定，则必须退回 offset-only 版本：

\[
u^{Y,\mathrm{offset}}_{i,h,n,c}
=
\frac{y_{i,h,n,c}-\alpha_{i,n,c}}
{s_{i,n,c}+\epsilon}.
\]

这两个版本必须作为显式消融，不能把趋势斜率和 level offset 一起加入后再声称“趋势模块有效”。

## 3. Future-Guided 关系目标

### 3.1 未来 teacher 分布

对 query 事件 \(i\) 和候选事件 \(j\)，只在满足时间因果、有效掩码和候选集合约束的候选中计算残差 future 距离：

\[
d_Y(i,j)
=
\frac{1}{|\Omega_{ij}|}
\sum_{(h,n,c)\in\Omega_{ij}}
\rho\left(u^Y_{i,h,n,c}-u^Y_{j,h,n,c}\right),
\]

其中 \(\Omega_{ij}\) 是两事件 future 都有效的位置，\(\rho\) 第一版使用平方误差；若异常值导致 teacher 过度尖锐，再单独比较 Huber 距离。

由此得到未来相似性 teacher：

\[
q_{ij}
=
\frac{\exp(-d_Y(i,j)/\tau_Y)}
{\sum_{k\in\mathcal{C}_i}\exp(-d_Y(i,k)/\tau_Y)}.
\]

\(q_{ij}\) 只在训练历史事件上构造。推理阶段不能使用 query 的真实未来来计算它。

### 3.2 encoder 检索分布

令现有 STAnchor encoder 输出 query/key：

\[
h_i=E_\theta(X_i),\quad h_j=E_\theta(X_j),
\]

其中 \(h_i,h_j\in\mathbb{R}^{d}\)。第一版使用 cosine similarity：

\[
p_{ij}
=
\frac{\exp(\operatorname{cos}(h_i,h_j)/\tau_X)}
{\sum_{k\in\mathcal{C}_i}\exp(\operatorname{cos}(h_i,h_k)/\tau_X)}.
\]

Future-guided trend-residual loss 为：

\[
\mathcal{L}_{\mathrm{FTR}}
=
\frac{1}{|\mathcal{B}|}
\sum_{i\in\mathcal{B}}
\operatorname{CE}(q_i,p_i).
\]

可选的对称形式为 \(\frac{1}{2}[\operatorname{KL}(q_i\|p_i)+\operatorname{KL}(p_i\|q_i)]\)，但第一阶段不同时比较 CE、KL、InfoNCE 等多个目标，以免实验失去归因。

### 3.3 原始关系目标的含义

当前 E3 的关系是“原始或归一化 future 是否接近”。E5 的关系变为：

> 两个历史事件是否会产生相似的、相对于各自局部趋势和尺度的未来变化形状。

T0 已证明该 trend/scale 版本不成立。当前 T1-A 将关系目标收敛为：两个历史事件的 \(Y-\lambda\alpha\) 是否相似。该定义在近端比较 level-offset future，在远端恢复 RawFuture relation，并与 OffsetDecay 推理的加权对象严格一致。T1-B 才进一步比较真实 FutureIncrement。

## 4. 检索与 memory payload

### 4.1 T0 保留的 OffsetDecay payload

Bank 继续保存 raw future，不修改 schema。对 query \(q\) 和检索事件 \(j\)，先用各自可见历史末端 level \(\alpha_q,\alpha_j\) 得到完整对齐版本：

\[
\widehat Y^{\mathrm{offset}}_{q,h,n,c}
=
\alpha_{q,n,c}
+\sum_{j\in\mathcal K_q}w_{qj}
\left(Y_{j,h,n,c}-\alpha_{j,n,c}\right).
\]

Raw retrieval 为：

\[
\widehat Y^{\mathrm{raw}}_{q,h,n,c}
=
\sum_{j\in\mathcal K_q}w_{qj}Y_{j,h,n,c}.
\]

最终使用零参数 horizon decay：

\[
\widehat Y^{\mathrm{OD}}_{q,h,n,c}
=
\widehat Y^{\mathrm{raw}}_{q,h,n,c}
+\lambda_h
\left(
\widehat Y^{\mathrm{offset}}_{q,h,n,c}
-\widehat Y^{\mathrm{raw}}_{q,h,n,c}
\right),
\qquad
\lambda_h=1-\frac{h-1}{H-1}.
\]

其中 \(\sum_jw_{qj}=1\)。这一步只使用 query history 与 causal Bank：近端充分利用当前 level，远端逐步退回 learned raw future。局部斜率和 local scale 不再进入 payload。

跨数据集时只迁移 encoder 参数 \(\theta\)，不把 METR-LA future 搬到 PEMS-BAY。目标域 Bank 仍只由目标域 training history 构造。

### 4.2 检索评分的最小路线

按以下顺序进行：

1. **Fixed past-residual Pearson**：推理时只对 query 和 candidate 的历史残差 \(r^X_q,r^X_j\) 计算 Pearson/余弦相似度，作为不依赖 encoder 的可部署基线；
2. **Future-oracle residual ranking**：只在离线诊断中用 \(u^Y_q,u^Y_j\) 计算上限，严禁作为推理方法或正式模型结果；
3. **Current encoder + OffsetDecay teacher**：只训练现有 key encoder，使历史 key 分布逼近经过 anchor-mean 尺度校准的 \(Y-\lambda\alpha\) future teacher；
4. **Learned key + OffsetDecay payload**：只有当第 3 步在 source 和 target 都有信号，才进入正式 memory 检索。

Fixed past-residual 相似度定义为：

\[
s_X^{\mathrm{fixed}}(q,j)
=
\operatorname{cos}\left(
\operatorname{vec}(r^X_q),
\operatorname{vec}(r^X_j)
\right),
\]

其中只使用共同有效的历史位置。它与 future-oracle 必须在结果表中分栏，避免把未来信息诊断误写成可部署检索性能。

不得在第一步同时加入 weekday、slot、level_weight、frequency branch 和 learned gate。每个新增因素都必须有独立的候选集合和消融。

### 4.3 confidence 的位置

仍使用现有融合形式：

\[
\hat y=b+q(m-b),\quad q\in[0,1],
\]

其中 \(b\) 是 base prediction，\(m=\hat y^{\mathrm{mem}})。confidence 只在 memory 方向与 base 误差存在正相关时放大它；它不能修复 memory payload 本身没有有效动态方向的问题。

因此实验顺序必须是：

1. 先比较不带 confidence 的 memory 质量；
2. 再固定检索器，比较 confidence 的增量；
3. 如果只有第 2 步有效而第 1 步无效，不能把 confidence 的收益归因于 future-guided pretraining。

## 5. 方案版本与消融

| 版本 | future teacher | memory payload | 目的 |
|---|---|---|---|
| E3-Base | 当前 E3 future relation | 当前 raw/normalized future | 历史基线 |
| E5-Offset | 去末端 level 的 future residual | 完整 horizon offset residual | T0 定位版本，跨域有效但同域 aggregate 失败 |
| **E5A-OffsetDecay** | \(Y-\lambda\alpha\) relation + anchor-mean distance normalization | 近端 level 对齐、远端退回 raw | **T1-A，当前唯一先执行版本** |
| **E5B-FutureIncrement** | 0.5 normalized OD relation + 0.5 normalized future increment relation | 与 T1-A 相同的 OffsetDecay payload | **T1-B，仅在 T1-A 通过后执行** |
| E5-Trend | 去 level + 局部线性趋势 | trend residual | T0 双数据集失败，停止 |
| E5-Scale | 局部变化尺度归一化 | scale-transferred residual | 极端放大，停止 |
| E5-Fixed | 无 encoder，past-residual Pearson | offset/trend residual | T0 定位基线，双数据集失败 |
| E5-Conf | 最终 E5A/E5B + 原 confidence | OffsetDecay memory | 两阶段和多 seed 均成功后才测 confidence 增量 |

必做的对照关系是：

\[
\text{E3-Base}
\rightarrow
\text{E5A-OffsetDecay}
\rightarrow
\text{E5B-FutureIncrement（条件）}.
\]

E5-Offset、E5-Trend、E5-Scale 和 E5-Fixed 已在 T0 完成定位，不再训练对应新模型。T0 中 fixed offset/trend 均劣于 learned raw，说明手工历史 Pearson 不能替代 learned relation；T1-A 仍需证明新的部署对齐 teacher 相对 E3 teacher 的增量。

## 6. 分阶段实验计划

### T0：已完成的表示和检索上限诊断

T0 不训练模型，固定 E3 history/key 排序、候选集合、learned weights 和历史 Bank，比较 RawFuture、Offset、OffsetDecay、Trend、past-residual fixed ranking 与 future-oracle ranking。future-oracle 读取 query future，只是离线上限，不属于可部署方法。

**T0 决策**：OffsetDecay 在 METR-LA/PEMS-BAY 分别改善 MAE `4.72%/12.36%`，保留；Trend 分别恶化 `41.33%/26.67%`，删除；local scale transfer 因最大比例约 `4151` 的无界放大而删除；fixed past-residual ranking 不能替代 learned selector。

### Stage 0：E3 Level-0 固定对照

`Level-0` 表示检索 node reranking 中手工 level similarity 的权重 `level_weight=0`，使排序只由 learned key 决定。T0 旧表使用 `level_weight=0.25`，因此 T1 前必须在 METR-LA 与 PEMS-BAY validation 重跑 E3 Level-0 OffsetDecay 诊断。该步骤不训练新模型，也不改变 Bank。

### T1-A：部署对齐 teacher

只训练 METR-LA source encoder。两个 checkpoint 候选分别是 validation total loss 最低的 `pretrain_best.pt` 和 relation loss 最低的 `pretrain_best_relation.pt`；只能用 METR-LA validation 的 OffsetDecay retrieval MAE 选择一次，之后冻结 checkpoint kind。PEMS-BAY 只构建 target-local training-history Bank，不能参与 source checkpoint 选择。

T1-A retrieval 必须全部满足：

- 相对 E3 Level-0，METR-LA 和 PEMS-BAY 的 OffsetDecay MAE 都至少改善 `0.5%`；
- 至少一个数据集改善 `1.0%`；
- 两个数据集 RMSE 均不恶化超过 `0.5%`；
- 每个数据集至少 8/12 个 horizon 的 MAE 不劣于 E3；
- common coverage 下降不超过 `0.2` 个百分点。

通过 retrieval 门后，固定 seed 42 比较 `base_only`、E3 OffsetDecay、E5A OffsetDecay 和 random OffsetDecay。`base_only` 不读取 memory；其余三组使用相同的无 confidence `learned_topk_offset_decay_horizon` 融合。

T1-A downstream 必须全部满足：

- 两个数据集的 E5A memory MAE 都低于 E3；
- 两个数据集 final validation MAE 都相对 E3 改善至少 `0.3%`；
- 两个数据集 final MAE 都低于 random 和 base-only；
- 15、30、60 分钟中至少两个位置不劣于 E3；
- confidence loss 和 confidence trainable parameters 都为 0；
- 相对 E3 不增加推理参数，单 batch 检索延迟增加不超过 `5%`。

任一 retrieval 或 downstream 条件失败均停止 E5，不运行 T1-B，不用 confidence 补救。

### T1-B：FutureIncrement 条件增量

只有 T1-A 全部门槛通过后，才将 teacher 从 \(\widetilde d^{\mathrm{OD}}\) 改为 \(0.5\widetilde d^{\mathrm{OD}}+0.5\widetilde d^{\mathrm{inc}}\)。encoder、candidate protocol、Bank schema、OffsetDecay payload、downstream backbone、训练预算和 seed 42 均保持不变。

T1-B retrieval 必须在两个数据集都相对 T1-A 再改善 OffsetDecay MAE 至少 `0.5%`；H6/H12 至少一个改善、另一个恶化不超过 `0.3%`；整体 RMSE 恶化不超过 `0.3%`；coverage 下降不超过 `0.2` 个百分点。

通过后才训练无 confidence downstream。T1-B 只有在两个数据集 final validation MAE 都相对 T1-A 改善至少 `0.3%`，且 15、30、60 分钟中至少两个位置不劣于 T1-A 时保留。否则删除 FutureIncrement，最终保留 T1-A；不搜索多个 increment 权重补救。

### 多随机种子、test 与 confidence

只有最终 T1-A 或 T1-B 通过 seed 42 门槛后，才补预训练 seed `2024/2025`，并固定最终 seed 42 encoder/Bank 运行下游 seed `42/2024/2025`。checkpoint kind、Level-0、OffsetDecay 公式、increment 权重和全部门槛必须先冻结。

`Test` 是全部模型选择结束后的最终留出数据，只做一次性报告，不得反向选择 teacher、checkpoint、seed 或 fusion。Confidence 只有最终无 confidence 多 seed 结论成立后才允许作为独立后置增量；目标域适配当前不在 E5 主实验中。

## 7. 复杂度预算与明确不做的事情

E5 第一版的复杂度预算只有：

- 现有 STAnchor encoder；
- 固定的 endpoint level 统计量和零参数 horizon decay；
- 一个 future-guided relation loss 和无参数 anchor-mean distance normalization；
- 仅在 T1-A 通过后加入的 FutureIncrement teacher distance；
- residual future payload 的重构。

第一版明确不加入：

- FFT/DWT 双分支；
- 第二个时空 Transformer；
- 可学习频率字典；
- 复杂动态图传播；
- 多个 confidence/gate 串联；
- 目标域无标签伪标签迭代；
- 为了追平 ST-SSDL 数字而直接复现 GCRU。

这些机制只有在一个更小的实验明确暴露出对应缺失能力时才可以打开。每打开一个机制，必须删除或冻结一个已有机制，保持可归因性。

## 8. 与相关工作的边界

以下工作已经覆盖了部分相邻思想，因此不能把这些单独包装成创新：

1. RAFT 已使用 offset removal、趋势相关性和多尺度检索；
2. SARAF 已指出“历史相似不保证未来相似”，并使用 dataset-level stationarity、时间对齐、diversity selection 和自适应聚合增强检索；
3. RAST 已讨论面向时空预测的 retrieval-augmented framework；
4. PIR 已将预测后的 identification/revision 与 uncertainty gate 结合；
5. STWave+ 使用小波和趋势/事件解耦；
6. TimeMixer 展示了轻量级多尺度 trend/seasonal mixing；
7. FEPCross 讨论了频率增强预训练和跨城市迁移。

因此 E5 只能检验下面这个更窄的组合，T1 成功前不能把它写成既成贡献：

> 在同物理变量的跨域交通预测中，用未来轨迹监督历史检索关系，并将 causal target-local Bank 中的 future memory 通过 node/instance-level、horizon-decayed query-level alignment 映射到当前 query，使检索表示在不依赖 confidence 的条件下产生预测收益。

与 SARAF 的预期区别必须明确：SARAF 根据 dataset-level stationarity 调节候选多样性和聚合；E5 研究的是用 **训练期 future relation** 教会历史 encoder 识别相似未来动力学，并在 **node/instance-level residual coordinate** 中迁移 future payload。若完整阅读 SARAF 后发现其正文或补充材料已经覆盖这一组合，则 E5 的创新表述必须继续收窄，不能依赖标题差异规避重叠。

T0 已证明 payload 机制成立，但尚未证明新的预训练目标成立。Linear trend 与 local scale 只能作为失败消融记录；论文主线不再使用“趋势分解”作为贡献名称。

## 9. 已准备的论文、源码与核对任务

### 第一优先级：实现 T1 时必须核对

1. **RAFT**：论文和官方实现。重点核对 offset removal、Pearson trend similarity、多尺度下采样和 retrieval future reconstruction。论文：[ICML 2025 RAFT](https://proceedings.mlr.press/v267/han25d.html)，源码：[archon159/RAFT](https://github.com/archon159/RAFT)。
2. **STWave+**：论文和源码。只借鉴其趋势/事件定义和数据处理方式，不直接移植完整双分支。论文作者页：[STWave+ PDF](https://zheng-kai.com/paper/tkde_fang_2023.pdf)，代码仓库：[LMissher/STWave](https://github.com/LMissher/STWave)。
3. **TimeMixer**：论文和源码。用于核对最小化的多尺度 trend/seasonal mixing 是否有必要，不作为 E5 第一版结构。论文：[ICLR 2024 OpenReview](https://openreview.net/forum?id=7oLshfEIC2)，源码：[kwuking/TimeMixer](https://github.com/kwuking/TimeMixer)。
4. **SARAF**：论文和官方源码。该工作与 E5 的非平稳检索问题高度相关，必须在动手实现 T1 前完整核对方法与消融。论文：[KDD 2026 / arXiv](https://arxiv.org/abs/2606.04135)，源码：[ShiqiaoZhou/SARAF](https://github.com/ShiqiaoZhou/SARAF)。

### 第二优先级：创新边界和后续适配前拿到

5. **RAST**：论文、补充材料和官方代码。用于检查 retrieval store、query generator 和目标域迁移是否与 E5 重叠。论文：[AAAI 2026 RAST](https://ojs.aaai.org/index.php/AAAI/article/view/41264)，源码：[RWLinno/RAST](https://github.com/RWLinno/RAST)。
6. **PIR**：论文和官方源码。用于区分 confidence selector、post-forecasting revision 与 E5 memory correction 的边界。论文：[NeurIPS 2025 PIR](https://proceedings.neurips.cc/paper_files/paper/2025/hash/331c41353b053683e17f7c88a797701d-Abstract-Conference.html)，源码：[ustc-time-series/PIR](https://github.com/ustc-time-series/PIR)。
7. **FEPCross**：论文和源码。用于比较跨城市频域迁移与 E5 的局部 trend-residual 迁移是否属于不同层次的问题。论文：[ECML-PKDD 2024 FEPCross](https://jhc.sjtu.edu.cn/~gjzheng/files/papers/pkdd2024_FEPCross/pkdd2024_FEPCross_paper.pdf)。
8. **ST-SSDL**：当前项目已有论文和源码，无需为 E5 立刻复现 GCRU；后续只用于公平下游比较和误差上限分析。论文：[NeurIPS 2025 ST-SSDL](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d7b351608d824a4680344a02b180a947-Abstract-Conference.html)。

相关资料已经准备完成，不再作为 T1-A 实现的阻塞条件。实现和论文写作时仍必须记录各资料的本地版本、官方链接、核心公式和与 E5 的逐项差异；如果某项官方代码未公开，不使用非官方复现替代结论性证据。

## 10. 后续实现与执行边界

T0 已实现纯张量 payload 工具和独立诊断。T1 已按以下可独立测试的单元完成工程接入；这些实现是实验条件，不是正式结果：

1. `build_offset_decay_signature`：输入 source-train future、future mask、最近 12 步 forecast context 和 history mask，输出 \(Y-\lambda\alpha\) 及有效 mask；
2. `anchor_mean_distance_normalization`：输入 `[B,B,N]` pairwise distance 和候选 mask，输出尺度不变、有限的无量纲距离；
3. `build_future_increment`：第一步计算 \(Y_1-\alpha\)，其余步计算 \(Y_h-Y_{h-1}\)，并严格传播缺失 mask；
4. `offset_decay_relation_target`：构造 T1-A teacher；`offset_decay_increment_relation_target` 只在 T1-A 过门后构造 T1-B teacher；
5. `learned_topk_offset_decay_horizon`：复用 learned Top-K 和权重，构造 OffsetDecay memory，只保留 horizon fusion，禁用 confidence head；
6. causal test：推理路径不能读取 query future、validation/test future 不能写入 Bank；
7. attribution test：同候选协议、Level-0、无 confidence 比较 E3/E5/random/base-only。

T1 已固定：endpoint level 来自最近 12 步可见 context 末端，候选集合和缺失 mask 沿用 E3，\(\lambda_h=1-(h-1)/(H-1)\)，distance normalization 固定为 anchor mean，T1-B increment weight 固定为 `0.5`。不得同时改变 downstream backbone、candidate sampling 和 confidence 网络。T1-A/T1-B 代码路径存在不等于 T1-B 可以立即实验；T1-B 正式运行仍必须等待 T1-A 过门。

## 11. 结论与决策门

本方案不是“再增加一个趋势模块”，而是把现有 future-guided 创新重新对准其失败点：**训练期 teacher 必须监督推理期真正被检索聚合的 future relation；OffsetDecay 负责部署坐标对齐，FutureIncrement 只作为条件性的真实未来动力学监督，confidence 只能作为最终后置校准。**

下一步只执行 T1-A 正式 seed 42；不复现 GCRU，不进入 E4 目标域适配，不接 confidence。T1-A 只有同时优于 E3 relation、random 和 base-only，并在 METR-LA/PEMS-BAY 的 retrieval 与无 confidence 下游指标中成立，才开放 T1-B。T1-B 失败不推翻 T1-A，而是删除 FutureIncrement 分支。最终版本只有通过多 seed 后，才能作为论文第二创新点候选。
