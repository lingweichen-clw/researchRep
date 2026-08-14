# E5 Latent48 + Context-Conditioned FGDA 优化方案

> 文档状态：机制设计、编码和最小验证已完成。本文将该版本简称为 `CC-FGDA`。本文所有公式均说明其目的、输入、输出和 future 信息边界，避免通过无明确作用的结构堆叠扩大模型。

## 1. 本轮决策

最终检索 key 继续使用纯 48 维 latent key，不重新引入 CFDP profile。图结构继续使用各数据集已有的静态图，本轮不学习动态图，也不使用节点 ID embedding。

本轮只优化 `FGDA`（Future-Guided Dynamics Adapter，未来关系引导的动态适配器）。FGDA 是主时空编码器之后的历史动态辅助模块：它从历史序列的一阶变化和静态图邻居变化中构造残差，修正主编码器隐藏状态。`future-guided` 只表示源域预训练期间，既有 OffsetDecay relation teacher 的梯度会指导该模块；FGDA 的输入中没有 future。

当前 FGDA-v1 的门控输出已经随样本、patch 和节点改变，但仍有三个能力限制：

1. 所有 token 共用同一个 `96 -> 16 -> 96` 残差映射，16 维瓶颈可能限制不同动态模式的拟合能力。
2. 最终融合门控是单个标量，只能整体放大或缩小 96 维残差，无法分别控制不同隐藏通道中的动态修正。
3. 当前模型没有显式利用主编码器上下文去调节残差映射。同一种局部上升在早高峰、平峰和拥堵消散阶段可能具有不同含义。

因此，本轮采用 B 方案：在保留共享参数的前提下，引入逐 token 的低秩条件调制和分组融合门控。新版本称为 `CC-FGDA`（Context-Conditioned FGDA，上下文条件化 FGDA）。

## 2. 设计原则

### 2.1 共享参数不等于相同修正

模型仍对所有节点和 patch 共享参数，保证节点数变化时参数形状不变，并支持 METR-LA 到 PEMS-BAY 等跨数据集迁移。不同 token 的行为差异不通过节点专属参数实现，而由其历史动态和主编码上下文生成条件系数。

### 2.2 每个新增计算只解决一个问题

| 计算 | 解决的问题 |
|---|---|
| 一阶差分 | 显式保留相邻时间步的上升、下降和局部变化 |
| 静态图聚合 | 表示相邻道路上已经发生的动态传播 |
| 标量空间门控 | 判断当前 token 应引入多少邻居动态 |
| 48 维低秩动态因子 | 提高残差表达能力，同时限制参数规模 |
| 上下文条件调制 | 让每个节点、每个 patch 使用不同的有效残差映射 |
| 完整动态直连 | 防止所有动态信息都必须经过低秩瓶颈 |
| 8 组融合门控 | 分组决定哪些动态修正应写回主隐藏状态 |

本方案不增加新的 future target、辅助预测头、对比损失、频域分支、多专家路由或动态图学习。

## 3. 符号与张量

| 符号 | 形状 | 含义 |
|---|---:|---|
| \(B\) | 标量 | batch 中的事件数 |
| \(T\) | 标量 | 历史输入长度；Global288 中为 288 |
| \(L\) | 标量 | patch 长度；当前为 12 |
| \(P=T/L\) | 标量 | patch 数；Global288 中为 24 |
| \(N\) | 标量 | 当前数据集的节点数 |
| \(C\) | 标量 | 输入物理变量通道数；当前速度数据为 1 |
| \(D_h\) | 标量 | 主编码隐藏维度；当前为 96 |
| \(D_b\) | 标量 | CC-FGDA 动态瓶颈维度；本方案为 48 |
| \(G\) | 标量 | 融合门控组数；本方案为 8 |
| \(\bar X\) | \([B,T,N,C]\) | 只用可见历史计算并归一化后的输入 |
| \(Z\) | \([B,P,N,96]\) | FactorizedSTEncoder 的输出 |
| \(A\) | \([N,N]\) 的稀疏表示 | 当前数据集提供的静态图边权 |

## 4. CC-FGDA 数据流

### 4.1 第一步：构造相邻历史变化

对归一化历史序列计算一阶差分：

\[
\Delta \bar X_{b,t,n,c}
=
\bar X_{b,t,n,c}-\bar X_{b,t-1,n,c}.
\]

该公式回答“当前值相对于上一个历史时刻发生了什么变化”。正值表示上升，负值表示下降，绝对值表示变化强度。只有相邻两个位置都可见时该差分有效：

\[
v_{b,t,n,c}
=
\mathbb I
\left[
o_{b,t,n,c}=1
\ \land\
o_{b,t-1,n,c}=1
\right].
\]

其中 \(o\) 是历史观测掩码。无效差分置零，第一个时间步由于没有前驱也置零。该步骤只读取 history。

### 4.2 第二步：保留 patch 内有序变化

每个 patch 内的 \(L\) 个差分按时间顺序拼接：

\[
\delta_{b,p,n}
=
\left[
\Delta\bar X_{b,(p-1)L+1,n,:};
\ldots;
\Delta\bar X_{b,pL,n,:}
\right]
\in\mathbb R^{LC}.
\]

方括号中的分号表示按原时间顺序拼接，不是求均值，也不是只保留 patch endpoint。\(:\) 表示保留全部输入通道。也就是说，\(\delta_{b,p,n}\) 包含节点 \(n\) 在第 \(p\) 个 patch 中的全部 \(L\) 个相邻变化。

随后执行共享线性投影：

\[
D_{b,p,n}=W_\Delta\delta_{b,p,n}
\in\mathbb R^{96}.
\]

当前 \(L=12,C=1\)，输入是 12 维并被映射到 96 维，因此这里不是维度压缩。该投影的目的只是把有物理意义的差分序列映射到与主编码器一致的隐藏空间。后续低秩分支负责非线性组合，本步骤不再叠加卷积或额外 attention。

### 4.3 第三步：提取静态图邻居动态

对节点 \(n\) 的有效非自环邻居执行边权归一化聚合：

\[
D^G_{b,p,n}
=
\frac{
\sum_{m\in\mathcal N(n)}
A_{nm}v_{b,p,m}D_{b,p,m}
}{
\sum_{m\in\mathcal N(n)}
A_{nm}v_{b,p,m}+\epsilon
}.
\]

其中：

- \(\mathcal N(n)\) 是静态图中指向节点 \(n\) 的邻居集合；
- \(A_{nm}\) 是邻居 \(m\) 到节点 \(n\) 的图边权；
- \(v_{b,p,m}\) 表示该邻居在当前 patch 是否存在有效历史差分；
- \(\epsilon\) 防止无有效邻居时除零。

分母将加权和转换为加权平均，避免邻居数量多或边权总和大的节点天然得到更大的特征幅值。输出 \(D^G\in\mathbb R^{B\times P\times N\times96}\) 表示邻居道路上已经发生的历史动态，不代表未来传播结果。

### 4.4 第四步：区分局部变化与邻居传播

空间门控为：

\[
a_{b,p,n}
=
\sigma
\left(
W_a[D_{b,p,n};D^G_{b,p,n}]+b_a
\right)
\in(0,1).
\]

其中分号表示特征拼接，\(\sigma\) 是 Sigmoid 函数。\(a_{b,p,n}\) 是标量，含义明确：当前样本、patch、节点应引入多少邻居动态。融合后的动态状态为：

\[
F_{b,p,n}
=
D_{b,p,n}+a_{b,p,n}D^G_{b,p,n}
\in\mathbb R^{96}.
\]

这里保留标量空间门控，不改成多头或多组空间门控，因为该标量可以直接解释为“邻居传播贡献强度”。

### 4.5 第五步：形成紧凑的动态因子

将 96 维动态状态投影到 48 维动态因子空间：

\[
u_{b,p,n}
=
\operatorname{GELU}
\left(
W_{\mathrm{down}}F_{b,p,n}+b_{\mathrm{down}}
\right)
\in\mathbb R^{48}.
\]

48 维瓶颈不是时间步数、节点数或 retrieval key 维度。它表示一组可学习的紧凑动态因子。相比旧版 16 维，48 维减少快速上升、下降、波动和传播状态被迫挤入过小空间的风险；相比直接使用 96 维全连接残差，它仍限制参数量和过拟合能力。

### 4.6 第六步：让每个 token 调节自己的动态因子

主编码状态 \(Z\) 包含绝对速度水平、日历位置、跨 patch 时间关系和图空间关系；动态状态 \(F\) 包含一阶变化及邻居变化。二者经过独立 LayerNorm 后拼接：

\[
c_{b,p,n}
=
[\operatorname{LN}_Z(Z_{b,p,n});
\operatorname{LN}_F(F_{b,p,n})]
\in\mathbb R^{192}.
\]

由该上下文生成 48 维调制向量：

\[
m_{b,p,n}
=
\tanh
\left(
W_m c_{b,p,n}+b_m
\right)
\in(-1,1)^{48}.
\]

再逐元素调节动态因子：

\[
\widetilde u_{b,p,n}
=
u_{b,p,n}
\odot
(1+m_{b,p,n}).
\]

其中 \(\odot\) 表示逐元素乘法。\(m\) 接近 \(-1\) 时抑制对应动态因子，接近 0 时保持原强度，接近 1 时增强对应动态因子。使用 \(1+m\) 而不是直接乘 \(m\)，是为了让零初始化的调制分支对应自然基准 \(\widetilde u=u\)。

虽然 \(W_m\)、\(W_{\mathrm{down}}\) 和 \(W_{\mathrm{up}}\) 在节点间共享，但每个 token 都有自己的 \(m_{b,p,n}\)。其有效低秩映射可写成：

\[
W_{\mathrm{up}}
\operatorname{Diag}(1+m_{b,p,n})
W_{\mathrm{down}}.
\]

因此，不同节点和 patch 使用的是不同的有效映射，但模型没有节点专属参数。这正是条件化设计相对于单纯扩大 MLP 的核心区别。

### 4.7 第七步：保留未经过瓶颈的完整动态通路

低秩动态残差为：

\[
R^{\mathrm{low}}_{b,p,n}
=
W_{\mathrm{up}}\widetilde u_{b,p,n}
+b_{\mathrm{up}}
\in\mathbb R^{96}.
\]

同时增加一个仅含 96 个参数的通道缩放向量 \(s\)：

\[
R^{\mathrm{direct}}_{b,p,n}
=
s\odot F_{b,p,n}.
\]

最终动态残差为：

\[
R_{b,p,n}
=
R^{\mathrm{low}}_{b,p,n}
+R^{\mathrm{direct}}_{b,p,n}.
\]

低秩通路负责学习上下文条件化的非线性动态组合；直连通路允许已经位于 96 维空间中的动态信息绕过 48 维瓶颈。直连不是新的特征分支，也不进行额外预测，只用于降低瓶颈造成信息丢失的风险。

### 4.8 第八步：分组决定动态残差写回多少

旧版使用单个标量 \(g_{b,p,n}\) 同时控制全部 96 个通道。新版本将 96 个通道分成 \(G=8\) 组，每组 12 维：

\[
g_{b,p,n}
=
\sigma
\left(
W_g
[\operatorname{LN}_Z(Z_{b,p,n});
\operatorname{LN}_R(R_{b,p,n})]
+b_g
\right)
\in(0,1)^8.
\]

将每个组门控值复制到对应的 12 个通道，得到：

\[
\widetilde g_{b,p,n}
=
\operatorname{ExpandGroup}(g_{b,p,n})
\in(0,1)^{96}.
\]

最终输出为：

\[
Z'_{b,p,n}
=
Z_{b,p,n}
+\widetilde g_{b,p,n}\odot R_{b,p,n}.
\]

8 组门控位于单标量门控和 96 维逐通道门控之间：它允许不同类型的动态残差被独立抑制或增强，同时避免逐通道门控带来的额外参数和过拟合。输出形状仍为 \([B,P,N,96]\)，因此后续共享 temporal pooling 和 48 维 latent retrieval head 无需改变。

## 5. 初始化与训练稳定性

为使新模块开始时不破坏原编码器：

- \(W_{\mathrm{up}}\) 和 \(b_{\mathrm{up}}\) 初始化为零；
- 直连缩放 \(s\) 初始化为零；
- 条件调制 \(W_m,b_m\) 初始化为零，此时 \(m=0\)；
- 分组融合门控权重初始化为零，偏置初始化为 \(-2\)；
- 空间门控继续采用保守的负偏置初始化。

因此初始动态残差严格为零：

\[
R=0,\qquad Z'=Z.
\]

训练初期先由已有 reconstruction loss 和 OffsetDecay relation loss 学习残差投影，随后条件调制和门控逐步获得有效梯度。该设计不新增损失函数，避免模型改进来源被多个同时变化的目标混淆。

## 6. Future 信息边界

### 6.1 训练阶段

- clean CC-FGDA 只读取完整 source-train history、历史观测掩码和 source 静态图。
- masked CC-FGDA 只读取遮挡后仍可见的 source-train history，并重新计算历史归一化和相邻差分。
- source-train future 只由 `torch.no_grad()` 下的 OffsetDecay relation teacher 使用，用于产生样本关系监督。
- future 不进入 \(D\)、\(D^G\)、\(F\)、\(m\)、\(R\)、\(g\) 或 retrieval key。

### 6.2 建库与部署阶段

- Bank key 由已发生历史经过 encoder、CC-FGDA 和 retrieval head 得到。
- query key 只使用 query 当前可见 history 和目标数据集静态图。
- query future 不参与条件调制、候选排序或 OffsetDecay memory 构造。

因此 CC-FGDA 学习的是“由 future relation 监督的历史动态表示”，不是把 future 编码到部署输入中。

## 7. 参数量与计算预算

在 \(L=12,C=1,D_h=96,D_b=48,G=8\) 下，预计参数量如下：

| 组件 | 预计参数量 |
|---|---:|
| 差分 patch 投影 `12 -> 96` | 1,152 |
| 标量空间门控 `192 -> 1` | 193 |
| 低秩残差 `96 -> 48 -> 96` | 9,360 |
| 条件调制 `192 -> 48` | 9,264 |
| 8 组融合门控 `192 -> 8` | 1,544 |
| 完整动态直连缩放 | 96 |
| 三个 96 维 LayerNorm | 576 |
| **合计** | **约 22,185** |

该估算约为当前主模型参数量的 5.5% 至 6%，位于已确认的 5% 至 10% 预算内。正式编码后必须用实际模型实例重新统计，而不能把本表估算当作最终效率结果。

静态图聚合与 FGDA-v1 相同，不增加新的图注意力层。推理延迟目标为相对 Latent48 不超过 10%，显存增量需要与参数量和中间激活一并实测。

## 8. 必须记录的解释性诊断

下列指标只用于判断模块是否真正工作，不参与训练目标：

| 指标 | 计算 | 传达的信号 |
|---|---|---|
| 空间门控均值与标准差 | 有效 token 上统计 \(a\) | 模型是否区分局部变化和邻居传播 |
| 条件调制强度 | 统计 \(\lVert m\rVert_1/48\) | 条件分支是否被实际使用 |
| 条件调制 token 方差 | 跨节点、patch 统计 \(m\) 的方差 | 不同 token 是否得到不同有效映射 |
| 分组门控均值与方差 | 分别统计 8 个 gate | 是否仍坍缩为近似单标量门控 |
| 低秩残差比例 | \(\lVert R^{\mathrm{low}}\rVert/\lVert Z\rVert\) | 条件化低秩通路贡献多少 |
| 直连残差比例 | \(\lVert R^{\mathrm{direct}}\rVert/\lVert Z\rVert\) | 是否确实需要绕过瓶颈 |
| 总修正比例 | \(\lVert\widetilde g\odot R\rVert/\lVert Z\rVert\) | FGDA 是否被忽略或过度主导主编码器 |

若 \(m\) 的 token 方差长期接近零，说明条件化映射退化为共享映射；若 8 组 gate 长期完全相同，说明分组门控没有学习到比标量门控更多的信息；若直连残差始终接近零，应删除直连通路而不是保留无效结构。

## 9. 最小对照实验

为区分“参数变多”和“条件机制有效”，正式比较必须保持同一数据、teacher、mask、候选协议、seed、Bank payload 和下游设置。

| 实验 | 结构 | 回答的问题 |
|---|---|---|
| B0：Latent48 | 不使用 FGDA | 纯 latent key 基线 |
| B1：FGDA-v1 | 16 维瓶颈 + 标量融合门控 | 当前轻量 adapter 基线 |
| B2：Wide-FGDA | 48 维瓶颈 + 标量融合门控，不使用条件调制和直连 | 单纯增加容量是否足够 |
| B3：CC-FGDA | 48 维瓶颈 + 条件调制 + 直连 + 8 组融合门控 | 完整 B 方案是否优于等方向扩容 |

第一阶段只运行 seed 42。若 B3 不优于 B2，则不能声称上下文条件化有效，应回退到更简单的 B2 或删除 FGDA。只有 B3 同时改善关系、memory 和下游结果，才进入组件消融：去掉条件调制、将8组 gate恢复为标量、去掉直连通路。

## 10. Keep / Remove / Stop 规则

### 10.1 保留完整 CC-FGDA

相对 Latent48 和 FGDA-v1，seed 42 同时满足：

- OD relation Spearman 至少提高 0.02；
- Recall@5 至少提高 1 个百分点；
- 无 confidence memory MAE 至少下降 0.5%；
- 无 confidence 下游 MAE 至少下降 0.5%；
- B3 明确优于仅扩容的 B2；
- 参数增量不超过 10%；
- 推理延迟增量不超过 10%；
- 条件调制和分组门控没有坍缩为常量。

满足后才进行多 seed、PEMS-BAY 迁移和完整下游验证。

### 10.2 简化模块

- B2 与 B3 相当：条件化没有独立收益，保留更简单的 Wide-FGDA。
- 直连残差长期接近零：删除直连缩放 \(s\)。
- 8 组 gate 行为近似相同：恢复标量融合门控。
- 图门控长期接近零且 Local 与 LocalGraph 相当：删除静态图动态分支，只保留 Local-FGDA。

### 10.3 停止 FGDA 优化

若 Wide-FGDA 和 CC-FGDA 均不能稳定改善关系排序、memory MAE 和无 confidence 下游 MAE，则停止继续扩大 adapter，不进入多专家、动态图、频域或趋势分解。最终主线回退到纯 Latent48 encoder，把研究重点转向检索 payload 和下游风险感知校准。

## 11. 暂不实施的方向

- 不使用节点 ID embedding，避免参数绑定固定传感器集合。
- 不学习动态图，继续使用每个数据集自己的静态图。
- 不使用 Mixture-of-Experts；只有 CC-FGDA 已证明条件化有益但容量仍不足时才重新评估。
- 不加入 FFT、DWT 或显式趋势分解，避免偏离未来相似关系检索的核心问题。
- 不新增 future profile、profile key 或辅助 future 回归头。
- 不修改 48 维 latent key、OffsetDecay payload 和 confidence calibrator 的接口。

## 12. 实施边界

编码已在现有 `HistoryDynamicsAdapter` 内完成版本化扩展，并保持 `none`、`local`、`local_graph` 旧配置可严格加载。新配置和 checkpoint 使用独立名称，不能覆盖 FGDA-v1 的模型、Bank 或日志。

### 12.1 已实现文件

- `stanchor/models/dynamics_adapter.py`：CC-FGDA 分支、输出诊断和参数初始化；
- `stanchor/config.py`、`stanchor/models/pretraining.py`、`stanchor/engine/pretrainer.py`：配置、模型接入和 epoch 级诊断日志；
- `configs/metrla_e5_final_latent48_cc_fgda_global288_v1.yaml`：Global288、纯 Latent48、静态图、OffsetDecay + SymmetricNorm 的唯一新配置；
- `scripts/run_e5_latent48_cc_fgda_global288_queue.ps1`：预训练与后处理队列，使用独立产物目录并拒绝覆盖已有结果；
- `tests/test_dynamics_adapter.py`、`tests/test_experiment_queues.py`：形状、恒等初始化、梯度、参数预算、配置单变量和队列契约测试。

### 12.2 当前验证结论

在 `research` 环境中，全仓库 `177` 个 unittest 通过；`python -m compileall -q stanchor scripts tests` 通过；PowerShell 队列解析通过。Global288 CC-FGDA 总参数为 `414,021`，adapter 参数为 `22,185`，相对纯 Latent48 增量约 `5.66%`。这些是实现验证结果，不是模型效果结论；关系排序、Bank、可视化和下游指标必须等待实验机正式预训练完成后再评估。
