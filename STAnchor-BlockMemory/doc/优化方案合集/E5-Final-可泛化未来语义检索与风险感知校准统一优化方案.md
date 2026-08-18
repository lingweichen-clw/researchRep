# E5-Final：可泛化未来关系检索与 PostHoc 风险修正方案

> 当前主线：`Global288 + Latent48 + SymNorm + OffsetDecay + PostHoc Error-Aware Fusion`  
> 文档状态：2026-08-17 主线精炼版。CFDP/profile、CC-FGDA、Trend Residual T0 和旧六特征 confidence 已退出活动主线。当前 PostHoc BaseCap/Wide 对照仍在运行，本文不把未完成结果写成最终结论。

## 1. 核心研究问题

本工作的核心不是把真实 future 放入部署 key，而是回答两个可检验问题：

1. source-train 中的 future 关系监督，能否把只读取历史的编码器训练成更好的历史事件检索器；
2. 检索到的历史 future，能否作为一个对任意已训练下游预测器可插拔的诊断与修正信号。

最终方案只保留四个有明确作用的计算：

| 机制 | 解决的问题 | 新增部署参数 | 当前 query future |
|---|---|---:|---|
| `Latent48` | 用历史表示未来关系相似性 | 48 维检索头 | 不读取 |
| `SymNorm` | 消除 teacher 距离与余弦相似度的对称性冲突 | 0 | 仅 source-train teacher 读取训练 future |
| `OffsetDecay` | 对齐历史 payload 与当前 query 的近端速度 level | 0 | 不读取 |
| `PostHoc Error-Aware Fusion` | 判断基础预测风险以及 memory 应修正多少 | 轻量风险头与加性融合器 | 仅训练标签读取，推理不读取 |

论文主张必须限制为：future 用于 source-train 关系监督和 target-calibration 标签构造；部署 query key、候选选择、融合特征和最终预测都只使用当前可见历史、已训练 base 输出及因果历史 Bank。

## 2. 张量、时间范围与信息边界

给定短期下游输入和检索输入：

\[
X^{\mathrm{short}}\in\mathbb R^{B\times 12\times N\times C},
\qquad
X^{\mathrm{ret}}\in\mathbb R^{B\times 288\times N\times C}.
\]

其中，\(B\) 为 batch 大小，\(N\) 为节点数，METR-LA 当前 \(C=1\)，时间分辨率为 5 分钟。下游模型读取最近 60 分钟；检索编码器读取最近 24 小时；预测 horizon 为：

\[
Y\in\mathbb R^{B\times 12\times N\times C}.
\]

未来信息边界如下：

- source 预训练：训练事件 future 只在 `torch.no_grad()` teacher 中定义事件关系；encoder 输入仍只有历史；
- Bank 构建：只写入训练时间段内已经完整发生的事件；
- target 检索：候选必须满足候选 future 已在 query context 开始前结束；
- PostHoc 训练：真实 target 只构造 risk/blend 监督标签；
- validation/test：真实 future 只计算离线指标；
- 部署推理：不能读取当前 query future。

## 3. Latent48 历史编码器

`Global288` 表示把 288 个历史时间步按 `patch_size=12` 聚合成 24 个时间 patch。时空编码器输出：

\[
Z\in\mathbb R^{B\times 24\times N\times 96}.
\]

编码器使用共享时间注意力和静态图约束的空间注意力。时间池化后，每个节点得到：

\[
h_{b,n}\in\mathbb R^{96}.
\]

纯 `Latent48` 检索头将其映射为 L2 归一化节点 key：

\[
q_{b,n}
=
\operatorname{L2Norm}\!\left(g_{\theta}(h_{b,n})\right)
\in\mathbb R^{48}.
\]

这里的 48 维向量不分解为 profile 与 latent 子空间。它的语义由 future-relation teacher 监督，而不是由某一维对应某个未来预测步。事件 key 是节点 key 的平均后再归一化，用于 event-level 粗筛。

### 3.1 v2 编码器的非局部空间扩展（待验证）

当前已完成的 v1 结果仍以静态图空间注意力为准。v2 在保留该局部分支的同时，增加历史条件的非局部稀疏路由，用于处理多跳传播或物理图未显式连接但历史变化一致的节点。

对时间编码状态 (Zinmathbb R^{B	imes24	imes N	imes D})，先构造每个节点的历史状态和趋势摘要：

\[
s_{b,n}=\left[\operatorname{Mean}_{p}Z_{b,p,n};Z_{b,24,n}-Z_{b,1,n}\right].
\]

通过共享的低维目标/源投影得到 (q_{b,n},k_{b,m})，并计算：

\[
r_{b,n,m}=\frac{q_{b,n}^{\top}k_{b,m}}{\sqrt{d_r}}
+\lambda_{\mathrm{diff}}\log\left(1+(A_{\mathrm{rw}}^2+A_{\mathrm{rw}}^3)_{n,m}\right),
\qquad A_{\mathrm{rw}}=D^{-1}\bar A,
\]

其中 \(\bar A\) 是移除 self-loop 后的静态邻接，矩阵平方和立方分别累计二跳、三跳路径权重；它只是固定图先验，不是可学习的节点关系矩阵。

路由候选只排除自身，每个节点默认保留 Top-10 混合范围节点，其中 4 个来自一阶邻居、6 个来自多阶/远端集合；选中的消息通过负偏置门控注入局部图表示。该分支只读取当前 query 的历史，不读取 query future，不使用节点 ID 专属参数，也不改变 Latent48、Bank 或因果候选协议。完整公式、参数预算和消融见 [E5-Latent48-TGGE-Structured-Error-Corrector-v2优化方案.md](E5-Latent48-TGGE-Structured-Error-Corrector-v2优化方案.md)。

当前实现配置 `hidden_dim=80`、`route_dim=16`、3 个编码块；3 个路由分支实测 25,827 个参数，完整 retrieval state 实测 302,755 个参数。时间分支仍沿用 v1 的因子化时间注意力，后续时间卷积替换必须作为独立消融，不与本次空间路由结果混报。

## 4. SymNorm future-relation 预训练

### 4.1 OffsetDecay 关系对象

`OffsetDecay relation signature` 是训练期用于比较 future 关系的对象。对 source-train 事件 \(i\)，设 \(Y_{i,h,n,c}\) 为未来值，\(\alpha_{i,n,c}\) 为最近 12 个可见历史点的平均 level，定义：

\[
S^{\mathrm{OD}}_{i,h,n,c}
=
Y_{i,h,n,c}-\lambda_h\alpha_{i,n,c},
\qquad
\lambda_h=1-\frac{h-1}{H-1}.
\]

对事件 \(i,j\) 和节点 \(n\)，在共同有效 future 位置计算 masked MAE：

\[
d^{\mathrm{OD}}_{ij,n}
=
\frac{1}{|\Omega_{ij,n}|}
\sum_{(h,c)\in\Omega_{ij,n}}
\left|S^{\mathrm{OD}}_{i,h,n,c}-S^{\mathrm{OD}}_{j,h,n,c}\right|.
\]

该对象只在 source-train teacher 中构造，不在部署 query 上构造。

### 4.2 对称几何均值归一化

`SymNorm` 是 Symmetric Geometric Mean Normalization，即对称几何均值归一化。对事件 \(i\) 的有效候选集合 \(\mathcal C_{i,n}\)，先计算锚点尺度：

\[
\mu_{i,n}
=
\frac{1}{|\mathcal C_{i,n}|}
\sum_{k\in\mathcal C_{i,n}}d^{\mathrm{OD}}_{ik,n}.
\]

再定义：

\[
\widetilde d_{ij,n}
=
\frac{d^{\mathrm{OD}}_{ij,n}}
{\sqrt{(\mu_{i,n}+\varepsilon)(\mu_{j,n}+\varepsilon)}}.
\]

当候选 mask 对称时，\(\widetilde d_{ij,n}=\widetilde d_{ji,n}\)。这与节点 key 的余弦相似度

\[
s_{ij,n}=q_{i,n}^{\top}q_{j,n}=s_{ji,n}
\]

具有一致的对称结构。SymNorm 没有可学习参数，也不增加 Bank 检索开销。

### 4.3 关系分布匹配

teacher 分布与 student 分布分别为：

\[
p^{T}_{ij,n}
=
\operatorname{Softmax}_{j}
\left(-\widetilde d_{ij,n}/\tau_T\right),
\]

\[
p^{S}_{ij,n}
=
\operatorname{Softmax}_{j}
\left(s_{ij,n}/\tau_S\right).
\]

关系损失是有效 anchor 上的交叉熵：

\[
\mathcal L_{\mathrm{relation}}
=
-\frac{1}{|\mathcal A|}
\sum_{(i,n)\in\mathcal A}
\sum_j p^{T}_{ij,n}\log p^{S}_{ij,n}.
\]

完整预训练损失只保留遮挡重建与关系监督：

\[
\mathcal L_{\mathrm{pretrain}}
=
\mathcal L_{\mathrm{reconstruction}}
+\lambda_{\mathrm{relation}}\mathcal L_{\mathrm{relation}}.
\]

当前主线不再包含 CFDP profile loss、future increment loss 或 FGDA auxiliary loss。

## 5. 因果 Bank 与两阶段检索

`Bank` 是目标数据集训练时间段的因果历史记忆库。每个事件保存：事件 key、节点 key、level 特征、已发生的历史 future payload 及有效 mask。

对 query \(q\)：

1. `exact_calendar` 只保留同星期、同时间槽且满足因果约束的历史事件；METR-LA validation 中候选池平均约 8 个；
2. 事件 key 粗筛上限为 `event_top_r=32`，当前精确日历池不会被该上限截断；
3. 节点 key 余弦相似度对每个节点独立排序；
4. 每个节点取 `node_top_k=5`；
5. `search_temperature=0.10` 对 Top-5 分数做 softmax 聚合。

这里的 `node_top_k=5` 是 Bank 检索阶段的历史事件候选数，与 v2 编码器空间分支中的混合范围路由候选数相互独立；后者默认 `K_g=10`，配额为 4 个一阶加 6 个多阶/远端节点。

主线 `level_weight=0`，因此候选排序不混入手工 level 距离。level 只在 PostHoc 诊断特征和 OffsetDecay payload 对齐中使用。

## 6. OffsetDecay 历史 future 聚合

设节点 \(n\) 的 Top-5 历史事件为 \(\mathcal R(q,n)\)，权重为 \(\pi_{qjn}\)。Raw memory 为：

\[
\widehat Y^{\mathrm{raw}}_{q,h,n,c}
=
\sum_{j\in\mathcal R(q,n)}\pi_{qjn}Y_{j,h,n,c}.
\]

设 \(\alpha_q\) 与 \(\alpha_j\) 分别为 query 和历史事件最近 12 个可见点的平均 level，完整 offset memory 为：

\[
\widehat Y^{\mathrm{offset}}_{q,h,n,c}
=
\alpha_{q,n,c}
+\sum_{j\in\mathcal R(q,n)}\pi_{qjn}
\left(Y_{j,h,n,c}-\alpha_{j,n,c}\right).
\]

最终 OffsetDecay memory 为：

\[
\widehat Y^{\mathrm{mem}}_{q,h,n,c}
=
\widehat Y^{\mathrm{raw}}_{q,h,n,c}
+\lambda_h
\left(
\widehat Y^{\mathrm{offset}}_{q,h,n,c}
-\widehat Y^{\mathrm{raw}}_{q,h,n,c}
\right).
\]

第 1 个预测步完整使用 level 对齐，第 12 个预测步回到 raw historical future。该机制为零参数，并且只读取 query history 与因果 Bank。

## 7. PostHoc 风险诊断与检索修正

### 7.1 冻结基础模型

`PostHoc` 表示下游 base 模型先独立训练完成，再加载其 checkpoint 并冻结。基础预测为：

\[
\widehat Y^{\mathrm{base}}
=
f_{\theta^\star}(X^{\mathrm{short}}),
\]

其中 \(\theta^\star\) 在 PostHoc 训练中不更新。风险头只读取可见历史和停止梯度的 base 输出：

\[
\widehat r
=
g_{\phi}\!\left(
X^{\mathrm{short}},
\operatorname{StopGrad}(\widehat Y^{\mathrm{base}})
\right)
\in\mathbb R^{B\times H\times N\times1}.
\]

\(\widehat r\) 预测基础模型在每个 horizon-node 上的 SmoothL1 风险，而不是读取真实误差作为输入。

### 7.2 九个部署可用特征

每个 \((b,h,n)\) 位置构造九维特征 \(z_{b,h,n}\)：

| 序号 | 特征 | 计算与作用 |
|---:|---|---|
| 1 | predicted base risk | 风险头预测的基础模型误差 |
| 2 | retrieval similarity | Top-5 完整 Latent48 key 相似度的加权均值 |
| 3 | score margin | Top-1 与 Top-2 检索总分之差 |
| 4 | effective support | \((\sum_j\pi_j^2)^{-1}/K\)，衡量权重是否过度集中 |
| 5 | payload dispersion | `log1p` 后的候选 future 加权标准差 |
| 6 | direction agreement | 候选相对 base 修正方向的一致程度 |
| 7 | level match | 历史事件与 query 的 level 距离匹配度 |
| 8 | memory/base disagreement | `log1p` 后的 \(|\widehat Y^{mem}-\widehat Y^{base}|\) |
| 9 | horizon position | 从 0 到 1 的相对预测位置 |

旧 10 特征实现把缺失的 `profile_scores` 和 `latent_scores` 同时回退到完整 `shape_scores`，在纯 Latent48 Bank 下形成重复输入。主线只保留一个 `retrieval_similarity`。

### 7.3 可解释加性融合

每个特征由独立的一维 shape function \(f_d\) 映射为 logit 贡献：

\[
w_{b,h,n}
=
\sigma\!\left(
b+\sum_{d=1}^{9}f_d(z_{b,h,n,d})
\right).
\]

最终预测为：

\[
\widehat Y^{\mathrm{final}}
=
\widehat Y^{\mathrm{base}}
+w\left(
\widehat Y^{\mathrm{mem}}-
\widehat Y^{\mathrm{base}}
\right).
\]

当 memory 无效时强制 \(w=0\)，最终输出严格等于 base。各 \(f_d\) 的输出可以单独记录，因此能够解释某个位置的修正是由高风险、相似度、候选共识还是 horizon 等因素推动。

### 7.4 训练标签与损失

风险标签是 base prediction 与真实 target 的通道平均 SmoothL1 误差。最优 blend 标签是在 memory-base 方向上的截断最小二乘步长：

\[
w^*
=
\operatorname{clip}_{[0,1]}
\frac{
\langle Y-\widehat Y^{\mathrm{base}},
\widehat Y^{\mathrm{mem}}-\widehat Y^{\mathrm{base}}\rangle
}{
\|\widehat Y^{\mathrm{mem}}-\widehat Y^{\mathrm{base}}\|_2^2+\varepsilon
}.
\]

训练目标为：

\[
\mathcal L_{\mathrm{posthoc}}
=
\mathcal L_{\mathrm{forecast}}
+\lambda_r\mathcal L_{\mathrm{risk}}
+\lambda_b\mathcal L_{\mathrm{blend}}.
\]

真实 future 只用于上述训练标签和离线指标；推理时风险与 blend 都由历史、base 输出和检索诊断预测。

## 8. 已有证据与退出机制

以下结果均为 validation 阶段证据，不能替代最终 test、多 seed 或跨数据集下游结论。

| 证据 | 结果 | 决策 |
|---|---|---|
| OffsetDecay vs Raw | METR-LA memory MAE 改善 4.72%，PEMS-BAY 改善 12.36% | 保留 OffsetDecay |
| E5A vs E3 selector | 两个数据集 memory MAE 均改善约 3%，METR-LA 无 confidence 下游 MAE 再改善 0.62% | 保留 deployment-aligned relation teacher |
| SymNorm 零训练诊断 | 完全消除 distance/logit asymmetry；固定 key Spearman `+0.047629`；Oracle Top-5 MAE 改善 1.43% | 保留 SymNorm，并继续以实际预训练/迁移验证约束结论 |
| CFDP/profile | `gamma=0.25` memory MAE 仅改善约 0.077%，profile-only 破坏关系排序 | 删除 profile key、profile loss 和 Bank v2 分解 |
| Trend Residual T0 | METR-LA/PEMS-BAY memory MAE 分别明显恶化 | 删除趋势残差执行路径 |
| CC-FGDA | 未达到预先规定的检索和下游联合门槛，且增加额外结构 | 删除 adapter 主线 |
| 旧六特征 confidence | 风险排序弱，且与 PostHoc 诊断职责重叠 | 删除旧 confidence 模式 |

这些删除不是否定所有趋势、profile 或 adapter 方法，而是表示当前实验没有证明它们对本研究主张提供足够的独立增益。

## 9. 泛化实验路线

下一阶段只按以下顺序关闭不确定性：

1. 完成 9 特征 PostHoc BaseCap/Wide；Wide 只有在相对 BaseCap 有稳定收益时保留；
2. 固定 source checkpoint，在目标数据集重建 target-local 因果 Bank，比较 pretrained 与 target-random encoder；
3. 比较冻结 encoder、轻量参数高效微调和全参数微调，首先确认是否存在稳定 transfer gap；
4. 泛化成立后，再接入第二类下游 backbone，验证 PostHoc 是否真正与 backbone 解耦；
5. 最后运行多 seed，并报告均值、标准差、显著性、参数量、训练时间、推理延迟和显存。

每一步必须有明确决策：

- pretrained 不优于 random：停止宣称跨数据集表示泛化，先诊断域偏移；
- 轻量微调不优于冻结：删除微调模块；
- 全参数微调只提高 target 结果但破坏 source：只作为上界，不作为主协议；
- PostHoc 不能稳定超过同一 frozen base：删除 PostHoc 主张，保留检索 memory 作为独立结果；
- Wide 不优于 BaseCap：保留更小模型，停止扩容。

## 10. 最终论文叙事边界

当前最有说服力的组合贡献是：

> 使用 source-train future 关系监督一个只读取历史的时空检索编码器，在目标数据集构建因果历史 Bank，通过零参数 OffsetDecay 对齐历史 future，并以可解释 PostHoc 风险模型决定它对冻结下游预测的修正幅度。

当前不能宣称：

- 48 维 key 的每一维具有独立物理语义；
- 单个 Top-1 候选已经可靠，当前部署仍依赖 Top-5 平滑；
- seed 42 的结果代表统计显著性；
- PEMS-BAY 检索层改善已等价转化为跨数据集下游改善；
- 对速度数据的结论自动适用于流量、需求、天气或不同采样频率。

主方案到此停止机制扩张。后续只允许由泛化实验中的具体失败模式触发单变量修改。
