# TGGE 当前版本迁移 E2 Hard-Negative 监督方案

## 1. 方案目的

当前 TGGE 主线使用 Latent48 检索 key、OffsetDecay future 表示和连续 relation supervision。本方案验证：在保持当前 encoder、索引优化、OffsetDecay、Bank 协议和下游结构不变的前提下，E2 的离散正负样本监督是否比当前 soft relation supervision 更有利于候选内部排序。

不回退 E2 旧 encoder，不恢复 profile，不加入 Rank Loss，不改变 future payload 的 OffsetDecay 定义。

## 2. 监督定义

历史上下文使用已有局部归一化表示：

$$
\tilde X_{q,t,n,c}=\frac{X_{q,t,n,c}-\mu^{ctx}_{q,n,c}}{\sigma^{ctx}_{q,n,c}+\varepsilon}.
$$

context distance 只使用 query 时刻以前的历史：

$$
d^{ctx}_{q,j,n}=\operatorname{MAE}(\tilde X_{q,:,n},\tilde X_{j,:,n}).
$$

future distance 沿用当前 TGGE 的 OffsetDecay：

$$
d^{future}_{q,j,n}=d^{OD}_{q,j,n}.
$$

有效候选集合为 $\mathcal V_{q,n}$。正样本取 OffsetDecay future 距离最小的 10%：

$$
\mathcal P_{q,n}=\{j\in\mathcal V_{q,n}:d^{future}_{q,j,n}\le Q_{0.1}\}.
$$

普通负样本为 $\mathcal V_{q,n}\setminus\mathcal P_{q,n}$。hard negative 为历史 context 相似但 future dynamics 不相似的候选：

$$
\mathcal N^{hard}_{q,n}=\{j:d^{ctx}_{q,j,n}\le Q_{0.2},\ d^{future}_{q,j,n}\ge Q_{0.8}\}.
$$

因此完整监督同时保留正样本、普通负样本和加权 hard negative，而不是只使用 hard negative。

key 相似度为 $s_{q,j,n}=\cos(z_{q,n},z_{j,n})$，logit 为 $\ell_{q,j,n}=s_{q,j,n}/\tau$。hard negative 使用 E2 的替代式权重 $w_h=2.0$：普通候选（正样本和普通负样本）权重为 1，hard negative 权重为 $w_h$，而不是在原权重上再叠加一次：

$$
L_{HN-OD}=\log\left(\sum_{j\in\mathcal P\cup\mathcal D}\exp(\ell_{q,j,n})+w_h\sum_{j\in\mathcal N^{hard}}\exp(\ell_{q,j,n})\right)-\log\sum_{j\in\mathcal P}\exp(\ell_{q,j,n}).
$$

因此，若 $w_h=2$，每个 hard negative 在分母中贡献 2 倍，而不是 $1+2=3$ 倍。

## 3. 代码和实验计划

新增 hard_negative_offset_decay retrieval loss 模式，复用当前 future relation targets、masked pairwise MAE 和 output.clean.statistics.normalized。不改变 encoder 参数量、索引优化、Bank payload、profile、Rank Loss、downstream 校正器或 Memory 聚合器。

第一版保持当前训练权重，只替换 retrieval loss 形式，不把 retrieval weight 改回 E2 的 0.1，以保证单变量归因。

代码级验证包括 compileall、synthetic forward、有限 loss/gradient、hard-negative 计数和现有 relation 接口回归。随后做 3–5 epoch 短实验，对比当前 soft relation 与 HN-OffsetDecay，记录 retrieval loss、valid anchors、candidate/hard-negative count、key-future Spearman、top-1 future distance 和单 epoch 时间。smoke 产物验证后清理。

同时继续上一轮下游方向：forecast-only + residual-additive，即 $\hat Y=Y^{base}+\alpha R^{memory}+\beta$，做 5 epoch 短验证，确认该结构是否稳定优于 scalar gate 和 base-only。两个实验分开，避免同时改变 encoder 和校正器。

## 4. 决策标准

- 保留 HN-OffsetDecay：top-1、top-k memory 或最终下游 MAE 稳定改善；
- 降低 hard-negative weight：top-1 改善但 weighted memory 或长 horizon 变差；
- 删除 HN-OffsetDecay：key-future 排序和下游 MAE 都没有改善；
- 优先继续优化校正器：检索指标改善但最终 forecast MAE 不改善。

资源约束：单轮预训练目标不超过 10 分钟；不增加 encoder 参数量；不引入额外 encoder forward；不破坏向量化检索和索引优化。

