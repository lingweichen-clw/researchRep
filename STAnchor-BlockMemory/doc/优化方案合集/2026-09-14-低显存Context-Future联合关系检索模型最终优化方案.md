# 低显存 Context–Future 联合关系检索模型最终优化方案

## 1. 文档状态与结论

本文给出 STAnchor-BlockMemory 检索模型在当前证据和资源约束下的最终优化设计。方案处于“设计定稿、尚未实现和验证”状态。

最终选择是：**保持现有 Patch Embedding、Factorized ST Encoder、RetrievalHead、64 维 Key、Bank schema 和下游 Router 架构不变，只把当前 future-only 关系 teacher 改为一个统一的 Context–Future 联合关系 teacher。主版本不启用 `HistoryDynamicsAdapter`。**

该选择的直接结果是：

- 检索模型参数量保持 `958,704`，不增加任何训练或推理参数；
- 编码器前向图完全不变，推理显存和延迟原则上不变；
- 训练阶段复用当前已经计算过的 context pairwise distance，只增加极小的 `[B,B,N]` 联合距离与 teacher 分布；
- 新目标直接约束 Key 同时表达 future 演化关系和窗口内部 Context shape，而不是再增加一个可学习分支；
- `HistoryDynamicsAdapter(mode=local)` 仅在主版本验证后仍存在明确的 Context 表达能力缺口时启用，不能与联合 teacher 同时作为第一轮变化。

## 2. 已观测问题与证据边界

### 2.1 已排除全局 Key 坍缩

Offset-only epoch 32 检查点的 64 维 Key 有效秩为 `9.52`，平均 pair cosine 为 `0.627`；learned pooling 平均有效使用 `19.30 / 24` 个 patch。因此当前问题不是全部 Key 收缩到一个点，也不是池化只看极少数 patch。

### 2.2 存在相对的 Context-shape 使用不足

在相同日历控制下，Key 距离与三类因素的 Spearman 相关性为：

| 因素 | 相关性 |
|---|---:|
| Context shape distance | 0.082 |
| Context level distance | 0.552 |
| Offset-only future distance | 0.554 |

`shape_swap` 引起的 Key cosine change 为 `0.0759`，`level_swap` 为 `0.2195`。这说明 Key 会使用 Context，但窗口内部动态形状相对于 `[mean,std,last,slope]` 和 future teacher 的影响偏弱。

本方案的目标不是删除 level，而是让窗口归一化后的 Context shape 获得可验证的增量作用。

### 2.3 长尾错误主要来自 payload，不应错误归因给 Key

Future Surprise 二乘二归因显示：Top20 的 payload 平均主效应约为 `+0.5162 MAE`，Key 主效应约为 `+0.0380`；Top5 的 payload 主效应约为 `+2.0885`，Key 主效应接近零。

因此本方案只解决检索表征中的 Context/level 平衡，不宣称解决 Offset-only payload 的远期锚定误差。下游仍需依靠现有 Context-aware Router 在 Base 与历史候选之间进行逐 horizon 校准。

## 3. 三种候选路线与选择

### 3.1 路线 A：联合关系 teacher，编码器不变（采用）

用窗口归一化后的 Context shape distance 细化现有 future relation teacher。该路线新增参数为零，也不增加编码器激活。

选择理由：现有干预实验已经表明 encoder hidden 会响应 Context shape 和 patch 顺序，而 Key 的变化较小，说明主要缺口更接近“关系监督和 Key 压缩没有充分保留形状”，而不是 encoder 完全缺乏动态表达能力。

### 3.2 路线 B：联合 teacher + `HistoryDynamicsAdapter(local)`（条件后备）

在 encoder 后注入窗口归一化历史增量。实测总参数量从 `958,704` 增加到 `964,737`，增加 `6,033`，约 `0.63%`。

该路线参数增加很小，但会保存多组 `[B,24,N,128]` 训练激活。以 `B=16,N=207,D=128` 计算，单个 FP32 张量约为 `38.8 MiB`，结合 residual、gate 和 autograd 缓存，训练峰值可能增加数百 MiB。因此它不进入第一版。

### 3.3 路线 C：`local_graph` 或 `context_conditioned` Adapter（否决为主线）

| Adapter 模式 | 总参数量 | 增量 | 决策 |
|---|---:|---:|---|
| none | 958,704 | 0 | 主方案 |
| local | 964,737 | +6,033 | 条件后备 |
| local_graph | 964,994 | +6,290 | 暂不采用 |
| context_conditioned | 971,801 | +13,097 | 不采用 |

当前 Factorized ST Encoder 已包含空间图交互，额外的 graph dynamics 容易重复建模，并增加图聚合激活；`context_conditioned` 还包含更多 normalization、modulation 和 gate 中间量。两者的预期收益不足以抵消 10GB 基线上的显存风险。

## 4. 输入、归一化与表示路径

### 4.1 输入定义

Retrieval Context 为过去 288 个五分钟时间步：

\[
X\in\mathbb{R}^{B\times288\times N\times1}.
\]

其中：

- \(B\) 是 batch size，当前为 16；
- \(N\) 是节点数，METR-LA 为 207；
- 最后一维是交通速度通道。

输入首先经过只在训练时间段拟合的逐节点全局 scaler。模型接收的是全局标准化后的值 \(X^g\)，validation/test 统计量不得参与 scaler 拟合。

### 4.2 窗口归一化与 level

对每个事件和节点使用当前 288 步历史计算：

\[
\mu^w_{b,n}
=\frac{\sum_t M_{b,t,n}X^g_{b,t,n}}
{\sum_tM_{b,t,n}},
\]

\[
\sigma^w_{b,n}
=\sqrt{
\frac{\sum_tM_{b,t,n}(X^g_{b,t,n}-\mu^w_{b,n})^2}
{\sum_tM_{b,t,n}}
+\epsilon
},
\]

\[
\widetilde X_{b,t,n}
=\frac{X^g_{b,t,n}-\mu^w_{b,n}}
{\sigma^w_{b,n}+\epsilon}.
\]

\(M\) 是历史观测有效掩码。Level feature 沿用：

\[
L_{b,n}=[\mu^w_{b,n},\sigma^w_{b,n},last_{b,n},\Delta^{endpoint}_{b,n}],
\]

其中 \(\Delta^{endpoint}=last-first\)。它是 288 步窗口首尾位移，不在论文中称为严格的局部 slope，避免物理含义误导。

### 4.3 原有编码器保持不变

288 步切成 24 个长度为 12 的 patch：

\[
H^0
=\operatorname{PatchEmbedding}
(\widetilde X,L,calendar)
\in\mathbb{R}^{B\times24\times N\times128},
\]

\[
H^{enc}
=\operatorname{FactorizedSTEncoder}(H^0,G)
\in\mathbb{R}^{B\times24\times N\times128},
\]

\[
K
=\operatorname{RetrievalHead}(H^{enc})
\in\mathbb{R}^{B\times N\times64}.
\]

主方案设置：

```yaml
model:
  dynamics_adapter_mode: none
```

## 5. Context–Future 联合关系 teacher

### 5.1 Offset-only Future Relation

Offset-only Future Signature 是用训练样本的真实 future 减去 forecast context 最后可见端点：

\[
\Phi^f_{b,h,n}
=Y_{b,h,n}-e_{b,n},
\qquad
\Phi^f\in\mathbb{R}^{B\times12\times N\times1}.
\]

其中 \(Y\) 只在预训练时作为 teacher 标签使用；\(e\) 沿用现有 Offset-only helper 的 endpoint 与缺失值处理规则。推理时 query future 不进入 Key、候选筛选或 Router 输入。

对 batch 内两个事件 \(i,j\) 的同一节点 \(n\)，计算 masked MAE：

\[
d^f_{ij,n}
=\operatorname{MAE}_{h,c}
(\Phi^f_{i,:,n,:},\Phi^f_{j,:,n,:}).
\]

输出形状为：

\[
d^f\in\mathbb{R}^{B\times B\times N}.
\]

### 5.2 Context Shape Relation

Context Shape 使用干净、完整但只含历史的窗口归一化序列：

\[
d^c_{ij,n}
=\operatorname{MAE}_{t,c}
(\widetilde X^{clean}_{i,:,n,:},
 \widetilde X^{clean}_{j,:,n,:}),
\]

\[
d^c\in\mathbb{R}^{B\times B\times N}.
\]

由于每个事件节点已经独立做窗口归一化，\(d^c\) 主要表达 288 步内部的相对变化形状，而不是绝对速度 level。

Future distance 与 Context distance 分别使用现有 `symmetric_geometric_mean` 方法做尺度归一化，得到 \(\bar d^f\) 与 \(\bar d^c\)。不能在相加后才统一归一化，否则数值范围更大的分量会隐式主导 teacher。

### 5.3 联合关系距离

把每个事件对看作二维关系向量：

\[
r_{ij,n}
=
\begin{bmatrix}
\sqrt{1-\alpha}\,\bar d^f_{ij,n}\\
\sqrt{\alpha}\,\bar d^c_{ij,n}
\end{bmatrix}.
\]

联合距离为该二维向量的欧氏范数：

\[
\boxed{
D^{joint}_{ij,n}
=\sqrt{
(1-\alpha)(\bar d^f_{ij,n})^2
+\alpha(\bar d^c_{ij,n})^2
}
}.
\]

第一版固定：

\[
\alpha=0.2.
\]

这里的 \(0.2\) 是关系几何中的 Context 权重，不解释为梯度或信息量严格占 20%。当 \(\alpha=0\) 时：

\[
D^{joint}=\bar d^f,
\]

能够严格退化为原有 future-only relation teacher。

### 5.4 有效候选集合

对 anchor \((i,n)\)，teacher softmax 只在以下候选集合内计算：

\[
\mathcal C_{i,n}
=\{j:\ j\neq i,
\operatorname{nonoverlap}(i,j),
future\_valid(i,j,n),
context\_valid(i,j,n)\}.
\]

每个 anchor 至少需要两个有效候选。自身 pair、时间重叠 pair 和观测不足 pair 必须 mask，防止零距离自身样本或近重复窗口主导 teacher。

## 6. 单一分布式关系损失

Teacher 分布为：

\[
p^T_{ij,n}
=\operatorname{softmax}_{j\in\mathcal C_{i,n}}
\left(-\frac{D^{joint}_{ij,n}}{\tau_T}\right),
\qquad \tau_T=0.1.
\]

Student 使用 L2-normalized node Key 的余弦相似度：

\[
s^S_{ij,n}=K_{i,n}^{\top}K_{j,n},
\]

\[
p^S_{ij,n}
=\operatorname{softmax}_{j\in\mathcal C_{i,n}}
\left(\frac{s^S_{ij,n}}{\tau_S}\right),
\qquad \tau_S=0.1.
\]

唯一的 retrieval loss 是 teacher 到 student 的交叉熵：

\[
\boxed{
\mathcal L_{joint-rel}
=-\frac{1}{|\mathcal A|}
\sum_{(i,n)\in\mathcal A}
\sum_{j\in\mathcal C_{i,n}}
p^T_{ij,n}\log p^S_{ij,n}
}.
\]

主方案不再同时叠加 hard-negative loss、rank loss、level penalty、orthogonal loss 或额外 context loss。旧 HN-Offset-only 是外部正式基线，不和新 teacher 混写在一个目标中。

## 7. Masked Reconstruction 与泄漏边界

总预训练目标保持两项：

\[
\boxed{
\mathcal L
=2\mathcal L_{rec}
+\mathcal L_{joint-rel}
}.
\]

为保持当前约 10GB 显存的 single-view 优化，encoder 仍只执行一次 masked forward：

1. 采样 mask，得到 `visible = observed & ~value_mask`；
2. student embedding、encoder、RetrievalHead 和 reconstruction head 只使用 `visible` 计算的窗口统计；
3. \(d^c\) 的 clean history teacher 只执行一次无梯度 `normalize_window(X, observed)`，不执行第二次 encoder；
4. clean Context 不得作为 residual、adapter 输入或 reconstruction feature；
5. \(d^f,d^c,D^{joint},p^T\) 全部在 `torch.no_grad()` 下构造。

这样 teacher 可以用完整历史定义稳定关系，同时不会把被掩码值泄漏给重构路径。

## 8. 参数、显存与时间预算

### 8.1 参数量

主方案只修改 loss/teacher：

\[
958{,}704\rightarrow958{,}704.
\]

Bank Key 维度、RetrievalHead 输入输出和 Router 接口均不变。

### 8.2 显存

在 `B=16,N=207` 时，最终联合距离张量 `[B,B,N]` 只有约 `52,992` 个 float，FP32 小于 `0.3 MiB`。Context pairwise MAE 的最大瞬时中间量约为：

\[
16\times16\times288\times207
\approx15.26\text{ million floats},
\]

FP32 约 `58.2 MiB`。当前 HN loss 已经计算同形状的 context pairwise distance，因此实现时应复用现有函数和缓冲，不同时保留重复的 full pairwise tensor。

资源验收条件：

- batch size 保持 16；
- 继续使用 `masked_relation_single_view`；
- 不创建 clean encoder forward；
- 新方案相对同机器、同 batch 的峰值 `cuda allocated` 增量目标不超过 `0.2 GB`；
- 若超过 `0.3 GB`，优先检查重复 pairwise tensor 生命周期和 autograd 引用，不通过降低 batch size掩盖实现问题。

### 8.3 推理成本

联合 teacher 只存在于训练阶段。推理仍为：

\[
X\rightarrow\text{Embedding}\rightarrow\text{Encoder}
\rightarrow\text{RetrievalHead}\rightarrow K.
\]

因此推理参数、FLOPs、Key/Bank 存储和在线显存均不增加。

## 9. 实现影响范围

预计只涉及：

- `stanchor/config.py`：增加联合 teacher mode 和 `context_relation_weight`；
- `stanchor/losses/pretraining.py`：构造 clean Context distance、联合距离和 teacher distribution；
- `stanchor/engine/pretrainer.py`：向 loss 传入 clean retrieval history/observed，记录新配置与 teacher support；
- `tests/test_future_relation_loss.py`：覆盖退化性、mask、non-overlap、无 future 泄漏和数值稳定性；
- 新增一个正式配置，不修改现有 Offset-only 与 OffsetDecay 配置。

不修改：

- `patch_embedding.py`；
- `encoder.py`；
- `retrieval_head.py`；
- `dynamics_adapter.py`；
- Bank schema；
- downstream Router 架构；
- payload 公式。

由于训练后的 encoder/Key 参数会改变，必须建立新的 checkpoint、retrieval fingerprint 和匹配 Bank。下游 Router 架构可以复用，但必须在新 Bank 和新候选分布上重新训练，不能复用旧 Router 权重。

## 10. 最小验证与进度安排

### 10.1 Stage 0：无需完整预训练

在固定 batch/validation 子集上计算 \(\alpha\in\{0,0.1,0.2,0.3\}\) 的 teacher：

- 与 future-only teacher 的 Top-K Jaccard；
- teacher effective support 和熵；
- Top-K future distance；
- Top-K Context-shape distance；
- Context-similar/future-different mirage 进入 teacher Top-K 的比例。

默认正式值仍为 `0.2`。只有当 `0.2` 造成 future Top-K 明显恶化或 teacher support 异常收缩时，才改用 `0.1`；不同时搜索温度。

### 10.2 Stage 1：低成本筛选

从同一 Offset-only checkpoint 初始化两个短程分支，重置相同优化器并固定数据顺序：

| 分支 | Teacher | Adapter | 用途 |
|---|---|---|---|
| R0 | future-only relation，\(\alpha=0\) | none | soft relation 对照 |
| R1 | joint relation，\(\alpha=0.2\) | none | 主候选 |

先训练 3–5 个 epoch，只用于方向筛选，不能作为论文最终数值。旧 HN-Offset-only checkpoint 作为已完成外部基线。

### 10.3 Stage 2：正式预训练和 Bank

只有 R1 同时通过表示与 future-ranking 门槛时，才进行完整训练并重建匹配 Bank。正式论文归因至少保留：

1. HN-Offset-only 已完成主线；
2. future-only relation \(\alpha=0\)；
3. joint relation \(\alpha=0.2\)。

### 10.4 Stage 3：条件 Adapter

仅当联合 teacher 已改善 loss/检索，但 `shape_swap`、Context–Key alignment 仍显示 encoder 输出缺少足够动态信息时，增加：

```yaml
model:
  dynamics_adapter_mode: local
  dynamics_bottleneck_dim: 16
```

该实验必须与“joint relation + adapter none”单变量比较。若 adapter 没有稳定增益，立即删除；不继续尝试 `local_graph` 或 `context_conditioned`。

## 11. 验收指标与决策规则

### 11.1 表示层

- Key effective rank 与 pair cosine 不出现新的几何坍缩；
- pooling effective patches 不显著减少；
- Context shape–Key 距离相关性和 `shape_swap` 敏感性应提高；
- 不能只追求 shape 相关性：Offset-only future Spearman、Recall@5、NDCG@5 不得显著下降；
- `level_swap` 敏感性允许保留，但 level 相对 shape 的垄断程度应减弱。

### 11.2 Frozen Memory 层

- 总体 MAE/RMSE/MAPE；
- 12 个 horizon 的 MAE；
- ordinary 80%、Future Surprise Top20、Top5；
- Context-similar/future-similar、Context-similar/future-different、Context-different/future-similar四象限 Key 距离。

### 11.3 下游层

使用相同 Base、数据划分、candidate protocol、payload、Router 架构、训练轮数和初始化规则：

- Base-only；
- 新 Key + 新 Bank + horizon-only；
- 新 Key + 新 Bank + 当前 Context-aware Router。

Router 必须重新训练。第一轮保持 Offset-only payload 不变，避免同时改变 Key teacher 与 payload；已知 payload 长尾问题继续单独报告。

### 11.4 Keep / Remove / Stop

**Keep joint teacher：** Context-shape 证据改善，future ranking 不显著下降，Frozen Memory 或下游至少一项获得稳定改善，且资源预算通过。

**Remove joint teacher：** shape 指标提高但 future ranking、Memory MAE 或下游 MAE恶化，说明模型只学会了表面 Context 相似性。

**Keep local adapter：** 只有它在 joint-teacher-only 之上带来独立增益，并且峰值显存增量可接受。

**Stop architecture expansion：** joint teacher 无效时先回到当前编码器，不继续堆叠 patch-level level、额外图分支、多损失或更大的 Key；因为当前证据不足以证明容量是瓶颈。

## 12. 最终科研表述

如果实验成立，可以把机制表述为：

> We learn retrieval keys from a unified relation geometry that preserves future-evolution similarity while explicitly retaining level-invariant historical context shape. The Context term is used only as source-training supervision; query futures remain unavailable to the encoder, Memory Bank search, and downstream inference.

不能表述为：

- 模型原先发生了检索坍缩；
- level 信息是错误或无用的；
- 联合 teacher 已经解决 Offset-only payload 的全部长尾问题；
- 单个 seed 的结果证明跨数据集普适性。
