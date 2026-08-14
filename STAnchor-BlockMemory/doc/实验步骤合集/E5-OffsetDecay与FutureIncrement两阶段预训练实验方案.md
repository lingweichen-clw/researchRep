# E5 OffsetDecay 与 FutureIncrement 两阶段预训练实验方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not start T1-B until the T1-A decision gate has passed.

**Goal:** 在不增加 future encoder、不接入 confidence、不更换下游 backbone 的条件下，先验证与 OffsetDecay 推理严格对齐的 future relation teacher；只有该目标在同域、跨域和无 confidence 下游同时成立后，才加入真实 future increment 监督。

**Architecture:** 保留现有 STAnchor encoder、掩码重建任务、relation distribution matching 和 causal target-local Bank。T1-A 将训练期 future teacher 改为经过无参数距离校准的 OffsetDecay signature，并在推理期使用零参数 OffsetDecay payload；T1-B 只在 T1-A 成功后，为 teacher distance 增加真实 future increment 距离。两个阶段都不增加推理网络参数。

**Tech Stack:** Python 3.10、PyTorch、NumPy、YAML、项目内 `unittest`、现有 `pretrain.py` / `build_bank.py` / diagnostics / downstream CLI，运行环境为 Conda `research`。

---

## 0. 文档状态与执行边界

本文档是一份自包含实验方案。读者不需要翻阅其他报告才能理解本文中的实验名称、公式或命令。

当前状态：

- T0 表示诊断已完成；
- T1-A/T1-B 的工程接口、配置和单元/集成测试已实现；T1-A 已完成工程 smoke，T1-B 仍必须等 T1-A 过门后才允许启动实验；
- E3 默认配置仍使用 `context_normalized` teacher；E5 配置显式选择 `offset_decay` 或 `offset_decay_increment` teacher；
- 截至 2026-08-03，没有任何正式 E5 checkpoint 是用 `ODSignature` 预训练得到的；
- 本文中的新增配置、模式和命令已具备工程运行条件；下一步从第 10 节 T1-A METR-LA 正式 seed 42 开始；
- 当前禁止直接运行 T1-B 正式预训练，除非第 16 节 T1-A 下游决策门已经通过；
- 当前禁止读取 METR-LA 和 PEMS-BAY test；
- 当前不复现 GCRU，不加入 confidence，不实现 TimeMixer、DWT/FFT 或 learned future encoder。

本文采用两阶段门控：

1. **T1-A**：只验证 OffsetDecay teacher 与 OffsetDecay 推理是否对齐；
2. **T1-B**：只有 T1-A 通过预设门槛，才加入真实 future increment 监督。

任一阶段失败都执行 `Stop`，不能通过增加网络复杂度继续补救。

## 1. 特殊名词与实验名称

本节重新定义本文使用的所有特殊名称。即使这些名称在其他报告中出现过，也以本节定义为准。

### 1.0 `E3`、`T0`、`E5`、`E5A` 与 `E5B`：实验代号

- `E3`：当前 relation 预训练基线。它用 288 步可见历史生成 node key；训练期将真实 future 按各样本历史 mean/std 归一化，计算 future pair distance 作为 teacher distribution，再让历史 key 的 cosine distribution 匹配 teacher。E3 的 query 推理只使用历史，不读取 query future。
- `T0`：已经完成的零训练表示诊断。它冻结 E3 checkpoint、候选集合、learned weights 和历史 Bank，只替换历史 future 的 payload 表示，用来判断问题来自 selector 还是 payload；T0 不产生新模型。
- `E5`：本文提出的下一组预训练检索优化实验总称，不是一个已经训练完成的模型。E5 只改变训练期 future-relation teacher 和下游 payload，不更换历史 encoder 或下游 backbone。
- `E5A`：T1-A 通过后保存的模型/Bank/下游 artifact 前缀，例如 `metrla_e5a_*`。其 teacher 只使用 OffsetDecay relation。
- `E5B`：T1-B 通过后保存的 artifact 前缀，例如 `metrla_e5b_*`。其 teacher 在 OffsetDecay relation 上增加 FutureIncrement relation。

这五个名称只是实验追踪标签，不参与任何张量计算，也不改变 future 信息边界。

E3 的训练期 future signature 具体为：

\[
S^{E3}_{i,h,n,c}
=\frac{Y_{i,h,n,c}-\mu^X_{i,n,c}}
{\sigma^X_{i,n,c}+\epsilon},
\]

其中 \(\mu^X\) 和 \(\sigma^X\) 只由该样本 288 步可见历史计算。E3 对 \(S^{E3}\) 做 pairwise masked MAE 并 softmax 得到 teacher；因此 E3 只在 source-train teacher 中使用 future，部署 selector 不使用 query future。

### 1.1 `EndpointLevel`：历史末端水平

`EndpointLevel` 指每个样本、节点、通道在可见历史末端的当前数值，记为：

\[
\alpha_{i,n,c}\in\mathbb R.
\]

其中：

- \(i\)：样本或历史事件索引；
- \(n\)：交通传感器节点；
- \(c\)：物理变量通道，当前速度数据中 \(C=1\)；
- \(\alpha_{i,n,c}\)：预测起点前最后一个有效历史值。

输入为最近 12 步 forecast context：

\[
X_i\in\mathbb R^{12\times N\times C}.
\]

如果第 12 步有效，直接取第 12 步；如果末端缺失，则在最近 12 步有效值上使用 `offset` 统计回退，即有效值均值。该统计只读取历史，不读取 query future。

### 1.2 `RawFuture`：原始历史未来轨迹

`RawFuture` 指历史事件 \(j\) 发生后真实观测到的 12 步 future：

\[
Y_j\in\mathbb R^{H\times N\times C},\qquad H=12.
\]

它在构建训练历史 Bank 时已经发生，因此推理时读取历史事件的 `RawFuture` 是因果的。当前 query 的 future \(Y_q\) 在推理时不可读取。

### 1.3 `Offset`：去除历史末端水平的未来变化

`Offset` 不是网络，而是一种 future 表示。对历史事件 \(i\)，定义：

\[
U^{\mathrm{off}}_{i,h,n,c}
=Y_{i,h,n,c}-\alpha_{i,n,c}.
\]

它表达“相对于事件当前水平，未来变化了多少”。输入是历史末端水平 \(\alpha_i\) 和该历史事件已经发生的 future \(Y_i\)，输出仍为 \([H,N,C]\)。

训练时可以使用训练样本 future 构造 teacher；推理时只能对 Bank 中已经发生的历史 future 计算 Offset，不能使用 query future。

### 1.4 `HorizonDecay`：随预测距离衰减的固定系数

`HorizonDecay` 指一个没有可学习参数的预测步系数：

\[
\lambda_h=1-\frac{h-1}{H-1},
\qquad h=1,\ldots,H.
\]

当 \(H=12\) 时：

- 第 1 步 \(\lambda_1=1\)；
- 中间预测步在 1 和 0 之间线性下降；
- 第 12 步 \(\lambda_{12}=0\)。

它的作用是：当前 query 的 level 对近端预测影响较强，预测越远，该 level 修正越不可靠，因此逐步退回 RawFuture。

`HorizonDecay` 不读取 future，不训练参数，也不根据 validation 自动拟合曲线。

### 1.5 `OffsetDecay`：随预测距离衰减的 level 对齐

`OffsetDecay` 是本文的核心 payload 机制。它不是模型名，也不是额外神经网络。

给定 E3/E5 selector 选出的历史事件集合 \(\mathcal K_{q,n}\) 和权重 \(w_{qjn}\)，RawFuture memory 为：

\[
\widehat Y^{\mathrm{raw}}_{q,h,n,c}
=\sum_{j\in\mathcal K_{q,n}}w_{qjn}Y_{j,h,n,c}.
\]

完整 Offset memory 为：

\[
\widehat Y^{\mathrm{off}}_{q,h,n,c}
=\alpha_{q,n,c}
+\sum_{j\in\mathcal K_{q,n}}w_{qjn}
\left(Y_{j,h,n,c}-\alpha_{j,n,c}\right).
\]

OffsetDecay 输出为：

\[
\boxed{
\widehat Y^{\mathrm{OD}}_{q,h,n,c}
=\widehat Y^{\mathrm{raw}}_{q,h,n,c}
+\lambda_h
\left(
\widehat Y^{\mathrm{off}}_{q,h,n,c}
-\widehat Y^{\mathrm{raw}}_{q,h,n,c}
\right)
}.
\]

将 \(\widehat Y^{\mathrm{off}}\) 展开后，OffsetDecay 也可以直接写成：

\[
\widehat Y^{\mathrm{OD}}_{q,h,n,c}
=\lambda_h\alpha_{q,n,c}
+\sum_{j\in\mathcal K_{q,n}}w_{qjn}
\left(Y_{j,h,n,c}-\lambda_h\alpha_{j,n,c}\right).
\]

这里是对历史候选项求**加权求和**，不是减去加权和。若公式渲染成
\(\alpha_q-\sum_jw_{qjn}(Y_j-\alpha_j)\)，那是排版错误；当前 T0 实现和本文采用的正确形式都是上面的加号。

输入包括 query history、历史 Bank future、历史候选 endpoint level、selector 权重；输出为 memory prediction \([B,H,N,C]\)。推理时全部输入都来自当前历史和已经发生的 Bank 历史，不读取 query future。

本文不使用 `HorizonOffset` 作为另一个独立方法名。若旧笔记中出现 `HorizonOffset`，它只表示“按 horizon 衰减的 offset 修正”，本文统一称为 `OffsetDecay`，避免同一机制出现两个名字。

### 1.6 `DeploymentAlignedOffsetDecaySignature`：与推理对齐的训练期 future signature

`DeploymentAlignedOffsetDecaySignature` 可译为“与部署推理对齐的 OffsetDecay 未来表示”，简称 `ODSignature`。这是训练期 teacher 使用的 future signature，不是推理网络。

对训练事件 \(i\)，定义：

\[
\boxed{
S^{\mathrm{OD}}_{i,h,n,c}
=Y_{i,h,n,c}-\lambda_h\alpha_{i,n,c}
}.
\]

其来源是将 OffsetDecay 推理公式整理为：

\[
\widehat Y^{\mathrm{OD}}_{q,h,n,c}
=\lambda_h\alpha_{q,n,c}
+\sum_jw_{qjn}S^{\mathrm{OD}}_{j,h,n,c}.
\]

因此训练时让 key 学习 \(S^{\mathrm{OD}}\) 的相似关系，与推理时真正被加权检索的对象一致。

需要区分实现阶段：T0 只执行上一小节的推理 payload；T1-A 预训练才使用下面的 \(S^{\mathrm{OD}}\) 作为 teacher signature。T1-A 尚未运行前，不能把 E3 checkpoint 称为 OffsetDecay-pretrained checkpoint。

重要区别：不能先计算 \(Y_i-\alpha_i\)，再仅用 \(\lambda_h\) 加权距离。那种做法会在远期忽略 future relation；本文的 \(Y_i-\lambda_h\alpha_i\) 会在远期恢复 RawFuture relation。

`ODSignature` 只在源域预训练 teacher 中读取训练 future。teacher 在 `torch.no_grad()` 下构造；梯度只更新历史 encoder 和 retrieval head。推理时不构造 query `ODSignature`。

### 1.6.1 `AnchorMeanDistanceNormalization`：按锚点校准 teacher 距离尺度

`AnchorMeanDistanceNormalization` 可译为“按锚点平均距离归一化”。这里的 `anchor` 是发起关系匹配的样本-节点对 \((i,n)\)，候选是同一 batch 中满足 non-self、future non-overlap 和观测有效条件的样本 \(j\)。它不是神经网络，也不增加推理步骤。

原始 signature 处在交通速度物理单位，而 E3 的 `relation_teacher_temperature=0.1` 面向归一化 future。直接把原始速度距离除以 `0.1` 会使 teacher 极易坍缩为近 one-hot，导致实验实际比较的是距离量纲，而不是 future relation。因此对每个有效 anchor 的候选距离执行：

\[
\bar d^m_{i,n}
=\frac{1}{|\mathcal C_{i,n}|}
\sum_{j\in\mathcal C_{i,n}}d^m_{ij,n},
\]

\[
\boxed{
\widetilde d^m_{ij,n}
=\frac{d^m_{ij,n}}{\bar d^m_{i,n}+\epsilon}
},
\qquad m\in\{\mathrm{OD},\mathrm{inc}\}.
\]

输入是 teacher 内部已经计算出的 pairwise masked MAE 和候选 mask，输出是同形状的无量纲距离 `[B,B,N]`。均值和除法都在 `torch.no_grad()` 中完成；无有效候选的 anchor 保持无效，平均距离接近 0 时用 \(\epsilon\) 保证有限值。该变换只读取 source-train future 形成的训练期距离，推理完全不运行。

它保留同一 anchor 内候选的距离排序，并使“正比例缩放整个物理变量”不会改变 teacher distribution。T1-A 对 OD distance 使用该归一化；T1-B 对 OD 和 FutureIncrement distance 分别归一化后再组合，因此固定 \(\eta=0.5\) 才表示两个关系各占一半，而不是被原始数值尺度支配。

### 1.7 `T1-A`：OffsetDecay teacher 对齐阶段

`T1-A` 是本文第一阶段的实验标签，完整含义是：

> 保持 E3 网络结构、掩码重建、student key distribution 和训练预算不变，只把 E3 的 future teacher 改为经过 anchor-mean 尺度校准的 `ODSignature` relation，并在下游使用 OffsetDecay memory。

它回答：训练期 future relation 与推理 payload 对齐后，是否能让历史 key 更准确地检索可迁移 future。

T1-A 不加入 future increment，不加入 future encoder，不加入 confidence。

### 1.8 `FutureIncrement`：真实未来变化量

`FutureIncrement` 指真实 future 相邻预测步之间的变化，记为 \(G_i\)：

\[
G_{i,1,n,c}=Y_{i,1,n,c}-\alpha_{i,n,c},
\]

\[
G_{i,h,n,c}=Y_{i,h,n,c}-Y_{i,h-1,n,c},
\qquad h=2,\ldots,H.
\]

它不是从历史拟合斜率，也不是把一条直线外推到未来。它直接来自训练样本真实 future，用于描述未来上升、下降、拥堵形成、拥堵释放和方向反转。

输入为训练事件 endpoint level 和真实 future，输出为 \([H,N,C]\)。它只在预训练 teacher 和离线指标中使用；推理 query 不可访问自己的 FutureIncrement。

### 1.9 `OffsetDecayIncrementTeacher`：OffsetDecay 与未来变化量联合 teacher

`OffsetDecayIncrementTeacher` 是 T1-B 使用的 teacher，中文含义是“OffsetDecay 关系与真实未来变化关系的联合监督”。

先分别计算节点级 masked MAE：

\[
d^{\mathrm{OD}}_{ij,n}
=\operatorname{MaskedMAE}
\left(S^{\mathrm{OD}}_{i,:,n,:},S^{\mathrm{OD}}_{j,:,n,:}\right),
\]

\[
d^{\mathrm{inc}}_{ij,n}
=\operatorname{MaskedMAE}
\left(G_{i,:,n,:},G_{j,:,n,:}\right).
\]

分别使用 `AnchorMeanDistanceNormalization` 得到 \(\widetilde d^{\mathrm{OD}}\) 和 \(\widetilde d^{\mathrm{inc}}\)，再使用固定权重 \(\eta=0.5\) 构造联合距离：

\[
\boxed{
\widetilde d^{\mathrm{OD+Inc}}_{ij,n}
=(1-\eta)\widetilde d^{\mathrm{OD}}_{ij,n}
+\eta \widetilde d^{\mathrm{inc}}_{ij,n},
\qquad \eta=0.5
}.
\]

`eta` 表示经过尺度校准后的 FutureIncrement relation 在联合 teacher 中的固定占比。第一轮不通过 validation 搜索 `eta`，避免把 T1-B 变成多超参数实验。

### 1.10 `T1-B`：真实未来变化监督阶段

`T1-B` 是第二阶段实验标签，完整含义是：

> 在已经通过门槛的 T1-A 上，只把 teacher distance 从 \(\widetilde d^{\mathrm{OD}}\) 改为 \(\widetilde d^{\mathrm{OD+Inc}}\)，其余模型、Bank、下游和训练预算保持不变。

它回答：在训练部署已经对齐后，真实 future 的逐步变化方向能否进一步改善 selector，尤其是中远期预测。

T1-B 只有在 T1-A 通过时才允许执行。

### 1.11 `learned_topk_offset_decay_horizon`：无 confidence 下游模式

`learned_topk_offset_decay_horizon` 是计划新增的下游模式名，含义是：

1. 使用冻结的 E3/E5 encoder node key 选择每个节点的 learned Top-K 历史事件；
2. 使用相同 learned weights 构造 OffsetDecay memory \(\widehat Y^{\mathrm{OD}}\)；
3. 不运行 confidence head；
4. 只训练轻量 backbone 和每个 horizon 一个共享融合权重。

下游最终预测为：

\[
\widehat Y^{\mathrm{final}}_{q,h,n,c}
=\widehat Y^{\mathrm{base}}_{q,h,n,c}
+a_h
\left(
\widehat Y^{\mathrm{OD}}_{q,h,n,c}
-\widehat Y^{\mathrm{base}}_{q,h,n,c}
\right),
\]

其中 \(a_h\in(0,1)\) 是下游 validation 训练得到的 horizon-only fusion weight。

必须区分：

- \(\lambda_h\)：OffsetDecay 内部固定的 level 修正衰减，不训练；
- \(a_h\)：memory 与 backbone 的下游融合权重，需要训练；
- 本模式没有节点级、样本级 confidence。

### 1.12 `Confidence`：被本方案禁用的样本级检索修正

`Confidence` 指现有下游中的可学习可信度 \(c_{q,h,n}\in[0,1]\)，它根据 Top-K shape score、分数间隔、权重集中度、历史 future 分歧、level match 和 memory/base 分歧六类特征，预测当前 query、horizon、节点是否应信任 memory。现有 confidence 模式的融合为：

\[
\widehat Y^{\mathrm{final}}_{q,h,n,c}
=\widehat Y^{\mathrm{base}}_{q,h,n,c}
+a_hc_{q,h,n}
\left(
\widehat Y^{\mathrm{memory}}_{q,h,n,c}
-\widehat Y^{\mathrm{base}}_{q,h,n,c}
\right).
\]

训练时 confidence soft target 会比较 memory 与 base 相对真实 target 的误差收益，因此训练标签使用 train future；推理时 confidence 输入本身不读取 query future。它输出 `[B,H,N,1]` 的乘法门控。本文禁用该 head，即有效 memory 位置固定 \(c=1\)、无效位置 \(c=0\)，只保留 horizon fusion \(a_h\)。这样可以直接判断预训练 selector 和 OffsetDecay memory 是否有效，避免 confidence 掩盖检索失败。

## 2. 当前证据与目标失败

T0 在同一 E3 selector、同一 Bank、同一 learned weights、无 confidence 条件下得到：

| Dataset | RawFuture MAE | OffsetDecay MAE | 相对改善 | LinearTrend MAE |
|---|---:|---:|---:|---:|
| METR-LA | 3.8027 | 3.6231 | +4.72% | 5.3743 |
| PEMS-BAY | 2.2028 | 1.9305 | +12.36% | 2.7904 |

已确认的失败：

- 历史局部线性斜率向未来外推在两个数据集都恶化；
- candidate/query local scale transfer 会因接近零的候选尺度产生极端放大；
- fixed past-residual Pearson 不足以替代 learned selector；
- 当前 E3 预训练相对 random 的迁移优势仍弱。

因此下一步缺失能力不是“更复杂的趋势网络”，而是：

> 当前 E3 teacher 学习的 future relation 与实际有效的 OffsetDecay payload 不一致，encoder 没有被直接监督去学习部署阶段真正需要的 future relation。

## 3. 三种候选路线与选择

### 路线 A：部署对齐的 `ODSignature` relation teacher

- 改动：修改 teacher signature，并用无参数 anchor-mean normalization 消除原始速度量纲；二者共同定义一个新的 relation teacher；
- 参数增量：0；
- 推理开销增量：0；
- 优点：训练和部署严格对齐，可直接归因；
- 风险：可能只修复 level relation，仍不能区分拥堵形成与释放。

结论：作为 T1-A，优先执行。

### 路线 B：`ODSignature + FutureIncrement`

- 改动：在 T1-A teacher distance 上增加真实 future increment distance；
- 参数增量：0；
- 推理开销增量：0；
- 优点：直接监督真实未来变化方向，不依赖历史斜率外推；
- 风险：increment 噪声可能降低 teacher 稳定性。

结论：作为 T1-B，仅在 T1-A 成功后执行。

### 路线 C：Learned Future Encoder

- 改动：用额外 Transformer、TimeMixer、DWT 或 temporal CNN 编码训练 future；
- 参数和训练成本：增加；
- 风险：需要防止 teacher collapse，收益难以归因，论文复杂度增加；
- 当前证据：固定 teacher 尚未失败，缺少增加网络的必要性。

结论：本方案不实现。T1-A/T1-B 均失败时直接停止 E5，而不是自动开放路线 C。

## 4. 数据、张量与未来信息边界

### 4.1 数据协议

- 源预训练：METR-LA train；
- 同域选择：METR-LA validation；
- 跨域检验：PEMS-BAY validation；
- 目标 Bank：只使用 PEMS-BAY training history；
- 数据变量：两者均为交通速度；
- 采样间隔：5 分钟；
- retrieval context：288 步，即一天；
- endpoint context：最近 12 步；
- prediction horizon：12 步，即 60 分钟；
- 主筛选 seed：42；
- 稳定性 seed：42、2024、2025，仅在单 seed 门槛通过后运行。

### 4.2 张量契约

| 张量 | Shape | 来源 | 用途 |
|---|---|---|---|
| `retrieval_x` | `[B,288,N,C]` | query/source history | encoder 输入 |
| `x` | `[B,12,N,C]` | retrieval history 的预测端尾部 | endpoint level |
| `y` | `[B,12,N,C]` | source train future | teacher；推理不可见 |
| `node_keys` | `[B,N,D_r]` | history encoder | student relation |
| `S_OD` | `[B,12,N,C]` | `y - lambda * alpha` | T1-A teacher signature |
| `FutureIncrement` | `[B,12,N,C]` | train future differences | T1-B teacher signature |
| `teacher_distribution` | `[B,B,N]` | future pair distance | 监督 key relation |
| `memory_prediction` | `[B,12,N,C]` | causal Bank | 无 confidence 下游 memory |

### 4.3 未来信息边界

允许使用 future 的位置：

- METR-LA source train 中构造 teacher；
- 构建历史 Bank 时保存已经发生的 training future；
- validation/test 离线计算指标；
- oracle 诊断，必须标记为不可部署。

禁止使用 future 的位置：

- query inference selector；
- query endpoint level；
- OffsetDecay 推理修正；
- Bank candidate filtering；
- validation/test future 写入 Bank；
- confidence 输入，本方案本身也不运行 confidence。

合法 Bank 候选始终满足：

\[
\operatorname{futureEnd}(j)<\operatorname{contextStart}(q).
\]

## 5. T1-A 与 T1-B 的训练目标

### 5.1 共同 student distribution

历史 encoder 产生 node key：

\[
k_{i,n}\in\mathbb R^{D_r}.
\]

节点级 student 分布为：

\[
p_{ij,n}
=\operatorname{Softmax}_j
\left(
\frac{\operatorname{cos}(k_{i,n},k_{j,n})}{\tau_S}
\right).
\]

只在 non-self、future non-overlap、共同有效的 batch 内候选上归一化。

### 5.2 T1-A teacher distribution

使用 `ODSignature` 距离：

\[
d^{\mathrm{OD}}_{ij,n}
=\frac{
\sum_{h,c}\mathcal O_{ijhnc}
\left|S^{\mathrm{OD}}_{i,h,n,c}-S^{\mathrm{OD}}_{j,h,n,c}\right|
}{
\sum_{h,c}\mathcal O_{ijhnc}+\epsilon
}.
\]

按第 1.6.1 节的公式在每个有效 anchor 内计算无量纲距离 \(\widetilde d^{\mathrm{OD}}\)。该步骤只改变数值尺度，不改变同一 anchor 内候选的排序。

teacher 分布为：

\[
q^{\mathrm{OD}}_{ij,n}
=\operatorname{Softmax}_j
\left(-\widetilde d^{\mathrm{OD}}_{ij,n}/\tau_T\right).
\]

relation loss 为：

\[
\mathcal L_{\mathrm{ODR}}
=-\frac{1}{|\mathcal A|}
\sum_{(i,n)\in\mathcal A}
\sum_j q^{\mathrm{OD}}_{ij,n}\log p_{ij,n}.
\]

总预训练损失继续使用：

\[
\mathcal L
=\mathcal L_{\mathrm{mask}}
+0.1\mathcal L_{\mathrm{ODR}}.
\]

除上述 relation teacher 定义外，E3 的 encoder、掩码重建、student distribution、loss weight、训练预算和数据协议均不改变。

### 5.3 T1-B teacher distribution

将 T1-A 的 \(\widetilde d^{\mathrm{OD}}\) 替换为第 1.9 节定义的：

\[
\widetilde d^{\mathrm{OD+Inc}}
=0.5\widetilde d^{\mathrm{OD}}+0.5\widetilde d^{\mathrm{inc}}.
\]

teacher distribution 和 cross-entropy 形式不变。T1-B 不增加第二个 loss weight，不训练 future branch。

## 6. 实现前置任务

以下任务已完成工程接入和测试验证；本节保留为实现契约与复核清单。第 10 节以后的 T1-A 正式命令已经具备运行条件；第 17 节 T1-B 命令仍必须等待第 16 节门控通过。

### Task 1：新增 teacher mode 配置

**Files:**

- Modify: `stanchor/config.py`
- Create: `configs/metrla_e5_offset_decay_relation_v1.yaml`
- Create: `configs/metrla_e5_offset_decay_relation_level0_v1.yaml`
- Create: `configs/metrla_e5_offset_decay_increment_relation_v1.yaml`
- Create: `configs/metrla_e5_offset_decay_increment_relation_level0_v1.yaml`
- Create: `configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml`
- Create: `configs/pemsbay_e5_offset_decay_increment_transfer_level0_v1.yaml`

新增配置字段：

```yaml
pretrain:
  retrieval_loss_mode: relation
  relation_teacher_mode: offset_decay
  relation_distance_normalization: anchor_mean
  future_increment_weight: 0.0
```

允许值及其当前文档内含义：

- `context_normalized`：当前 E3 teacher，使用 context mean/std 归一化 future；
- `offset_decay`：T1-A，使用 `ODSignature = y - lambda * alpha`；
- `offset_decay_increment`：T1-B，使用 `0.5 * normalized OD distance + 0.5 * normalized FutureIncrement distance`。

`relation_distance_normalization: anchor_mean` 表示对每个样本-节点 anchor 的有效候选距离除以该 anchor 的候选平均距离。它是第 1.6.1 节定义的训练期无参数尺度校准，推理不运行。E5 固定为 `anchor_mean`，本轮不搜索其他归一化方式。

T1-B 配置固定：

```yaml
pretrain:
  relation_teacher_mode: offset_decay_increment
  relation_distance_normalization: anchor_mean
  future_increment_weight: 0.5
```

配置必须拒绝未知 mode、负权重和大于 1 的权重。

### Task 2：实现 teacher signature 与 relation distance

**Files:**

- Modify: `stanchor/losses/pretraining.py`
- Modify: `stanchor/engine/pretrainer.py`
- Test: `tests/test_future_relation_loss.py`
- Test: `tests/test_pretraining_flow.py`

需要新增的纯函数契约：

```python
def build_offset_decay_signature(
    future_model: torch.Tensor,       # [B,H,N,C]
    future_observed: torch.Tensor,    # [B,H,N,C]
    forecast_context: torch.Tensor,   # [B,T,N,C]
    context_observed: torch.Tensor,   # [B,T,N,C]
) -> tuple[torch.Tensor, torch.Tensor]: ...

def build_future_increment(
    future_model: torch.Tensor,
    future_observed: torch.Tensor,
    endpoint_level: torch.Tensor,     # [B,N,C]
    endpoint_valid: torch.Tensor,     # [B,N,C]
) -> tuple[torch.Tensor, torch.Tensor]: ...
```

必须测试：

- H1 等于 `future - endpoint`；
- H12 等于 RawFuture；
- 中间 horizon 使用固定线性 \(\lambda_h\)；
- FutureIncrement 第一步从 endpoint 出发，之后使用相邻 future 差；
- OD 与 FutureIncrement 距离分别执行 anchor-mean normalization；
- 将 signature 乘任意正常数不改变 teacher distribution；
- `eta=0.5` 在两个归一化距离上组合，而不是在原始速度距离上组合；
- 缺失 endpoint、缺失相邻 future、空候选均不会产生 NaN；
- teacher 在 `no_grad` 中，梯度只到 `node_keys`；
- `context_normalized` 完全复现 E3 原行为。

### Task 3：实现 OffsetDecay 下游聚合模式

**Files:**

- Modify: `stanchor/modes.py`
- Modify: `stanchor/retrieval/strategies.py`
- Modify: `stanchor/engine/target.py`
- Test: `tests/test_retrieval_strategies.py`
- Test: `tests/test_downstream_flow.py`

新增 mode：

```python
LEARNED_TOPK_OFFSET_DECAY_HORIZON = "learned_topk_offset_decay_horizon"
```

该 mode 必须：

- 复用 learned event/node Top-K；
- 保持 learned weights 不变；
- 从 causal Bank history 计算 candidate endpoint；
- 从 query 最近 12 步历史计算 query endpoint；
- 输出 OffsetDecay memory；
- confidence 恒为 1 或无效位置为 0，不训练 confidence head；
- 只训练 backbone 和 horizon fusion；
- 空候选时 final prediction 精确回退到 base。

### Task 4：补充预训练 CLI seed override

**Files:**

- Modify: `scripts/pretrain.py`
- Test: `tests/test_pretraining_cli.py`

新增：

```text
--seed <int>
```

它只覆盖 `runtime.seed`，用于门槛通过后的预训练 seed 稳定性实验。单 seed 42 阶段不依赖该参数。

### Task 5：回归验证

```powershell
conda run -n research python -m unittest discover -s tests -v
conda run -n research python -m compileall -q stanchor scripts tests
```

预期：全量测试通过，`compileall` exit code 为 0；现有 E3 `context_normalized` 测试结果不变。

## 7. 统一环境命令

从 PowerShell 执行：

```powershell
conda activate research
Set-Location -LiteralPath D:\projects\researchProjects\TrafficRobustST\STAnchor-BlockMemory
```

配置检查：

```powershell
python -c "from stanchor.config import load_config; a=load_config('configs/metrla_e5_offset_decay_relation_v1.yaml'); b=load_config('configs/metrla_e5_offset_decay_increment_relation_v1.yaml'); print(a.pretrain.relation_teacher_mode, a.pretrain.relation_distance_normalization, a.pretrain.future_increment_weight); print(b.pretrain.relation_teacher_mode, b.pretrain.relation_distance_normalization, b.pretrain.future_increment_weight)"
```

预期：

```text
offset_decay anchor_mean 0.0
offset_decay_increment anchor_mean 0.5
```

## 8. Stage 0：重新建立 E3 Level-0 对照

`Level-0` 表示 node reranking 中手工 level similarity 的权重设为 0。这样排序只由 learned key 决定，用于隔离预训练表示本身的贡献。它不改变 checkpoint 或 Bank。

T0 的旧正式结果使用 `level_weight=0.25`，不能直接作为 T1 learned-key 主对照，因此先运行两个只读诊断：

```powershell
python scripts/diagnose_trend_residual.py `
  --config configs/metrla_e3_relation_level0_v1.yaml `
  --checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e3_relation_relation `
  --split val `
  --trend-length 12 `
  --output artifacts/e5_t1_baselines/metrla_e3_level0_offset_decay_val.json
```

```powershell
python scripts/diagnose_trend_residual.py `
  --config configs/pemsbay_e3_transfer_level0_v1.yaml `
  --checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt `
  --bank artifacts/pemsbay_bank_from_metrla_e3_relation `
  --split val `
  --trend-length 12 `
  --output artifacts/e5_t1_baselines/pemsbay_e3_level0_offset_decay_val.json
```

这两份 JSON 是 T1-A/T1-B 的固定 selector baseline。后续不得根据 E5 结果重新切换 E3 baseline。

## 9. T1-A Stage 1：工程 smoke

`Smoke` 表示短训练工程检查，不是正式实验，不得把其指标写入论文结果。

```powershell
python scripts/pretrain.py `
  --config configs/metrla_e5_offset_decay_relation_v1.yaml `
  --epochs 2 `
  --max-batches 4 `
  --seed 42 `
  --run-name metrla_e5a_offset_decay_smoke_seed42
```

检查：

- 日志显示 `relation_teacher_mode=offset_decay`；
- total/reconstruction/retrieval loss 有限；
- `valid_retrieval_anchors > 0`；
- `relation_candidate_pairs > 0`；
- `teacher_effective_support` 和 `student_effective_support` 有限；
- 保存/加载 checkpoint 指纹一致；
- 没有 query future 进入 encode/inference path。

通过后删除 smoke 目录。删除前验证绝对路径：

```powershell
$ArtifactRoot = (Resolve-Path artifacts).Path
$SmokePath = (Resolve-Path artifacts/metrla_e5a_offset_decay_smoke_seed42).Path
if ((Split-Path -Parent $SmokePath) -ne $ArtifactRoot) { throw "Unexpected smoke path: $SmokePath" }
Remove-Item -LiteralPath $SmokePath -Recurse
```

## 10. T1-A Stage 2：METR-LA 正式 seed 42 预训练

```powershell
python scripts/pretrain.py `
  --config configs/metrla_e5_offset_decay_relation_v1.yaml `
  --seed 42 `
  --run-name metrla_e5a_offset_decay_seed42
```

正式目录：

```text
artifacts/metrla_e5a_offset_decay_seed42/
  pretrain.log
  pretrain_metrics.jsonl
  pretrain_best.pt
  pretrain_best_relation.pt
```

两个 checkpoint 的含义：

- `pretrain_best.pt`：validation 总预训练损失最低；
- `pretrain_best_relation.pt`：validation relation loss 最低。

不能仅根据 relation loss 选择 checkpoint。两者都进入 METR-LA validation 检索诊断。

## 11. T1-A Stage 3：两个 checkpoint 建立 METR-LA Bank

```powershell
python scripts/build_bank.py `
  --config configs/metrla_e5_offset_decay_relation_level0_v1.yaml `
  --checkpoint artifacts/metrla_e5a_offset_decay_seed42/pretrain_best.pt `
  --output-dir artifacts/metrla_bank_e5a_offset_decay_total_seed42 `
  --dataset-name METR-LA
```

```powershell
python scripts/build_bank.py `
  --config configs/metrla_e5_offset_decay_relation_level0_v1.yaml `
  --checkpoint artifacts/metrla_e5a_offset_decay_seed42/pretrain_best_relation.pt `
  --output-dir artifacts/metrla_bank_e5a_offset_decay_relation_seed42 `
  --dataset-name METR-LA
```

每个 Bank 必须与对应 checkpoint fingerprint 完全匹配，禁止交叉混用。

## 12. T1-A Stage 4：METR-LA validation 选择 checkpoint

```powershell
python scripts/diagnose_trend_residual.py `
  --config configs/metrla_e5_offset_decay_relation_level0_v1.yaml `
  --checkpoint artifacts/metrla_e5a_offset_decay_seed42/pretrain_best.pt `
  --bank artifacts/metrla_bank_e5a_offset_decay_total_seed42 `
  --split val `
  --trend-length 12 `
  --output artifacts/e5_t1a/metrla_e5a_total_level0_val.json
```

```powershell
python scripts/diagnose_trend_residual.py `
  --config configs/metrla_e5_offset_decay_relation_level0_v1.yaml `
  --checkpoint artifacts/metrla_e5a_offset_decay_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5a_offset_decay_relation_seed42 `
  --split val `
  --trend-length 12 `
  --output artifacts/e5_t1a/metrla_e5a_relation_level0_val.json
```

选择规则：

1. 主指标为 `learned_offset_decay_topk.mae`，越低越好；
2. 次指标为 `learned_offset_decay_topk.rmse`；
3. 若 MAE 差异小于 `0.01`，选择 `learned_raw_topk.mae` 更低者；
4. coverage 差异不得超过 `0.2` 个百分点；
5. 只使用 METR-LA validation 选择 checkpoint；不得使用 PEMS-BAY 或任一 test 反向选择。

选择后在当前 PowerShell 会话冻结变量。例如，若 relation checkpoint 胜出：

```powershell
$E5A_CHECKPOINT = "artifacts/metrla_e5a_offset_decay_seed42/pretrain_best_relation.pt"
$E5A_METR_BANK = "artifacts/metrla_bank_e5a_offset_decay_relation_seed42"
$E5A_CHECKPOINT_KIND = "relation"
```

若 total checkpoint 胜出，只能改成：

```powershell
$E5A_CHECKPOINT = "artifacts/metrla_e5a_offset_decay_seed42/pretrain_best.pt"
$E5A_METR_BANK = "artifacts/metrla_bank_e5a_offset_decay_total_seed42"
$E5A_CHECKPOINT_KIND = "total"
```

该选择一旦冻结，后续 PEMS-BAY 和 downstream 不得切换。

## 13. T1-A Stage 5：PEMS-BAY target-local Bank 与跨域诊断

`Target-local Bank` 表示 Bank 的 keys 由冻结源 encoder 计算，但 payload、节点集合、图、scaler 和历史 future 全部来自 PEMS-BAY training history；它不是把 METR-LA future 搬到 PEMS-BAY。

在新 PowerShell 会话中必须重新设置 checkpoint 变量。若 METR-LA validation 选择了 relation-loss checkpoint，执行：

```powershell
$E5A_CHECKPOINT = "artifacts/metrla_e5a_offset_decay_seed42/pretrain_best_relation.pt"
$E5A_CHECKPOINT_KIND = "relation"
```

若选择了 total-loss checkpoint，执行：

```powershell
$E5A_CHECKPOINT = "artifacts/metrla_e5a_offset_decay_seed42/pretrain_best.pt"
$E5A_CHECKPOINT_KIND = "total"
```

只能执行与已冻结 METR-LA validation 选择一致的一组赋值。然后构建 PEMS-BAY Bank：

```powershell
$E5A_PEMS_BANK = "artifacts/pemsbay_bank_e5a_offset_decay_$($E5A_CHECKPOINT_KIND)_seed42"

python scripts/build_bank.py `
  --config configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml `
  --checkpoint $E5A_CHECKPOINT `
  --output-dir $E5A_PEMS_BANK `
  --dataset-name PEMS-BAY
```

```powershell
python scripts/diagnose_trend_residual.py `
  --config configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml `
  --checkpoint $E5A_CHECKPOINT `
  --bank $E5A_PEMS_BANK `
  --split val `
  --trend-length 12 `
  --output artifacts/e5_t1a/pemsbay_e5a_level0_val.json
```

## 14. T1-A Retrieval 决策门

设 E3 Level-0 OffsetDecay MAE 为 \(M^{E3}_{D}\)，T1-A 为 \(M^{A}_{D}\)，数据集 \(D\in\{\text{METR},\text{PEMS}\}\)。相对改善为：

\[
G^{A}_{D}
=\frac{M^{E3}_{D}-M^{A}_{D}}{M^{E3}_{D}}\times100\%.
\]

进入无 confidence downstream 的条件：

- METR-LA 与 PEMS-BAY 的 \(G^{A}_{D}\) 均至少为 `0.5%`；
- 至少一个数据集的改善达到 `1.0%`；
- 两个数据集的 OffsetDecay RMSE 均不恶化超过 `0.5%`；
- 两个数据集至少 8/12 个 horizon 的 MAE 不劣于 E3；
- common coverage 相对 E3 下降不超过 `0.2` 个百分点；
- 所有指标均为有限值，Bank/checkpoint fingerprint 匹配。

决策：

- `Keep`：全部满足，进入第 15 节；
- `Remove`：仅 teacher loss 下降但实际 retrieval 未过门，删除 T1-A；
- `Stop`：任一数据集 OffsetDecay MAE 恶化，不运行 T1-B，不新增 future encoder。

## 15. T1-A Stage 6：无 confidence downstream seed 42

本阶段比较四种来源：

- `base_only`：只有 12 步轻量 backbone，没有 memory；
- `E3 OffsetDecay`：E3 selector + OffsetDecay memory；
- `E5A OffsetDecay`：T1-A selector + OffsetDecay memory；
- `random OffsetDecay`：相同结构但未经预训练的 random selector + OffsetDecay memory。

### 15.1 METR-LA 命令

```powershell
python scripts/train_downstream.py `
  --config configs/metrla_e5_offset_decay_relation_level0_v1.yaml `
  --pretrained-checkpoint $E5A_CHECKPOINT `
  --bank $E5A_METR_BANK `
  --mode base_only `
  --seed 42 `
  --run-name metrla_e5a_base_only_seed42
```

```powershell
python scripts/train_downstream.py `
  --config configs/metrla_e5_offset_decay_relation_level0_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e3_relation_relation `
  --mode learned_topk_offset_decay_horizon `
  --seed 42 `
  --run-name metrla_e3_offset_decay_horizon_seed42
```

```powershell
python scripts/train_downstream.py `
  --config configs/metrla_e5_offset_decay_relation_level0_v1.yaml `
  --pretrained-checkpoint $E5A_CHECKPOINT `
  --bank $E5A_METR_BANK `
  --mode learned_topk_offset_decay_horizon `
  --seed 42 `
  --run-name metrla_e5a_offset_decay_horizon_seed42
```

```powershell
python scripts/train_downstream.py `
  --config configs/metrla_e5_offset_decay_relation_level0_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e3_target_random_seed42/random_checkpoint.pt `
  --bank artifacts/metrla_bank_e3_target_random_seed42 `
  --mode learned_topk_offset_decay_horizon `
  --seed 42 `
  --run-name metrla_random_offset_decay_horizon_seed42
```

### 15.2 PEMS-BAY 命令

```powershell
python scripts/train_downstream.py `
  --config configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml `
  --pretrained-checkpoint $E5A_CHECKPOINT `
  --bank $E5A_PEMS_BANK `
  --mode base_only `
  --seed 42 `
  --run-name pemsbay_e5a_base_only_seed42
```

```powershell
python scripts/train_downstream.py `
  --config configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt `
  --bank artifacts/pemsbay_bank_from_metrla_e3_relation `
  --mode learned_topk_offset_decay_horizon `
  --seed 42 `
  --run-name pemsbay_e3_offset_decay_horizon_seed42
```

```powershell
python scripts/train_downstream.py `
  --config configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml `
  --pretrained-checkpoint $E5A_CHECKPOINT `
  --bank $E5A_PEMS_BANK `
  --mode learned_topk_offset_decay_horizon `
  --seed 42 `
  --run-name pemsbay_e5a_offset_decay_horizon_seed42
```

```powershell
python scripts/train_downstream.py `
  --config configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml `
  --pretrained-checkpoint artifacts/pemsbay_e3_target_random_seed42/random_checkpoint.pt `
  --bank artifacts/pemsbay_bank_target_random_seed42 `
  --mode learned_topk_offset_decay_horizon `
  --seed 42 `
  --run-name pemsbay_random_offset_decay_horizon_seed42
```

### 15.3 下游分支诊断

```powershell
python scripts/diagnose_downstream.py `
  --config configs/metrla_e5_offset_decay_relation_level0_v1.yaml `
  --pretrained-checkpoint $E5A_CHECKPOINT `
  --downstream-checkpoint artifacts/metrla_e5a_offset_decay_horizon_seed42/downstream_best.pt `
  --bank $E5A_METR_BANK `
  --split val `
  --output artifacts/e5_t1a/metrla_e5a_offset_decay_downstream_seed42_val.json
```

```powershell
python scripts/diagnose_downstream.py `
  --config configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml `
  --pretrained-checkpoint $E5A_CHECKPOINT `
  --downstream-checkpoint artifacts/pemsbay_e5a_offset_decay_horizon_seed42/downstream_best.pt `
  --bank $E5A_PEMS_BANK `
  --split val `
  --output artifacts/e5_t1a/pemsbay_e5a_offset_decay_downstream_seed42_val.json
```

## 16. T1-A 下游决策门

T1-A 允许进入 T1-B 的条件：

- E5A 的 `memory_mae` 在两个数据集都低于 E3；
- E5A 的 final validation MAE 在两个数据集都低于 E3，且相对改善至少 `0.3%`；
- E5A final MAE 在两个数据集都低于 random；
- E5A final MAE 在两个数据集都低于对应 `base_only`；
- 15、30、60 分钟 MAE 中至少 2 个位置不劣于 E3；
- confidence loss 必须为 0，confidence head trainable parameters 必须为 0；
- 推理参数量相对 E3 不增加，单 batch 检索延迟增加不超过 `5%`。

决策：

- `Keep T1-A`：全部满足；
- `Open T1-B`：只有 `Keep T1-A` 后才允许；
- `Stop E5`：任一数据集不如 E3 或 random，不进行 T1-B；
- 不允许使用 confidence 把未过门的结果修正为通过。

## 17. T1-B：FutureIncrement 条件实验

本节只有第 16 节通过后执行。T1-B 除 teacher distance 外，完全复用 T1-A。

### 17.1 Smoke

```powershell
python scripts/pretrain.py `
  --config configs/metrla_e5_offset_decay_increment_relation_v1.yaml `
  --epochs 2 `
  --max-batches 4 `
  --seed 42 `
  --run-name metrla_e5b_offset_decay_increment_smoke_seed42
```

检查 FutureIncrement mask、finite loss 和 gradient 后删除 smoke 目录。`Smoke` 目录只包含最多 4 个 batch、2 个 epoch 的工程检查结果，不属于正式实验资产。删除前必须解析并核对绝对路径：

```powershell
$ArtifactRoot = (Resolve-Path artifacts).Path
$SmokePath = (Resolve-Path artifacts/metrla_e5b_offset_decay_increment_smoke_seed42).Path
if ((Split-Path -Parent $SmokePath) -ne $ArtifactRoot) { throw "Unexpected smoke path: $SmokePath" }
Remove-Item -LiteralPath $SmokePath -Recurse
```

### 17.2 正式 seed 42 预训练

```powershell
python scripts/pretrain.py `
  --config configs/metrla_e5_offset_decay_increment_relation_v1.yaml `
  --seed 42 `
  --run-name metrla_e5b_offset_decay_increment_seed42
```

### 17.3 两个 METR-LA Bank

```powershell
python scripts/build_bank.py `
  --config configs/metrla_e5_offset_decay_increment_relation_level0_v1.yaml `
  --checkpoint artifacts/metrla_e5b_offset_decay_increment_seed42/pretrain_best.pt `
  --output-dir artifacts/metrla_bank_e5b_offset_decay_increment_total_seed42 `
  --dataset-name METR-LA
```

```powershell
python scripts/build_bank.py `
  --config configs/metrla_e5_offset_decay_increment_relation_level0_v1.yaml `
  --checkpoint artifacts/metrla_e5b_offset_decay_increment_seed42/pretrain_best_relation.pt `
  --output-dir artifacts/metrla_bank_e5b_offset_decay_increment_relation_seed42 `
  --dataset-name METR-LA
```

### 17.4 METR-LA checkpoint 选择

对 total-loss checkpoint 运行完整诊断：

```powershell
python scripts/diagnose_trend_residual.py `
  --config configs/metrla_e5_offset_decay_increment_relation_level0_v1.yaml `
  --checkpoint artifacts/metrla_e5b_offset_decay_increment_seed42/pretrain_best.pt `
  --bank artifacts/metrla_bank_e5b_offset_decay_increment_total_seed42 `
  --split val `
  --trend-length 12 `
  --output artifacts/e5_t1b/metrla_e5b_total_level0_val.json
```

对 relation-loss checkpoint 运行完整诊断：

```powershell
python scripts/diagnose_trend_residual.py `
  --config configs/metrla_e5_offset_decay_increment_relation_level0_v1.yaml `
  --checkpoint artifacts/metrla_e5b_offset_decay_increment_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5b_offset_decay_increment_relation_seed42 `
  --split val `
  --trend-length 12 `
  --output artifacts/e5_t1b/metrla_e5b_relation_level0_val.json
```

主指标是 `learned_offset_decay_topk.mae`，次指标是 `learned_offset_decay_topk.rmse`；若 MAE 差异小于 `0.01`，以 `learned_raw_topk.mae` 较低者胜出，且 coverage 差异不得超过 `0.2` 个百分点。只能使用 METR-LA validation 选择，不得读取 PEMS-BAY validation 或任一 test 来决定 checkpoint。

若 relation-loss checkpoint 胜出，冻结：

```powershell
$E5B_CHECKPOINT = "artifacts/metrla_e5b_offset_decay_increment_seed42/pretrain_best_relation.pt"
$E5B_METR_BANK = "artifacts/metrla_bank_e5b_offset_decay_increment_relation_seed42"
$E5B_CHECKPOINT_KIND = "relation"
```

若 total-loss checkpoint 胜出，冻结：

```powershell
$E5B_CHECKPOINT = "artifacts/metrla_e5b_offset_decay_increment_seed42/pretrain_best.pt"
$E5B_METR_BANK = "artifacts/metrla_bank_e5b_offset_decay_increment_total_seed42"
$E5B_CHECKPOINT_KIND = "total"
```

该选择一旦冻结，后续 PEMS-BAY 和 downstream 不得切换。

### 17.5 PEMS-BAY Bank 与诊断

若在新 PowerShell 会话执行本节，必须恢复 T1-B 已冻结的 checkpoint 变量。relation-loss checkpoint 胜出时执行：

```powershell
$E5B_CHECKPOINT = "artifacts/metrla_e5b_offset_decay_increment_seed42/pretrain_best_relation.pt"
$E5B_CHECKPOINT_KIND = "relation"
```

total-loss checkpoint 胜出时执行：

```powershell
$E5B_CHECKPOINT = "artifacts/metrla_e5b_offset_decay_increment_seed42/pretrain_best.pt"
$E5B_CHECKPOINT_KIND = "total"
```

只能执行与已冻结 METR-LA validation 选择一致的一组赋值。随后执行：

```powershell
$E5B_PEMS_BANK = "artifacts/pemsbay_bank_e5b_offset_decay_increment_$($E5B_CHECKPOINT_KIND)_seed42"

python scripts/build_bank.py `
  --config configs/pemsbay_e5_offset_decay_increment_transfer_level0_v1.yaml `
  --checkpoint $E5B_CHECKPOINT `
  --output-dir $E5B_PEMS_BANK `
  --dataset-name PEMS-BAY
```

```powershell
python scripts/diagnose_trend_residual.py `
  --config configs/pemsbay_e5_offset_decay_increment_transfer_level0_v1.yaml `
  --checkpoint $E5B_CHECKPOINT `
  --bank $E5B_PEMS_BANK `
  --split val `
  --trend-length 12 `
  --output artifacts/e5_t1b/pemsbay_e5b_level0_val.json
```

### 17.6 T1-B 检索门

T1-B 必须相对 T1-A 而不是相对 E3 比较：

- 两个数据集 OffsetDecay memory MAE 均至少再改善 `0.5%`；
- 两个数据集 H6、H12 中至少一个改善，另一个恶化不超过 `0.3%`；
- RMSE 不恶化超过 `0.3%`；
- coverage 下降不超过 `0.2` 个百分点；
- teacher support 不坍缩到接近 1，也不因 increment 变成近似均匀分布。

未通过：删除 T1-B，最终模型保留 T1-A；不得搜索多个 `eta` 补救。

### 17.7 T1-B 无 confidence downstream

只有检索门通过后才执行。本阶段比较五种来源：

- `base_only`：只使用下游 backbone，不读取 memory；
- `E3 OffsetDecay`：E3 learned selector 检索历史事件，并用 OffsetDecay 构造 memory；
- `T1-A OffsetDecay`：T1-A learned selector 与 OffsetDecay memory；
- `T1-B OffsetDecay`：T1-B learned selector 与 OffsetDecay memory；
- `random OffsetDecay`：未经预训练的 random selector 与 OffsetDecay memory。

`FutureIncrement` 只改变 T1-B 的预训练 teacher，不作为 downstream query 输入，也不作为 Bank payload。五组都禁用 confidence。`base_only`、E3、T1-A 和 random 直接复用 seed 42 已冻结的下列正式运行，不得为迁就 T1-B 结果而重新训练：

| Dataset | Source | Frozen run |
|---|---|---|
| METR-LA | base_only | `artifacts/metrla_e5a_base_only_seed42` |
| METR-LA | E3 | `artifacts/metrla_e3_offset_decay_horizon_seed42` |
| METR-LA | T1-A | `artifacts/metrla_e5a_offset_decay_horizon_seed42` |
| METR-LA | random | `artifacts/metrla_random_offset_decay_horizon_seed42` |
| PEMS-BAY | base_only | `artifacts/pemsbay_e5a_base_only_seed42` |
| PEMS-BAY | E3 | `artifacts/pemsbay_e3_offset_decay_horizon_seed42` |
| PEMS-BAY | T1-A | `artifacts/pemsbay_e5a_offset_decay_horizon_seed42` |
| PEMS-BAY | random | `artifacts/pemsbay_random_offset_decay_horizon_seed42` |

新增的 METR-LA T1-B 训练命令：

```powershell
python scripts/train_downstream.py `
  --config configs/metrla_e5_offset_decay_increment_relation_level0_v1.yaml `
  --pretrained-checkpoint $E5B_CHECKPOINT `
  --bank $E5B_METR_BANK `
  --mode learned_topk_offset_decay_horizon `
  --seed 42 `
  --run-name metrla_e5b_offset_decay_horizon_seed42
```

新增的 PEMS-BAY T1-B 训练命令：

```powershell
python scripts/train_downstream.py `
  --config configs/pemsbay_e5_offset_decay_increment_transfer_level0_v1.yaml `
  --pretrained-checkpoint $E5B_CHECKPOINT `
  --bank $E5B_PEMS_BANK `
  --mode learned_topk_offset_decay_horizon `
  --seed 42 `
  --run-name pemsbay_e5b_offset_decay_horizon_seed42
```

METR-LA T1-B 分支诊断命令：

```powershell
python scripts/diagnose_downstream.py `
  --config configs/metrla_e5_offset_decay_increment_relation_level0_v1.yaml `
  --pretrained-checkpoint $E5B_CHECKPOINT `
  --downstream-checkpoint artifacts/metrla_e5b_offset_decay_horizon_seed42/downstream_best.pt `
  --bank $E5B_METR_BANK `
  --split val `
  --output artifacts/e5_t1b/metrla_e5b_offset_decay_downstream_seed42_val.json
```

PEMS-BAY T1-B 分支诊断命令：

```powershell
python scripts/diagnose_downstream.py `
  --config configs/pemsbay_e5_offset_decay_increment_transfer_level0_v1.yaml `
  --pretrained-checkpoint $E5B_CHECKPOINT `
  --downstream-checkpoint artifacts/pemsbay_e5b_offset_decay_horizon_seed42/downstream_best.pt `
  --bank $E5B_PEMS_BANK `
  --split val `
  --output artifacts/e5_t1b/pemsbay_e5b_offset_decay_downstream_seed42_val.json
```

保留 T1-B 必须同时满足：两个数据集 final validation MAE 都低于 T1-A 至少 `0.3%`；15、30、60 分钟 MAE 中至少两个位置不劣于 T1-A；confidence loss 为 0；confidence trainable parameters 为 0；推理参数量不增加。任一条件失败均删除 T1-B，最终保留已经通过门槛的 T1-A。

## 18. 多随机种子阶段

只有最终保留 T1-A 或 T1-B 后执行。该阶段分开测量两类随机性，不能混写。

### 18.1 预训练随机性

`Pretraining seed stability` 表示改变 encoder 预训练初始化，但不改变 teacher、数据划分和训练预算。

冻结最终配置和 checkpoint 选择规则后，对 E3 和最终 E5 分别补 seed 2024、2025：

```powershell
foreach ($Seed in 2024, 2025) {
  python scripts/pretrain.py `
    --config configs/metrla_e3_relation_v1.yaml `
    --seed $Seed `
    --run-name "metrla_e3_relation_seed$Seed"
}
```

如果最终保留 T1-A：

```powershell
foreach ($Seed in 2024, 2025) {
  python scripts/pretrain.py `
    --config configs/metrla_e5_offset_decay_relation_v1.yaml `
    --seed $Seed `
    --run-name "metrla_e5a_offset_decay_seed$Seed"
}
```

如果最终保留 T1-B，则改用：

```powershell
foreach ($Seed in 2024, 2025) {
  python scripts/pretrain.py `
    --config configs/metrla_e5_offset_decay_increment_relation_v1.yaml `
    --seed $Seed `
    --run-name "metrla_e5b_offset_decay_increment_seed$Seed"
}
```

`Checkpoint kind` 表示使用 `pretrain_best.pt`（total）还是 `pretrain_best_relation.pt`（relation）。它必须由 seed 42 的 METR-LA validation 预先冻结，不能为 seed 2024/2025 分别选择更有利的版本。

E3 已冻结为 relation kind，因此 seed 2024/2025 的完整 Bank 与 validation 诊断命令为：

```powershell
foreach ($Seed in 2024, 2025) {
  $Checkpoint = "artifacts/metrla_e3_relation_seed$Seed/pretrain_best_relation.pt"
  $MetrBank = "artifacts/metrla_bank_e3_relation_seed$Seed"
  $PemsBank = "artifacts/pemsbay_bank_from_metrla_e3_relation_seed$Seed"

  python scripts/build_bank.py `
    --config configs/metrla_e3_relation_level0_v1.yaml `
    --checkpoint $Checkpoint `
    --output-dir $MetrBank `
    --dataset-name METR-LA

  python scripts/diagnose_trend_residual.py `
    --config configs/metrla_e3_relation_level0_v1.yaml `
    --checkpoint $Checkpoint `
    --bank $MetrBank `
    --split val `
    --trend-length 12 `
    --output "artifacts/e5_multiseed/metrla_e3_relation_seed${Seed}_val.json"

  python scripts/build_bank.py `
    --config configs/pemsbay_e3_transfer_level0_v1.yaml `
    --checkpoint $Checkpoint `
    --output-dir $PemsBank `
    --dataset-name PEMS-BAY

  python scripts/diagnose_trend_residual.py `
    --config configs/pemsbay_e3_transfer_level0_v1.yaml `
    --checkpoint $Checkpoint `
    --bank $PemsBank `
    --split val `
    --trend-length 12 `
    --output "artifacts/e5_multiseed/pemsbay_e3_relation_seed${Seed}_val.json"
}
```

最终 E5 的完整命令先在当前 PowerShell 会话内设置三个冻结变量。`$FINAL_E5_STAGE` 只能是 `T1-A` 或 `T1-B`；`$FINAL_E5_CHECKPOINT_KIND` 只能是 `total` 或 `relation`。下面示例假设 T1-A 和 relation 胜出，实际值必须来自 seed 42 的门控结果：

```powershell
$FINAL_E5_STAGE = "T1-A"
$FINAL_E5_CHECKPOINT_KIND = "relation"

if ($FINAL_E5_STAGE -eq "T1-A") {
  $RunPrefix = "metrla_e5a_offset_decay"
  $MetrConfig = "configs/metrla_e5_offset_decay_relation_level0_v1.yaml"
  $PemsConfig = "configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml"
} elseif ($FINAL_E5_STAGE -eq "T1-B") {
  $RunPrefix = "metrla_e5b_offset_decay_increment"
  $MetrConfig = "configs/metrla_e5_offset_decay_increment_relation_level0_v1.yaml"
  $PemsConfig = "configs/pemsbay_e5_offset_decay_increment_transfer_level0_v1.yaml"
} else {
  throw "FINAL_E5_STAGE must be T1-A or T1-B"
}

if ($FINAL_E5_CHECKPOINT_KIND -eq "relation") {
  $CheckpointFile = "pretrain_best_relation.pt"
} elseif ($FINAL_E5_CHECKPOINT_KIND -eq "total") {
  $CheckpointFile = "pretrain_best.pt"
} else {
  throw "FINAL_E5_CHECKPOINT_KIND must be total or relation"
}

foreach ($Seed in 2024, 2025) {
  $Checkpoint = "artifacts/${RunPrefix}_seed$Seed/$CheckpointFile"
  $MetrBank = "artifacts/metrla_bank_final_e5_$($FINAL_E5_CHECKPOINT_KIND)_seed$Seed"
  $PemsBank = "artifacts/pemsbay_bank_final_e5_$($FINAL_E5_CHECKPOINT_KIND)_seed$Seed"

  python scripts/build_bank.py `
    --config $MetrConfig `
    --checkpoint $Checkpoint `
    --output-dir $MetrBank `
    --dataset-name METR-LA

  python scripts/diagnose_trend_residual.py `
    --config $MetrConfig `
    --checkpoint $Checkpoint `
    --bank $MetrBank `
    --split val `
    --trend-length 12 `
    --output "artifacts/e5_multiseed/metrla_final_e5_seed${Seed}_val.json"

  python scripts/build_bank.py `
    --config $PemsConfig `
    --checkpoint $Checkpoint `
    --output-dir $PemsBank `
    --dataset-name PEMS-BAY

  python scripts/diagnose_trend_residual.py `
    --config $PemsConfig `
    --checkpoint $Checkpoint `
    --bank $PemsBank `
    --split val `
    --trend-length 12 `
    --output "artifacts/e5_multiseed/pemsbay_final_e5_seed${Seed}_val.json"
}
```

### 18.2 下游初始化随机性

`Downstream seed stability` 表示固定最终 E5 seed 42 encoder、METR-LA Bank 和 PEMS-BAY Bank，只改变下游 backbone 与 horizon fusion 的初始化。先在本节重新设置所有变量，不依赖前文 PowerShell 会话：

```powershell
$FINAL_E5_STAGE = "T1-A"  # 按门控结果填写 T1-A 或 T1-B
$FINAL_E5_CHECKPOINT = "artifacts/metrla_e5a_offset_decay_seed42/pretrain_best_relation.pt"
$FINAL_E5_METR_BANK = "artifacts/metrla_bank_e5a_offset_decay_relation_seed42"
$FINAL_E5_PEMS_BANK = "artifacts/pemsbay_bank_e5a_offset_decay_relation_seed42"

if ($FINAL_E5_STAGE -eq "T1-A") {
  $FINAL_E5_METR_CONFIG = "configs/metrla_e5_offset_decay_relation_level0_v1.yaml"
  $FINAL_E5_PEMS_CONFIG = "configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml"
} elseif ($FINAL_E5_STAGE -eq "T1-B") {
  $FINAL_E5_METR_CONFIG = "configs/metrla_e5_offset_decay_increment_relation_level0_v1.yaml"
  $FINAL_E5_PEMS_CONFIG = "configs/pemsbay_e5_offset_decay_increment_transfer_level0_v1.yaml"
} else {
  throw "FINAL_E5_STAGE must be T1-A or T1-B"
}
```

上面三个 artifact 路径是 T1-A relation 胜出时的具体示例；若 T1-A total 或 T1-B 胜出，必须把三个路径改成已冻结的对应 checkpoint/Bank。随后执行 METR-LA 三个下游 seed：

```powershell
foreach ($Seed in 42, 2024, 2025) {
  python scripts/train_downstream.py `
    --config $FINAL_E5_METR_CONFIG `
    --pretrained-checkpoint $FINAL_E5_CHECKPOINT `
    --bank $FINAL_E5_METR_BANK `
    --mode learned_topk_offset_decay_horizon `
    --seed $Seed `
    --run-name "metrla_final_e5_offset_decay_horizon_seed$Seed"
}
```

PEMS-BAY 三个下游 seed：

```powershell
foreach ($Seed in 42, 2024, 2025) {
  python scripts/train_downstream.py `
    --config $FINAL_E5_PEMS_CONFIG `
    --pretrained-checkpoint $FINAL_E5_CHECKPOINT `
    --bank $FINAL_E5_PEMS_BANK `
    --mode learned_topk_offset_decay_horizon `
    --seed $Seed `
    --run-name "pemsbay_final_e5_offset_decay_horizon_seed$Seed"
}
```

无论最终保留 T1-A 还是 T1-B，payload mode 都是 OffsetDecay；FutureIncrement 只改变训练 teacher，不进入推理输入。

## 19. Test 使用规则与命令

`Test` 指训练、checkpoint 选择、超参数选择和方法取舍全部完成后才读取的最终留出数据。它只用于一次性无偏报告，不用于选择 T1-A/T1-B、checkpoint kind、fusion 或随机种子。

只有下列内容全部冻结后才允许读取 test：

- 最终 teacher 是 T1-A 还是 T1-B；
- checkpoint kind 是 total 还是 relation；
- `level_weight=0`；
- OffsetDecay 使用线性 \(\lambda_h\)；
- downstream mode；
- 三个 downstream seeds；
- 所有门槛和报告指标。

先在本节重新填写最终选择，不依赖前面任何 PowerShell 变量。以下路径展示 T1-A relation 胜出的情况；若最终保留 T1-A total 或 T1-B，必须填写已冻结的对应路径：

```powershell
$FINAL_E5_STAGE = "T1-A"
$FINAL_E5_CHECKPOINT = "artifacts/metrla_e5a_offset_decay_seed42/pretrain_best_relation.pt"
$FINAL_E5_METR_BANK = "artifacts/metrla_bank_e5a_offset_decay_relation_seed42"
$FINAL_E5_PEMS_BANK = "artifacts/pemsbay_bank_e5a_offset_decay_relation_seed42"

if ($FINAL_E5_STAGE -eq "T1-A") {
  $FINAL_E5_METR_CONFIG = "configs/metrla_e5_offset_decay_relation_level0_v1.yaml"
  $FINAL_E5_PEMS_CONFIG = "configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml"
} elseif ($FINAL_E5_STAGE -eq "T1-B") {
  $FINAL_E5_METR_CONFIG = "configs/metrla_e5_offset_decay_increment_relation_level0_v1.yaml"
  $FINAL_E5_PEMS_CONFIG = "configs/pemsbay_e5_offset_decay_increment_transfer_level0_v1.yaml"
} else {
  throw "FINAL_E5_STAGE must be T1-A or T1-B"
}
```

METR-LA 最终 test 命令：

```powershell
foreach ($Seed in 42, 2024, 2025) {
  python scripts/evaluate.py `
    --config $FINAL_E5_METR_CONFIG `
    --pretrained-checkpoint $FINAL_E5_CHECKPOINT `
    --downstream-checkpoint "artifacts/metrla_final_e5_offset_decay_horizon_seed$Seed/downstream_best.pt" `
    --bank $FINAL_E5_METR_BANK `
    --split test 2>&1 | Tee-Object -FilePath "artifacts/metrla_final_e5_offset_decay_horizon_seed$Seed/test.log"
}
```

PEMS-BAY 最终 test 命令：

```powershell
foreach ($Seed in 42, 2024, 2025) {
  python scripts/evaluate.py `
    --config $FINAL_E5_PEMS_CONFIG `
    --pretrained-checkpoint $FINAL_E5_CHECKPOINT `
    --downstream-checkpoint "artifacts/pemsbay_final_e5_offset_decay_horizon_seed$Seed/downstream_best.pt" `
    --bank $FINAL_E5_PEMS_BANK `
    --split test 2>&1 | Tee-Object -FilePath "artifacts/pemsbay_final_e5_offset_decay_horizon_seed$Seed/test.log"
}
```

Test 不用于重新选择 checkpoint、teacher、seed、level weight 或 fusion 形式。

## 20. 必须报告的指标

### 20.1 预训练

- validation total/reconstruction/relation loss；
- valid anchors；
- relation candidate pairs；
- teacher/student effective support；
- best epoch、stop epoch；
- 每 epoch 时间、峰值 GPU memory；
- retrieval encoder 参数量。

### 20.2 检索 memory

- RawFuture MAE/RMSE/MAPE；
- OffsetDecay MAE/RMSE/MAPE；
- 12 个 horizon 的 MAE/RMSE；
- common coverage；
- future oracle，只标记为 diagnostic-only；
- E3/E5/random 的同候选协议差异；
- METR-LA/PEMS-BAY 分开报告。

### 20.3 无 confidence 下游

- BaseMAE、MemoryMAE、FinalMAE；
- overall MAE/RMSE/MAPE；
- 15、30、60 分钟指标；
- 12 个 horizon fusion weights \(a_h\)；
- OffsetDecay 固定系数 \(\lambda_h\)；
- confidence loss 必须为 0；
- trainable parameter count；
- inference latency 与峰值显存。

### 20.4 随机性

- 预训练 seed 变化与下游 seed 变化分表报告；
- 均值、样本标准差、逐 seed 值；
- 不得把同一个 frozen memory 在三个下游 seed 下的相同 MemoryMAE 写成三次独立预训练复现。

## 21. 最终 Keep / Remove / Stop 决策树

1. T1-A retrieval 未通过：`Stop E5`，保留 T0 为失败诊断，不运行 T1-B。
2. T1-A retrieval 通过、无 confidence downstream 未通过：`Remove deployment teacher claim`，停止，不接 confidence。
3. T1-A 全部通过：`Keep T1-A`，允许 T1-B。
4. T1-B retrieval 未通过：`Remove FutureIncrement`，最终保留 T1-A。
5. T1-B retrieval 通过、downstream 未通过：`Remove FutureIncrement`，最终保留 T1-A。
6. T1-B 双数据集、无 confidence downstream 均通过：`Keep T1-B`，进入多 seed。
7. 多 seed 不稳定或不优于 random：不得作为论文主创新；不增加模型复杂度补救。
8. 只有最终变体通过多 seed 后，才允许单独测 confidence 增量；confidence 结果不能替代 retrieval 表示证据。

## 22. 方案自检

- 没有同时修改 encoder architecture、teacher、candidate pool 和 downstream backbone；
- T1-A/T1-B 每阶段只有一个主要变量；
- 每个特殊名称都在本文首次出现处解释了含义、输入、计算、输出和 future 边界；
- 没有把历史斜率外推重新包装成 future trend learning；
- 没有引入 learned future encoder；
- PEMS-BAY validation 不参与 source checkpoint 选择；
- test 在全部选择冻结前不可访问；
- 所有正式实验都有明确的 Keep/Remove/Stop 后果；
- smoke/debug 结果不进入正式证据，验证后删除。

## Execution Note

当前仓库位于带既有未提交实验资产的 `main`。本轮已完成 T1-A/T1-B 工程接入、配置、测试和 T1-A smoke 验证；T1-A smoke 及下游 smoke 目录均已删除，不能作为正式实验结果引用。截至 2026-08-03 尚未运行任何正式 E5 预训练、Bank 构建、validation 诊断、下游训练或 test 评估。实施正式实验时不得覆盖现有 E3 checkpoint、Bank、正式 JSON 或用户已有改动；所有 E5 产物使用独立目录。
