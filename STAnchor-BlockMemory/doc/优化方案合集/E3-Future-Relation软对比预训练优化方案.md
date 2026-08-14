# STAnchor-BlockMemory：Future-Relation 软对比预训练优化方案

更新时间：2026-07-23

文档定位：E3 研究方案审查版。章节安排与 `doc/计划方案.md` 完全一致；未改变的模块直接引用原计划，改变的预训练检索目标在第 9 节完整说明。

## 1. 当前研究范围

本方案继承 `doc/计划方案.md` 的单源预训练、同物理量跨数据集迁移边界，并以 E2 一天历史编码器为当前基线：

1. retrieval encoder 输入完整一天，即 288 个 5 分钟时间步；
2. forecasting backbone 仍只输入最近 12 步；
3. encoder、retrieval head、目标域 Bank、两阶段检索、confidence 和 safe fusion 均保持不变；
4. 只将原来的 quantile positive、context-near hard negative 和 hard-negative InfoNCE 替换为 Future-Relation 软对比损失。

E3 不增加 trainable future encoder。真实 future 只用于构造固定关系教师，训练完成后不进入 checkpoint 的可迁移模块，也不参与建库或推理。

当前实验优先实现 E3。`E2预训练加速优化方案.md` 暂时保留为工程备选，不与 E3 首轮同时启用，避免把损失机制变化与 AMP、分块计算或任务路由混在一起。

当前仍不做：

- 多源联合预训练；
- 速度与流量跨物理量迁移；
- ST-Norm、FFT 或新的时频分支；
- 跨 batch 对比队列；
- 更复杂的 selector、decoder 或动态图。

本轮只回答：

> 用连续 future 邻居关系直接监督 context key，能否比原离散 hard-negative 目标学到更好的检索空间，并减少预训练损失的计算成本？

## 2. 本文公式的阅读规则

本节与 `doc/计划方案.md` 相同。新增公式继续遵守以下顺序：术语、上游来源、数学公式、符号与维度、科学意义、未来信息边界。

为避免两类 mask 混淆，E3 统一使用：

- \(\mathbf O\)：原始数据观测掩码，1 表示真实有效，0 表示原始缺失；
- \(\mathbf M\)：预训练阶段人为生成的时间或空间掩码。

预测张量仍使用轴顺序：

$$
(\text{batch},\ \text{time},\ \text{node},\ \text{channel}).
$$

## 3. 专业术语解释

### 3.1 Context 和 Forecast Horizon

与 `doc/计划方案.md` 相同，但 E3 明确区分两种 context：

- retrieval context：\(T_{\mathrm{ret}}=288\)，供预训练 encoder 和 learned retrieval 使用；
- forecast context：\(T_{\mathrm{pred}}=12\)，供下游预测骨干和 raw-L1 对照使用；
- forecast horizon：\(H=12\)。

### 3.2 Historical Block

与 `doc/计划方案.md` 相同。每个历史事件由 288 步 retrieval context、最后 12 步 forecast context、紧接其后的 12 步真实 future 及时间元数据构成。

### 3.3 Memory Bank

与 `doc/计划方案.md` 相同。Bank 只存目标数据集训练历史的 context key、历史 future、观测掩码和元数据；E3 的 future 教师分布不写入 Bank。

### 3.4 Query 和 Key

与 `doc/计划方案.md` 相同。E3 优化的核心仍是节点 key：future 相近的历史事件应在 context key 空间形成相近的检索邻居。

### 3.5 Temporal Patch

机制与 `doc/计划方案.md` 相同。E2/E3 使用 \(p=12\)，所以一天 retrieval context 产生：

$$
P=\frac{T_{\mathrm{ret}}}{p}=\frac{288}{12}=24
$$

个小时级时间 token。

### 3.6 Factorized Spatio-Temporal Attention

与 `doc/计划方案.md` 相同：每层先对同一节点的 24 个时间 token 做时间注意力，再对同一 patch 中图边相连的节点做稀疏空间注意力。

### 3.7 Spatio-Temporal Mirage

定义与 `doc/计划方案.md` 相同。E3 不再显式构造二值 mirage hard-negative 标签，而是用 future 教师分布自然处理：

- context key 给某个候选高概率；
- 但该候选 future 与 anchor 明显不同，教师只给它低概率；
- 软对比交叉熵因此直接惩罚这种错误高相似度。

### 3.8 第一版不做 Level Alignment

与 `doc/计划方案.md` 相同。E3 不平移、不缩放历史 future，避免把检索表示的变化与数值对齐混在一起。

### 3.9 Confidence Calibration

与 `doc/计划方案.md` 相同。confidence 仍在目标域判断检索历史是否可能帮助基础预测，不参与源域 Future-Relation 教师构造。

### 3.10 Safe Residual Fusion

与 `doc/计划方案.md` 相同。confidence 为 0 时最终结果精确退回 base prediction。

### 3.11 常用网络与损失术语

原计划中的 LayerNorm、GELU、Softmax、L2 normalization、cosine similarity、Huber loss、Brier 和 ECE 定义保持不变。E3 新增三个术语：

**软对比学习**：不把候选简单标为正样本或负样本，而是为所有合法候选分配连续目标概率。

**关系对齐**：不要求 context 表示复制 future 数值，而是要求两者对“哪些事件更相似”的相对排序保持一致。

**关系蒸馏**：将教师给出的样本间关系传递给学生表示。E3 没有神经教师模型，更准确的名称是 Future-Guided Relational Contrastive Learning，而不是经典 model-to-model knowledge distillation。

## 4. 符号与维度

### 4.1 基本维度

| 符号 | 含义 | E3 设置 |
|---|---|---:|
| \(B\) | 预训练 batch size | 16 |
| \(T_{\mathrm{ret}}\) | retrieval context 长度 | 288 |
| \(T_{\mathrm{pred}}\) | 下游 forecast context 长度 | 12 |
| \(H\) | 预测 future 长度 | 12 |
| \(N_s\) | 源数据集节点数 | METR-LA 为 207 |
| \(N_t\) | 目标数据集节点数 | PEMS-BAY 为 325 |
| \(C\) | 交通变量通道数 | 1 |
| \(p\) | temporal patch 长度 | 12 |
| \(P=T_{\mathrm{ret}}/p\) | retrieval token 数 | 24 |
| \(D\) | encoder 隐藏维度 | 96 |
| \(D_r\) | node/event key 维度 | 48 |
| \(L_{\mathrm{enc}}\) | 因子化时空层数量 | 3 |
| \(M\) | Bank 历史事件数 | 数据决定 |
| \(R\) | 事件级粗检索保留数 | 32 |
| \(K\) | 节点级最终候选数 | 5 |

目标 adapter、confidence 和图边等其余维度符号与 `doc/计划方案.md` 相同。

### 4.2 核心数据张量

| 张量 | 来源 | 维度 | 含义 |
|---|---|---|---|
| \(\mathbf X^{\mathrm{ret}}\) | 数据窗口 | \(B\times288\times N_s\times C\) | 干净 retrieval context |
| \(\mathbf X^{\mathrm{pred}}\) | retrieval context 的末 12 步 | \(B\times12\times N_s\times C\) | 下游短历史输入 |
| \(\mathbf Y\) | context 后真实发生的数据 | \(B\times12\times N_s\times C\) | 只在训练监督和评估时可见的 future |
| \(\mathbf O^X\) | 原始数据 | 与 \(\mathbf X^{\mathrm{ret}}\) 相同 | context 原始观测掩码 |
| \(\mathbf O^Y\) | 原始数据 | 与 \(\mathbf Y\) 相同 | future 原始观测掩码 |
| \(\mathbf K^X\) | context encoder + retrieval head | \(B\times N_s\times D_r\) | 学生节点 key |
| \(\mathbf D^Y\) | future signature 两两距离 | \(B\times B\times N_s\) | future 教师距离矩阵 |
| \(\mathbf Q^Y\) | \(\mathbf D^Y\) 的 masked softmax | \(B\times B\times N_s\) | future 教师邻居分布 |
| \(\mathbf S^X\) | node key 余弦相似度 | \(B\times B\times N_s\) | 学生相似度矩阵 |
| \(\mathbf P^X\) | \(\mathbf S^X\) 的 masked softmax | \(B\times B\times N_s\) | 学生检索分布 |

### 4.3 索引

| 索引 | 范围 | 含义 |
|---|---|---|
| \(b\) | \(1,\ldots,B\) | batch 样本 |
| \(t\) | \(1,\ldots,T_{\mathrm{ret}}\) | retrieval context 时间位置 |
| \(h\) | \(1,\ldots,H\) | future 时间位置 |
| \(n\) | \(1,\ldots,N\) | 节点 |
| \(c\) | \(1,\ldots,C\) | 变量通道 |
| \(r\) | \(1,\ldots,P\) | temporal patch |
| \(i\) | \(1,\ldots,B\) | 当前 anchor event |
| \(j\) | \(1,\ldots,B\) | batch 内候选 event |
| \(k\) | \(1,\ldots,K\) | Bank 检索后的候选编号 |

### 4.4 标量超参数

| 符号 | 含义 | E3 初始设置或确定方法 |
|---|---|---|
| \(\varepsilon\) | 防止除零 | 固定小常数 |
| \(\tau_Y\) | future 教师分布温度 | 根据训练集教师有效支持数选择 |
| \(\tau_X\) | context key 学生分布温度 | 初始沿用 0.1 |
| \(\lambda_{\mathrm{rel}}\) | 关系损失权重 | 初始沿用旧 retrieval weight 0.1 |
| \(\tau_{\mathrm{search}}\) | Bank top-\(K\) 聚合温度 | 与原计划相同 |

旧损失中的 \(q_{\mathrm{pos}}\)、\(q_{\mathrm{ctx}}\)、\(q_{\mathrm{neg}}\) 和 \(\lambda_{\mathrm{hard}}\) 在 E3 中删除。其他目标域超参数与 `doc/计划方案.md` 相同。

## 5. 整体流程

整体预测流程与 `doc/计划方案.md` 相同。E3 只改变源域预训练中的 retrieval supervision：

$$
\mathbf X^{\mathrm{ret}}
\xrightarrow{E_\theta,R_\phi}
\mathbf K^X
\xrightarrow{\text{pairwise cosine}}
\mathbf P^X,
$$

$$
(\mathbf Y,\mathbf O^Y,\text{context statistics})
\xrightarrow{\text{fixed future relation}}
\mathbf Q^Y,
$$

$$
(\mathbf Q^Y,\mathbf P^X)
\xrightarrow{\text{soft contrastive loss}}
\mathcal L_{\mathrm{rel}}.
$$

训练完成后只保留 \(E_\theta\) 和 \(R_\phi\)。目标域仍按照“context 编码、Bank 检索、历史 future 聚合、confidence、safe fusion”的原流程运行。

## 6. 数据构造与归一化

### 6.1 数据集级标准化

与 `doc/计划方案.md` 相同。scaler 只在当前数据集训练段拟合，再用于 train、validation 和 test，禁止使用未来划分拟合统计量。

### 6.2 窗口内部归一化

与 `doc/计划方案.md` 的 mask-aware 归一化相同，但统计窗口为一天 retrieval context。对样本 \(i\)、节点 \(n\)、通道 \(c\) 得到：

$$
\mu^X_{i,n,c},\qquad \sigma^X_{i,n,c}.
$$

二者形状均为 \(B\times N_s\times C\)，只由 \(\mathbf X^{\mathrm{ret}}\) 中真实可观测位置计算。

### 6.3 水平特征

与 `doc/计划方案.md` 相同：均值、标准差、末值和首末差进入 patch embedding、Bank level features 和 confidence，但不直接平移历史 future。

### 6.4 历史事件切分

与 E2 数据契约相同。每个样本的 288 步 context 和 12 步 future 必须完整位于自己的 train、validation 或 test 划分内，不允许跨划分借用一天历史。

## 7. 时空检索编码器

本节机制与 `doc/计划方案.md` 相同，采用 E2 参数规模。E3 不修改 encoder 结构。

### 7.1 Temporal Patch Embedding

输入 \(\mathbf X^{\mathrm{ret}}\in\mathbb R^{B\times288\times N_s\times C}\)，输出：

$$
\mathbf Z^{(0)}\in\mathbb R^{B\times24\times N_s\times96}.
$$

### 7.2 时间注意力

与 `doc/计划方案.md` 相同。每个节点独立在 24 个小时级 token 上做 multi-head self-attention。

### 7.3 稀疏图空间注意力

与 `doc/计划方案.md` 相同。空间注意力只沿当前数据集图边计算，并使用当前数据集自己的邻接结构。

### 7.4 时空层输出

三层编码后：

$$
\mathbf H\in\mathbb R^{B\times24\times N_s\times96}.
$$

该表示同时接收 masked reconstruction 和 Future-Relation loss 的梯度。

## 8. Retrieval Head

机制与 `doc/计划方案.md` 相同。E3 不增加 future head。

### 8.1 节点级时间池化

对 24 个时间 token 做注意力池化，得到 \(B\times N_s\times96\) 的节点表示。

### 8.2 节点级 key

通过 MLP 和 L2 normalization 得到：

$$
\mathbf K^X\in\mathbb R^{B\times N_s\times48}.
$$

Future-Relation loss 只监督该节点 key 的相对几何关系，不要求 key 逐元素重建 future。

### 8.3 事件级 key

与 `doc/计划方案.md` 相同。节点 key 平均后再次 L2 normalization，得到 \(B\times48\) 的事件级 key，用于 Bank 事件粗检索。

## 9. 单源预训练

E3 在单个源数据集训练段上联合执行 masked reconstruction 和 Future-Relation 软对比学习，不读取其他数据集。

### 9.1 Masked Reconstruction

本节整体与 `doc/计划方案.md` 相同，仅将 E2 的一天 context 和 24 个 patch 代入原流程。

#### 9.1.1 前人工作给出的边界

与 `doc/计划方案.md` 相同：采用结构化时间/空间掩码和轻量线性重建头，不通过加深 decoder 获得表面重建优势。

#### 9.1.2 干净检索视图与掩码重建视图

与 `doc/计划方案.md` 相同。干净视图产生 retrieval key，掩码视图产生 reconstruction；两个视图共享 encoder 参数。

#### 9.1.3 时间 patch 掩码

与 E2 相同：时间掩码率 0.25，连续遮蔽 36 个原始时间步，即 3 个连续小时 patch。

#### 9.1.4 空间节点掩码

与 `doc/计划方案.md` 相同：按样本遮蔽完整节点历史，并保证被遮节点至少有一个未遮挡的一阶邻居。

#### 9.1.5 解耦训练调度

与 `doc/计划方案.md` 相同：每个 reconstruction batch 选择时间掩码或空间掩码，不使用逐元素随机混合掩码。

#### 9.1.6 防止水平特征泄漏

与 `doc/计划方案.md` 相同：掩码视图的均值、标准差、末值和首末差只能由可见位置重新计算。

#### 9.1.7 mask token 与 encoder 输入

与 `doc/计划方案.md` 相同。mask token 保留时间和节点位置，但不暴露被遮挡真实数值。

#### 9.1.8 decoder 选择

与 `doc/计划方案.md` 相同，继续使用线性 reconstruction head。

#### 9.1.9 重建目标与损失

与 `doc/计划方案.md` 相同，在“原始有观测且被人工遮蔽”的集合上计算 Smooth-L1：

$$
\mathcal L_{\mathrm{mask}}
=
\frac{1}{|\Omega_{\mathrm{mask}}|}
\sum_{(i,t,n,c)\in\Omega_{\mathrm{mask}}}
\operatorname{SmoothL1}
\left(
\widehat X^{\mathrm{mask}}_{i,t,n,c},
X^{\mathrm{target}}_{i,t,n,c}
\right).
$$

\(\Omega_{\mathrm{mask}}\) 只包含 \(O^X_{i,t,n,c}=1\) 且人工掩码 \(M_{i,t,n,c}=1\) 的位置。

### 9.2 Future Signature

Future signature 的作用是比较“两个历史事件之后的演化是否相似”，而不是比较绝对速度水平。

上游输入包括：

- 数据集级标准化后的真实 future \(\mathbf Y\in\mathbb R^{B\times H\times N_s\times C}\)；
- 由干净 retrieval context 计算的 \(\boldsymbol\mu^X,\boldsymbol\sigma^X\in\mathbb R^{B\times N_s\times C}\)；
- future 原始观测掩码 \(\mathbf O^Y\)。

定义：

$$
\widetilde Y_{i,h,n,c}
=
\frac{
Y_{i,h,n,c}-\mu^X_{i,n,c}
}{
\sigma^X_{i,n,c}+\varepsilon
}.
$$

其中：

- \(i\) 是 batch 中事件；
- \(h\) 是 12 个 future 时间步；
- \(n\) 是节点；
- \(c\) 是变量通道；
- \(\widetilde{\mathbf Y}\in\mathbb R^{B\times12\times N_s\times C}\) 是 future shape signature。

该步骤只在源域预训练损失和 validation loss 中使用真实 future。建库、目标域检索和部署推理均不计算该 signature。

### 9.3 Context 和 Future 距离

E3 保留 future 距离，删除原方案的 288 步 context 两两距离。

对事件 \(i,j\) 在节点 \(n\) 上的共同 future 观测数定义为：

$$
C^Y_{i,j,n}
=
\sum_{h=1}^{H}\sum_{c=1}^{C}
O^Y_{i,h,n,c}O^Y_{j,h,n,c}.
$$

future masked L1 距离为：

$$
d^Y_{i,j,n}
=
\frac{
\sum_{h=1}^{H}\sum_{c=1}^{C}
O^Y_{i,h,n,c}O^Y_{j,h,n,c}
\left|
\widetilde Y_{i,h,n,c}-\widetilde Y_{j,h,n,c}
\right|
}{
C^Y_{i,j,n}
}.
$$

符号和维度：

- \(\mathbf C^Y\in\mathbb N^{B\times B\times N_s}\)：每个事件对、每个节点的共同 future 观测数；
- \(\mathbf D^Y\in\mathbb R^{B\times B\times N_s}\)：future 距离矩阵；
- 当 \(C^Y_{i,j,n}=0\) 时，该 pair 在节点 \(n\) 上无效，不进入 softmax 或 loss；
- \(O^Y_iO^Y_j\) 保证只有双方都真实有观测的位置才参与距离。

旧损失还构造 \(B\times B\times288\times N_s\times C\) 的 context 广播张量。E3 只构造长度为 \(H=12\) 的 future 距离，原始时间比较规模由：

$$
B^2T_{\mathrm{ret}}N_sC
$$

降为：

$$
B^2HN_sC.
$$

这里删除 context 距离是预训练目标变化，不应只描述为工程优化。

### 9.4 Positive 和 Mirage Hard Negative

E3 删除二值 positive 和 mirage hard-negative 标签，不再使用 q10、q20、q80 分位点。

首先定义合法候选集合 \(\mathcal V_{i,n}\)。候选事件 \(j\) 必须同时满足：

1. \(j\neq i\)；
2. 事件 \(i,j\) 的 context/future 时间区间不重叠；
3. \(C^Y_{i,j,n}>0\)。

对固定 anchor \(i\) 和节点 \(n\)，future 教师分布为：

$$
q^Y_{i,j,n}
=
\frac{
\exp\left(-d^Y_{i,j,n}/\tau_Y\right)
}{
\sum_{k\in\mathcal V_{i,n}}
\exp\left(-d^Y_{i,k,n}/\tau_Y\right)
},
\qquad j\in\mathcal V_{i,n}.
$$

其中：

- \(q^Y_{i,j,n}\in[0,1]\)：候选 \(j\) 应成为 anchor \(i\) 在节点 \(n\) 上检索邻居的目标概率；
- \(\tau_Y>0\)：future 教师温度；越小则分布越接近 hard positive，越大则越接近均匀分布；
- \(\mathbf Q^Y\in\mathbb R^{B\times B\times N_s}\)；
- 对固定 \(i,n\)，合法候选概率和为 1。

Mirage 不再依靠显式加权标签处理。如果某候选 context key 很像但 future 很不同，学生会给它较高概率而教师给低概率，关系损失会自动惩罚该错误。

为避免教师退化成 top-1 硬标签或近似均匀分布，记录教师有效支持数：

$$
K^Y_{\mathrm{eff}}(i,n)
=
\frac{1}{
\sum_{j\in\mathcal V_{i,n}}
(q^Y_{i,j,n})^2
}.
$$

第一版只从一个很小的 \(\tau_Y\) 候选集合中选择使教师中位有效支持数处于 2 到 5 的设置，不扩大成大规模超参数搜索。

### 9.5 Future-Guided Contrastive Loss

干净 context 经过 encoder 和 retrieval head 得到 L2 归一化节点 key：

$$
\mathbf K^X
\in
\mathbb R^{B\times N_s\times D_r}.
$$

事件 \(i,j\) 在节点 \(n\) 上的学生相似度为：

$$
s^X_{i,j,n}
=
\left(\mathbf k^X_{i,n}\right)^\top
\mathbf k^X_{j,n}.
$$

由于 key 已做 L2 normalization，\(s^X_{i,j,n}\) 就是 cosine similarity，\(\mathbf S^X\in\mathbb R^{B\times B\times N_s}\)。

学生检索分布为：

$$
p^X_{i,j,n}
=
\frac{
\exp\left(s^X_{i,j,n}/\tau_X\right)
}{
\sum_{k\in\mathcal V_{i,n}}
\exp\left(s^X_{i,k,n}/\tau_X\right)
},
\qquad j\in\mathcal V_{i,n}.
$$

\(\mathbf P^X\in\mathbb R^{B\times B\times N_s}\)，其合法候选轴概率和为 1。

有效 anchor 集合定义为：

$$
\mathcal A
=
\left\{
(i,n)\mid |\mathcal V_{i,n}|\geq2
\right\}.
$$

至少需要两个合法候选，因为只有一个候选时教师和学生 softmax 都恒等于 1，该 anchor 的关系损失恒为 0，无法提供 key 排序梯度。

Future-Relation 软对比损失为：

$$
\mathcal L_{\mathrm{rel}}
=
-\frac{1}{|\mathcal A|}
\sum_{(i,n)\in\mathcal A}
\sum_{j\in\mathcal V_{i,n}}
q^Y_{i,j,n}\log p^X_{i,j,n}.
$$

等价地，在教师分布熵不参与梯度时，可以理解为最小化：

$$
D_{\mathrm{KL}}
\left(
\mathbf Q^Y\Vert\mathbf P^X
\right).
$$

该目标只要求 context key 复现 future 邻居排序，不要求 key 与 future 数值处在同一坐标系，也不增加 future encoder。若全部 context key 坍缩成相同向量，\(\mathbf P^X\) 会变成均匀分布；只要 \(\mathbf Q^Y\) 非均匀，损失就会产生分离 key 的梯度。

### 9.6 总预训练损失

E3 总损失为：

$$
\mathcal L_{\mathrm{pretrain}}
=
\mathcal L_{\mathrm{mask}}
+
\lambda_{\mathrm{rel}}
\mathcal L_{\mathrm{rel}}.
$$

其中：

- \(\mathcal L_{\mathrm{mask}}\)：保持通用时空上下文建模能力；
- \(\mathcal L_{\mathrm{rel}}\)：直接训练用于历史 future 检索的节点 key；
- \(\lambda_{\mathrm{rel}}\)：第一版设为 0.1，与旧 retrieval weight 保持一致，完成单变量归因。

checkpoint 同时保存 `best_total` 和 `best_relation`。最终用于 Bank 的 checkpoint 只能根据源 validation retrieval diagnostics 选择，不能根据目标 test 指标选择。

## 10. 目标域适配与模式库

本节与 `doc/计划方案.md` 相同。E3 不改变目标域适配协议。

### 10.1 纯冻结迁移

默认冻结 source-pretrained embedding、encoder 和 retrieval head，在目标训练历史上重新编码并建库。

### 10.2 可选 Target Adapter

与 `doc/计划方案.md` 相同。只有纯冻结迁移证据不足时才单独评估轻量 adapter，不与 E3 loss 首轮实验同时启用。

### 10.3 Bank 的数学定义

与 `doc/计划方案.md` 相同。E3 Bank 中不存在 future teacher key 或教师分布，只保存由目标历史 context 产生的 event/node key、历史 future、掩码、level features 和时间元数据。

## 11. 两阶段历史检索

本节与 `doc/计划方案.md` 相同。E3 只期望获得更好的 key，不改变实际检索算法。

### 11.1 Calendar Filter

与 `doc/计划方案.md` 相同：使用 weekday-slot 和严格因果条件过滤历史事件。

### 11.2 事件级粗检索

与 `doc/计划方案.md` 相同：使用 event key cosine score 保留 top-\(R\)。

#### 如何选择 \(R\)

与 `doc/计划方案.md` 相同。若合法日历候选数始终小于 \(R\)，事件级粗检索不产生截断，应在最终简化版本中删除该冗余步骤。

### 11.3 节点级精排

与 `doc/计划方案.md` 相同：在事件候选子集内按节点 key 和 level score 精排。

### 11.4 检索权重

与 `doc/计划方案.md` 相同：节点 top-\(K\) score 通过 search-temperature softmax 形成聚合权重。

## 12. 历史 Future 与节点级聚合

本节与 `doc/计划方案.md` 相同。

### 12.1 取出节点候选 Future

仍从目标 Bank 读取每个节点选中事件的真实历史 future，形成 \(B\times H\times N_t\times K\times C\)。

### 12.2 第一版不做数值对齐

与 `doc/计划方案.md` 相同，固定 No Alignment。

### 12.3 历史记忆预测 \(\mathbf Y^{\mathrm{mem}}\)

与 `doc/计划方案.md` 相同，对节点 top-\(K\) 历史 future 做 mask-aware 加权聚合。

### 12.4 节点候选分歧 \(\mathbf V^{\mathrm{mem}}\)

与 `doc/计划方案.md` 相同，计算候选 future 围绕 \(\mathbf Y^{\mathrm{mem}}\) 的加权方差。

## 13. 下游基础预测

与 `doc/计划方案.md` 相同。下游 backbone 只读取最近 12 步，E3 不改变其网络、输入和损失。

## 14. Mirage Confidence Head

本节与 `doc/计划方案.md` 相同。Future-Relation 只优化离线 key，不能替代目标域 confidence。

### 14.1 Confidence 要回答的问题

与 `doc/计划方案.md` 相同：估计当前节点和 horizon 上使用历史 memory 是否比 base prediction 更可能有帮助。

### 14.2 六类置信度特征

特征种类和维度与 `doc/计划方案.md` 相同。

#### 特征一：最佳形状相似度

使用 E3 key 的最佳节点检索相似度，定义与原计划相同。

#### 特征二：Top-1 与 Top-2 间隔

与 `doc/计划方案.md` 相同。

#### 特征三：权重集中度

与 `doc/计划方案.md` 相同。

#### 特征四：候选 future 分歧

与 `doc/计划方案.md` 相同，读取 \(\mathbf V^{\mathrm{mem}}\)。

#### 特征五：水平匹配程度

与 `doc/计划方案.md` 相同。

#### 特征六：Memory 与 Base 的分歧

与 `doc/计划方案.md` 相同。

#### 特征来源汇总

与 `doc/计划方案.md` 相同。E3 不增加 teacher entropy 等源域训练统计作为目标域 confidence 输入，避免训练/推理语义不一致。

### 14.3 Confidence MLP

与 `doc/计划方案.md` 相同。

### 14.4 Confidence 监督标签

与 `doc/计划方案.md` 相同，只允许使用目标训练/calibration 段真实 future 构造，不读取 test。

## 15. Safe Residual Fusion

与 `doc/计划方案.md` 相同：

$$
\mathbf Y^{\mathrm{final}}
=
\mathbf Y^{\mathrm{base}}
+
\boldsymbol\rho\odot
\left(
\mathbf Y^{\mathrm{mem}}-mathbf Y^{\mathrm{base}}
\right)
$$

并保留无有效 memory 时精确回退 base prediction 的约束。

#### 各模块的决策粒度

与 `doc/计划方案.md` 相同：事件级粗检索按样本，节点级精排按节点，confidence 和 fusion 按节点与 horizon。

## 16. 训练、建库与推理顺序

### 16.1 源数据集预训练

1. 构造 288 步 retrieval context、12 步 future 和原始观测掩码。
2. 干净 context 经过 encoder 与 retrieval head 得到 \(\mathbf K^X\)。
3. 掩码 context 经过共享 encoder 与线性 head 计算 \(\mathcal L_{\mathrm{mask}}\)。
4. 使用真实 future 计算固定教师分布 \(\mathbf Q^Y\)。
5. 使用 node key 计算学生分布 \(\mathbf P^X\)。
6. 计算 \(\mathcal L_{\mathrm{rel}}\) 和总损失，更新 embedding、encoder、retrieval head 和 reconstruction head。
7. 保存 best-total 与 best-relation checkpoint。

### 16.2 目标 adapter 适配，可选

与 `doc/计划方案.md` 相同。第一轮 E3 实验不启用 adapter。

### 16.3 构建目标 bank

加载选定 E3 checkpoint，仅用目标训练历史 context 编码 key。未来教师分布不参与建库。

### 16.4 训练 confidence 与 fusion

与 `doc/计划方案.md` 相同，并与现有 E2 下游模式保持完全一致以完成单变量归因。

### 16.5 推理

推理只执行：当前 context 编码、目标 Bank 检索、历史 future 聚合、base prediction、confidence 和 safe fusion。待预测真实 future 不可访问。

## 17. METR-LA 到 PEMS-BAY 的维度示例

### 17.1 源域预训练

| 张量 | 维度 | 含义 |
|---|---|---|
| METR-LA retrieval context | \(16\times288\times207\times1\) | 一天历史 |
| clean/masked token | \(16\times24\times207\times96\) | 两个 encoder 视图 |
| node key | \(16\times207\times48\) | 学生检索 key |
| true future | \(16\times12\times207\times1\) | 固定教师来源 |
| future distance \(\mathbf D^Y\) | \(16\times16\times207\) | 节点级 future 关系 |
| teacher/student distribution | \(16\times16\times207\) | 软关系监督 |

### 17.2 目标 query 与 bank

| 张量 | 维度 | 含义 |
|---|---|---|
| PEMS-BAY query context | \(B\times288\times325\times1\) | learned retrieval 输入 |
| PEMS-BAY forecast context | \(B\times12\times325\times1\) | base/raw-L1 输入 |
| query node key | \(B\times325\times48\) | 节点 query |
| query event key | \(B\times48\) | 事件 query |
| bank node key | \(M\times325\times48\) | 目标历史 key |
| bank future | \(M\times12\times325\times1\) | 目标历史真实 future |

### 17.3 检索、聚合与融合

与 `doc/计划方案.md` 相同，只将 retrieval key 维度更新为 48；top-\(K\) 权重、候选 future、memory prediction、variance、confidence 和 final prediction 的形状均不变。

## 18. 可行性审查要点

### 18.1 历史 block 是否真的有信息

沿用 `doc/计划方案.md` 的 historical statistics、raw-L1、learned retrieval、top-1/top-\(K\) 对照。当前 E2 已证明候选池有价值，但 E3 仍必须在同一候选池和共同有效位置上重新比较。

### 18.2 预训练是否真的迁移

本节保留原计划的 random、source frozen、adapter 和 target-specific 对照，并增加源域关系学习诊断：

1. \(\mathcal L_{\mathrm{rel}}\) 是否下降且没有教师/学生分布坍缩；
2. context key 排名与 future 排名的 Future NDCG@K 是否提高；
3. E3 learned Top-1 和 weighted Top-K 是否优于当前 E2；
4. 在 PEMS-BAY 重建 Bank 后，source frozen E3 是否优于 source frozen E2 和 target random。

只有源域检索改善和目标域迁移至少一项成立，才能保留 E3；若只降低训练时间但检索质量下降，则不能作为模型优化保留。

### 18.3 Confidence 是否真的识别有害历史

与 `doc/计划方案.md` 相同，报告 AUROC、AUPRC、Brier、ECE、分桶 gain 和 harmful-memory subset。

### 18.4 是否有必要研究数值对齐

与 `doc/计划方案.md` 相同。E3 首轮继续固定 No Alignment。

## 19. 防泄漏与因果约束

沿用 `doc/计划方案.md` 的 train-only scaler、严格时间划分、候选完整 future 早于 query context、validation/test 不写入 Bank 等规则。

E3 额外规定：

1. \(\mathbf Y\) 只在源域预训练 loss 和 validation loss 中生成教师分布；
2. 建库和推理接口不得接受 query future；
3. teacher distribution 必须 `stop-gradient`，它由数据计算而非可学习参数产生；
4. test future 只能用于最终评估，不能选择 \(\tau_Y\)、\(\tau_X\)、checkpoint 或 Bank；
5. batch 内关系候选必须排除自身与时间重叠事件。

因此 E3 属于利用自然未来结果构造预训练监督，不属于推理时未来泄漏。

## 20. 必做消融与指标

### 20.1 编码器与损失

第一组只替换一个变量：

| 实验 | Encoder | Reconstruction | Retrieval supervision |
|---|---|---|---|
| E2 baseline | 一天 E2 | 保留 | quantile positive + mirage hard negative |
| E3 main | 与 E2 完全相同 | 保留 | Future-Relation soft contrastive |

E3 通过后再做必要消融：

- \(\mathcal L_{\mathrm{mask}}\) only；
- \(\mathcal L_{\mathrm{rel}}\) only；
- 两者联合，主方法；
- hard one-hot future teacher，检验软标签价值；
- \(\tau_Y\) 的小范围教师支持数对照；
- 保留/移除 temporal attention 和 spatial attention，沿用原计划；
- 不增加 future encoder，除非固定教师方案已被证据否定。

### 20.2 检索与聚合

与 `doc/计划方案.md` 相同：calendar、raw-L1、event/node retrieval、shape/level、\(K\) 和 \(R\) 对照。E2 与 E3 必须使用相同 Bank 历史范围、候选过滤和聚合参数。

### 20.3 Confidence 与 Fusion

与 `doc/计划方案.md` 相同。只有 E3 learned retrieval 先通过 horizon-only 对照，才训练 confidence，避免 confidence 掩盖 key 质量。

### 20.4 指标

预训练关系指标：

- teacher-student KL/cross-entropy；
- teacher \(K^Y_{\mathrm{eff}}\) 的均值、q10、中位数和 q90；
- student \(K^X_{\mathrm{eff}}\)；
- key 方差、平均两两 cosine 和有效秩，用于检查坍缩。

检索指标：Future Recall@K、Future NDCG@K、learned Top-1 future MAE、learned uniform/weighted Top-K future MAE、Oracle gap。Mirage separation AUC 保留为离线诊断，不再作为训练标签。

效率指标：epoch 时间、train batch 时间、峰值 CUDA 显存和损失计算时间。不得宣称整体加速 24 倍，因为一天 encoder 本身仍然存在。

预测、confidence 和正式多 seed 统计要求与 `doc/计划方案.md` 相同。

当前 E2 validation 参考值：

| 指标 | E2 |
|---|---:|
| learned Top-1 MAE | 4.2631 |
| learned weighted Top-K MAE | 3.7600 |
| raw-L1 Top-K MAE | 3.9892 |
| learned Top-1 - Oracle Top-1 | 1.4918 |
| epoch 时间 | 10.78 分钟 |

E3 保留规则：

- 明确保留：Top-1 和 weighted Top-K 至少一个优于 E2，另一个退化不超过 0.03，同时 Future NDCG@K 提高；
- 条件保留：检索指标基本持平但 epoch 时间下降至少 15%，进入一个下游 seed 验证；
- 删除：weighted Top-K MAE 不低于 3.80、Top-1 不低于 4.31，或出现 key/教师分布坍缩。

## 21. 实施顺序

1. 固定当前 E2 数据、encoder、mask、batch、seed、Bank 和诊断协议。
2. 为 future masked distance、合法 pair mask、教师 softmax 和学生 softmax 编写独立张量测试。
3. 验证缺失 future、零合法候选、单合法候选被排除，以及非重叠约束。
4. 实现 \(\mathcal L_{\mathrm{rel}}\)，删除 E3 路径中的 context distance、quantile positive 和 hard-negative 权重。
5. 增加 \(\tau_Y\)、\(\tau_X\)、\(\lambda_{\mathrm{rel}}\) 配置，同时保留旧 E2 loss 模式用于单变量对照。
6. 使用固定真实 batch 比较 E2/E3 的 forward、loss、梯度和耗时，删除 smoke 产物。
7. 运行 3 epoch pilot，检查 teacher/student 支持数、key 方差、有限梯度和速度。
8. pilot 通过后运行一个 seed 的完整 E3 预训练，同时保存 best-total 和 best-relation。
9. 使用两个 checkpoint 分别构建完整 METR-LA Bank，并运行完整 retrieval diagnostics。
10. 按第 20.4 节规则决定保留 E3、条件进入下游或删除 E3。
11. 只有 E3 检索通过后，才运行 horizon-only 和 confidence 下游对照。
12. 单 seed 机制成立后再补三 seed，不提前扩展到多源、频域或 future encoder。

## 22. 论文主张边界与相关工作

若实验成立，E3 最合适的机制描述是：

> Future-guided relational soft contrastive pretraining for transferable spatio-temporal historical retrieval.

当前不能直接声称：

- 首次提出软对比学习或关系蒸馏；
- 已构建通用时空基础模型；
- 已解决时空海市蜃楼；
- 对任意城市、节点数、采样间隔或物理变量都能迁移。

与相关方向的边界：

- 与普通 supervised contrastive learning 的区别是：E3 使用连续 future 距离形成候选分布，而不是类别标签；
- 与经典 knowledge distillation 的区别是：E3 没有 trainable teacher network，只蒸馏由真实 future 定义的关系；
- 与原 Future-Guided hard-negative loss 的区别是：E3 删除 context 距离和二值分位数标签，直接学习完整候选排序；
- masked reconstruction、单源迁移、目标 Bank、confidence memory 和 safe fusion 的相关工作关系与 `doc/计划方案.md` 相同。

论文中只有在 E3 稳定提高 Future NDCG、retrieved future MAE 和下游预测，并通过 E2 hard-label 对照后，才能将“future relation supervision”作为核心创新点。若只获得速度收益，应将其定位为训练目标简化，而不是主要科学贡献。
