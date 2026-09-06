# STAnchor-BlockMemory 当前最终结构说明
## Retrieval-Aware MHA Residual Router

> 文档版本：2026-09-01 主线冻结版
> 适用范围：METR-LA 主线、同物理量下游迁移与后续跨数据集验证
> 对应代码：stanchor/models/encoder.py、stanchor/models/pretraining.py、stanchor/models/retrieval_head.py、stanchor/retrieval/retriever.py、stanchor/retrieval/strategies.py、stanchor/models/retrieval_router.py、stanchor/engine/target.py

## 1. 版本定位与研究目标

### 1.1 当前版本解决的问题

STAnchor-BlockMemory 是一个“冻结下游基础预测、利用历史事件检索进行后验校准”的插件式系统。给定当前交通观测窗口，系统先由一个下游预测 backbone 产生基础预测，再从目标数据集训练历史中检索若干相似事件，最后由校准器决定是否以及在哪些预测 horizon 上施加历史候选的修正。

当前最终版本的核心假设是：

1. 检索器负责学习哪些历史事件在时空和未来演化上相似；
2. Bank 负责以事件为单位保存检索所需的 key、未来片段和元数据；
3. 下游 backbone 只负责产生 Base prediction；
4. Router 负责在 Base 与历史候选之间进行逐 horizon 的残差路由；
5. 当历史候选不可靠或不存在时，Base 必须能够被显式保留。

因此，本版本的研究重点不是重新训练或替换下游 backbone，而是验证一个可解释的候选选择机制能否稳定改善冻结的基础预测。

### 1.2 当前冻结的主线

当前主线固定为：

~~~text
HN-OffsetDecay v2 检索编码器
    -> 目标训练历史 Bank
    -> weekday_radius1_overlap 候选协议
    -> node-level Top-12
    -> Retrieval-Aware 4-head MHA Residual Router
    -> 冻结 backbone 的后验校准
~~~

下列内容不属于当前主线：

- 旧版独立 Alpha memory gate；
- 旧版自由 Beta additive correction；
- 旧版 confidence/fusion 作为 error-aware 主路径；
- 已暂停的 candidate quality teacher loss；
- 已提出但尚未纳入当前实现的 event-aligned candidate graph；
- 多源联合预训练、FAISS 近似检索和跨物理量迁移。

旧接口仍保留少量兼容字段，使历史 checkpoint 和诊断脚本可以被读取；兼容字段不代表旧模块仍参与当前 Router 的预测计算。

## 2. 数据契约与符号约定

### 2.1 原始数据

原始交通序列记为：

\[
X^{raw}\in\mathbb{R}^{L\times N\times C},
\]

其中 \(L\) 是全序列时间步数，\(N\) 是传感器节点数，\(C\) 是输入变量通道数，METR-LA 当前为 \(C=1\)。

每个时间步带有 timestamp、weekday、slot 和 observed。5 分钟采样时，一天有 \(S_{day}=288\) 个 slot。observed 表示某个节点和通道是否有有效观测。

数据按时间顺序划分为 train、validation 和 test。节点标准化器只使用 train 段拟合：

\[
\widetilde X_{t,n,c}
=
\frac{X^{raw}_{t,n,c}-\mu_{n,c}}
{\sigma_{n,c}+\epsilon}.
\]

所有训练损失和评估指标都使用 observed mask 排除无效值。

### 2.2 两种 context 长度

当前代码中有两个不同的时间窗口：

| 名称 | 符号 | 当前值 | 用途 |
|---|---:|---:|---|
| 下游 context | \(T\) | 12 | backbone 与校准器的即时输入 |
| 检索 context | \(T_r\) | 288 | 检索编码器构造历史事件 key |

对一个样本，数据集返回：

- x：最近 \(T=12\) 步，形状 [B,T,N,C]；
- retrieval_x：最近 \(T_r=288\) 步，形状 [B,T_r,N,C]；
- y：接下来的 \(H=12\) 步，形状 [B,H,N,C]；
- context_start、context_end、future_end；
- query 的 weekday 和 slot；
- 稳定的 sample_id=context_end。

288 步仅用于检索器预训练、Bank key 构造和查询 key 提取。校准器不再重复处理 288 步序列，而是使用 12 步下游 context、Base 输出和已经缓存的 retrieval node key。

### 2.3 统一张量轴

除特别说明外，当前项目使用：

~~~text
x / y                    [B,T/H,N,C]
retrieval_x              [B,T_r,N,C]
patch token              [B,P,N,D_e]
node key                 [B,N,D_r]
event key                [B,D_r]
event candidate ids      [B,R]
node candidate ids       [B,N,K]
candidate future         [B,H,N,K,C]
router weights            [B,N,H,K+1]
final prediction          [B,H,N,C]
~~~

其中 \(B\) 是 batch size，\(P=T_r/\text{patch\_size}=24\)，\(K=12\) 是历史候选数，\(K+1=13\) 还包含 Base token。

## 3. 从样本到事件的时间窗口

对数据集中的一个事件，以 context_end=t 表示当前观测窗口的最后一步：

\[
\begin{aligned}
\text{retrieval context} &= [t-T_r+1,\ t],\\
\text{downstream context} &= [t-T+1,\ t],\\
\text{query future} &= [t+1,\ t+H].
\end{aligned}
\]

事件的 sample_id 等于 \(t\)。Bank 中每条事件记录保存自己的 context/future 时间边界，检索协议据此判断候选是否合法。

查询样本的真实 future \(Y^{query}\) 只在训练标签、验证指标和离线 oracle 诊断中使用。它不会进入 query key、calendar 候选筛选、node key 相似度、Router 候选特征或 frozen-path cache。

## 4. HN-OffsetDecay v2 检索编码器

### 4.1 编码器输入和目标

检索编码器接收：

\[
\widetilde X^{r}\in\mathbb{R}^{B\times288\times N\times1}
\]

及其 observed mask、weekday 和 slot。它学习将每个历史事件映射为：

- 每节点 key：\(z^{node}\in\mathbb{R}^{N\times64}\)；
- 事件 key：\(z^{event}\in\mathbb{R}^{64}\)。

检索器的学习目标不是直接预测未来数值，而是让 key 空间中的相似度反映事件未来演化的相似程度。未来值因此只在预训练 teacher 中出现，部署时不需要当前 query 的未来。

### 4.2 TemporalPatchEmbedding

时间序列首先按 patch_size=12 切成不重叠 patch：

\[
P=\frac{T_r}{12}=24.
\]

每个 patch 将连续 12 步的数值、level 特征和时间信息投影到 \(D_e=128\) 维 token：

\[
E_{b,p,n}
=
E_{\mathrm{patch}}
\left(
\widetilde X^{r}_{b,\mathcal{I}_p,n,:},
\mathrm{level}_{b,n},
\mathrm{weekday}_{b,\mathcal{I}_p},
\mathrm{slot}_{b,\mathcal{I}_p}
\right)
\in\mathbb{R}^{128}.
\]

输出形状为：

\[
E\in\mathbb{R}^{B\times24\times N\times128}.
\]

Patch 化的作用是把 288 步长序列压缩成 24 个时间 token，避免在完整 288 步上执行高成本注意力。

### 4.3 FactorizedSTEncoder

FactorizedSTEncoder 由 4 个相同结构的 block 组成，每个 block 顺序包含：

1. 时间多头自注意力；
2. 基于真实图边的稀疏空间注意力；
3. 可选的 history-conditioned route 分支；
4. 前馈网络和残差归一化。

#### 时间交互

对每个节点独立处理 patch 序列：

\[
E^{time}
=
\operatorname{MHA}_{time}
\left(
E_{:, :, n, :}
\right).
\]

实现中将 [B,P,N,D_e] 重排为 [B*N,P,D_e]，因此时间注意力只在同一节点的 24 个 patch 之间发生，不会把不同节点错误拼为同一序列。

#### 图空间交互

图空间分支接收交通图的 edge_index 和 edge_weight，只在显式边集合上计算多头注意力：

\[
s_{b,p,(i,j),h}
=
\frac{
q_{b,p,i,h}^{\top}k_{b,p,j,h}
}{
\sqrt{d_h}
}
+
\gamma\log(A_{ij}+\epsilon).
\]

对每个目标节点 \(i\) 的入边做 softmax，再聚合源节点 \(j\) 的 value。该实现不会 materialize 完整的 \(N\times N\) 注意力矩阵，空间复杂度随边数 \(E\) 增长。

#### History-conditioned route

配置中 route_enabled=true、route_top_k=6、route_local_quota=0。该分支根据历史 token summary 为每个节点选择固定数量的远程节点，使用低秩 value_down/value_up 传递补充消息。它是检索编码器内部的空间补充机制，不是下游 Router 的候选事件图传播。

经过 4 层后得到：

\[
H\in\mathbb{R}^{B\times24\times N\times128}.
\]

### 4.4 RetrievalHead

RetrievalHead 首先在 24 个 patch 上学习池化权重：

\[
a_{b,p,n}
=
\operatorname{softmax}_p
\left(
w^\top\tanh(W_pH_{b,p,n})
\right),
\]

再得到节点级隐藏状态：

\[
u_{b,n}
=
\sum_{p=1}^{24}a_{b,p,n}H_{b,p,n}
\in\mathbb{R}^{128}.
\]

当前配置启用 adapter_bottleneck_dim=96 的轻量 domain adapter：

\[
u'_{b,n}=u_{b,n}
+
W_{up}\,\operatorname{GELU}(W_{down}u_{b,n}).
\]

最后通过 key_mlp: 128 -> 128 -> 64 并做 L2 归一化：

\[
z^{node}_{b,n}
=
\frac{
E_{key}(u'_{b,n})
}{
\lVert E_{key}(u'_{b,n})\rVert_2+\epsilon
}
\in\mathbb{R}^{64}.
\]

事件 key 是节点 key 的均值后归一化：

\[
z^{event}_{b}
=
\operatorname{normalize}
\left(
\frac{1}{N}\sum_{n=1}^{N}z^{node}_{b,n}
\right)
\in\mathbb{R}^{64}.
\]

不同节点的 node key 保留节点差异；event key 只是兼容事件级粗检索和 Bank schema 的全局摘要。

### 4.5 单视图 masked relation 预训练

当前正式配置使用：

~~~yaml
pretrain:
  objective: masked_relation_single_view
  retrieval_loss_mode: relation
  reconstruction_weight: 2.0
  retrieval_weight: 1.0
~~~

forward_pretrain_single_view 对一个 batch 只执行一次 masked encoder forward，同时服务于 reconstruction 和 retrieval relation 两个目标，避免 clean view 与 masked view 的双倍计算。

#### 掩码

- time mask ratio：0.25；
- space mask ratio：0.25；
- time block size：36 个原始时间步；
- 训练阶段按配置采样 time/space 任务；
- 验证阶段交替使用 time 和 space mask。

掩码只作用于输入可见性，不会把真实 future 放入模型输入。

#### OffsetDecay teacher relation

对 batch 内事件 \(i,j\)，先根据 context endpoint 和 future 构造 OffsetDecay 语义签名。该签名在近端 horizon 对齐事件 level，在远端 horizon 逐渐回到原始 future：

\[
\phi_{i,h,n}
=
Y_{i,h,n}
-
\lambda_h\,\ell_{i,n},
\qquad
\lambda_h\ \text{从 1 线性衰减到 0}.
\]

使用 symmetric_geometric_mean 对 pairwise future distance 做尺度归一化，得到 teacher 距离 \(d^{OD}_{ij,n}\)。teacher 分布为：

\[
p^{T}_{ij,n}
=
\operatorname{softmax}_{j}
\left(
-\frac{d^{OD}_{ij,n}}{\tau_T}
\right),
\qquad \tau_T=0.1.
\]

student 使用 L2-normalized node key 的余弦相似度：

\[
s^{S}_{ij,n}
=
\left(z^{node}_{i,n}\right)^\top z^{node}_{j,n},
\]

\[
p^{S}_{ij,n}
=
\operatorname{softmax}_{j}
\left(
\frac{s^{S}_{ij,n}}{\tau_S}
\right),
\qquad \tau_S=0.1.
\]

关系损失是对有效 anchor 的交叉熵：

\[
\mathcal{L}_{rel}
=
-\frac{1}{|\mathcal{A}|}
\sum_{(i,n)\in\mathcal{A}}
\sum_j
p^{T}_{ij,n}\log p^{S}_{ij,n}.
\]

有效 pair 需有可比较的 context/future 观测并满足时间非重叠约束。真实 future 只用于训练 teacher；部署时只计算 key。

#### Reconstruction loss

masked reconstruction head 将 masked hidden 投影回 patch 数值：

\[
\widehat X^{mask}
=
D_{rec}(H^{mask}).
\]

只在人工 mask 且原始观测有效的位置计算 SmoothL1：

\[
\mathcal{L}_{rec}
=
\operatorname{SmoothL1}
\left(
\widehat X^{mask},X^{target}
\right)_{\text{masked, observed}}.
\]

总预训练目标为：

\[
\mathcal{L}_{pretrain}
=
2.0\,\mathcal{L}_{rec}
+
1.0\,\mathcal{L}_{rel}.
\]

当前 profile loss、rank loss 和额外 relation projection 分支均未启用。

## 5. 事件 Bank 与缓存

### 5.1 Bank 的作用

Bank 是由目标数据集训练历史构建的不可变事件库。它把昂贵的历史事件编码从每个下游训练 epoch 中移到离线阶段，使训练时只需读取 query 的 retrieval node key、按时间协议获得合法事件池、用 node key 做节点级 Top-K，并从 Bank 读取候选 future 和元数据。

### 5.2 Bank 构建范围

当前 build_bank.py 对 data.train 的前 memory_fraction=0.7 事件构建 Bank。validation/test 不写入 Bank。

每条事件保存：

~~~text
event_keys.npy       [M,64]       float16
node_keys.npy        [M,N,64]     float16
future_values.npy    [M,H,N,C]    float32
future_masks.npy     [M,H,N,C]    uint8
level_features.npy   [M,N,4C]     float32
weekday.npy          [M]
slot.npy             [M]
context_start.npy    [M]
context_end.npy      [M]
future_end.npy       [M]
sample_id.npy        [M]
~~~

同时保存 calendar_offsets.npy、calendar_event_ids.npy 和 manifest.json。

manifest 校验 schema version、节点数、horizon、retrieval dimension、encoder fingerprint、graph fingerprint 和 scaler 状态，防止不同编码器或不同标准化器生成的 Bank 被混用。

### 5.3 CPU mmap 与 GPU 张量边界

Bank 的大数组使用 NumPy read-only mmap，存放在磁盘和 CPU 地址空间。事件 key 的小数组会复制到内存供粗粒度查询。每个 batch 只把当前候选相关的 key、future 和 mask 搬到 GPU。

因此缓存不等于把完整 Bank 放进 GPU，也不会构造全数据集的 hidden 表示。frozen_path_cache 是训练过程中的 CPU 字典，按 sample_id 保存 base_prediction、node_candidates、aggregation 和 retrieval_node_keys；这些张量均 detached 后存放。训练和 validation 各自维护独立 cache，避免路径顺序或 split 混用。

## 6. 当前候选协议：weekday_radius1_overlap

### 6.1 语义

weekday_radius1_overlap 不是“前后五分钟放宽”，而是沿星期维度扩展同一日内时间槽。

对于 query 的星期 \(w_q\) 和时间槽 \(s_q\)，候选事件来自：

\[
(w_q-1,s_q),\quad
(w_q,s_q),\quad
(w_q+1,s_q),
\]

其中星期按模 7 处理。

以周三 08:00 为例，候选池包括周二 08:00、周三 08:00 和周四 08:00。这使候选池相对于 exact calendar 具有约三倍的时间来源，同时保留同一日内时刻的可解释语义。

### 6.2 因果和重叠边界

当前协议允许历史候选的 future 与 query 最近 12 步 context 重叠，但要求候选 future 已经结束于 query context 结束之前：

\[
future\_end(e)\le context\_end(q).
\]

因此候选 future 中最多包含 query 在当前时刻已经观察到的历史值，不包含 query 的未来。该 overlap 是有意保留的协议设计，用于让模型看到前后相邻日期同一时段的历史演化。

exact/relaxed 的其他协议仍使用更严格的历史边界，但不属于当前正式 Router 主线。

### 6.3 Event pool 与 node rerank

在当前 error-aware 路径中：

1. calendar_event_candidates 根据 weekday、slot 和时间边界建立合法事件池；
2. 对合法事件数量执行 event_top_r=32 的容量检查；
3. 使用 query 的 node key 与 Bank node key 计算逐节点相似度；
4. 取 node_top_k=12；
5. 使用 level_weight=0，因此当前精排只由 node-key 相似度决定；
6. 对 Top-12 分数以 search_temperature=0.1 softmax，得到初始 node candidate weights。

Event key 仍保存在 Bank，并被通用 TwoStageRetriever.search_events 使用；当前 weekday-radius 主路径先由 calendar 协议定义合法事件集合，再由 node key 完成节点级语义排序。这样既避免把不同节点平均成一个 event key，也保留了事件级接口兼容性。

### 6.4 NodeCandidates 与 AggregationOutput

节点级候选结构为：

~~~text
event_ids       [B,N,K]
total_scores    [B,N,K]
shape_scores    [B,N,K]
level_distances [B,N,K]
weights         [B,N,K]
valid           [B,N,K]
~~~

候选 future 读取后为：

~~~text
candidate_futures [B,H,N,K,C]
candidate_masks   [B,H,N,K,C]
~~~

这是数值候选张量，而不是 256 维 hidden 张量。以 METR-LA、B=32、H=12、N=207、K=12、C=1 为例，单个 float32 candidate future 约为 3.8 MB，candidate mask 约为 1 MB；它不会产生文档早期提到的 [B,N,K,H,64] hidden 张量。

## 7. OffsetDecay 候选 future

当前 error-aware 路径不会直接把 Bank 中的原始 future 当作校准候选，而是先执行 offset_decay_aggregation。

对 query 最近 12 步 context 估计 endpoint level \(\ell^q_n\)，对每个候选事件估计 endpoint level \(\ell^k_n\)。对原始候选 future \(Y^{raw}_{h,n,k}\) 先做 level 对齐：

\[
Y^{align}_{h,n,k}
=
Y^{raw}_{h,n,k}
+
\ell^q_n-\ell^k_n.
\]

然后按 horizon 使用从 1 衰减到 0 的系数：

\[
Y^{cand}_{h,n,k}
=
Y^{raw}_{h,n,k}
+
\lambda_h
\left(
Y^{align}_{h,n,k}-Y^{raw}_{h,n,k}
\right),
\qquad
\lambda_h:1\rightarrow0.
\]

因此近端预测更多继承 query 当前 level，远端预测更多保留历史候选自身 future 形状，候选缺失位置通过 candidate mask 排除。Router 使用的 \(Y^{cand}\) 指这个经过 OffsetDecay 的候选 future。

## 8. 下游基础预测与训练边界

### 8.1 Backbone 接口

当前支持四个下游 backbone：

- Graph WaveNet（GWN）；
- STGCN；
- STAEformer；
- ARGCN。

它们统一接收：

\[
x\in\mathbb{R}^{B\times12\times N\times1}
\]

并输出：

\[
Y^{base}\in\mathbb{R}^{B\times12\times N\times1}.
\]

STAEformer 额外接收数据集提供的 slot 与 weekday calendar covariates；其他 backbone 使用各自图或时序接口。

### 8.2 Post-hoc frozen-base

当前正式 Router 实验使用：

~~~yaml
target:
  training_protocol: posthoc_frozen_base
  training_data_scope: full_train
  downstream_mode: learned_topk_error_aware
  calibrator_arch: retrieval_aware_mha_router
~~~

执行顺序为：

1. 加载已训练好的 HN-OffsetDecay v2 checkpoint，并冻结；
2. 加载与当前 backbone、split、seed 匹配的 Base-only checkpoint；
3. 冻结 backbone；
4. 读取或构建 retrieval path cache；
5. 只更新 Router 参数。

Base checkpoint 的 backbone fingerprint 在每个 epoch 后核对，发生变化就停止训练。这样可以保证校准器收益不被 backbone 重新训练混淆。

## 9. Retrieval-Aware MHA Residual Router

### 9.1 Router 的输入输出

Router 输入：

~~~text
history             [B,T=12,N,C]
base                [B,H=12,N,C]
candidate_futures   [B,H,N,K,C]
candidate_masks     [B,H,N,K,C]
NodeCandidates      [B,N,K]
retrieval_node_keys [B,N,64]
~~~

Router 输出：

~~~text
final_prediction      [B,H,N,C]
historical_mass       [B,H,N,1]
contributions         [B,H,N,2]
learned_memory        [B,H,N,C]
routing_weights       [B,N,H,K+1]
mha_attention_weights [B,N,4,H,K+1]
~~~

### 9.2 Query state encoder

Router 的 _state 对 12 步下游 context 按 observed mask 做节点内时间标准化，并与 Base prediction 拼接：

\[
S^{in}_{b,n}
=
\operatorname{flatten}
\left(
\operatorname{normalize}(X^{ctx}_{b,:,n,:}),
Y^{base}_{b,:,n,:}
\right)
\in\mathbb{R}^{24}.
\]

当前 \(C=1,T=H=12\)，所以输入维度为 24。状态编码器为：

~~~text
Linear(24,256)
-> GELU
-> Linear(256,256)
-> GELU
~~~

得到 \(S_{b,n}\in\mathbb{R}^{256}\)。该状态只使用已观测 context 和 Base，不使用 query future。

### 9.3 Retrieval node key 条件

缓存中的 query node key：

\[
z^q_{b,n}\in\mathbb{R}^{64}
\]

经过 Linear(64,128) -> GELU -> Linear(128,256)，映射为 key condition，再与 \(S_{b,n}\) 拼接后经 query_fusion：

~~~text
Linear(512,256)
-> GELU
-> Linear(256,256)
~~~

得到节点级 query base state \(Q^0_{b,n}\)。

对每个 horizon 加可学习 horizon embedding：

\[
Q_{b,n,h}
=
Q^0_{b,n}+e_h,
\qquad
Q\in\mathbb{R}^{B\times N\times H\times256}.
\]

### 9.4 Candidate residual

对第 \(k\) 个历史候选定义 residual：

\[
\Delta_{b,h,n,k}
=
Y^{cand}_{b,h,n,k}
-
Y^{base}_{b,h,n}.
\]

形状为：

\[
\Delta\in\mathbb{R}^{B\times H\times N\times K\times C}.
\]

候选 validity 同时考虑 candidate_masks 的 horizon/channel 观测、NodeCandidates.valid 以及候选 future 是否存在有限值。代码将候选在任一有效 horizon/channel 上有观测且 NodeCandidates.valid 为真视为可路由历史 expert；完全没有有效观测的候选会被屏蔽。Base token 永远有效。

### 9.5 Candidate summary encoder

每个节点、每个候选计算以下摘要：

\[
[
\operatorname{mean}_h(\Delta),
\operatorname{std}_h(\Delta),
\Delta_{H},
\operatorname{mean}_h(|\Delta|),
\operatorname{mean}_h(\mathbf{1}_{\Delta>0})
]
\]

以及四个元数据：shape score、负 level distance、从 1 到 0 的 candidate rank、valid ratio。

当前 \(C=1\)，输入维度为 \(5C+4=9\)。编码器为：

~~~text
Linear(9,64)
-> GELU
-> Linear(64,256)
~~~

得到 \(C^{summary}_{b,n,k}\in\mathbb{R}^{256}\)。

### 9.6 Candidate trajectory encoder

为了保留 horizon 轨迹形状，Router 另外拼接：

\[
[\Delta_{1:H,n,k},|\Delta_{1:H,n,k}|]
\in\mathbb{R}^{2HC}
=
\mathbb{R}^{24}.
\]

其实际批量形状先从 [B,H,N,K,C] 重排为 [B,N,K,H*C*2]，再经过：

~~~text
Linear(24,64)
-> GELU
-> Linear(64,256)
~~~

得到 trajectory token，并与 summary token 相加：

\[
C_{b,n,k}
=
C^{summary}_{b,n,k}
+
C^{traj}_{b,n,k}
\in\mathbb{R}^{256}.
\]

最终只生成：

\[
C\in\mathbb{R}^{B\times N\times K\times256},
\]

不会生成 [B,H,N,K,256]。无效历史候选的 token 被置零。

### 9.7 Base token

Base 不是一个全零输入占位符，而是一个有语义的 fallback token。它使用 query state、retrieval node key condition、context 波动性和 Router 的 Base risk probe 输出。

输入维度为 \(256+256+2=514\)，编码器为：

~~~text
Linear(514,256)
-> GELU
-> Linear(256,256)
~~~

再加上可学习 base_type embedding：

\[
C^{base}_{b,n}\in\mathbb{R}^{256}.
\]

将历史候选与 Base token 拼接：

\[
C^{all}
=
[
C_{b,n,1},\ldots,C_{b,n,K},C^{base}_{b,n}
]
\in\mathbb{R}^{B\times N\times(K+1)\times256}.
\]

Base token 的 residual 定义为零：

\[
\Delta_{b,h,n,K+1}=0.
\]

### 9.8 四头标准 MHA

MHA 的 query、key、value 输入为：

~~~text
query: [B*N,H,256]
key:   [B*N,K+1,256]
value: [B*N,K+1,256]
~~~

使用 PyTorch 官方 nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.05, batch_first=True)。每个 head 的维度为 64。历史候选 validity 通过 key_padding_mask 传入，Base token 永远不 mask。

MHA 输出为：

\[
O=\operatorname{MHA}(Q,C^{all},C^{all})
\in\mathbb{R}^{B\times N\times H\times256}.
\]

然后做 query residual 和 LayerNorm：

\[
U
=
\operatorname{LN}(Q+O).
\]

MHA 的 attn_output 会进入后续 routing head，而不是只读取 attention weights。因此 Q/K/V/O 四组投影都处在 forecast loss 的梯度路径上，避免 value projection 形成零梯度死分支。

MHA 诊断权重形状为：

\[
A^{mha}\in\mathbb{R}^{B\times N\times4\times H\times(K+1)}.
\]

### 9.9 Bilinear routing head

MHA 输出 \(U\) 与 candidate token 分别映射到 \(D_r=128\) 维：

\[
q^r_{b,n,h}=W_q^rU_{b,n,h},
\qquad
c^r_{b,n,k}=W_c^rC^{all}_{b,n,k}.
\]

路由 logit 为：

\[
\ell_{b,n,h,k}
=
\frac{
(q^r_{b,n,h})^\top c^r_{b,n,k}
}{
\sqrt{128}
}.
\]

Base logit 增加可学习 bias，初始值为 1.0：

\[
\ell_{b,n,h,K+1}
\leftarrow
\ell_{b,n,h,K+1}+b_{base}.
\]

对 K 个历史候选和 Base token 统一 softmax：

\[
\pi_{b,n,h,:}
=
\operatorname{softmax}_{k=1}^{K+1}
\left(
\ell_{b,n,h,:}
\right).
\]

无效历史候选在 softmax 前被填为 dtype 的最小有限值。由于 Base 永远有效，softmax 始终有合法分母。

路由权重形状为：

\[
\pi\in\mathbb{R}^{B\times N\times H\times(K+1)}.
\]

### 9.10 Residual mixture 和最终预测

只对历史候选 residual 做加权：

\[
R_{b,h,n}
=
\sum_{k=1}^{K}
\pi_{b,n,h,k}\,
\Delta_{b,h,n,k}.
\]

Base token 的零 residual 不需要显式相加，但它会通过 softmax 分母关闭历史修正。最终预测为：

\[
\widehat Y_{b,h,n}
=
Y^{base}_{b,h,n}
+
R_{b,h,n}.
\]

当历史候选全部无效时：

\[
\pi_{b,n,h,K+1}=1,
\qquad
R_{b,h,n}=0,
\qquad
\widehat Y_{b,h,n}=Y^{base}_{b,h,n}.
\]

代码还对没有任何有效候选的节点执行显式 torch.where fallback，确保数值上严格等于 Base。

Router 同时输出 historical_mass、base_usage、routing_entropy、residual absolute contribution、candidate dispersion 和 learned_memory=Base+residual。这些量用于诊断和 CaseStudy，不参与额外损失。

## 10. 下游损失与优化策略

### 10.1 当前 forecast loss

正式配置使用：

~~~yaml
target:
  forecast_loss_space: physical
  validation_loss_variant: forecast_only
~~~

模型内部仍使用标准化输入和输出，但在计算 forecast loss 前执行 scaler inverse transform：

\[
\widehat Y^{phy}
=
\sigma\widehat Y^{model}+\mu,
\qquad
Y^{phy}
=
\sigma Y^{model}+\mu.
\]

训练损失是 observed mask 下的物理单位 MAE：

\[
\mathcal{L}_{forecast}
=
\operatorname{MAE}
\left(
\widehat Y^{phy},Y^{phy}
\right)_{\text{observed}}.
\]

训练日志中的 train_forecast/val_total 是该物理空间损失，因此可以与物理单位 MAE 直接比较。RMSE、MAPE 和分 horizon 指标也统一在 inverse-transformed physical space 计算。

### 10.2 已关闭的辅助损失

当前主线明确关闭：

~~~yaml
candidate_quality_weight: 0.0
risk_weight: 0.0
blend_weight: 0.0
~~~

代码仍保留 candidate-quality、base-risk 和 blend target 的兼容实现，但它们不改变当前 Router 的训练目标。Router 的 risk_probe 仍会通过 Base token 路径参与 forecast loss；日志中的 confidence/risk/blend 在 forecast_only 配置下应为零。

不重新引入旧 confidence teacher、旧 Alpha/Beta、usage balance 正则、label smoothing 或未经匹配实验验证的额外正则项。

### 10.3 优化器和调度器

当前正式 Router 配置：

~~~yaml
batch_size: 32
epochs: 50
optimizer_name: adam
learning_rate: 0.0005
weight_decay: 0.0001
scheduler_name: step_lr
scheduler_step_size: 10
scheduler_gamma: 0.5
early_stopping_enabled: false
~~~

每轮反向传播后执行 clip_grad_norm_(downstream.parameters(), max_norm=5.0)。在 post-hoc frozen-base 阶段，只有 Router 参数 requires_grad=True；backbone、检索器和缓存均不更新。

### 10.4 为什么不把 loss 数值和指标混为一谈

当配置为 normalized loss 时，loss 数值处于标准化坐标；当配置为 physical 时，loss 处于原始交通单位。无论哪种配置，日志中的 RMSE/MAPE/MAE 指标都由 inverse transform 后的数据计算。比较不同实验时必须先确认 forecast_loss_space 一致，不能仅凭 train_total 的数值判断模型优劣。

## 11. 训练和推理流程

### 11.1 预训练阶段

~~~text
raw train windows
    -> 288-step retrieval_x
    -> mask sampler
    -> TemporalPatchEmbedding
    -> FactorizedSTEncoder
    -> RetrievalHead
    -> masked reconstruction + OffsetDecay relation loss
    -> pretrained checkpoint
~~~

真实 future 只用于 relation teacher 和 reconstruction target。

### 11.2 Bank 阶段

~~~text
target train history
    -> encode_clean(retrieval_x)
    -> event/node keys
    -> future values and masks
    -> immutable mmap Bank
~~~

Bank 生成后不再反向传播，不写入 validation/test 事件。

### 11.3 下游训练阶段

~~~text
query x (12 steps)
    -> frozen backbone
    -> Base prediction

query retrieval_x (288 steps)
    -> frozen retrieval encoder
    -> node key
    -> weekday_radius1_overlap event pool
    -> node Top-12
    -> OffsetDecay candidate futures
    -> Retrieval-Aware MHA Residual Router
    -> physical-space forecast MAE
~~~

开启 frozen_path_cache 后，Base、候选和 node key 只为每个 sample_id 计算一次并放到 CPU cache；Router 每个 epoch 仍会重新执行可训练部分。

### 11.4 推理阶段

推理时允许使用当前已观测的 12 步 context、冻结 backbone 输出、Bank 中历史事件的 context/future 以及已训练的检索器和 Router。

推理时禁止使用 query 的真实 future、由 query future 计算的 candidate error、target-derived teacher distribution 和 validation/test 写入的 Bank 事件。

候选历史 future 即使与 query 最近 context 重叠，也属于 query 时刻之前已可观察的历史，不是 query future 泄漏。

## 12. 参数量、显存和时间成本

### 12.1 Router 参数量

当前配置下，Router 参数可按代码精确分解为：

| 子模块 | 参数量 |
|---|---:|
| state_encoder | 72,192 |
| retrieval_encoder | 41,344 |
| query_fusion | 197,120 |
| candidate_summary_encoder | 17,280 |
| trajectory_encoder | 18,240 |
| base_encoder | 197,632 |
| 4-head MHA（Q/K/V/O） | 263,168 |
| query_norm | 512 |
| query_routing | 32,896 |
| candidate_routing | 32,896 |
| horizon_embedding | 3,072 |
| base_type | 256 |
| base_bias | 1 |
| risk_probe | 3,084 |
| **Router 合计** | **879,693** |

因此日志中 downstream_trainable 约为 879,693。downstream_total 还包括冻结的 backbone、兼容性的 ConfidenceHead 和 SafeResidualFusion，具体数值随 GWN、STGCN、STAEformer、ARGCN 而变化。

### 12.2 不生成的超大张量

Router 明确不生成：

\[
[B,N,H,K,D]
\quad\text{或}\quad
[B,H,N,K,D].
\]

它只保留 candidate numerical future [B,H,N,K,C]、candidate token [B,N,K,D]、query token [B,N,H,D]、MHA weights [B,N,4,H,K+1] 和 routing logits/weights [B,N,H,K+1]。

因此显存主要由当前 batch 的 candidate future、MHA 中间激活和反向图决定，而不是由完整历史 Bank 或 hidden 级五维候选张量决定。

### 12.3 当前资源观测边界

此前 16 GB 实验机上的最终 Router smoke/formal 测试已观测到约 3--5 GB CUDA peak、单 epoch 约 1 分钟量级；该数值依赖 backbone、batch、CUDA allocator 和 cache 命中状态，不作为所有机器的硬保证。正式训练仍需记录 cuda_peak_allocated_mb、每 epoch 秒数、batch 数、参数量、NaN/Inf 和 cache 命中情况。

## 13. 四个下游 backbone 的接口边界

### 13.1 Graph WaveNet

Graph WaveNet 负责图卷积、扩张时间卷积和基础未来预测。Router 不修改其 residual、dilation、skip 结构，只读取输出的 Base prediction。

### 13.2 STGCN

STGCN 负责固定交通图上的时空卷积预测。Router 与其通过统一 [B,H,N,C] Base 接口连接，不改变 STGCN 的图邻接或时间核。

### 13.3 STAEformer

STAEformer 使用 12 步输入、dataset-provided slot/weekday calendar covariates、空间/自适应 embedding 和时间注意力。Router 不重复注入 calendar feature，只接收 STAEformer 的 Base output 与独立 retrieval node key。

### 13.4 ARGCN

ARGCN 负责自回归图卷积基础预测。Router 只进行 post-hoc residual routing，确保 ARGCN checkpoint fingerprint 在训练期间保持不变。

四个 backbone 的公平比较必须固定数据 split、scaler、seed、Base checkpoint、Bank 和 encoder fingerprint、candidate protocol、node_top_k、batch size、optimizer、scheduler、50 epoch 训练预算和 best-checkpoint 规则。

## 14. 日志与可解释性输出

### 14.1 训练日志

train_downstream.py 同时输出控制台日志和 downstream.log，每轮记录 train/val forecast loss、physical-space MAE/RMSE/MAPE、15/30/60 分钟 horizon 指标、batch 数、epoch 秒数、learning rate、CUDA peak memory 和 parameter counts。

### 14.2 Router 诊断量

Router 在 forward 后保留 last_routing_weights、last_mha_attention、last_base_usage、last_routing_entropy 和 current_attention，并输出 residual contribution 与 candidate dispersion。

这些量可以用于判断是否长期只使用 Base、是否塌缩到单个历史候选、比较不同 horizon 的 history mass、分析候选误差与 learned weight 的相关性以及选择论文 CaseStudy 的代表性样本。它们不是额外监督，也不改变 forecast loss。

## 15. 数据泄漏与复现控制

### 15.1 scaler

scaler 只由当前数据集 train 段拟合。Bank manifest 中保存 scaler 状态；加载 Bank 时与当前目标数据 scaler 逐项核对。

### 15.2 encoder 与 graph fingerprint

Bank manifest 保存 retrieval encoder fingerprint、graph fingerprint、retrieval context length、retrieval dimension 和 schema version。目标训练加载 Bank 前必须通过 _validate_bank，否则停止运行。

### 15.3 时间边界

每个候选事件由 Bank 的 future_end 和 query 的 context_end 检查合法性。当前 weekday_radius1_overlap 允许候选 future 结束在 query context_end 时或之前，但不允许使用 query context_end 之后的历史事件。

### 15.4 随机性

正式实验固定 seed=42，并保存 downstream initialization hash、Base checkpoint fingerprint、Bank/encoder fingerprint 和完整配置。smoke 或 max_batches 运行产物不能作为论文正式结果。

## 16. 当前主线的兼容层说明

STAnchorDownstreamModel 仍包含 ConfidenceHead、SafeResidualFusion 和可选 horizon_aggregator。这些字段是历史接口兼容层。

对于 learned_topk_error_aware + retrieval_aware_mha_router：

1. error_corrector 指向当前 Router；
2. legacy confidence/fusion 不参与最终预测；
3. configure_error_aware_stage 只将 Router 参数设为可训练；
4. confidence_head、fusion 和 backbone 保持冻结；
5. forecast_only 下 confidence/risk/blend 辅助损失为 0。

因此日志中出现 confidence_head 或 fusion 不表示旧 Alpha/Beta 架构重新启用。

## 17. 当前实验结果的证据边界

当前文档规定结构和实验接口，不把某一次 smoke、单 batch 或未完成训练的中间 epoch 当作最终结果。正式报告应从完整 target_metrics.jsonl 中读取：

- 最佳验证 MAE 对应的 epoch；
- 同一 epoch 的 horizon MAE/RMSE/MAPE；
- 测试集指标；
- Base-only 对照；
- Bank/encoder fingerprint；
- candidate protocol 和 node Top-K；
- 训练时间与 CUDA peak。

如果不同实验的 loss space、Base checkpoint、split 或 Bank fingerprint 不一致，不能直接作因果比较。

## 18. 当前实现与方案的对应关系

| 研究组件 | 代码位置 | 主要输出 |
|---|---|---|
| 时间窗口与 mask | stanchor/data/dataset.py、stanchor/data/masking.py | retrieval_x、x、mask、时间边界 |
| Patch embedding | stanchor/models/patch_embedding.py | [B,24,N,128] |
| 时空检索 encoder | stanchor/models/encoder.py | [B,24,N,128] |
| Retrieval head | stanchor/models/retrieval_head.py | node/event key |
| 预训练目标 | stanchor/losses/pretraining.py | reconstruction + relation loss |
| Bank schema/storage | stanchor/bank/schema.py、storage.py、builder.py | mmap arrays + manifest |
| 候选协议 | stanchor/retrieval/strategies.py | legal event pool |
| 节点级检索/聚合 | stanchor/retrieval/retriever.py | Top-K candidates + future |
| OffsetDecay | stanchor/retrieval/strategies.py | adjusted candidate future |
| 下游 Router | stanchor/models/retrieval_router.py | MHA routing + residual mixture |
| 训练与冻结 | stanchor/engine/target.py | post-hoc frozen-base |
| 评估与诊断 | stanchor/diagnostics/、scripts/evaluate.py | metrics/attention/case study |

## 19. 后续跨数据集验证接口

后续跨数据集实验只替换目标数据 raw HDF、目标数据 adjacency、目标数据 train-only scaler、目标数据 Bank 和对应 Base-only checkpoint。

保持不变：

- HN-OffsetDecay v2 的模型结构和 checkpoint；
- Router 的结构和参数；
- Bank schema；
- weekday_radius1_overlap 的协议语义；
- node-level Top-12；
- quality teacher loss 关闭；
- post-hoc frozen-base 训练原则。

跨数据集结果应至少报告 Target Base-only、source encoder retrieval + Router，以及 source encoder 相同初始化协议下的 target-adapted encoder（若进行微调），并报告 MAE、RMSE、MAPE、15/30/60 分钟 horizon 指标、候选有效率、Base usage、history mass、routing entropy、encoder/Bank fingerprint 和训练成本。

跨数据集验证的目标是判断检索表示是否能提供可迁移的历史相似性，而不是在目标数据上重新改变 Router 结构。

## 20. 最终冻结声明

从本版本起，以下机制视为当前论文主线：

\[
\boxed{
\text{HN-OffsetDecay v2}
+
\text{weekday-radius-1 overlap retrieval}
+
\text{node Top-12}
+
\text{4-head MHA residual Router}
}
\]

其中：

- 检索器学习历史事件表示；
- Bank 保存可复用历史证据；
- 候选协议控制时间合法性和候选范围；
- Router 在 Base 与历史 residual experts 之间做逐 horizon 选择；
- Base token 提供显式 fallback；
- 所有真实 query future 只用于训练标签和离线评估，不进入部署输入。

后续仅允许进行学习率、调度器、随机种子、数据集路径和跨数据集适配层面的实验。任何架构级改变都必须作为独立方案重新评审，不能混入当前正式结果。
