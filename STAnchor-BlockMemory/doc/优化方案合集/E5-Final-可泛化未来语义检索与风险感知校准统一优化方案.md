# E5-Final：可泛化未来语义检索与风险感知校准统一优化方案

> 文档状态：机制设计与第一版工程实现已完成，正式预训练与下游对比尚未开始。本文是 E5 的最后一次机制级大改方案。后续只允许进行由实验结果直接触发的小范围调整，不再并行增加趋势分解、频域分支、新 backbone 或第二套检索系统。

## 1. 核心判断

本方案保留并强化 STAnchor-BlockMemory 的核心创新：

> 使用 source-train future 监督历史 encoder 学习“未来关系”，部署时只输入 query history，从 causal target-local Bank 中检索已经完整发生的历史事件，并用这些历史事件的 future 校正任意下游预测模型。

当前系统需要解决两个相互独立的问题：

1. **检索关系不够准确且 key 不可解释。** 当前 48 维 key 由 patch attention pooling 和普通 MLP 产生。虽然 pretrained key 相对 random 已学到弱但真实的 future relation，key 的每一维没有明确语义，跨数据集时也无法判断模型共享的是未来动力学还是源域数值分布。
2. **confidence 只能粗略判断 memory 是否有帮助。** 当前 confidence 使用六个诊断特征，并通过 `horizon_limit x confidence` 双门控融合。三随机种子结果证明它能稳定改善预测，但 AUROC 约为 0.55，说明它对误差原因和最佳修正幅度的理解仍然较弱。

因此最终大改只引入三个可独立关闭的机制：

| 机制 | 解决的问题 | 是否增加部署网络 | 是否读取 query future |
|---|---|---:|---:|
| `SymNormTeacher` | teacher 距离与对称 key 相似度的几何冲突 | 否 | 只在 source-train teacher 中读取 |
| `CanonicalFutureDynamicsProfile` | key 缺乏可解释、可跨数据集共享的未来语义 | 一个小型线性 head | 只在 source-train 监督中读取 |
| `ErrorAwareAdditiveFusion` | confidence 不理解基础预测风险，也没有直接学习修正幅度 | 一个轻量风险 head 和加性融合器 | 只在 target calibration 构造标签时读取 |

未来引导检索是论文主创新。PIR-inspired 校准只负责使用检索结果，不能替代 future-guided pretraining，也不能用其收益掩盖 selector 失败。

## 2. 证据起点

本方案不是无证据堆叠模块，而是由已经完成的诊断逐步收敛得到：

1. E2、E3、E5A 的 pretrained key-future Spearman 和 Recall@5 均优于对应 random，说明 future-guided relation 方向有效。
2. E5A pretrained OffsetDecay memory 的 MAE 优于 random，且同一 selector 下 RawFuture 到 OffsetDecay 的物理误差明显下降，说明 payload 对齐必须保留。
3. HardTop1 明显失败；Top-5 近均匀聚合更稳健。因此本方案不追求 one-hot teacher，不收紧检索温度，也不删除多候选共识。
4. E5A-SymNorm 零训练诊断消除了 teacher logit asymmetry，并改善固定 key 的 Spearman、Recall@5 和 Oracle Top-5 MAE，但 Top-1 变差、异常稳定性略降。因此 SymNorm 值得训练，但必须保留 Top-5 和异常诊断。
5. 线性趋势外推和 query/candidate local-scale transfer 已被 T0 诊断否定。特别是 local-scale ratio 曾达到约 4151，说明不能把候选残差乘以不受控的 query/candidate 尺度比。
6. E3 confidence 三随机种子平均改善 validation MAE 约 2.47%，但 AUROC 只有约 0.548 至 0.551。当前融合能稳定降低误差，但其概率语义和误差辨别能力偏弱。

## 3. 适用范围与明确不做的事情

### 3.1 当前泛化范围

第一版泛化目标是：

- 单一源数据集预训练；
- 迁移到具有相同物理变量类型的目标时空数据集；
- 节点数和图结构可以不同；
- 预测 horizon 和采样间隔可以不同，但必须能够映射到固定的相对预测时间网格；
- 当前工程仍固定单值通道 `C=1`，例如交通速度。多物理变量、多通道联合 pretraining 不在本轮范围内。

“可泛化”表示模型设计消除绝对 level、数据集 scaler、节点数量和实际 horizon 数量对语义 key 维度的绑定，并通过跨数据集实验验证。它不是未经实验即可成立的 domain-invariant 声明。

### 3.2 本轮不做

- 不加入线性趋势 payload、local-scale transfer 或趋势分解网络；
- 不加入 FFT、DWT、频率字典或第二个时空 Transformer；
- 不修改 Top-5、`search_temperature=0.10` 和 OffsetDecay 主公式；
- 不引入 PIR 的 Local Revision Transformer；
- 不把 confidence 与 horizon gate 继续相乘；
- 不把 validation/test future 写入 Bank；
- 不宣称速度到流量、交通到天气等跨物理量零样本泛化；
- 不同时改变 candidate protocol、下游 backbone 和预训练目标。

## 4. 统一张量与信息边界

| 符号 | 含义 | 形状 |
|---|---|---|
| `X` | query 历史输入 | `[B,T,N,C]` |
| `X_long` | 检索 encoder 的长历史，当前为一天 288 步 | `[B,288,N,1]` |
| `Y` | 真实待预测 future | `[B,H,N,C]` |
| `Hid` | 时空 encoder patch hidden | `[B,P,N,D]` |
| `K_node` | 节点级总检索 key | `[B,N,48]` |
| `K_event` | 对节点 key 求均值并归一化得到的事件 key | `[B,48]` |
| `Y_mem` | Top-5 OffsetDecay memory prediction | `[B,H,N,C]` |
| `Y_base` | 任意下游 backbone 的基础预测 | `[B,H,N,C]` |
| `w_fuse` | 最终融合权重 | `[B,H,N,1]` |
| `Y_final` | 最终校正预测 | `[B,H,N,C]` |

未来信息边界必须固定：

- source pretraining 可以在 teacher 的 `torch.no_grad()` 分支读取 source-train future；
- encoder、profile head 和 key head 的输入始终只有 history、history mask、日历和图；
- target Bank 只保存已经完整发生的 target-train 历史事件及其 future；
- target calibration 可以用 calibration future 监督风险和融合权重；
- validation/test query future 只能在预测完成后计算指标，不能进入 encoder、retriever、Bank、风险 head 或融合器。

## 5. 完整架构

```text
Source pretraining
history [B,288,N,1]
        |
        v
shared ST encoder [B,P,N,96]
        |
        +---------------------------+
        |                           |
        v                           v
canonical future profile head      latent relation head
12-D interpretable profile         36-D latent relation
        |                           |
        +------------+--------------+
                     v
          composed 48-D retrieval key
                     |
                     v
       target-local causal Bank rebuild
                     |
                     v
        Top-5 retrieval + OffsetDecay
                     |
                     v
           memory and diagnostics
                     |
Target fine-tuning   +       downstream backbone
                     |              |
                     v              v
              error-aware additive fusion
                     |
                     v
                final prediction
```

预训练检索 encoder 与 Bank 对下游 backbone 即插即用。风险估计器和融合器需要针对每个下游 backbone 重新微调，因为不同 backbone 的基础误差分布不同。

## 6. 检索优化 A：SymNormTeacher

### 6.1 名称、目的和输入输出

`SymNormTeacher` 是“使用对称几何均值归一化的 OffsetDecay future-relation teacher”。它不是神经网络，也不进入部署推理。

输入：source-train future、future mask、各事件 forecast context、合法 pair mask。

输出：无量纲 pairwise teacher distance 和 teacher distribution，形状分别为 `[B,B,N]`。

### 6.2 DeploymentAlignedOffsetDecaySignature

`DeploymentAlignedOffsetDecaySignature` 简称 `ODSignature`，表示训练期与 OffsetDecay 推理代数一致的 future relation 对象。对训练事件 `i`：

\[
S^{OD}_{i,h,n,c}
=
Y_{i,h,n,c}-\lambda_h\alpha_{i,n,c},
\]

\[
\lambda_h=1-\frac{h-1}{H-1}.
\]

其中 `alpha` 是事件 `i` 自己 forecast context 的 endpoint level，不是另一个 future 近端值。`ODSignature` 只在训练 teacher 中构造，推理不构造 query signature。

事件 `i` 和 `j` 的 masked OD 距离为：

\[
d^{OD}_{ij,n}
=
\operatorname{MaskedMAE}
\left(S^{OD}_{i,:,n,:},S^{OD}_{j,:,n,:}\right).
\]

### 6.3 对称几何均值归一化

对 anchor `(i,n)` 的有效候选集合 `C_{i,n}`：

\[
\mu_{i,n}
=
\frac{1}{|C_{i,n}|}
\sum_{k\in C_{i,n}}d^{OD}_{ik,n}.
\]

SymNorm 距离定义为：

\[
\widetilde d^{SYM}_{ij,n}
=
\frac{d^{OD}_{ij,n}}
{\sqrt{(\mu_{i,n}+\epsilon)(\mu_{j,n}+\epsilon)}}.
\]

teacher distribution 为：

\[
p^T_{ij,n}
=
\operatorname{Softmax}_{j\in C_{i,n}}
\left(-\frac{\widetilde d^{SYM}_{ij,n}}{\tau_T}\right).
\]

该定义满足 `d_tilde_ij = d_tilde_ji`，与余弦 key logit 的对称结构一致。SymNorm 不改变候选合法性，也不读取部署 query future。

## 7. 检索优化 A：CanonicalFutureDynamicsProfile

### 7.1 为什么不能直接把 raw future 放进 semantic key

直接预测 `Y_{1:H}` 会把源数据集的速度 level、节点 scaler、采样间隔和 horizon 数量固化到 key 中。这样的 key 即使在源域可解释，也不能作为可共享的预训练检索表示。

因此本方案定义 `CanonicalFutureDynamicsProfile`，中文为“规范化未来动力学轮廓”，简称 `CFDP`。它的目的不是精确预测物理单位 future，而是把不同数据集的 future 变成固定长度、无量纲、相对时间对齐的动态语义。

### 7.2 固定相对时间网格

设 profile 固定长度 `Kp=12`，规范化预测位置为：

\[
u_k=\frac{k-1}{K_p-1},\qquad k=1,\ldots,K_p.
\]

对任意实际 horizon `H`，将 future 通过 mask-aware interpolation 映射到 `u_k`，得到：

\[
\overline Y_{i,k,n,c}
=
\operatorname{Resample}(Y_{i,:,n,c},u_k).
\]

METR-LA 和 PEMS-BAY 当前都是 `H=12`，所以该操作等价于恒等映射。以后 `H` 改变时，profile 维度仍固定为 12。缺失位置只有在插值两端均有效时才视为有效，不跨越不可观测区间强行补值。

### 7.3 无量纲部署对齐坐标

从事件 `i` 的可见 forecast context 得到：

- `alpha_{i,n,c}`：context endpoint level；
- `m_{i,n,c}`：context 可见均值；
- `s_{i,n,c}`：context 可见标准差。

为避免平坦窗口产生近零除数：

\[
\overline s_{i,n,c}=\max(s_{i,n,c},s_{min}).
\]

`s_min` 在经过 train-only scaler 的模型坐标中固定，首轮设为 `0.1`，不按数据集搜索。它只用于 profile teacher 数值稳定，不用于 payload 重构。

对规范化位置 `u_k`，令：

\[
\lambda_k=1-u_k.
\]

CFDP 定义为：

\[
G_{i,k,n,c}
=
\frac{
\overline Y_{i,k,n,c}
-\lambda_k\alpha_{i,n,c}
-(1-\lambda_k)m_{i,n,c}
}{\overline s_{i,n,c}}.
\]

解释如下：

- 近端 `lambda=1` 时，比较 `(future - endpoint) / scale`，表示从当前 level 向上或向下变化多少；
- 远端 `lambda=0` 时，比较 `(future - context mean) / scale`，表示未来相对于自己的历史分布处于什么状态；
- 中间 horizon 平滑连接两个坐标，与 OffsetDecay 的 horizon decay 保持同一方向；
- 输出没有原始速度单位，也不绑定节点 scaler。

CFDP 不执行线性趋势外推，不估计 season/trend 分量，也不把 candidate scale 转移到 query。之前失败的 local-scale payload 会乘以 `query_scale / candidate_scale`；CFDP 只在 source teacher 内把每个事件自身无量纲化，推理时不使用该比例重构 future，因此不存在 4151 倍放大路径。

### 7.4 可解释 profile key 与 latent key

现有 attention pooling 得到节点历史表示：

\[
h_{i,n}\in\mathbb R^{96}.
\]

新增一个线性 profile head：

\[
\widehat G_{i,n}
=
W_p h_{i,n}+b_p
\in\mathbb R^{12}.
\]

`G_hat[k]` 表示 encoder 仅根据 history 预测的第 `k` 个相对 horizon 的 CFDP 值。profile key 为：

\[
p_{i,n}=\operatorname{L2Norm}(\widehat G_{i,n})
\in\mathbb R^{12}.
\]

latent head 输出：

\[
z_{i,n}=\operatorname{L2Norm}(\operatorname{MLP}(h_{i,n}))
\in\mathbb R^{36}.
\]

总 key 保持 48 维：

\[
k_{i,n}
=
\left[
\sqrt{\gamma}\,p_{i,n};
\sqrt{1-\gamma}\,z_{i,n}
\right]
\in\mathbb R^{48}.
\]

首轮固定 `gamma=0.25`，与 12/48 的 profile 维度占比一致，不进行 validation 搜索。两个 key 都单位归一化，所以总相似度可以精确拆解：

\[
s^{total}_{ij,n}
=
\gamma s^{profile}_{ij,n}
+(1-\gamma)s^{latent}_{ij,n}.
\]

其中：

\[
s^{profile}_{ij,n}=p_{i,n}^{\top}p_{j,n},
\qquad
s^{latent}_{ij,n}=z_{i,n}^{\top}z_{j,n}.
\]

这使每次检索都能解释为“规范化未来动力学相似度”和“潜在部署关系相似度”两部分，而不是一个不可分解的 48 维黑盒。

在当前 `hidden_dim=96`、`retrieval_dim=48` 下，将原 `96 -> 96 -> 48` MLP 改成 `96 -> 12` profile head 和 `96 -> 96 -> 36` latent head，retrieval head 参数量基本不增加。

### 7.5 Profile 监督与 relation 监督

profile loss 为 mask-aware SmoothL1：

\[
L_{profile}
=
\frac{1}{|M_G|}
\sum_{i,k,n,c}
M^G_{i,k,n,c}
\operatorname{SmoothL1}
\left(\widehat G_{i,k,n,c},G_{i,k,n,c}\right).
\]

relation student 仍由总 48 维 key 形成：

\[
p^S_{ij,n}
=
\operatorname{Softmax}_{j\in C_{i,n}}
\left(
\frac{k_{i,n}^{\top}k_{j,n}}{\tau_S}
\right).
\]

relation loss 为 teacher 到 student 的交叉熵：

\[
L_{relation}
=
-\frac{1}{|A|}
\sum_{(i,n)\in A}
\sum_{j\in C_{i,n}}
p^T_{ij,n}\log p^S_{ij,n}.
\]

预训练总损失为：

\[
L_{pretrain}
=
L_{reconstruction}
+\lambda_{relation}L_{relation}
+\lambda_{profile}L_{profile}.
\]

首轮保持 `lambda_relation=0.1`，固定 `lambda_profile=0.1`。不增加 derivative loss、对比损失或 profile consistency loss。只有 profile 预测指标显示模型只学 level 而未学动态时，才允许单独评估一个 increment loss。

梯度边界：

- `L_reconstruction` 更新 embedding、encoder 和 reconstruction head；
- `L_relation` 更新 embedding、encoder、profile head 和 latent head；
- `L_profile` 更新 embedding、encoder 和 profile head；
- CFDP 和 SymNorm teacher 均在 `torch.no_grad()` 中构造；
- encoder 输入从不读取 future。

## 8. Bank 与检索部署

### 8.1 Bank v2

虽然总 key 仍为 48 维，但语义已经改变，旧 Bank 不能复用。Bank manifest 升级到 `schema_version=2`，新增：

- `key_layout: canonical_profile_latent`；
- `profile_dim: 12`；
- `latent_dim: 36`；
- `profile_weight: 0.25`；
- `profile_grid_size: 12`；
- `profile_scale_floor: 0.1`；
- `relation_distance_normalization: symmetric_geometric_mean`。

Bank 仍只保存：

- 48 维 event/node key；
- raw historical future 和 mask；
- level features、日历和时间边界。

不额外保存 profile tensor，因为 profile 和 latent 已按固定布局写入 48 维 key。Bank 大小基本不变，但 encoder fingerprint 和 manifest 必须与 query encoder 完全一致。

### 8.2 两阶段检索

事件级 coarse retrieval 和节点级 Top-5 rerank 保持不变：

\[
s^{total}_{qj,n}
=
0.25s^{profile}_{qj,n}
+0.75s^{latent}_{qj,n}.
\]

部署仍使用 `search_temperature=0.10`。同时把 selected Top-5 的 profile similarity 和 latent similarity作为诊断输出，供风险感知融合器使用，但不改变候选 ID。

## 9. OffsetDecay payload 保持不变

Bank 保存的是 raw historical future。检索到候选 `j` 后，部署时根据 query endpoint 和 candidate endpoint 生成：

\[
Z^{OD}_{qj,h,n,c}
=
Y_{j,h,n,c}
+\lambda_h
\left(\alpha_{q,n,c}-\alpha_{j,n,c}\right).
\]

Top-5 mask-aware 聚合为：

\[
Y^{mem}_{q,h,n,c}
=
\frac{
\sum_j w_{qj,n}M_{qj,h,n,c}Z^{OD}_{qj,h,n,c}
}{
\sum_j w_{qj,n}M_{qj,h,n,c}
}.
\]

OffsetDecay 只读取 query history、candidate context 和已经发生的 Bank raw future。CFDP 不替换该 payload，也不把无量纲 profile 反变换成预测值。

## 10. 校准优化：ErrorAwareAdditiveFusion

### 10.1 名称、目的和与 PIR 的边界

`ErrorAwareAdditiveFusion` 中文为“误差感知加性融合器”。它是下游微调模块，不是检索器，也不是概率意义上的 uncertainty calibration。

它借鉴 PIR 的 Failure Identification 思想：根据 history 和基础预测估计当前实例的预测误差；但不复制 PIR 的 Local Revision Transformer，也不直接把 raw retrieval future 加到基础预测上。

输入：query 最近历史、下游基础预测、Top-5 检索诊断、OffsetDecay memory。

输出：基础预测风险 `e_hat_base [B,H,N,1]`、每个可解释特征的 logit contribution、融合权重 `w_fuse [B,H,N,1]` 和最终预测 `[B,H,N,C]`。

### 10.2 PredictedBaseRisk

`PredictedBaseRisk` 中文为“预测的基础模型误差”，表示模型在不知道真实 future 时，对基础预测绝对误差大小的估计。

使用 query 最近 12 步的 sample/node-local normalized history 和完整基础预测：

\[
\widehat e^{base}_{q,h,n}
=
\operatorname{Softplus}
\left(
f_{err}
\left(
X^{local}_{q,:,n,:},
Y^{base}_{q,:,n,:},
u_h
\right)
\right).
\]

其中 `f_err` 只使用共享线性投影和一个小型 MLP，不使用节点 ID embedding。这样节点数和传感器编号变化时不需要修改参数。

训练风险标签使用标准化模型坐标中的 Huber absolute error：

\[
e^{base}_{q,h,n}
=
\frac{1}{C}
\sum_c
\operatorname{Huber}
\left(
Y^{base}_{q,h,n,c}-Y_{q,h,n,c}
\right).
\]

风险损失为：

\[
L_{risk}
=
\operatorname{SmoothL1}
\left(
\widehat e^{base},
\operatorname{stopgrad}(e^{base})
\right).
\]

使用 Huber 而不是 PIR 的 MSE，是为了降低单点异常对误差监督的支配。物理单位风险指标在诊断时通过 target scaler 逆变换后另行报告。

### 10.3 十个可部署融合特征

每个 `(query,horizon,node)` 使用以下特征：

| 特征 | 含义与计算 | 形状 |
|---|---|---|
| predicted base risk | 上一节的 `e_hat_base` | `[B,H,N,1]` |
| profile similarity | selected Top-5 的 CFDP profile cosine 加权均值 | `[B,H,N,1]` |
| latent similarity | selected Top-5 的 latent cosine 加权均值 | `[B,H,N,1]` |
| score margin | total Top-1 与 Top-2 score 之差 | `[B,H,N,1]` |
| normalized effective support | `(1 / sum_j w_j^2) / K` | `[B,H,N,1]` |
| payload dispersion | `log(1 + mean_c sqrt(candidate variance))` | `[B,H,N,1]` |
| direction agreement | Top-5 候选相对 base 的修正方向一致程度 | `[B,H,N,1]` |
| level match | query/candidate endpoint level 距离的加权匹配分数 | `[B,H,N,1]` |
| memory-base disagreement | `log(1 + mean_c |Y_mem - Y_base|)` | `[B,H,N,1]` |
| horizon position | `u_h=(h-1)/(H-1)` | `[B,H,N,1]` |

`direction agreement` 定义为：

\[
a^{dir}_{q,h,n}
=
\frac{1}{C}
\sum_c
\left|
\sum_j
\overline w_{qjhnc}
\operatorname{sign}
\left(Z^{OD}_{qjhnc}-Y^{base}_{qhnc}\right)
\right|.
\]

若所有候选都建议向同一方向修正，`a_dir` 接近 1；若候选方向相互抵消，则接近 0。全部输入都来自 history、base prediction 和 causal Bank，不读取 query future。

### 10.4 GroupedAdditiveFusion

`GroupedAdditiveFusion` 中文为“分组加性融合”。它不是另一套独立模型，而是代码类 `ErrorAwareAdditiveFusion` 内部使用的可解释加性结构：用独立的小型形状函数处理每个可解释特征，再把贡献相加：

\[
\ell_{q,h,n}
=
b_0
+\sum_{r=1}^{10}g_r(f^{(r)}_{q,h,n}),
\]

\[
w^{fuse}_{q,h,n}=\sigma(\ell_{q,h,n}).
\]

每个 `g_r` 使用 `1 -> 8 -> 1` 的两层小 MLP。模块同时返回每个 `g_r` 的 contribution，因此可以直接解释某次融合是被“高基础风险”“高候选一致性”还是“高 memory-base disagreement”推动或抑制。

初始化时所有 `g_r` 的最后一层为 0，`b_0=logit(0.1)`，因此初始融合权重为 0.1。旧的 `horizon_limit x confidence` 双门控删除；horizon 影响只通过可解释的 `horizon position` 进入一个融合权重，避免两个门控互相补偿而失去语义。

### 10.5 最优融合标签

最终预测沿着 memory-base 方向修正：

\[
Y^{final}_{q,h,n,c}
=
Y^{base}_{q,h,n,c}
+w^{fuse}_{q,h,n}
\left(
Y^{mem}_{q,h,n,c}-Y^{base}_{q,h,n,c}
\right).
\]

训练期对每个 `(q,h,n)` 定义 convex blend 的 oracle 权重：

\[
w^*_{q,h,n}
=
\operatorname{clip}
\left(
\frac{
\sum_c
(Y_{qhnc}-Y^{base}_{qhnc})
(Y^{mem}_{qhnc}-Y^{base}_{qhnc})
}{
\sum_c
(Y^{mem}_{qhnc}-Y^{base}_{qhnc})^2
+\epsilon
},
0,1
\right).
\]

它直接表示“沿 memory-base 修正方向应该走多少”。当 `||Y_mem-Y_base||` 小于固定数值门槛时，该位置的融合结果对 `w` 不敏感，因此不计算 blend supervision，避免不稳定标签。

blend loss 为：

\[
L_{blend}
=
\operatorname{SmoothL1}
\left(w^{fuse},\operatorname{stopgrad}(w^*)\right).
\]

该 `w_fuse` 是修正幅度，不是“memory 有帮助的概率”。因此不能把 Brier 或 ECE 当作主校准指标；可以用它计算 helpfulness AUROC 作为排序诊断，但概率语义需要单独说明。

### 10.6 下游总损失

\[
L_{target}
=
L_{forecast}
+\lambda_{risk}L_{risk}
+\lambda_{blend}L_{blend}.
\]

首轮固定：

- `L_forecast`：现有 masked MAE；
- `lambda_risk=0.1`；
- `lambda_blend=0.1`。

不同时搜索两个权重。只有风险或 blend loss 的梯度量级与 forecast loss 相差超过一个数量级时，才根据记录的 gradient norm 调整一次。

## 11. 训练阶段与梯度边界

### 11.1 Stage A：source pretraining

训练 embedding、时空 encoder、reconstruction head、profile head 和 latent head。SymNorm 与 CFDP teacher 使用 source-train future，但不接收梯度。

### 11.2 Stage B：target Bank rebuild

冻结整个预训练检索模块，使用目标数据集 train-only scaler、目标图和目标训练历史重建 Bank v2。目标 Bank 依赖目标数据集，但不依赖下游 backbone，所以同一 Bank 可被多个 backbone 共用。

### 11.3 Stage C：base warm-up

对每个下游 backbone 先训练或加载 base-only checkpoint。原因是 PIR 论文和源码都表明，从随机 backbone 开始同时学习误差估计会产生移动且不稳定的风险标签。

### 11.4 Stage D：calibrator warm-up

冻结预训练检索、Bank 和下游 backbone，训练风险 head 与加性融合器 5 个 epoch。memory、retrieval diagnostics 和 base prediction 全部 `detach`。

### 11.5 Stage E：joint fine-tuning

保持预训练检索和 Bank 冻结，解冻下游 backbone，与校准器联合微调：

- calibrator learning rate 使用 target 配置主学习率；
- backbone learning rate 固定为 calibrator 的 0.1 倍；
- `e_base` 和 `w_star` 标签始终由 detached base/memory 构造；
- final forecast loss 可以更新 backbone 和 calibrator；
- gradient norm 分模块记录，防止校准辅助损失支配 backbone。

## 12. 可泛化性设计说明

### 12.1 数值分布偏移

encoder 输入已经使用 sample/node-local history normalization。CFDP 进一步用事件自己的 context endpoint、mean 和 scale 表达 future，不携带源城市绝对速度 level。SymNorm 对 pair distance 再做无量纲对称校准。

目标数据集仍使用自己的 train-only scaler。scaler 是数据读取与物理指标恢复工具，不写入 semantic key 的固定维度。

### 12.2 不同节点数和图结构

时间 attention 按节点共享参数，空间 attention 只读取当前数据集的 `edge_index/edge_weight`。profile/latent head 逐节点共享，不包含 node ID embedding。因此 `N` 变化不改变模型参数形状。

目标数据集必须提供自己的图并重建 Bank。共享的是 encoder 参数和 48 维语义，不是源数据集的节点编号或源 Bank。

### 12.3 不同 horizon 和采样间隔

CFDP 使用固定的 12 个相对预测位置。实际 `H` 变化时只改变 mask-aware resampling，不改变 profile/key 维度。该机制对相对轨迹形状泛化，但不能自动保证“5 分钟急剧变化”和“1 小时急剧变化”具有相同物理含义，因此正式论文必须报告实际采样间隔，并在跨频率实验中单独分析。

### 12.4 不同下游 backbone

预训练 encoder 和 Bank 可共用；ErrorAwareAdditiveFusion 必须针对每个 backbone 微调。原因是 `PredictedBaseRisk` 的监督对象就是该 backbone 的误差。

### 12.5 当前不可宣称的泛化

- 不宣称 C=1 checkpoint 可直接处理任意 C；
- 不宣称交通速度 profile 与流量、天气或需求 profile 完全同义；
- 不宣称无 target Bank 的纯 zero-shot retrieval；
- 不宣称没有目标 calibration 标签也能得到可靠融合权重。

## 13. 实验顺序与决策门

每个阶段只关闭一个不确定性。前一阶段失败时停止后续相关分支，不通过增加模块补救。

### 13.1 R0：实现正确性与泄漏检查

检查：

- CFDP 在 `H=12` 时 resampling 为恒等映射；
- 常数平移和正比例缩放输入后，CFDP 近似不变；
- profile/latent/total key 分别为 12/36/48 维；
- total cosine 等于 `0.25 x profile cosine + 0.75 x latent cosine`；
- SymNorm pair distance 严格对称；
- Bank v1 与 v2 互相拒绝加载；
- inference 调用路径删除 query future 后输出不变；
- one-batch forward、backward、checkpoint save/load 和 Bank build 均有限。

任何一项失败都停止正式训练。

### 13.2 R1：只训练 E5A-SymNorm

动机：先验证无参数 teacher 几何修正是否在重新训练后成立。

对照：

- 当前 E5A AnchorMean；
- E5A-SymNorm，其他结构和配置完全相同。

保留条件：SymNorm 训练后保持 teacher logit 对称，Recall@5 不低于 AnchorMean，并且 pretrained-vs-random memory 或无 confidence 下游 MAE 至少一个产生超过 seed 波动的改善；否则删除 SymNorm，后续 profile 仍可在 AnchorMean teacher 上单独验证。

### 13.3 R2：加入 CFDP semantic key

动机：检验可解释、无量纲未来 profile 是否真正改善 relation 和跨数据集共享，而不是只增加一个辅助预测 loss。

对照：

- R1 最优 teacher + 原 48-D latent key；
- R1 最优 teacher + 12-D CFDP profile + 36-D latent key；
- 相同新结构的 random initialization。

必须报告：

- CFDP prediction SmoothL1、MAE、cosine similarity；
- profile similarity 与真实 CFDP distance 的 Spearman；
- total key 与 OD teacher distance 的 Spearman、Recall@5；
- pretrained-vs-random Top-5 OffsetDecay MAE/RMSE/MAPE；
- METR-LA source 和 PEMS-BAY transfer 的相同口径结果；
- 参数量、epoch 时间、Bank 大小和检索 latency。

保留条件：CFDP 必须提高跨数据集 pretrained-vs-random gap，且 source no-confidence MAE 不出现系统性退化。若只改善 profile prediction 而不改善 retrieval/downstream，则删除 profile key，不能因为可视化好看而保留。

### 13.4 R3：profile weight 最小消融

只有 R2 通过后比较：

- `gamma=0`：纯 latent；
- `gamma=0.25`：主方案；
- `gamma=1`：纯 profile。

不进行连续网格搜索。若 `gamma=0` 最好，删除 profile；若 `gamma=1` 在跨域最好但源域退化，则保留 0.25 并报告权衡，不继续增加可学习 gate。

### 13.5 C1：固定检索的校准单变量实验

固定 R 阶段选出的同一个 encoder、Bank、Top-5 和 OffsetDecay，比较：

1. base-only；
2. horizon-only；
3. 当前六特征 confidence；
4. 当前融合器 + PredictedBaseRisk 额外特征；
5. ErrorAwareAdditiveFusion 完整版。

第 4 项只检验 PIR 风格风险信号是否有用；第 5 项检验删除双门控并直接学习修正幅度是否有用。

### 13.6 C2：校准器消融

只在 C1 完整版改善 validation MAE 后执行：

- 去掉 PredictedBaseRisk；
- 去掉 profile/latent 分解，只给 total similarity；
- 去掉 direction agreement；
- 用普通等参数 MLP 替换 GroupedAdditiveFusion；
- 去掉 `L_blend`，只用 final forecast loss；
- 去掉 `L_risk`，风险 head 只接受 final loss 间接训练。

该阶段用于回答每个机制是否必要以及可解释结构是否只是参数增加。

### 13.7 C3：多 backbone 与多 seed

只有 R2 和 C1 同时通过后：

- 固定一个最终 retrieval checkpoint 和 Bank；
- 在轻量 MLP、ST-SSDL 及一个不同结构 backbone 上分别训练校准器；
- 每个正式设置运行下游 seed 42、2024、2025；
- 最终 retrieval 预训练补 seed 2024、2025；
- test 只在结构、超参数、seed 和 checkpoint 选择全部冻结后执行一次。

## 14. 指标定义

### 14.1 检索与 profile 指标

- `Profile MAE`：预测 CFDP 与 teacher CFDP 的 mask-aware 平均绝对误差。
- `Profile cosine`：预测与真实 CFDP 方向余弦，越高表示未来动力学方向越一致。
- `Spearman`：key distance 与 future teacher distance 的秩相关。
- `Recall@5`：key Top-5 与 future teacher Top-5 的交集比例。
- `Top-5 Jaccard`：异常扰动前后 teacher 或 deployed Top-5 集合交并比。
- `TV`：异常扰动前后候选概率分布的 Total Variation。
- `memory MAE/RMSE/MAPE`：只评价 OffsetDecay memory，不混入下游 backbone。

### 14.2 风险与融合指标

- `Risk MAE`：预测基础风险与真实 Huber error 的差。
- `Risk Spearman`：预测风险与真实基础误差的排序相关。
- `Risk R2`：预测风险对真实基础误差方差的解释比例。
- `Blend target MAE`：`w_fuse` 与 `w_star` 的绝对差。
- `Helpfulness AUROC/AUPRC`：把 `w_fuse` 当排序分数，判断 memory 是否优于 base；只解释排序，不解释概率校准。
- `Weight quartile gain`：按 `w_fuse` 四分位统计真实 memory gain，检查权重越大是否越值得使用 memory。
- `Contribution curve`：每个 additive feature 的输入分箱与平均 logit contribution，用于解释融合器。

### 14.3 最终预测与效率指标

- overall MAE、RMSE、MAPE；
- 每个 5 分钟 horizon 的 MAE、RMSE、MAPE；
- node-wise MAE、peak-period MAE 和 worst 10% timestep MAE；
- 参数量、训练时间/epoch、推理 latency、peak GPU memory 和 Bank 大小；
- 三 seed mean、sample standard deviation 和配对差。

## 15. 计划修改的工程边界

实现阶段预计只修改以下边界：

1. `stanchor/config.py`：新增 profile、SymNorm、风险和 additive fusion 配置；
2. `stanchor/models/retrieval_head.py`：拆分 12-D profile 和 36-D latent；
3. `stanchor/models/pretraining.py`：输出 profile prediction；
4. `stanchor/losses/pretraining.py`：CFDP、SymNorm 和 profile loss；
5. `stanchor/bank/schema.py`、`builder.py`、`storage.py`：Bank v2 manifest；
6. `stanchor/retrieval/retriever.py`：返回 profile/latent score 与 consensus diagnostics；
7. `stanchor/models/downstream.py`：风险 head、`ErrorAwareAdditiveFusion` 内部的 grouped additive 结构和单一 residual blend；
8. `stanchor/losses/downstream.py`：risk/blend supervision；
9. `stanchor/engine/target.py`：base warm-up、calibrator warm-up 和分学习率 joint fine-tuning；
10. diagnostics、tests、config 和文档同步更新。

不修改图数据格式、时间切分、train-only scaler、raw Bank future、OffsetDecay 公式和预测指标实现。

## 15.1 当前实现状态与工程事实

截至 2026-08-12，以下代码已经接入：

- `stanchor/retrieval/semantic_profile.py`：实现 `CFDP` 的固定相对时间重采样、事件自身 endpoint/mean/std 无量纲化、`SymNorm` 对称几何均值归一化，以及 profile/latent key 组合。
- `stanchor/models/retrieval_head.py`：profile head 输出 12 维 profile prediction，latent head 输出 36 维 latent key；组合 key 仍为 48 维。节点级和事件级相似度都能拆成 `0.25 x profile + 0.75 x latent`。
- `stanchor/losses/pretraining.py`：支持 `symmetric_geometric_mean` teacher normalization 和 profile SmoothL1 loss。CFDP profile loss 缺少 forecast context 时会直接报错，不会用 query future 代替 context。
- `stanchor/bank/schema.py`、`builder.py`、`storage.py`：profile-enabled encoder 使用 Bank schema v2；manifest 保存 key layout、profile/latent 维度、profile weight、profile grid 和 SymNorm 标记。旧 v1 Bank 不会静默加载到 v2 encoder。
- `stanchor/retrieval/retriever.py`：v2 Bank 返回 Top-5 的 profile similarity 和 latent similarity 诊断，候选集合和 OffsetDecay payload 公式未改变。
- `stanchor/models/downstream.py`、`losses/downstream.py`、`engine/target.py`：新增 `learned_topk_error_aware` 模式、`PredictedBaseRisk`、`ErrorAwareAdditiveFusion`、Huber risk target、oracle blend target，以及 base/calibrator/joint 三阶段训练。
- `stanchor/diagnostics/downstream.py`：error-aware 模式只报告风险、blend 和 helpfulness 排序指标，不把修正权重当作概率计算 Brier/ECE；零 memory coverage 时输出 `null` memory 指标而不是报错。

当前尚未获得任何 E5-Final 正式效果结论。141 项单元测试、编译检查、真实 METR-LA 一批次预训练、Bank v2 小规模构建和下游一批次 smoke 仅证明接口、形状、梯度和 future 边界正确，不能作为论文结果。calibrator warm-up 阶段被冻结的 backbone 同时保持 eval 模式，避免 Dropout 改变校准器看到的 base prediction；joint fine-tuning 解冻 backbone 后恢复 train 模式。

## 16. 计划配置与命令

以下是已经接入代码的正式接口。命令必须按下一份“下一步实验步骤”文件的顺序执行；smoke 命令只用于工程检查，不得作为正式结果。

### 16.1 配置文件

```text
configs/metrla_e5_final_anchor_mean_v1.yaml
configs/metrla_e5_final_symnorm_v1.yaml
configs/metrla_e5_final_sym_profile_v1.yaml
configs/pemsbay_e5_final_transfer_v1.yaml
configs/metrla_e5_final_calibrator_v1.yaml
configs/pemsbay_e5_final_calibrator_v1.yaml
```

### 16.2 实现验证

```powershell
C:/Users/31396/.conda/envs/research/python.exe -m unittest discover -s tests -v
C:/Users/31396/.conda/envs/research/python.exe -m compileall -q stanchor scripts tests
```

### 16.3 R1：AnchorMean 与纯 SymNorm source pretraining

该命令关闭 CFDP（`profile_dim=0`、`profile_loss_weight=0`），只把 OffsetDecay teacher distance 的 `AnchorMean` 归一化替换为对称几何均值归一化。这样 R1 只回答 SymNorm 本身是否有效。

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/pretrain.py `
  --config configs/metrla_e5_final_symnorm_v1.yaml `
  --run-name metrla_e5_final_symnorm_seed42
```

当前工作区已确认旧 E5A AnchorMean checkpoint 与 Bank 的 retrieval fingerprint 一致，并且能由 `metrla_e5_final_anchor_mean_v1.yaml` 严格加载。因此新增正式训练只运行上面的 SymNorm 命令；旧 AnchorMean 直接作为参考组，但仍须运行相同的 validation 诊断。若实验机缺少旧产物，再用 AnchorMean 配置补跑，不能用 SymNorm 结果代替参考组。

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/build_bank.py `
  --config configs/metrla_e5_final_symnorm_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_symnorm_seed42/pretrain_best_relation.pt `
  --dataset-name METR-LA `
  --output-dir artifacts/metrla_bank_e5_final_symnorm_seed42

C:/Users/31396/.conda/envs/research/python.exe scripts/diagnose_teacher_metrics.py `
  --config configs/metrla_e5_final_anchor_mean_v1.yaml `
  --checkpoint artifacts/metrla_e5a_offset_decay_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5a_offset_decay_relation_seed42 `
  --split val `
  --candidate-protocol relaxed_calendar `
  --output-dir artifacts/metrla_e5_final_r1_anchor_mean_reference/teacher_metric_val

C:/Users/31396/.conda/envs/research/python.exe scripts/diagnose_teacher_metrics.py `
  --config configs/metrla_e5_final_symnorm_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_symnorm_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_final_symnorm_seed42 `
  --split val `
  --candidate-protocol relaxed_calendar `
  --output-dir artifacts/metrla_e5_final_symnorm_seed42/teacher_metric_val
```

### 16.4 R2：CFDP + latent source pretraining

该命令启用 `12-D CFDP profile + 36-D latent = 48-D key`。它必须在 R1 得出 Keep/Remove 结论后运行；若 R1 删除 SymNorm，应先把 R2 配置中的 teacher normalization 恢复为 R1 胜出的版本。

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/pretrain.py `
  --config configs/metrla_e5_final_sym_profile_v1.yaml `
  --run-name metrla_e5_final_cfdp_seed42
```

### 16.5 target-local Bank v2

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/build_bank.py `
  --config configs/pemsbay_e5_final_transfer_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_cfdp_seed42/pretrain_best_relation.pt `
  --dataset-name PEMS-BAY `
  --output-dir artifacts/pemsbay_bank_e5_final_cfdp_seed42
```

### 16.6 下游校准微调

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/train_downstream.py `
  --config configs/metrla_e5_final_calibrator_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e5_final_cfdp_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_final_cfdp_seed42
```

### 16.7 诊断与 validation

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/diagnose_cfdp.py `
  --config configs/metrla_e5_final_sym_profile_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_cfdp_seed42/pretrain_best_relation.pt `
  --split val `
  --output artifacts/metrla_e5_final_cfdp_seed42/cfdp_diagnostic_val.json

C:/Users/31396/.conda/envs/research/python.exe scripts/diagnose_retrieval.py `
  --config configs/metrla_e5_final_sym_profile_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_cfdp_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_final_cfdp_seed42 `
  --split val

C:/Users/31396/.conda/envs/research/python.exe scripts/diagnose_downstream.py `
  --config configs/metrla_e5_final_calibrator_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e5_final_cfdp_seed42/pretrain_best_relation.pt `
  --downstream-checkpoint artifacts/metrla_e5_final_calibrator_seed42/downstream_best.pt `
  --bank artifacts/metrla_bank_e5_final_cfdp_seed42 `
  --split val
```

## 17. 与相关工作的边界

1. [PIR, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/331c41353b053683e17f7c88a797701d-Abstract-Conference.html) 使用预测误差估计和 local/global revision。本方案借鉴其 error identification，但修正方向来自 future-guided OffsetDecay memory，融合标签直接监督 convex blend weight。
2. [Neural Additive Models, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/251bd0442dfcc53b5a761e050f8022b8-Abstract.html) 提供逐特征非线性加性解释结构。本方案只使用很小的 shape functions，不复制其完整 tabular pipeline。
3. [SARAF, KDD 2026](https://arxiv.org/abs/2606.04135) 研究 stationarity-aware retrieval 和 adaptive aggregation。本方案的核心区别是 source future relation pretraining、target-local causal Bank 和 node/horizon-level OffsetDecay residual correction。
4. [N-BEATS, ICLR 2020](https://arxiv.org/abs/1905.10437) 说明显式输出分量可以提升时序预测解释性。本方案的 CFDP 是 key 的可解释未来语义，不是 N-BEATS basis expansion。

不能把“uncertainty + retrieval weight”“显式 future profile”或“加性网络”单独声称为新贡献。论文可检验的组合贡献是：

> 用未来监督把历史编码成可分解的规范化未来动力学语义与部署关系语义，在不同目标图和节点集合上重建 causal Bank，并使用实例风险和候选共识决定 OffsetDecay memory 对任意下游预测器的修正幅度。

该表述只有在 source/random、cross-dataset、multi-backbone 和多随机种子实验全部通过后才能作为论文贡献。

## 18. 最终保留、删除与停止规则

| 结果 | 决策 |
|---|---|
| SymNorm 训练后不改善 relation 或 no-confidence downstream | 删除 SymNorm，回退 AnchorMean |
| CFDP 只改善辅助 loss，不改善 retrieval/transfer | 删除 profile key，保留纯 latent 48-D key |
| profile-only 最好但 total key 不好 | 只允许在固定 `gamma` 三点消融中选择，不增加可学习 gate |
| PredictedBaseRisk 排序无效 | 删除 risk head，不影响 future-guided retrieval |
| additive fusion 不优于等参数 MLP | 使用简单 MLP，不能为可解释性牺牲预测稳定性 |
| 完整校准器不优于当前 confidence | 保留当前 confidence，停止校准结构扩展 |
| source 有效、cross-dataset 无效 | 只能写同域 future relation，不宣称可泛化 retrieval |
| retrieval 和 calibrator 均通过 | 冻结大结构，进入小幅超参数确认、多 backbone、多 seed 和论文写作 |

本方案的复杂度上限已经固定。后续任何新增机制都必须先指出本文哪个明确能力缺口仍未解决，并同时删除或简化一个现有机制；不能再次通过堆叠分支扩大模型。

---

## 19. 本轮执行记录：pooling 归因与 Global288 下游归因

### 19.1 为什么先做 probe 而不是改正式 encoder

当前 profile cosine 偏低不能直接归因于 pooling。它可能来自共享 pooling、线性 profile head 太弱、Local12 历史信息不足，或 CFDP 训练目标与检索距离几何不匹配。为一次只关闭一个不确定性，本轮冻结正式 encoder，仅训练三个诊断 probe：

- **A `linear_pooled`**：复用正式 encoder 的共享 temporal pooling，只训练一个线性 profile head；
- **B `mlp_pooled`**：复用同一 pooling，只把 profile head 换成两层 GELU MLP；
- **C `horizon_specific`**：不使用共享 pooling，针对每个 profile horizon 从完整 token 序列学习独立的时间 attention；
- **`teacher_profile_oracle`**：直接用 validation 真 CFDP 作为 key 的离线几何上限，不可部署、不写入 Bank。

A/B/C 的输入始终是 history encoder 输出，训练 target 才读取 source-train future 构造的 CFDP；推理/部署不构造 query CFDP。若 B>A 且 C≈B，只需增强 head；若 C 同时超过 A/B，才考虑 horizon-specific pooling；若 oracle 也低，说明 CFDP/距离定义有问题，应删除或重定义 profile，而不是继续堆叠网络。

### 19.2 本轮正式命令

实验机：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_cfdp_probe_queue.ps1
```

脚本按 Local12、Global288 顺序分别读取已有 profile checkpoint，训练 5 个 probe epoch，输出 profile MAE/cosine、CFDP-MAE relation Spearman/Recall@5、OD relation Spearman/Recall@5 及 oracle。脚本拒绝覆盖已有正式目录。

本机：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_global288_downstream_attribution_queue.ps1
```

该队列按 `base_only -> pretrained selector + OffsetDecay -> random selector + OffsetDecay` 顺序串行运行。三组固定 `seed=42`、`exact_calendar` 候选协议和 `level_weight=0`；每组训练后自动执行 validation evaluation 与 downstream diagnosis。唯一研究变量是 memory/selector 来源，避免把 calendar 候选或 level score 的变化误认为 future relation 学习收益。

### 19.3 结果使用边界

probe 的 future teacher 和 oracle 只用于 source-train/validation 诊断，不能进入部署检索；下游 formal 结果才用于判断 OffsetDecay 是否真正校准任意下游 backbone。smoke 目录只证明接口、梯度、Bank 指纹和日志流程，不能作为论文证据。只有当预训练 selector 相对 random 在 memory MAE 或下游 MAE 上产生超过 seed 波动的稳定收益，才保留 CFDP/selector 优化；否则回退到更简单的纯 latent 或原有 relation encoder。
