# STAnchor-BlockMemory：新版下游校准器架构设计

更新时间：2026-08-31  
文档状态：架构设计稿，待下游验证与跨数据集实验后形成论文结果  
适用范围：HN-OffsetDecay 检索器、目标域历史 Memory Bank、冻结下游 backbone、posthoc_frozen_base

本文只描述新版下游校准器架构，不把尚未完成的下游训练和跨数据集实验写成既定结论。检索编码器、Bank 和候选协议在本文中作为固定上游输入。

---

## 1. 设计目标

当前 Base-as-candidate 校准器存在三个结构问题：

1. 配置声明四头，但实际候选 logit 仍是单个 hidden vector 的 scalar dot-product，并没有真正的多头候选交互。
2. 多个高维独立 MLP 在目标域 post-hoc 训练中容易形成容量冗余和过拟合。
3. 若只使用标准 MultiheadAttention 的 attention weights，而不使用 attn_output，Value 和 Output projection 不会影响最终预测，可能形成类似旧 value_proj 的死分支。

新版校准器的目标是：

- 使用真正的四头 Q/K/V/O 多头交叉注意力；
- 让 attn_output 进入最终候选评分，保证 Q/K/V/O 都有有效梯度；
- 将历史候选和 Base 放入统一的 K+1 路由决策；
- 保留 Base-as-candidate 的零残差 fallback；
- 最终只对原始 candidate residual 做数值聚合；
- 将 token hidden dimension 从 384 压到 256；
- 目标总参数量约 58--62 万；
- 不改变 Bank、候选协议或 frozen path cache 的语义。

---

## 2. 输入接口和信息边界

### 2.1 上游输入

校准器接收：

- 冻结下游 backbone 的 Base prediction；
- HN-OffsetDecay encoder 和目标 Bank 检索出的 candidate future；
- node-level key similarity；
- candidate level distance；
- candidate validity mask；
- 当前短 context。

校准器不更新 backbone、retrieval encoder 和 Bank。

### 2.2 张量形状

当前单变量交通预测使用：

$$
T=12,\qquad H=12,\qquad C=1.
$$

| 张量 | 形状 | 含义 |
|---|---|---|
| X_short | B x T x N x C | 下游 backbone 的短 context |
| Y_base | B x H x N x C | 冻结 backbone 预测 |
| F_cand | B x H x N x K x C | Top-K 历史 candidate future |
| M_cand | B x H x N x K x C | candidate validity mask |
| S_key | B x N x K | node key similarity |
| D_level | B x N x K | context level distance |

当前主实验使用 K=12，因此统一候选集合共有 13 个选项，其中第 13 个是 Base token。

### 2.3 Future-information boundary

真实 query future 只允许用于：

- forecast loss；
- training-time candidate quality teacher；
- validation/test 离线指标；
- oracle 和诊断分析。

真实 query future 不允许进入：

- state token；
- candidate token；
- Base token；
- MHA query/key/value；
- candidate retrieval；
- deployment-time routing。

---

## 3. 核心语义：Residual Candidate Mixture

第 k 个历史 candidate 相对于 Base 的 residual 定义为：

$$
\boldsymbol\Delta_{q,h,n,k}
=
\mathbf F^{cand}_{q,h,n,k}
-
\mathbf Y^{base}_{q,h,n}.
$$

Residual 表示：如果采用该历史 future，应对 Base 施加什么方向和幅度的修正。

Base 被建模为第 K+1 个候选，但它不提供外部修正：

$$
\boldsymbol\Delta_{q,h,n,base}=\mathbf0.
$$

最终预测始终为：

$$
\boxed{
\mathbf Y^{final}_{q,h,n}
=
\mathbf Y^{base}_{q,h,n}
+
\sum_{k=1}^{K}
\pi_{q,h,n,k}
\boldsymbol\Delta_{q,h,n,k}}
$$

其中 pi 是历史候选和 Base 上统一归一化的 routing distribution。

该公式的解释是：

- 历史候选权重高：对 Base 施加相应 residual；
- Base 权重高：保留 Base，关闭大部分历史修正；
- Base 权重为 1：最终预测严格等于 Base；
- MHA 和 routing head 只做候选选择，不直接生成自由预测值。

---

## 4. 架构组成和总体数据流

新版校准器由以下组件组成：

1. State encoder：编码当前短 context 和 Base 行为；
2. Local candidate encoder：编码每个 horizon 的候选局部特征；
3. Candidate trajectory encoder：编码候选完整 residual 轨迹；
4. Base encoder：编码 Base 风险、context 波动性和 horizon；
5. Horizon embedding：提供预测步身份；
6. Four-head standard Q/K/V/O MHA：形成 candidate-conditioned query；
7. Shared per-candidate routing head：对每个候选逐一输出 logit；
8. Residual aggregator：使用 routing weight 聚合历史 residual。

$$
\begin{aligned}
X^{short},Y^{base}&\to E_{state}\to s_{q,n},\\
\Delta_k&\to E_{local},E_{traj}\to z_{q,h,n,k},\\
s_{q,n},\{z_{q,h,n,i}\}_{i=1}^{K+1}&\to MHA\to q'_{q,h,n},\\
q'_{q,h,n},z_{q,h,n,i}&\to E_{route}\to \ell_{q,h,n,i},\\
\ell&\to masked\ softmax\to\pi,\\
\pi,\Delta&\to residual\ aggregation\to Y^{final}.
\end{aligned}
$$

新版不包含：

- independent Alpha gate；
- free Beta correction；
- memory-level second-stage confidence gate；
- hidden-to-prediction free value branch。

---

## 5. State Encoder 和 Horizon Query

### 5.1 State input

每个节点的 state input 使用当前可观测短 context 和 frozen Base prediction：

$$
\mathbf a_{q,n}
=
Concat\left(
Norm(\mathbf X^{short}_{q,:,n,:}),
\mathbf Y^{base}_{q,:,n,:}
\right)
\in\mathbb R^{(T+H)C}.
$$

当前输入维度为：

$$
(T+H)C=24.
$$

State encoder：

$$
E_{state}:24\to256\to256,
\qquad
s_{q,n}\in\mathbb R^{256}.
$$

Norm 只使用当前可观测 context 的统计量。

### 5.2 Horizon query

为每个预测步设置 learnable horizon embedding：

$$
e_h\in\mathbb R^{256},
\qquad h=1,\ldots,H.
$$

初始 query：

$$
q_{q,h,n}=W_qs_{q,n}+e_h
\in\mathbb R^{256}.
$$

这样同一个节点可以在近端和远端 horizon 使用不同候选选择策略。

---

## 6. Candidate Token

### 6.1 Local candidate feature

$$
x^{local}_{q,h,n,k}
=
Concat\left(
\Delta_{q,h,n,k},
|\Delta_{q,h,n,k}|,
s^{key}_{q,n,k},
-d^{level}_{q,n,k},
p_h
\right).
$$

其中：

- Delta：候选相对于 Base 的修正方向；
- absolute Delta：修正幅度；
- s_key：node key similarity；
- negative d_level：level 差异的相反数；
- p_h：归一化 horizon position。

单变量输入维度为 5：

$$
E_{local}:5\to128\to256.
$$

### 6.2 Full residual trajectory

同一个候选在所有 horizon 的 residual trajectory：

$$
\Delta^{traj}_{q,n,k}
=
Vec(\Delta_{q,1,n,k},\ldots,\Delta_{q,H,n,k})
\in\mathbb R^{HC}.
$$

当前 H=12、C=1，因此输入为 12 维：

$$
E_{traj}:12\to64\to256.
$$

该分支用于识别：

- 近端相似、远端发散；
- 全程同向但幅度不同；
- 发生转折或高波动的候选。

### 6.3 Historical candidate token

$$
z_{q,h,n,k}
=
LN\left(
E_{local}(x^{local}_{q,h,n,k})
+E_{traj}(\Delta^{traj}_{q,n,k})
+e_h
\right)
\in\mathbb R^{256}.
$$

Trajectory representation 沿 horizon 广播，horizon embedding 表示当前预测步。

### 6.4 Base token

Base token 使用部署可获得的行为特征：

$$
x^{base}_{q,h,n}
=
Concat(r^{base}_{q,h,n},v^{ctx}_{q,n},p_h,1).
$$

其中：

- r_base：由 context 和 Base 产生的风险代理；
- v_ctx：context 的观测波动性；
- p_h：horizon position；
- 1：Base token type indicator。

$$
E_{base}:4\to128\to256,
$$

$$
z_{q,h,n,base}
=
LN\left(E_{base}(x^{base}_{q,h,n})+e_h+e_{base}\right).
$$

Base token 始终有效，residual 恒为零。固定 dataset ID 或 backbone ID 不纳入 token，因为校准器会在每个目标数据集和 backbone 上重新训练。

---

## 7. Four-Head Standard Q/K/V/O MHA

### 7.1 Function boundary

MHA 根据当前 query 和候选集合形成 candidate-conditioned query，不直接生成预测值。最终预测仍使用 raw candidate residual aggregation。

### 7.2 Input shape

将每个 batch、horizon 和 node 作为一条独立 query：

| 张量 | 形状 |
|---|---|
| query | BHN x 1 x 256 |
| key/value | BHN x (K+1) x 256 |
| attention heads | 4 |
| head dimension | 64 |
| key padding mask | BHN x (K+1) |

Top-12 时 key/value sequence length 为 13。

### 7.3 Head-level computation

$$
Q^{(m)}=qW_Q^{(m)},\quad
K_i^{(m)}=z_iW_K^{(m)},\quad
V_i^{(m)}=z_iW_V^{(m)},
$$

$$
W_Q^{(m)},W_K^{(m)},W_V^{(m)}
\in\mathbb R^{256\times64}.
$$

$$
a_i^{(m)}
=
softmax_i\left(
\frac{Q^{(m)}(K_i^{(m)})^\top}{\sqrt{64}}
\right),
$$

$$
o^{(m)}=\sum_i a_i^{(m)}V_i^{(m)}.
$$

四头拼接后经过 output projection：

$$
o=Concat(o^{(1)},o^{(2)},o^{(3)},o^{(4)})W_O
\in\mathbb R^{256}.
$$

### 7.4 Effective gradient path

如果只使用 attention weights，V/O 不影响最终 prediction。新版必须使用 attn_output：

$$
q'_{q,h,n}=LN(q_{q,h,n}+o_{q,h,n}).
$$

q_prime 随后进入 routing head，因此：

$$
Y^{final}\to\pi\to q'\to MHA(Q,K,V,O)
$$

保证 Q/K/V/O 都获得有效梯度。MHA attention weights 只作为诊断量，不等于最终 routing weights。

---

## 8. Shared Per-Candidate Routing Head

### 8.1 Why not fixed-slot Linear

主架构不采用：

$$
q'\xrightarrow{Linear}\mathbb R^{K+1}.
$$

该写法把第 i 个输出绑定候选排序位置，不能保证候选重排后权重同步重排，也不能自然适配 K 的变化。

### 8.2 Content-conditioned score

对每个候选构造：

$$
r_{q,h,n,i}
=
Concat\left(
q'_{q,h,n},
z_{q,h,n,i},
q'_{q,h,n}\odot z_{q,h,n,i}
\right)
\in\mathbb R^{768}.
$$

所有候选共享：

$$
E_{route}:768\to128\to1,
$$

$$
\ell_{q,h,n,i}=E_{route}(r_{q,h,n,i}).
$$

候选重排时 logits 和 weights 同步重排，保持候选集合的 permutation equivariance。

### 8.3 K+1 masked softmax

Base mask 永远为 1：

$$
m=Concat(m^{cand}_{1:K},1).
$$

无效历史候选的 logit 设置为 negative infinity：

$$
\pi_i=softmax_i(masked(\ell_i)).
$$

满足：

$$
\pi_i\ge0,\qquad\sum_{i=1}^{K+1}\pi_i=1.
$$

Top-12 时输出 13 个 routing weights，最后一个是 Base usage。

---

## 9. Residual Aggregation and Base Fallback

$$
R^{mix}_{q,h,n}
=
\sum_{k=1}^{K}\pi_{q,h,n,k}\Delta_{q,h,n,k},
$$

$$
Y^{final}_{q,h,n}=Y^{base}_{q,h,n}+R^{mix}_{q,h,n}.
$$

当历史候选全部无效时：

$$
m_1=\cdots=m_K=0,\quad m_{base}=1,
$$

$$
\pi_{base}=1,\quad R^{mix}=0,\quad Y^{final}=Y^{base}.
$$

Base fallback 由 K+1 masked softmax 自然完成，不需要 independent Alpha。工程实现仍保留 finite check，防止错误 mask 造成整行无效。

---

## 10. Parameter and Resource Budget

按 T=12、H=12、C=1、K=12 估算：

| 模块 | 参数量约 |
|---|---:|
| State encoder 24 -> 256 -> 256 | 72,192 |
| Local encoder 5 -> 128 -> 256 | 33,792 |
| Base encoder 4 -> 128 -> 256 | 33,664 |
| Query projection 256 -> 256 | 65,792 |
| Base risk probe 256 -> 12 | 3,084 |
| Trajectory encoder 12 -> 64 -> 256 | 17,472 |
| Standard Q/K/V/O MHA | 263,168 |
| Shared routing head 768 -> 128 -> 1 | 98,561 |
| Horizon/type/bias/normalization | 4,865 |
| **Total** | **约 592,590** |

MHA 参数量：

$$
4D^2+4D
=
4\times256^2+4\times256
=263,168.
$$

参数、梯度和 Adam state 不是显存大头。主要显存来自：

$$
Z\in\mathbb R^{B\times H\times N\times(K+1)\times D}
$$

及其 backward activations。D 从 384 降到 256 后，candidate token activation 约减少三分之一。MHA sequence length 只有 K+1=13，不是对 288 步 context 做长序列 attention。

---

## 11. Loss and Training Boundary

### 11.1 Forecast loss

$$
L_{forecast}=MAE(Y^{final},Y).
$$

### 11.2 Candidate quality teacher

历史候选误差：

$$
e_k=mean_c|F_k^{cand}-Y|,
$$

Base error：

$$
e_{base}=mean_c|Y^{base}-Y|.
$$

构造 K+1 teacher：

$$
p_i^{teacher}=softmax_i(-\bar e_i/0.2),
$$

$$
L_{quality}=D_{KL}(p^{teacher}\Vert\pi).
$$

第一版总损失：

$$
L=L_{forecast}.
$$

Query future 不进入 inference token 或 routing。

### 11.3 Removed paths

新版不恢复：

- independent Alpha gate；
- free Beta additive correction；
- memory-level second-stage confidence gate；
- hidden-to-prediction free value branch；
- zero-coefficient value projection；
- forced Base/history usage balance regularizer。

---

## 12. Freeze and Cache Boundary

| 模块 | 是否更新 |
|---|---:|
| Downstream backbone | 否 |
| HN-OffsetDecay encoder | 否 |
| Bank and retrieval | 否 |
| Frozen path cache | 否 |
| New calibrator encoders | 是 |
| MHA Q/K/V/O | 是 |
| Routing head | 是 |

Frozen path cache 只保存 detached 的 frozen Base prediction、candidate set、candidate mask 和 aggregation output。

以下对象不得缓存：

- local candidate token；
- trajectory token；
- Base token；
- MHA output；
- routing logits；
- routing weights。

这些张量依赖当前校准器参数，必须在每次 forward 中重新计算。缓存机制和字段不改变；不同 candidate protocol 使用不同 run name，并在新进程重新建立 cache。

---

## 13. Configuration and Output Interface

推荐配置：

    calibrator_arch: transformer_candidate_router
    candidate_token_dim: 256
    calibrator_state_dim: 256
    candidate_attention_heads: 4
    candidate_trajectory_hidden_dim: 64
    routing_hidden_dim: 128
    mha_dropout: 0.05
    use_horizon_embedding: true
    base_logit_init_bias: 1.0
    base_warmup_epochs: 2
    candidate_quality_weight: 0.0
    candidate_quality_temperature: 0.2
    frozen_path_cache: true

候选配置：

    node_top_k: 12
    event_top_r: 32
    level_weight: 0.0

建议保持以下输出：

- final_prediction；
- routing_weights；
- mha_attention_weights；
- base_usage；
- historical_mass；
- routing_entropy；
- per-head attention entropy；
- correction_norm；
- trajectory_token_norm。

Historical mass 等于 1 - base_usage，只是统计量，不是 independent Alpha network。

不同 architecture、token dimension、head 数、node Top-K 和 candidate protocol 的 checkpoint 不允许 silent load。

---

## 14. Architecture Verification Requirements

正式训练前必须验证：

1. Total parameters 约 58--62 万；
2. 每个 head dimension 为 64；
3. Top-12 时 routing output 为 13；
4. routing weights 有限且每行和为 1；
5. candidate permutation 后 weights 同步 permutation；
6. 所有历史候选无效时 Base usage=1；
7. 所有历史候选无效时 final prediction 严格等于 Base；
8. MHA Q/K/V/O gradient 有限且非零；
9. routing head、trajectory encoder 和 token projection gradient 有限且非零；
10. 不存在 zero-coefficient dead branch；
11. state dict 和 architecture fingerprint 可保存/加载；
12. 真实 frozen path cache hit 路径可以 forward；
13. 记录 calibrator forward time 和 CUDA peak。

这些检查只验证结构和工程正确性，不等于性能已经提升。

---

## 15. Risks and Controls

### 15.1 V/O dead branch

检查 q_prime 是否真正进入 routing logits，并检查 Q/K/V/O gradient。

### 15.2 Fixed-slot candidate semantics

必须使用 shared per-candidate routing head，并加入 candidate permutation test。

### 15.3 Base collapse

若 Base usage 长期接近 1，先检查 Base bias、quality teacher 和 token scale，不新增 gate。

### 15.4 Historical residual amplification

若 Base usage 接近 0 且 correction norm 过大，检查 candidate mask、level feature 和 quality teacher，保留 Base-biased initialization。

### 15.5 Target-domain overfitting

比较 D=256 新版与 D=384 旧版的 train-validation gap、routing entropy 和 horizon MAE，不直接堆叠更多 MHA layer。

### 15.6 Runtime budget

分别测 cache build、cache hit 和 calibrator forward。Cache hit 后目标单轮不超过 300 秒，不通过修改 cache 语义隐藏耗时。

---

## 16. Experimental and Paper Boundary

完成结构测试后，应在相同 Base checkpoint、Bank、seed、data split、candidate protocol 和 optimizer 下比较：

1. 旧 27 万级 Base-as-candidate；
2. 旧 D=384 trajectory-conditioned scalar-dot 版；
3. 新 D=256 four-head MHA routing 版；
4. 新版移除 trajectory encoder 的 parameter-matched control。

先进行 3--5 epoch matched validation，再决定是否进入四个下游模型和跨数据集正式实验。评价同时报告 MAE、15/30/60 分钟指标、Base usage、routing entropy、candidate-error correlation、runtime 和 CUDA peak。

论文中的准确定位是：

> 在冻结 backbone 和固定检索候选的条件下，校准器利用真正的多头候选交互和共享逐候选路由，在 Base fallback 与历史 residual experts 之间进行 horizon-specific 选择。

该设计解决的是候选路由的统一决策和梯度路径问题，不预先声称消除 oracle gap，也不把目标域重新训练的校准器描述为跨数据集泛化模块。


