# E5 Latent48 TGGE-Structured Error Corrector v2 完整优化方案

## 1. 文档目的与主线

本文档给出一个自包含的 v2 设计，用于改造 STAnchor-BlockMemory 的检索编码器和下游后置修正模块。读者不需要阅读其他方案文档即可理解本方案的模型、训练协议、参数预算、信息边界和验收标准。

本方案保留项目的两个核心主张：

1. 由 Latent48 检索键表达可迁移的时空语义，并从历史事件 Bank 中检索相似事件。
2. 不替换已经训练好的下游预测器，而是根据基础模型风险和检索证据，学习一个有界、可解释的后置修正量。

本次调整只解决一个明确的容量失衡问题：当前检索侧约 39 万参数，而后置修正侧只有约 0.28 万参数。v2 将检索侧压缩到约 20--30 万参数，同时把修正头扩展到约 10--20 万参数。扩容不通过堆叠无明确用途的注意力或 MLP 完成，而是把参数分配给时间动态建模、局部图传播、历史条件非局部路由、基础风险估计和检索证据融合。

当前正在运行的旧版 STGCN + Latent48 实验、旧 checkpoint 和旧 Bank 均保持不变。v2 使用新的版本号、checkpoint 和 Bank 目录。

---

## 2. 术语、数据与信息边界

### 2.1 时空预测任务

输入历史窗口记为

$$
X\in\mathbb R^{B\times T\times N\times C},
$$

其中：

- $B$：batch 大小；
- $T=12$：下游预测使用的历史步数，即 60 分钟历史；
- $N=207$：METR-LA 的传感器节点数；
- $C=1$：速度变量通道数。

目标未来窗口记为

$$
Y\in\mathbb R^{B\times H\times N\times C},
\qquad H=12,
$$

即预测未来 12 个 5 分钟步。

检索编码器读取更长的 288 步历史窗口：

$$
X^{\mathrm{ret}}\in\mathbb R^{B\times 288\times N\times 1}.
$$

288 步被划分为 24 个长度为 12 的时间 patch。编码器输出每个节点一个 48 维 L2 归一化键：

$$
K\in\mathbb R^{B\times N\times 48}.
$$

### 2.2 Latent48、Bank 与检索聚合

Latent48 是只使用 48 维连续潜在向量的检索键布局，不包含独立的 calendar profile 分支。Bank 是由历史事件构成的只读记忆库，保存事件键、节点键和对应的已发生事件未来 payload。

检索采用两阶段协议。首先将一个事件的所有节点键求均值并再次 L2 归一化，得到 48 维 event key。对经过日历和因果过滤的历史候选，按 event key 相似度保留 Top-R 个事件；然后对每个 query node，按 node key 相似度从这些事件中保留 Top-K 个节点候选。主线配置使用 $R=32$、$K=5$，且纯 Latent48 归因实验令额外 level reranking weight 为 0。

对当前 query，检索器只允许使用满足时间因果约束的历史事件：候选事件的 future end 必须早于当前 query 的 context start。对候选集合 $\mathcal R(q)$，聚合结果为

$$
\widehat Y^{\mathrm{mem}}
=
\sum_{j\in\mathcal R(q)}\pi_jY^{\mathrm{future}}_j,
\qquad
\sum_j\pi_j=1.
$$

当前 query 的真实未来 $Y$ 不进入 query key、候选排序、检索特征或推理前向计算。历史 Bank 事件的 future payload 是在该事件已经结束后保存的历史数据，可以在部署时被读取。

### 2.3 PostHoc 修正

PostHoc 修正指先独立训练基础下游模型，再冻结基础模型，只训练检索诊断和修正头。本方案使用 STGCN（Spatio-Temporal Graph Convolutional Network，时空图卷积网络）作为下游基础模型，但修正接口不依赖 STGCN 内部隐藏状态，只依赖历史输入和基础预测。基础模型输出为

$$
\widehat Y^{\mathrm{base}}=f_{\theta^\star}(X),
$$

其中 $\theta^\star$ 在 v2 修正训练阶段保持不变。最终输出沿用有界残差形式：

$$
\widehat Y^{\mathrm{final}}
=
\widehat Y^{\mathrm{base}}
+w\odot
\left(
\widehat Y^{\mathrm{mem}}-
\widehat Y^{\mathrm{base}}
\right).
$$

当 Bank 候选无效时，强制 $w=0$，模型严格退化为基础模型。

### 2.4 预训练目标与未来信息用途

TGGE 仍采用两个既有预训练目标。

第一个是 masked reconstruction：只在 288 步历史窗口内部遮蔽一部分时间 patch 或节点观测，并重建被遮蔽的历史值。目标值仍然属于 query 时刻之前的可见历史范围，不涉及下游预测未来。

第二个是 OffsetDecay relation teacher。对两个预训练样本 $i,j$，先以各自历史窗口末端水平为参考，构造未来偏移轨迹 $\Delta Y_i$ 和 $\Delta Y_j$，再对较近 horizon 给予更高权重：

$$
D^{\mathrm{OD}}_{ij}
=
\frac{
\sum_{h=1}^{H}\rho^{h-1}M_{ijh}
\left|\Delta Y_{i,h}-\Delta Y_{j,h}\right|
}{
\sum_{h=1}^{H}\rho^{h-1}M_{ijh}+\epsilon
},
\qquad 0<\rho\leq1.
$$

$M_{ijh}$ 表示两个样本在 horizon $h$ 都有有效观测。该 future distance 只在预训练阶段作为 teacher，监督当前历史 key 的相对相似度；future trajectory 不输入 TGGE，也不保存在 query key 中。下游检索和部署阶段只计算历史编码，不再需要当前 query 的 future。

---

## 3. 当前版本的容量问题

当前纯 Latent48 检索侧使用：

~~~text
hidden_dim       = 96
encoder_layers   = 3
num_heads        = 4
ffn_multiplier   = 2
~~~

每个编码块依次包含时间 Multi-Head Attention、静态图稀疏 Attention 和 FFN。完整 retrieval state 还包括 patch embedding 和 retrieval head，当前参数量约为：

~~~text
retrieval state  ≈ 391,836
~~~

当前 PostHoc-Wide 修正模块为：

~~~text
risk head        ≈ 2,380
error-aware head ≈   442
total            = 2,822
~~~

当前风险头只把 24 个历史/base 输入映射到 12 个 horizon 风险值；融合器为 9 个相互独立的标量 shape function。它具有较强的可解释性，但跨特征交互和不同 horizon 的条件表达能力有限。

因此，v2 的目标不是让所有模块都变大，而是把容量放在以下四个必要位置：

1. 时间 patch 的局部和多尺度变化；
2. 静态图邻居传播；
3. 基础模型风险编码；
4. 检索证据与基础风险之间的条件门控。

---

## 4. v2 总体结构

v2 名称为 TGGE-Structured Error Corrector：

- TGGE（Temporal-Gated Graph Encoder）：时间门控图编码器，用局部多尺度时间混合替代全量时间 Attention，同时保留静态图空间传播，并增加历史条件的非局部稀疏路由；
- Structured Error Corrector：结构化误差修正头，由基础风险分支、检索证据分支、交互门控和可解释输出组成。

总体数据流为：

$$
X^{\mathrm{ret}}
\rightarrow
\text{Patch Embedding}
\rightarrow
\text{TGGE}
\rightarrow
\text{Latent48 Key}
\rightarrow
\text{Bank Retrieval}
\rightarrow
\text{Structured Error Corrector}
\rightarrow
\widehat Y^{\mathrm{final}}.
$$

下游 STGCN 只负责产生 $\widehat Y^{\mathrm{base}}$。在 PostHoc 训练中，STGCN 和 Latent48 编码器都冻结，只有 Structured Error Corrector 更新。

---

## 5. TGGE 检索编码器

### 5.1 Patch embedding

对每个长度为 12 的 patch，保留现有的数值、节点统计量、星期和日内 slot embedding：

$$
Z^{(0)}_{p,n}
=
\operatorname{LN}
\left(
W_v\operatorname{Patch}(X^{\mathrm{ret}}_{p,n})
+W_l L_n
+E_{\mathrm{weekday}(p)}
+E_{\mathrm{slot}(p)}
\right).
$$

输出形状为

$$
Z^{(0)}\in\mathbb R^{B\times 24\times N\times 80}.
$$

v2 使用 hidden_dim=80，保持 4 个 attention heads 在空间图分支中可整除，同时降低隐藏宽度。

### 5.2 时间门控分支

本次 v2 编码提交只启用下文 5.4 的历史条件混合范围路由，时间分支暂保持仓库已有的因子化时间 `MultiheadAttention`，以便先隔离空间路由的增益。下面的深度可分离卷积与 GLU 是后续独立消融项，不能当作本次 checkpoint 已实现的结构。

第 $l$ 个编码块接收

$$
Z^{(l)}\in\mathbb R^{B\times P\times N\times D},
\qquad P=24,\ D=80.
$$

时间分支沿 patch 轴做深度可分离卷积。深度卷积对每个隐藏通道独立提取局部变化，再用逐点线性层混合通道：

$$
U^{(l)}
=
\operatorname{DWConv}_{k=3,d_l}
\left(\operatorname{LN}_t(Z^{(l)})\right),
$$

$$
T^{(l)}
=
\operatorname{GLU}
\left(W^{(l)}_{\mathrm{pw}}U^{(l)}+b^{(l)}_{\mathrm{pw}}\right).
$$

这里 GLU 将逐点投影结果分成值分支和门控分支：

$$
\operatorname{GLU}([A;B])=A\odot\sigma(B).
$$

三层使用固定 dilation：

~~~text
d1 = 1
d2 = 2
d3 = 4
~~~

这不是堆叠多个无关模块，而是让 12 步 patch 分别观察相邻变化、短周期变化和更长时间间隔。卷积只读取历史 patch，不读取未来时间步。

### 5.3 静态图空间分支

空间分支保留当前项目的静态图稀疏 Attention。给定图边 $(n,m)\in\mathcal E$，对节点 $n$ 的邻居消息计算：

$$
q_n=W_qz_n,
\qquad
k_m=W_kz_m,
\qquad
v_m=W_vz_m,
$$

$$
\alpha_{nm}
=
\operatorname{softmax}_{m\in\mathcal N(n)}
\left(
\frac{q_n^\top k_m}{\sqrt d}
+\beta\log A_{nm}
\right),
$$

$$
G^{(l)}_n
=
\sum_{m\in\mathcal N(n)}\alpha_{nm}v_m.
$$

$A_{nm}$ 是固定图边权，$\beta$ 是图先验强度。该分支只负责物理图中的直接邻居传播，保留作为所有非局部关系的局部参照。v2 不学习无约束的稠密动态图，不添加节点专属参数，也不把未来值用于图构造。

### 5.4 历史条件混合范围稀疏路由

仅使用直接邻居会把空间感受野限制在静态图的一阶边上。完全排除一阶邻居也不合理，因为一阶邻居通常包含最直接的传播信息。因此 v2 在保留完整局部静态图聚合的同时，增加一个混合范围路由分支：它从一阶邻居和远端节点中按配额选择少量当前历史下最有用的节点。该分支的目的不是重新构造一张任意动态图，而是让模型在“局部传播”和“远端关联”之间做可解释的选择。

令时间分支输出为

$$
Z^{(l)}\in\mathbb R^{B\times P\times N\times D},
\qquad P=24,\ D=80.
$$

先为每个节点构造历史状态摘要：

$$
s^{(l)}_{b,n}
=
\left[
\operatorname{Mean}_{p}Z^{(l)}_{b,p,n};
Z^{(l)}_{b,P,n}-Z^{(l)}_{b,1,n}
\right]
\in\mathbb R^{2D}.
$$

第一项表示窗口内的平均状态，第二项表示从窗口起点到终点的历史变化趋势。二者都只来自 query 的历史上下文。

使用两个共享于节点的低维投影生成目标和源表示：

$$
q^{(l)}_{b,n}=W_q\operatorname{LN}(s^{(l)}_{b,n}),
\qquad
k^{(l)}_{b,m}=W_k\operatorname{LN}(s^{(l)}_{b,m}),
$$

其中 $q,k\in\mathbb R^{d_r}$，默认 $d_r=16$。$W_q$ 与 $W_k$ 不共享，使得路由分数可以表达有方向的传播关系：

$$
r^{(l)}_{b,n,m}
=
\frac{q^{(l)\top}_{b,n}k^{(l)}_{b,m}}{\sqrt{d_r}}.
$$

为了保留静态图的物理先验，先从静态邻接中移除 self-loop，记无自环邻接为 $\bar A$，再进行行归一化：

$$
D_{nn}=\sum_m \bar A_{nm},
\qquad
A_{\mathrm{rw}}=D^{-1}\bar A.
$$

由此构造二跳和三跳扩散先验：

$$
R^{(2:3)}=A_{\mathrm{rw}}^2+A_{\mathrm{rw}}^3,
$$

这里的幂是矩阵乘法，不是逐元素平方或立方。其元素满足：

$$
(A_{\mathrm{rw}}^2)_{n,m}
=
\sum_j A_{\mathrm{rw},n,j}A_{\mathrm{rw},j,m},
$$

表示从节点 $n$ 经过一个中间节点 $j$ 到达节点 $m$ 的所有二跳路径权重之和；三次幂同理表示三跳路径。$R^{(2:3)}\in\mathbb R^{N\times N}$ 是由静态图一次性计算的、没有可训练参数的软先验矩阵，不是新的可学习动态图。

并将其作为软偏置加入路由分数：

$$
\widetilde r^{(l)}_{b,n,m}
=
r^{(l)}_{b,n,m}
+\lambda_{\mathrm{diff}}\log\left(1+R^{(2:3)}_{n,m}\right).
$$

将候选集合按静态距离划分为一阶集合和远端集合：

$$
\mathcal C_n
=
\{m\mid m\ne n,\ (n,m)\in\mathcal E\},
$$

$$
\mathcal C^{(\ge2)}_n
=
\{m\mid m\ne n,\ m\notin\mathcal N_1(n)\}.
$$

局部分支仍然对全部一阶集合做静态图 Attention；混合范围路由默认保留 $K_g=10$ 个候选，其中 $K_1=4$ 个来自一阶集合，$K_{\ge2}=6$ 个来自远端集合：

$$
\mathcal I^{(1)}_{b,n}
=
\operatorname{TopK}_{K_1,m\in\mathcal C_n}
\widetilde r_{b,n,m},
\qquad
\mathcal I^{(\ge2)}_{b,n}
=
\operatorname{TopK}_{K_{\ge2},m\in\mathcal C^{(\ge2)}_n}
\widetilde r_{b,n,m}.
$$

$$
\mathcal I_{b,n}
=
\mathcal I^{(1)}_{b,n}\cup\mathcal I^{(\ge2)}_{b,n}.
$$

当某个节点的一阶度数小于 $K_1$，或远端候选数小于 $K_{\ge2}$ 时，缺少的槽位由另一集合按分数递补；因此该配额不会造成空候选。这样既不会把一阶信息全部丢掉，也不会让路由候选退化成全部一阶邻居。

在选中集合上进行归一化聚合：

$$
\alpha_{b,n,m}
=
\operatorname{softmax}_{m\in\mathcal I_{b,n}}
\left(\frac{\widetilde r_{b,n,m}}{\tau_g}\right),
$$

$$
G^{(l)}_{\mathrm{route},b,p,n}
=
\sum_{m\in\mathcal I_{b,n}}
\alpha_{b,n,m}V^{(l)}(Z^{(l)}_{b,p,m}).
$$

路由索引在整个 288 步历史上计算一次，并对所有 patch 复用；这使路由成本为 $O(BN^2d_r)$，而不是对每个 patch 构造完整的 $N\times N$ 图。非局部消息通过保守门控注入：

$$
g^{(l)}_{b,p,n}
=
\sigma\left(
W_g[\operatorname{LN}(Z^{(l)}_{b,p,n});
\operatorname{LN}(G^{(l)}_{\mathrm{route},b,p,n})]
+b_g
\right),
$$

其中 $b_g<0$，使模型在初始化时接近原有局部图编码器。该混合范围路由不使用节点 ID embedding，不改变 Bank schema，也不读取当前 query 的真实 future。future-guided retrieval loss 仍通过最终 Latent48 key 间接训练该路由。

### 5.5 低秩前馈残差与门控残差

时间和空间分支合并为：

$$
\widetilde Z^{(l)}
=
Z^{(l)}
+\alpha_t^{(l)}\odot T^{(l)}
+\alpha_s^{(l)}\odot G^{(l)}_{\mathrm{local}}
+g^{(l)}\odot G^{(l)}_{\mathrm{route}}.
$$

其中 $\alpha_t^{(l)}$ 和 $\alpha_s^{(l)}$ 是可学习的时间、局部空间通道门，初始化为较小值；$g^{(l)}$ 是第 5.4 节定义的历史条件非局部门控。三条残差通路都从较小贡献开始，使 v2 在训练开始时接近 patch embedding 的自然基准。

再使用低秩 FFN：

$$
R^{(l)}
=
W_{\mathrm{up}}^{(l)}
\operatorname{GELU}
\left(
W_{\mathrm{down}}^{(l)}
\operatorname{LN}_f(\widetilde Z^{(l)})
\right),
$$

其中

$$
W_{\mathrm{down}}\in\mathbb R^{128\times80},
\qquad
W_{\mathrm{up}}\in\mathbb R^{80\times128}.
$$

最终块输出为

$$
Z^{(l+1)}
=
\widetilde Z^{(l)}+R^{(l)}.
$$

### 5.6 Retrieval head 与 Latent48 输出

编码器输出经过 patch pooling 得到每个节点的隐藏表示：

$$
h_n
=
\sum_{p=1}^{24}\omega_{p,n}Z_{p,n},
\qquad
\sum_p\omega_{p,n}=1.
$$

纯 Latent48 retrieval head 使用：

$$
K_n
=
\operatorname{Normalize}_2
\left(
W_2\operatorname{GELU}(W_1h_n+b_1)+b_2
\right),
\qquad K_n\in\mathbb R^{48}.
$$

v2 不增加 profile key，不把未来趋势直接写入 key，也不改变 Bank 的候选协议。由于编码器结构发生变化，v2 必须重新预训练并重新建 Bank。

### 5.7 编码器参数预算

`metrla_e5_tgge_latent48_v2.yaml` 的 `count_parameters` 实测值如下。路由 value 先降到 16 维基底，再升回隐藏维度，因此其残差映射是低秩的。

| 部分 | 实测参数量 | 作用 |
|---|---:|---|
| Patch embedding | 25,360 | 数值、统计量和时间身份编码 |
| 3 个现有因子化时间分支、图分支与 FFN | 234,640 | 历史时间建模与静态邻居传播 |
| 3 个混合范围路由分支 | 25,827 | 历史条件的一阶/多阶节点选择与低秩聚合（默认 Top-10） |
| Latent48 retrieval head | 16,928 | 48 维归一化检索键 |
| **完整 retrieval state** | **302,755** | **接近 30 万上限** |

完整预训练模型还包含 972 个参数的重建头，总参数为 303,727；重建头不写入 retrieval checkpoint 的检索状态。

---

## 6. Structured Error Corrector

### 6.1 输入与未来信息边界

修正头在推理时可使用：

1. 下游历史输入 $X$；
2. 基础模型输出 $\operatorname{StopGrad}(\widehat Y^{\mathrm{base}})$；
3. 历史 Bank 的检索候选、权重和 future payload 聚合结果；
4. 由上述部署可用量计算出的风险和检索诊断特征。

修正头在推理时不可使用当前样本的真实未来 $Y$。训练时的 $Y$ 只用于构造监督标签和预测损失，不进入任何前向输入。

### 6.2 基础风险分支

将每个节点的历史和基础预测拼接为 24 维输入：

$$
s_x
=
\left[
\operatorname{Norm}(X_{:,n,:});
\operatorname{StopGrad}(\widehat Y^{\mathrm{base}}_{:,n,:})
\right]
\in\mathbb R^{24}.
$$

基础风险表示为：

$$
h_x
=
\phi_x(s_x)
\in\mathbb R^{128},
\qquad
\phi_x:24\rightarrow256\rightarrow128.
$$

风险头输出每个 horizon 的非负风险：

$$
\widehat r_{h,n}
=
\operatorname{Softplus}(W_rh_x+b_r),
\qquad
\widehat r\in\mathbb R^{B\times H\times N\times1}.
$$

$\widehat r$ 的含义是“基础模型在该节点和 horizon 上可能产生多大误差”，不是对真实未来值的直接预测。

### 6.3 检索证据分支

纯 Latent48 模式使用一个 9 维证据向量：

$$
e_{h,n}
=
[
\widehat r_{h,n},
s_{\mathrm{retrieval}},
\Delta_{\mathrm{rank}},
q_{\mathrm{support}},
d_{\mathrm{dispersion}},
a_{\mathrm{direction}},
r_{\mathrm{level}},
d_{\mathrm{memory}},
p_h
]\in\mathbb R^9.
$$

各分量定义如下：

- $\widehat r$：基础风险分支输出；
- $s_{\mathrm{retrieval}}$：Latent48 相似度；
- $\Delta_{\mathrm{rank}}$：Top-1 与 Top-2 分数差；
- $q_{\mathrm{support}}$：Top-5 有效支持度；
- $d_{\mathrm{dispersion}}$：候选 future payload 离散度；
- $a_{\mathrm{direction}}$：候选修正方向一致性；
- $r_{\mathrm{level}}$：节点水平匹配度；
- $d_{\mathrm{memory}}$：memory prediction 与 base prediction 差异；
- $p_h$：归一化 horizon 位置。

旧实现中的 profile_similarity 和 latent_similarity 不在 v2 中出现，因为纯 Latent48 Bank 没有两个独立子空间，这两个量会退化为重复的 shape score。

证据分支为：

$$
h_e
=
\phi_e(e)
\in\mathbb R^{128},
\qquad
\phi_e:9\rightarrow128\rightarrow128.
$$

### 6.4 风险-证据交互门控

基础风险和检索证据不能简单相加。先把 $h_x$ 沿 horizon 轴广播，再与逐 horizon 的 $h_e$ 拼接。只有当“基础模型确实有风险”且“检索证据足够可靠”时，才应该放大修正。

因此构造联合状态：

$$
h_c
=
\operatorname{GELU}
\left(
W_c[h_x;h_e]+b_c
\right)
\in\mathbb R^{256},
$$

$$
g
=
\sigma(W_gh_c+b_g)
\in(0,1)^{256},
\qquad
u=g\odot h_c.
$$

$g$ 表示风险与检索证据的条件一致性。它不是额外的预测分支，而是控制联合信息进入修正权重的门。

### 6.5 可解释融合 logit

为保留单特征归因，9 个证据各自贡献一个标量：

$$
a_i(e_i)=W_{i,2}\operatorname{GELU}(W_{i,1}e_i+b_{i,1})+b_{i,2}.
$$

联合分支提供跨特征交互，最终 logit 为：

$$
\ell_{h,n}
=
b_h
+\sum_{i=1}^{9}a_i(e_{h,n,i})
+W_o\operatorname{GELU}(W_u u+b_u).
$$

由于联合状态已经是逐 horizon 表示，输出头对每个 horizon 只输出一个 logit：

~~~text
256 -> 128 -> 1
~~~

最终修正权重使用 horizon-specific 上限：

$$
w_{h,n}
=
w_{\max,h}\sigma(\ell_{h,n}),
\qquad
0<w_{\max,h}<1.
$$

初始化时令 $w_{\max,h}$ 接近 0.1，使 v2 从保守的 base prediction 开始学习，而不是初始阶段强行使用 memory。

### 6.6 最终预测

$$
\widehat Y^{\mathrm{final}}_{h,n}
=
\widehat Y^{\mathrm{base}}_{h,n}
+m_{h,n}w_{h,n}
\left(
\widehat Y^{\mathrm{mem}}_{h,n}
-\widehat Y^{\mathrm{base}}_{h,n}
\right),
$$

其中 $m_{h,n}\in\{0,1\}$ 表示 Bank 是否对该节点和 horizon 提供完整有效的聚合结果。若 $m_{h,n}=0$，则最终输出严格等于 base prediction。

### 6.7 修正头参数预算

| 部分 | 结构 | 估算参数量 |
|---|---|---:|
| 基础风险分支 | 24 -> 256 -> 128 -> 12 | 约 43k |
| 检索证据分支 | 9 -> 128 -> 128 | 约 18k |
| 联合门控 | 256 -> 256 | 约 66k |
| horizon 输出头 | 256 -> 128 -> 1 | 约 33k |
| 9 个单特征 shape function | 1 -> 32 -> 1 | 约 1k |
| horizon 上限和偏置 | 每 horizon 一个标量 | 小于 1k |
| **Structured Error Corrector** |  | **约 154--164k** |

正式实现必须记录可训练参数量、非冻结总参数量和每个子模块参数量，不能只报告总模型参数。

---

## 7. 损失函数与训练方式

### 7.1 主预测损失

主损失比较最终预测与真实未来，只在有效观测位置计算 MAE。令 $O_{h,n,c}\in\{0,1\}$ 为未来观测掩码：

$$
\mathcal L_{\mathrm{forecast}}
=
\frac{
\sum_{h,n,c}O_{h,n,c}
\left|\widehat Y^{\mathrm{final}}_{h,n,c}-Y_{h,n,c}\right|
}{
\sum_{h,n,c}O_{h,n,c}
}.
$$

物理单位 MAE、RMSE 和 MAPE 只在反归一化后计算，不与代理损失混写。

### 7.2 风险监督

利用训练阶段可见的 base error 构造 Huber 风险标签。对单通道 METR-LA，标签为：

$$
r^\star_{h,n}
=
\operatorname{SmoothL1Element}
\left(
\widehat Y^{\mathrm{base}}_{h,n},Y_{h,n}
\right).
$$

其中逐元素 Huber 函数定义为

$$
\operatorname{SmoothL1Element}(a,b)
=
\begin{cases}
\frac{1}{2}(a-b)^2, & |a-b|<1,\\
|a-b|-\frac{1}{2}, & |a-b|\geq1.
\end{cases}
$$

风险分支训练目标为：

$$
\mathcal L_{\mathrm{risk}}
=
\operatorname{SmoothL1}(\widehat r,r^\star).
$$

推理阶段不计算 $r^\star$，只使用由历史和 base output 预测的 $\widehat r$。

### 7.3 最优凸步长标签

修正方向由 memory 和 base 的差决定：

$$
d_{h,n}
=
\widehat Y^{\mathrm{mem}}_{h,n}
-\widehat Y^{\mathrm{base}}_{h,n}.
$$

目标相对 base 的真实偏移为：

$$
o_{h,n}
=
Y_{h,n}-\widehat Y^{\mathrm{base}}_{h,n}.
$$

训练标签是在修正方向上最接近真实目标的凸组合步长：

$$
b^\star_{h,n}
=
\operatorname{Clip}_{[0,1]}
\left(
\frac{o_{h,n}^{\top}d_{h,n}}
{\|d_{h,n}\|_2^2+\epsilon}
\right).
$$

当 memory 无效或 $\|d_{h,n}\|_2$ 小于最小方向阈值时，该位置不参与 blend loss。融合监督为

$$
\mathcal L_{\mathrm{blend}}
=
\operatorname{SmoothL1}(w,b^\star).
$$

$b^\star$ 只在训练阶段由真实未来构造，不进入修正头输入；上式只对有效 memory、有效真实观测和有效修正方向位置求平均。

### 7.4 总损失

$$
\mathcal L
=
\mathcal L_{\mathrm{forecast}}
+\lambda_r\mathcal L_{\mathrm{risk}}
+\lambda_b\mathcal L_{\mathrm{blend}}.
$$

v2 初始建议沿用当前 risk_weight=0.1、blend_weight=0.1，先只改变结构，不同时修改损失权重。无效 memory 已由 $m_{h,n}$ 强制置零，不再额外堆叠 gate loss。

---

## 8. 训练和实验协议

### 8.1 版本隔离

v2 必须使用独立命名：

~~~text
pretrain checkpoint: metrla_e5_tgge_latent48_v2_seed42
Bank:                metrla_bank_e5_tgge_latent48_v2_seed42
downstream run:      metrla_stgcn_tgge_structured_corrector_v2_seed42
~~~

不能用 v1 encoder 生成的 Bank 配合 v2 encoder，也不能把 v2 key 写入旧 Bank 目录。Bank manifest 必须记录新的 encoder fingerprint、graph fingerprint、scaler 和 key layout。

### 8.2 阶段一：v2 检索编码器预训练

1. 使用与当前 Latent48 相同的数据切分、归一化、静态图和 288 步上下文。
2. 保留 masked reconstruction 与 relation/OffsetDecay retrieval loss。
3. 不增加新的 future target 分支，不把未来语义直接拼进 key。
4. 非局部路由只读取历史 token；二跳/三跳扩散先验由当前静态图预先计算，不参与未来目标构造。
5. 记录训练参数、精确参数量、每轮时间、峰值显存和 checkpoint fingerprint。

### 8.3 阶段二：构建 v2 Bank

1. 使用 v2 encoder 对历史事件编码。
2. 只写入通过因果过滤的历史事件 future payload。
3. 运行 Bank schema、节点数、图指纹、归一化和 encoder fingerprint 校验。
4. 先完成 retrieval-only 诊断，再启动下游适配。

### 8.4 阶段三：冻结 STGCN 的 PostHoc 适配

1. 加载已训练的 STGCN base-only checkpoint。
2. 严格加载并记录 base backbone fingerprint。
3. 冻结 STGCN、v2 encoder 和 Bank。
4. 只训练 Structured Error Corrector。
5. 每个 epoch 验证 base fingerprint 未变化。
6. 保存 best validation checkpoint，并在 test split 上只评估一次。

### 8.5 阶段四：多 seed 与泛化

单 seed 只用于结构筛选，不能作为最终优越性结论。通过单 seed 结构筛选后，使用相同协议补充 seed 2024 和 2025，报告均值、标准差和每个 seed 的结果。

---

## 9. 必须执行的对比与消融

### 9.1 主比较

所有比较使用同一个 STGCN base checkpoint 和同一候选协议：

1. STGCN base-only；
2. v1 Latent48 + 当前 PostHoc-Wide；
3. v2 TGGE Latent48 + Structured Error Corrector。

### 9.2 编码器消融

| 实验 | 时间分支 | 空间分支 | 目的 |
|---|---|---|---|
| Encoder-A | 时间 Attention | 图 Attention | 判断单纯压缩宽度的影响 |
| Encoder-B | 时间门控卷积 | 局部静态图 Attention | v2 局部空间主干 |
| Encoder-C | 时间门控卷积 | 静态图归一化聚合 | 判断空间 Attention 是否必要 |
| Encoder-D | 时间门控卷积 | 局部图 + 随机混合 Top-10（4+6） | 排除扩大感受野本身带来的假增益 |
| Encoder-E | 时间门控卷积 | 局部图 + 历史混合路由 Top-10（4+6） | 判断历史条件路由是否有效 |
| Encoder-F | 时间门控卷积 | 局部图 + 历史混合路由 + 扩散先验 Top-10（4+6） | 混合范围路由主方案 |

Encoder-F 只有在不牺牲 retrieval signal、且路由节点不发生明显 hub collapse 的前提下保留。Encoder-D/E/F 使用同一时间分支、同一训练损失和同一参数预算；只有空间候选机制变化。

### 9.3 修正头消融

| 实验 | 风险分支 | 检索分支 | 联合门控 | 目的 |
|---|---|---|---|---|
| Corrector-BaseCap | 小 MLP | 独立 shape function | 无 | 当前参考 |
| Corrector-Wide | 宽风险 MLP | 独立 shape function | 无 | 测试单纯扩容 |
| Corrector-Structured | 风险分支 | 证据分支 | 有 | v2 主方案 |
| Corrector-NoEvidence | 风险分支 | 删除 | 有 | 验证检索证据是否真正贡献 |

---

## 10. 评价指标与保留判据

### 10.1 检索指标

需要报告：

- 每个 query-node 的 anchor-wise Spearman；
- Kendall tau；
- Recall@1；
- NDCG@5；
- Top-5 有效候选率；
- 与 matched-random Bank 的差值；
- 非局部节点占比与平均静态跳数；
- 路由熵、Top-10 节点选择频率和跨 batch 选择稳定性；
- 非局部门控均值及其与检索相似度、候选一致性的关系。

评价应覆盖完整候选排序，而不是只看一个全局 batch 相关系数。

### 10.2 下游指标

需要报告：

- MAE、RMSE、MAPE；
- 15、30、45、60 分钟 horizon 指标；
- base-only 到最终输出的误差变化；
- memory 有效/无效样本的分组结果；
- 修正权重分位组的 helpful rate；
- 9 个特征的 additive contribution；
- 参数量、每轮训练时间、峰值显存和推理额外开销。

### 10.3 Keep/Remove/Stop 判据

Keep encoder：

1. 完整 retrieval state 约 30 万参数（当前 v2 实测 302,755），并单独报告路由增量 25,827；若后续需要严格低于 30 万，再以隐藏宽度作为单变量压缩消融；
2. anchor-wise Spearman、NDCG@5 相对 v1 不下降超过 2%；
3. 没有未来信息泄漏；
4. 训练和推理成本下降或至少不显著恶化；
5. 混合范围路由相对随机 Top-10（4+6）在至少一个检索排序指标和一个跨域指标上有稳定增益；
6. 非局部选择不能集中到少数固定节点，路由熵和选择频率必须通过诊断阈值。

Keep structured corrector：

1. 修正头在 10--20 万参数内；
2. 相对 v1 PostHoc-Wide，test MAE 有至少约 1% 的稳定改善，或风险排序指标在至少两个 seed 上一致改善；
3. RMSE 不出现超过 1% 的系统性退化；
4. 修正权重随风险、检索相似度和候选一致性呈现可解释变化。

Remove/Stop：

- 若 v2 encoder retrieval 指标下降且下游没有补偿，删除 TGGE 改造，保留 v1 encoder；
- 若 structured corrector 只降低训练损失但 test 不改善，删除联合门控或缩回 Wide 版本；
- 若历史混合路由与随机 Top-10（4+6）无差异，删除混合范围路由，只保留局部静态图；
- 若加入扩散先验后不如无先验版本，保留历史路由但删除扩散偏置；
- 若 v2 不能在多 seed 上稳定优于 v1，不继续增加新的注意力、动态路由或额外损失。

---

## 11. 实现影响与测试要求

### 11.1 预计修改文件

~~~text
stanchor/config.py
stanchor/models/encoder.py
stanchor/data/graph.py
stanchor/models/downstream.py
stanchor/engine/target.py
configs/metrla_e5_tgge_latent48_v2.yaml
configs/metrla_stgcn_tgge_structured_corrector_v2.yaml
~~~

新增 `encoder_variant`、`route_enabled`、`route_dim`、`route_top_k=10`、`route_local_quota=4`、`route_prior_weight`、`route_gate_bias` 和 `corrector_variant` 配置键，禁止通过旧配置静默切换结构。扩散先验的图指纹必须随 checkpoint 和 Bank manifest 保存。

### 11.2 必须通过的工程检查

1. [B,288,N,1] -> [B,N,48] 的 v2 encoder shape contract；
2. 静态图边索引、节点数和图指纹校验；
3. 路由候选排除 self，Top-10 总数以及 4+6 配额索引范围合法；
4. causal retrieval 检查；
5. Bank fingerprint 不匹配时拒绝加载；
6. memory 无效时最终输出等于 base；
7. PostHoc 训练期间 base fingerprint 不变；
8. 所有 loss、gate、路由分数和预测值均为 finite；
9. 各子模块参数量和 trainable/frozen 状态打印到日志；
10. 单 batch forward/backward；
11. 非局部路由小规模 overfit、路由熵和选择频率检查；
12. 小数据过拟合检查，用于确认修正头确实能改变融合权重。

### 11.3 不属于本方案的内容

本方案不包含：

- 无约束的稠密动态图构造或节点 ID 自适应邻接；
- 节点专属参数；
- 把未来真实值写入 Latent48 key；
- profile key 与 latent key 的重复拼接；
- 多源联合预训练；
- 为追求指标而堆叠额外 Attention、Diffusion 或 Prompt 模块。

---

## 12. 最终设计结论

v2 的核心变化可以概括为：

$$
\text{全量时间 Attention}
\rightarrow
\text{多尺度时间门控卷积},
$$

$$
\text{仅一阶静态邻居 Attention}
\rightarrow
\text{局部静态图 + 历史条件混合范围 Top-10 路由（4+6）},
$$

$$
\text{2.8k 参数标量修正头}
\rightarrow
\text{约 160k 参数的结构化风险-证据修正头}.
$$

编码器负责形成紧凑、可迁移的时空检索语义：局部静态图提供物理邻居传播，历史条件非局部路由补充多跳或图外但趋势一致的节点关系；修正头负责判断当前基础模型是否需要被历史事件证据修正。二者职责分离，且每个新增计算都有明确的输入、输出、损失和解释路径。

在 v2 通过 retrieval 指标、下游 test 指标、多 seed 稳定性和成本检查之前，不把它作为最终论文主模型；当前 v1 实验结果仍然保留为可复现实验基线。
