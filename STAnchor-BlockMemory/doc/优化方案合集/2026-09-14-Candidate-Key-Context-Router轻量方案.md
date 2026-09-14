# Candidate-Key Context Router 轻量方案

## 1. 目标与证据边界

第一部分的检索模型已经为查询节点和每个候选节点生成 64 维 Key。该 Key 是对 288 步历史 context 的压缩表示，并由 Offset-only future 关系与 context 关系联合约束。下游校准器需要利用候选 context，但没有必要再次读取并缓存每个候选的原始 12 步 context。

本方案只回答一个问题：在保留现有残差 Router、候选 future、检索分数和 Base 预测的前提下，显式加入不同候选的 Key，能否以很低成本改善校准效果。它不改变候选集合、Top-K、payload、Base 模型或预训练模型。

## 2. 输入与数据流

对查询节点 (n) 和候选 (k)，已有输入为：

- 查询 Key：(q_n\in\mathbb{R}^{64})；
- 候选 Key：(k_{n,k}\in\mathbb{R}^{64})；
- Base 预测：\(\hat Y^{base}_n\in\mathbb{R}^{H\times C}\)；
- 候选 future：\(Y^{mem}_{n,k}\in\mathbb{R}^{H\times C}\)；
- 检索分数、level 距离、候选有效性和排序位置。

查询 Key 继续经过原有 `retrieval_encoder`，与历史/Base 状态共同形成逐预测步查询 token。Base 预测继续形成 Base 候选 token。每个候选的 future 残差、统计量和检索标量继续形成原有候选 token。

新增路径只编码候选 Key：

\[
e^{key}_{n,k}
=W_{up}\,\mathrm{GELU}(W_{down}k_{n,k}),
\qquad 64\rightarrow16\rightarrow256.
\]

随后采用残差相加：

\[
\tilde e_{n,k}=e^{old}_{n,k}+e^{key}_{n,k}.
\]

MHA 和逐预测步 Router 仍在查询 token、(K) 个候选 token 和 Base token 之间完成交互，因此查询 context、不同候选的压缩 context、候选 future 和 Base 信息在原有结构内汇合。

## 3. 可回退设计

新增架构名为 `candidate_key_context_mha_router`；保留版继续使用 `retrieval_aware_mha_router`，其代码路径和输入协议不变。

`W_up` 的权重和偏置初始化为零，因此新版在训练开始时满足：

\[
e^{key}_{n,k}=0,
\qquad
\tilde e_{n,k}=e^{old}_{n,k}.
\]

也就是说，新版初始行为与保留版逐张量一致；如果短程实验没有收益，只需把 `calibrator_arch` 改回 `retrieval_aware_mha_router`，无需重建检索模型、Bank 或 Base checkpoint。

## 4. 成本

新增参数量为：

\[
(64\times16+16)+(16\times256+256)=5,392.
\]

相对保留版约 87.97 万个可训练参数，增幅约 0.61%。候选 Key 已经包含在 `NodeCandidates.node_keys` 中，因此不新增 Bank 字段，也不新增原始 context 缓存。被移除的高成本方案需要缓存 `[B,N,K,24]` context pair 特征，本方案不再生成或传递该张量。

## 5. 10 轮短程验证与决策

短程验证固定以下条件：最终 Joint-context Offset-only 检索 checkpoint、严格匹配的 epoch-9 Bank、同一个冻结 Graph WaveNet Base、`weekday_radius1_overlap` 候选协议、`learned_key` 排序和 `offset_only` payload。唯一变化是校准器增加候选 Key 分支。

主要比较验证集最优 physical MAE，同时记录每轮耗时、峰值显存和训练参数量。决策规则为：若 10 轮内能够接近或优于保留版 2.789273，且成本恢复到保留版附近，则保留该分支并进入完整训练；若仍明显落后或训练不稳定，则回退到 `retrieval_aware_mha_router`，停止继续增加 context 模块。
