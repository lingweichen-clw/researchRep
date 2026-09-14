# Context-Key Context Router 方案

## 1. 目的与证据边界

当前 `retrieval_aware_mha_router` 作为严格对比 baseline 保留。它已经使用 query 的历史状态、query retrieval key、候选 future residual 和检索元数据，但 candidate token 主要描述 residual，候选 key 只以标量 `shape_score` 进入 summary。Bank 也没有把候选 key 的逐维匹配关系传给 Router。

本方案只验证一个问题：将候选的 latent key compatibility 和 12 步局部 context compatibility 加入 candidate token 后，是否能改善下游校准性能和 hard/mirage pair 上的路由一致性。它不改变候选池、候选排序、Base checkpoint、Bank 内容、训练 split、归一化、seed 或 loss 口径。

## 2. 术语与信息来源

- **query context**：当前样本的下游历史窗口 `x`，形状 `[B,T,N,C]`，METR-LA 中 `T=12,C=1`。只包含 query context_end 及之前的观测。
- **candidate context**：Bank 中候选 event 的历史窗口，按候选 event 的 `context_end` 向前取 `T=12` 步，形状重排为 `[B,N,K,T,C]`。候选必须满足既有的 calendar/causal 规则。
- **query node key**：预训练 RetrievalHead 从当前 retrieval context 得到的归一化 latent key，形状 `[B,N,D_r]`，当前 `D_r=64`。
- **candidate node key**：Bank 保存的历史 event/node key，按 Top-K 节点候选 gather 后形状 `[B,N,K,D_r]`。
- **candidate residual**：候选 future 减去冻结 Base prediction，形状 `[B,H,N,K,C]`。
- **context pair feature**：候选 context 与 query context 在归一化空间的逐步差值及绝对差值，形状 `[B,N,K,2TC]`。它不使用 query true future。

## 3. 新旧模型

### 3.1 Baseline

原始 Router 的 candidate token 为：

\[
C^{base}_{b,n,k}
=
E_{summary}(s_{b,n,k})
+E_{trajectory}([\Delta_{1:H},|\Delta_{1:H}|]).
\]

它使用 9 维 residual/metadata summary（当前 `C=1` 时）和完整 future residual trajectory，输出 256 维 token。

### 3.2 Context-Key Context Router

保留原 residual token，增加两个并行分支：

\[
u^{key}_{b,n,k}
=
[q_{b,n}\odot z_{b,n,k}, |q_{b,n}-z_{b,n,k}|]
\in\mathbb{R}^{2D_r},
\]

\[
u^{ctx}_{b,n,k}
=
[\tilde{x}^{cand}_{b,:,n,k}-\tilde{x}^{query}_{b,:,n},
|\tilde{x}^{cand}_{b,:,n,k}-\tilde{x}^{query}_{b,:,n}|]
\in\mathbb{R}^{2TC}.
\]

其中 `q` 和 `z` 是已归一化 key；`⊙` 是逐维乘积，保留 latent 空间的逐维兼容关系；`\tilde{x}` 是模型归一化空间的 context。两个分支分别经过 `Linear -> GELU -> Linear` 映射到 256 维：

\[
C^{candidate}
=
C^{base}+E_{key}(u^{key})+E_{ctx}(u^{ctx}).
\]

最终 MHA 输入仍为 `[B*N,K+1,256]`，routing softmax、Base fallback 和 horizon-wise residual mixture 均不变。不引入第二个门控，也不增加新的 reliability loss。

两个新增分支的最后一层零初始化，使增强模型初始时与 baseline 的 candidate token 相同；训练后由 forecast-only loss 学习是否利用 context evidence。

## 4. 数据流与接口

1. `TwoStageRetriever.rerank_nodes` 在节点 Top-K 后同时 gather candidate node keys，并写入 `NodeCandidates.node_keys`。
2. 仅当 `calibrator_arch=context_retrieval_aware_mha_router` 时，target engine 调用 `candidate_context_pair_features`，从已有历史序列加载候选 12 步窗口。
3. context feature 随 frozen path cache 保存，避免每个 epoch 重复从 Bank/series 加载。
4. `STAnchorDownstreamModel` 将 candidate context features 传给 Router；原始 baseline 不生成也不消费该字段。
5. 所有候选窗口均通过既有 event `future_end/context_end` 规则确定，query true future 只进入训练 loss 和离线评估。

## 5. 文件与配置

- `stanchor/retrieval/retriever.py`：增加可选 candidate key，并在 Top-K gather。
- `stanchor/retrieval/strategies.py`：增加 context pair feature 提取函数。
- `stanchor/models/retrieval_router.py`：增加 key/context 两个 256 维分支。
- `stanchor/models/downstream.py`：传递 candidate context features。
- `stanchor/engine/target.py`：生成、冻结、合并和传递 context features。
- `stanchor/config.py`：注册 `context_retrieval_aware_mha_router`。
- `configs/formal_context_key_context_router_argcn.yaml`：与 formal baseline 匹配的增强配置。

## 6. 严格对照实验

固定：METR-LA split、train-only scaler、weekday-radius-1-overlap candidate protocol、event_top_r=32、node Top-K=12、同一 Bank、同一冻结 ARGCN Base checkpoint、seed=42、posthoc frozen-base、physical forecast MAE、forecast-only loss。

对照：

1. `formal_base_as_candidate_argcn.yaml`：原始 Router baseline；
2. `formal_context_key_context_router_argcn.yaml`：key + 12-step context 分支；
3. 增强模型关闭新增分支的诊断重放：验证差异来自 context token，而不是缓存或候选池变化。

报告整体 MAE/RMSE/MAPE、15/30/60 分钟 horizon 指标、Base usage、historical mass、routing entropy，以及两类 hard/mirage pair 的 key/context distance、routing weight 和 residual contribution。

## 7. Keep/Remove/Stop

- **Keep**：预测性能不低于 baseline，且 hard/mirage pair 上 context compatibility 与 routing/contribution 的一致性改善，运行成本在可接受范围内。
- **Remove**：性能无改善，或新增分支不改变候选路由，只增加参数和 I/O。
- **Stop**：出现任何未来信息泄漏、Bank/候选协议不匹配、缓存 fingerprint 不一致，或无法完成 baseline 的严格复现。
