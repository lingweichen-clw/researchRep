# Relation-only Pretraining and Vectorized Route Design

## Goal

验证 Future-Relation 是否可以独立训练检索用 Latent48 key，并在不改变路由候选语义的前提下减少预训练计算。

## Scientific Question

当前模型的 clean view 产生 Latent48 key，masked view 只经过共享 encoder 后由 reconstruction head 重建历史值。掩码损失因此只通过共享 encoder 间接影响 clean key，并不直接评价检索排序。Relation-only 将这个间接正则从主实验中移除，直接测量 future-relation teacher 对检索 key 的训练能力。

## Objective and Information Boundary

对 source-train batch，clean history 为

$$X^{ret}\in\mathbb R^{B\times288\times N\times1}.$$

编码器只接收当前历史，输出 node key

$$K=\operatorname{L2Norm}(g_\theta(\operatorname{Encoder}(X^{ret})))\in\mathbb R^{B\times N\times48}.$$

future relation teacher 使用训练样本真实 future 构造 OffsetDecay/SymNorm 距离，并在 `torch.no_grad()` 中形成目标分布。Relation-only 损失为

$$\mathcal L_{rel}=\operatorname{CE}(P_{teacher}(Y),P_{student}(K)).$$

真实 future 不进入 encoder、query key、Bank key 或部署候选排序。

Relation-only 配置显式设置 `objective=relation_only`、`reconstruction_weight=0`、`retrieval_weight=1`。现有 `joint` 配置继续保留，作为 `Joint-current` 或后续 `Joint-weak` 对照，不覆盖旧 checkpoint。

## Data Flow

```text
relation_only:
Xret -> patch embedding -> TGGE -> Latent48 -> future-relation loss

joint:
Xret -> clean TGGE -> Latent48
Xret(masked) -> masked TGGE -> reconstruction head
```

Relation-only 不创建 `MaskBatch`、不构造 masked token、不执行 reconstruction head，因此只保留一次 encoder forward。模型接口仍保留原有 `forward_pretrain`，新增 clean-only 路径供训练循环调用。

## Route Vectorization

路由分支的数学定义不变：每个 target node 排除 self，按历史条件分数和固定二/三跳先验选择 4 个一阶节点与 6 个远端节点。工程实现把静态图的一阶/远端候选索引在一次 encoder forward 前预计算，并在 `[B,N,N]` score tensor 上批量执行分区 `topk` 和配额递补。

向量化只改变索引组织和 kernel 调度，不改变 score、候选分区、配额、softmax 或 value aggregation。候选槽位不足时仍按原规则从另一分区递补；self 节点仍不可选。

## Validation Frequency

新增 `validation_interval`，默认 `1` 保持既有行为。Relation-only 实验配置使用 `2`，每两轮执行一次完整 validation，并在最后一轮强制执行。训练 batch 的顺序、loss、optimizer step 和参数更新不变；变化只在验证调用次数和 early-stopping 观测粒度，因此日志必须记录 `val_evaluated` 和实际验证 epoch。

## Experiments and Decisions

1. `Relation-only`: `L=L_rel`，判断 future relation 是否足够。
2. `Joint-current`: 保留当前 `L=L_rec+0.1L_rel`，作为历史基线。
3. `Joint-weak`: `L=0.1L_rec+L_rel`，判断掩码是否只需要作为鲁棒性正则。

主指标为 anchor-wise Spearman/Kendall、Recall@1、NDCG@5、Bank memory MAE 和跨域迁移结果；同时记录每轮时间、峰值显存和路由选择诊断。

Keep/Remove/Stop：

- Relation-only 检索与下游不下降且成本明显降低：保留为主线；
- Joint-weak 在缺失/跨域设置稳定更好：保留为鲁棒性消融；
- 掩码没有额外收益：删除 masked reconstruction 主线；
- 向量化候选集合与旧实现不等价：停止该优化，保留原实现。
