# E5-TGGE Single-View Masked Relation v3

## 1. 目标与当前证据

本版本针对两个已观察到的问题：

1. Relation-only 只优化 future-relation，METR-LA 验证 `val_retrieval` 在约第 20--24 轮稳定在 `2.195--2.206`；原 Joint v2 在第 26 轮为 `2.165991`。两者不能直接比较 `val_total`，因为 Relation-only 的 `retrieval_weight=1`，而 Joint v2 的 `retrieval_weight=0.1` 且还包含重建损失，但 relation 分量显示掩码辅助确实可能提供正则收益。
2. Joint v2 为 clean view 和 masked view 构造一个拼接 batch，再通过 Encoder。它虽然只有一次 Python Encoder 调用，但实际计算量和显存仍接近两份样本。

v3 的目标是保留掩码重建的鲁棒性正则，同时让每个 batch 只编码一次。

## 2. 单前向预训练策略

### 2.1 名称

`masked_relation_single_view` 表示 **单视图掩码关系联合预训练**。它使用一个 masked history view，同时计算：

- future-relation loss：约束 48 维 key 的候选排序接近 OffsetDecay teacher；
- masked reconstruction loss：只在被人工遮挡且原始观测有效的位置恢复输入。

### 2.2 数据流与张量形状

检索历史输入为 `x: [B, 288, N, 1]`，其中 `B` 是 batch，`N=207` 是节点数。采样器生成：

- `patch_mask: [B, 24, N]`；
- `value_mask: [B, 288, N, 1]`。

只使用可见值计算 masked normalization，然后经过：

```text
visible history -> patch embedding -> FactorizedSTEncoder -> hidden [B,24,N,80]
                                            |-> RetrievalHead -> key [B,N,48]
                                            |-> ReconstructionHead -> [B,288,N,1]
```

Encoder 只接收 masked tokens 一次。为了复用现有 loss 接口，该 masked encoding 暂存在 `PretrainForwardOutput.clean` 字段中；它并不意味着这个分支构造了 clean view。

### 2.3 目标函数

```math
\mathcal L_{v3}
= 1.0\,\mathcal L_{\mathrm{relation}}
+ 2.0\,\mathcal L_{\mathrm{reconstruction}}.
```

本次正式重跑采用 `reconstruction_weight=2.0`。在前一轮日志中，未加权
`val_mask` 约为 `0.35`、`val_retrieval` 约为 `2.25`；若仍使用 `0.1`，
重建项对总损失的数值贡献只有约 `0.035`，不足以发挥辅助正则作用。设置为
`2.0` 后其数值贡献约为 `0.70`，关系项仍保持主导，同时让掩码重建真正参与表示学习。

其中 relation teacher 只使用训练样本的真实 future 构造监督距离；真实 future 不进入 query key、候选构造或推理过程。重建项只在 `value_mask & observed` 上计算。

该设计借鉴 MAE/SimMIM 一类“masked input + 轻量重建头”的单编码器训练范式，但本项目的监督信号仍是 future relation，而不是把重建误认为最终任务。

### 2.4 为什么不会造成目标泄漏

- 训练时 future 仅作为 relation teacher 的标签，和现有 relation-only/J oint 协议相同；
- masked encoder 只能看检索历史的可见部分；
- 推理和建 Bank 仍调用 `encode_clean`，不带人工 mask；
- 下游 Bank schema 和 48 维 key 不变。

## 3. 图模块最终主线

### 3.1 一阶静态图注意力

`SparseGraphAttention` 在固定 `edge_index` 的全部一阶邻居上计算多头注意力。它保留物理拓扑传播，不做节点 ID embedding，也不依赖 query future。

### 3.2 高阶历史条件路由

`MixedRangeRouteAttention` 从非一阶候选中按历史 summary 和二/三阶 random-walk prior 选择最多 6 个节点：

```yaml
route_top_k: 6
route_local_quota: 0
```

因此最终配置的含义是：一阶信息由完整静态图分支负责；路由分支只补充高阶/远端关系。路由权重仍由历史 query-key 分数 softmax 得到，并经过低秩 value 投影和门控残差融合。

旧 `metrla_e5_tgge_latent48_v2.yaml` 的 4-direct + 6-remote 结果保留作为历史 case study，不被覆盖。

## 4. 预期成本

Relation-only 当前约 5 分钟/轮；Joint v2 约 18--19 分钟/轮。v3 只有一份 masked Encoder 激活，预计训练约 5--7 分钟/轮，验证约 0.3--0.5 分钟/轮，最终以完整 1418 train batches 的日志为准。

## 5. 代码与配置

- 单前向入口：`stanchor/models/pretraining.py::forward_pretrain_single_view`
- 训练分支：`stanchor/engine/pretrainer.py`
- 目标校验：`stanchor/config.py`
- 配置：`configs/metrla_e5_tgge_single_view_masked_relation_v3.yaml`
- 启动队列：`scripts/run_tgge_single_view_queue.ps1`

正式命令（前台查看日志）：

```powershell
python scripts/pretrain.py --config configs/metrla_e5_tgge_single_view_masked_relation_v3.yaml
```

后台队列会使用同一配置和独立输出目录 `artifacts/convergence/tgge_single_view_v3_higher_order/`，不覆盖已有 v2 artifacts。

## 6. 可视化协议

训练曲线对每个子图根据有限数据范围自动设置纵轴，跳过验证的 epoch 使用断点而不是错误连线。

Top-5 图新增 `top5_error_profiles.png`：它按候选 rank 绘制候选 future 相对 query future 的 MAE，并用虚线标出最终 memory MAE。这样即使候选轨迹在物理单位图中重叠，排序差异仍然可见。

候选池扩展只用于 broad causal 诊断，不改变 exact calendar 部署协议。例如：

```powershell
python scripts/visualize_retrieval.py `
  --version tgge_joint `
  --config configs/metrla_e5_tgge_single_view_masked_relation_v3.yaml `
  --checkpoint artifacts/metrla_e5_tgge_single_view_masked_relation_v3_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_tgge_single_view_masked_relation_v3_seed42 `
  --random-checkpoint <random-checkpoint> `
  --random-bank <random-bank> `
  --candidate-protocol pretrain_broad_causal `
  --event-top-r 64 `
  --output-dir artifacts/convergence/visualization/tgge_single_view_v3/broad_causal
```

`exact_calendar` 仍按真实日历候选计算，不能人为塞入无关候选来制造差异。

## 7. 保留/停止判据

- 保留：v3 的 `val_retrieval` 至少不劣于 Relation-only，并且单轮训练时间接近 Relation-only；
- 停止：若 v3 的 retrieval 和下游泛化均不优于 Relation-only，则不再增加新的预训练辅助头，直接进入跨数据集泛化；
- 图模块不再进行无决策后果的多组 quota 消融，后续只报告固定主线和已有 v2 历史对照。

## 8. 相关顶会工作

本策略借鉴 masked input 与轻量重建头的单编码器范式，相关原始工作可引用：

- MAE: He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022, https://arxiv.org/abs/2111.06377
- SimMIM: Xie et al., *SimMIM: A Simple Framework for Masked Image Modeling*, CVPR 2022, https://arxiv.org/abs/2111.09886

本项目的差异在于：future-relation teacher 仍是交通检索的主要监督，重建项仅作为 masked-view 鲁棒性正则；上述工作不直接提供本项目的 future relation 目标或 Bank 检索协议。
