# Retrieval-Aware MHA Residual Router 校准器方案

## 1. 目标与边界

本方案将下游校准器重构为 **Retrieval-Aware MHA Residual Router**。
它的职责是在冻结的 backbone 基础预测与检索得到的历史候选之间进行逐 horizon 的残差选择；检索器仍负责 288 步历史语义编码，校准器只处理下游 12 步 context、12 步 Base 预测和候选 future。

本轮不改变 Bank 格式、`weekday_radius1_overlap` 候选协议、Top-K 检索、`frozen_path_cache` 的 sample-id 语义，也不启用 event key、quality-teacher loss、独立 Alpha/Beta 或事件图传播。旧架构名称继续可加载，便于历史实验复现。

## 2. 术语

- **Base prediction**：冻结 backbone 对当前 12 步输入产生的预测，形状为 `[B,H,N,C]`。
- **Residual**：候选 future 相对 Base 的修正量，`delta = candidate_future - base[...,None,:]`，形状为 `[B,H,N,K,C]`。
- **Base token**：第 `K+1` 个零残差候选；它始终有效，选择它意味着不施加历史修正。
- **Routing weights**：统一 softmax 在 K 个历史候选和 Base token 上产生的概率，形状 `[B,N,H,K+1]`。
- **Retrieval node key**：检索器输出的每节点语义 key，形状 `[B,N,64]`。它只作为校准器的 query 条件输入，并在缓存中 detached 保存。
- **MHA**：标准四头 Multi-Head Attention。`attn_output` 必须进入最终 routing head，确保 Q/K/V/O 均有预测梯度。

## 3. 数据流与形状

1. Query 分支将 `[B,T,N,C]` context 与 `[B,H,N,C]` Base 展平为每节点状态，得到 `[B,N,S]`。`S=calibrator_state_dim`。同时将 detached node key `[B,N,64]` 投影到隐藏空间。
2. Candidate 分支只使用统计特征和完整 residual 轨迹摘要，不构造 hidden 级 `[B,N,H,K,D]`。候选 token 为 `[B,N,K,D]`。
3. Base token 为 `[B,N,1,D]`，与候选 token 拼接为 `[B,N,K+1,D]`。
4. 对每个节点展开 MHA：
   - query `[B*N,H,D]`；
   - key/value `[B*N,K+1,D]`；
   - attention weights `[B,N,heads,H,K+1]`。
5. MHA 输出与 query 残差相加、LayerNorm 后进入共享 routing head，产生 `[B,N,H,K+1]` logits，并只做一次 masked softmax。

## 4. 数学定义

历史候选残差为：
[
Delta_{b,h,n,k}=Y^{cand}_{b,h,n,k}-Y^{base}_{b,h,n},\quad k=1,\ldots,K.
]
Base token 的残差固定为：
[
Delta_{b,h,n,K+1}=0.
]
MHA 产生候选条件 query：
[
U=operatorname{LN}(Q+operatorname{MHA}(Q,C,C)).
]
路由 logits 由共享的 query/candidate 交互头计算：
[
ell_{b,n,h,k}=f_{route}(U_{b,n,h},C_{b,n,k}),
]
并在 K+1 维上对无效历史候选做 mask：
[
pi_{b,n,h,:}=operatorname{softmax}(ell_{b,n,h,:}).
]
最终预测为 residual mixture：
[
Y^{final}_{b,h,n}=Y^{base}_{b,h,n}+sum_{k=1}^{K}pi_{b,n,h,k}Delta_{b,h,n,k}.
]
当历史候选全部无效时，只有 Base token 有效，故 `pi_base=1` 且 `Y_final=Y_base`。

## 5. Future-information 边界

部署阶段 query、Base、node key、候选相似度和候选 future 可用；真实目标 future 不可用。真实 future 仅用于训练 forecast loss、离线指标和可选 teacher 诊断。本轮 `candidate_quality_weight=0`，因此路由器不读取 target，也不通过缓存保存 target-derived 内容。

## 6. 参数与资源约束

默认 `D=256`、4 头 MHA、routing hidden 128。MHA 的 Q/K/V/O 通过 `attn_output` 参与预测，避免 value projection 死分支。候选 hidden 始终为 `[B,N,K,D]`；不会 materialize `[B,N,H,K+1,D]`。缓存继续驻留 CPU，命中后只将当前 batch 张量搬到 GPU。

实际参数量、forward/backward 时间和 CUDA peak memory 必须由资源脚本测量，不以理论估算代替。

## 7. 接口兼容

`STAnchorDownstreamModel.forward` 增加可选 `retrieval_node_keys=None`。旧路由器忽略该参数；新路由器在缺失时使用零 key 投影，保证旧调用可运行，但正式检索实验应传入缓存中的 detached key。`FrozenPathEntry` 新增可选 `retrieval_node_keys`，split/merge 时保持 CPU 存储和 detached 属性。

新路由器输出仍保持 `(final, historical_mass, contributions, learned_memory)` 四元组；诊断字段包括 `last_routing_weights`、`last_mha_attention`、`last_base_usage`、`last_routing_entropy`。

## 8. 验证与决策

Smoke 必须验证：输出形状、K+1 权重归一化、无候选精确 Base fallback、MHA in/out projection 非零有限梯度、缓存 key split/merge 和 finite 检查。资源测试报告参数量、forward/backward 时间和 CUDA peak；不把单 batch 结果宣称为完整 epoch。

正式三轮验证固定数据、Bank、预训练 checkpoint、Base checkpoint、seed、候选协议和 batch，仅比较路由器行为与验证 MAE。若出现 NaN、fallback 错误、缓存重复编码或显存超出实验机预算，停止正式训练并回退旧架构；若 MAE 持平但路由可解释、梯度完整且成本可接受，可保留新架构。

## 9. 正式下游优化策略

本节只规定 Router 的下游训练超参数，不改变检索编码器预训练、Bank、候选协议、Base checkpoint、模型结构、损失函数或缓存机制。四个 backbone 的校准器采用完全相同的优化设置，保证跨下游比较时学习率与调度器严格匹配。

完整 AGCRN 实验使用旧优化设置 `1e-3 + StepLR(step_size=10, gamma=0.95)` 训练了 50 轮。其最佳验证 MAE 为 `2.852085`（第 29 轮），相对匹配的 Base-only 最佳 MAE `3.008657` 改善 `0.156572`，相对改善约 `5.204%`；15、30 和 60 分钟 MAE 分别由 `2.634069/3.029713/3.566936` 降至 `2.564193/2.882085/3.285953`。该结果支持冻结检索器和 Router 架构，不再增加或删除模型组件。

该曲线没有呈现严重的经典过拟合：第 29 轮时训练 MAE 为 `2.879283`，高于验证 MAE `2.852085`，没有不断扩大的训练优于验证的泛化间隙。但旧学习率在第 50 轮仍有 `8.15e-4`，第 29 轮以后训练 forecast loss 仅继续下降 `0.000088`，验证 MAE 却回升 `0.018075` 至 `2.870160`。因此主要问题是平台期内更新步长过大造成的验证波动，而不是模型容量必须继续调整。正式配置统一采用：

```yaml
target:
  batch_size: 32
  epochs: 50
  optimizer_name: adam
  learning_rate: 0.0005
  weight_decay: 0.0001
  scheduler_name: step_lr
  scheduler_step_size: 10
  scheduler_gamma: 0.5
  mha_dropout: 0.05
  patience: 20
  early_stopping_enabled: false
```

根据当前训练器中“每轮验证完成后再执行 `scheduler.step()`”的顺序，第 1--10、11--20、21--30、31--40 和 41--50 轮使用的学习率依次为 `5e-4`、`2.5e-4`、`1.25e-4`、`6.25e-5` 和 `3.125e-5`。该调整只影响 Router 可训练参数；冻结的 backbone、检索编码器及 CPU frozen-path cache 均不参与梯度更新，因此不会改变候选集合和 future-information 边界。

四个正式实验必须共同固定：METR-LA 划分与归一化、HN-OffsetDecay v2 checkpoint、同一 Bank、`weekday_radius1_overlap`、node-level Top-12、`level_weight=0`、`candidate_quality_weight=0`、batch size 32、seed 42 和完整 50 个 epoch。正式冲刺阶段关闭 early stopping，因为已完成曲线在第 13 轮到第 29 轮之间存在 16 轮的有效改进间隔，而当前每轮约一分钟，跑满 50 轮的成本可接受。训练始终保存验证 MAE 最低的 checkpoint，正式测试只能加载该 best checkpoint，不能使用最后一轮权重。

保持 `weight_decay=1e-4` 和 `mha_dropout=0.05` 不变。当前没有梯度爆炸、非有限损失或明显的容量型过拟合证据，因此不增加 weight decay、不提高 dropout、不加入新的正则损失，也不修改 Router、检索器、Bank 或缓存机制。若出现非有限损失、Base fallback 失效或缓存错误，应停止任务并按工程故障处理，而不能通过继续调参掩盖。

将 `879,693` 个参数除以 `22,681` 个时间窗得到 `38.79`，不能直接作为强正则依据。每个窗口对最多 `H*N*C=12*207*1=2,484` 个预测位置提供共享参数监督，约对应 5,634 万个带时空相关性的标量目标；这些目标并非相互独立，但足以说明“一个窗口等于一个监督样本”的参数/样本比不适用于当前节点共享、horizon 共享的 Router。

当前实现使用 `torch.optim.Adam`，而不是默认解耦权重衰减的 AdamW。`weight_decay` 通过 Adam 的耦合 L2 方式进入更新，且当前参数组没有排除 LayerNorm、bias、horizon embedding 或 Base bias；它也不会作为显式正则项加入日志中的 `train_total`。因此不能根据假设参数均值计算 `weight_decay * ||theta||^2`，再要求该值达到 forecast loss 的固定百分比。直接采用 `0.005--0.01` 会把当前衰减提高 50--100 倍，缺乏曲线证据且存在欠拟合风险。

本地公开实现也不支持必须使用强衰减的结论：STAEformer 在 METR-LA 上使用 `weight_decay=3e-4`、dropout `0.1`；Graph WaveNet 使用 `weight_decay=1e-4`、dropout `0.3`；AGCRN 使用 `weight_decay=0`；ICML 2025 RAFT 实现使用 Adam 且未设置 weight decay，默认 dropout 为 `0.1`。这些 dropout 作用于各自完整网络的不同位置，不能直接等价为本 Router 仅作用于 MHA attention weights 的 `mha_dropout`。本轮也不采用 label smoothing（分类正则，不适用于 MAE 回归）、无梯度爆炸证据的 gradient clipping、尚未实现且会引入新变量的 cosine/warmup，或会将每轮更新数从 709 降至约 355 的 batch size 64。

## 10. 架构冻结决策

自本版本起，HN-OffsetDecay v2 检索器与 Retrieval-Aware MHA Residual Router 作为最终机制版本冻结。后续允许修改的范围仅包括学习率、调度器、早停、随机种子和跨数据集所需的数据配置；不再改变 token 定义、MHA、residual mixture、候选协议语义、Bank 数据结构或 frozen-path cache。任何架构级变更必须作为独立后续工作提出，不能混入当前四下游和跨数据集验证。
