# 方案 B：双模块 Candidate Set Attention + Horizon Mixer 诊断校准优化方案

## 1. 目标与证据

本方案针对当前下游校准器在 MAE 约 2.84–2.86 附近的平台期。既有对照显示：

| 对照 | 最佳验证 MAE |
|---|---:|
| Forecast + Risk + Blend | 2.851375 |
| Forecast + Risk | 2.845433 |
| Forecast only | 2.845435 |
| Scalar gate | 2.851374 |
| Vector residual | 2.858691 |
| Residual additive | 2.836799 |
| Residual additive + forecast-only，5 epoch | 2.830616 |

因此删除 blend loss；risk loss 不再反向传播，只保留风险诊断统计。主优化目标改为最终预测的 masked MAE。

## 2. 设计原则

将原来的三个概念部分精简为两个：

1. **Base Reliability Encoder**：从历史 context 和冻结 base forecast 编码 base 的状态，不再单独回归 risk scalar。
2. **Candidate-Aware Memory Corrector**：合并候选聚合与 memory 校正，直接在候选维度上比较和加权，不先把候选 future 压成单一 memory。

不使用大 Transformer。候选数 K≤5、horizon H=12，采用轻量候选集合注意力和深度可分离 horizon mixer，控制单轮时间和参数量。

## 3. 输入输出与符号

冻结 base 输出：

$$Y^{base}\in\mathbb{R}^{B\times H\times N\times C}.$$

候选 future：

$$Y^{cand}\in\mathbb{R}^{B\times H\times N\times K\times C}.$$

候选残差：

$$\Delta^{cand}_{q,h,n,k,c}=Y^{cand}_{q,h,n,k,c}-Y^{base}_{q,h,n,c}.$$

候选 token 输入包含：候选残差、绝对残差、已有 key similarity、level distance、有效 mask、horizon position。它不使用 query 真实 future。

## 4. 模块 A：Base Reliability Encoder

输入历史 context 和 base prediction，先按节点对历史做标准化，再拼接 base forecast：

$$S^{base}=E_b([Normalize(X^{ctx}),Y^{base}])
\in\mathbb{R}^{B\times N\times D_b}.$$

该状态表示节点当前是否处于突变、过平滑或外推不可靠状态。它不再通过独立 risk regression loss 训练，而是通过最终 forecast loss 学习。需要保留可解释性时，离线计算 predicted risk 与真实 base error 的 Spearman，但不把该指标当成最终预测收益。

## 5. 模块 B：Candidate-Aware Memory Corrector

### 5.1 候选 token

对每个 horizon 和候选构造：

$$e_{q,h,n,k}=E_c([\Delta^{cand},|\Delta^{cand}|,s^{key},-d^{level},m,p_h]).$$

### 5.2 候选集合注意力

使用 base state 作为 query，候选 token 作为 key/value，得到候选权重：

$$a_{q,h,n,k}=Score(S^{base}_{q,n},e_{q,h,n,k}),\qquad
\pi_{q,h,n,:}=Softmax_k(a).$$

mask 无效候选；当候选无效时输出零 residual。由于 softmax 在候选维度进行，候选排列不会改变结果。

### 5.3 Residual 聚合

$$R^{memory}_{q,h,n,c}=\sum_k\pi_{q,h,n,k}\Delta^{cand}_{q,h,n,k,c}.$$

同时计算候选残差方差作为诊断量，不把方差直接作为监督目标。

### 5.4 Horizon Mixer

把每个节点的候选聚合状态沿 horizon 送入轻量 depthwise temporal convolution：

```text
Linear projection → depthwise Conv1d(kernel=3) → GELU → pointwise Linear → gated residual
```

它显式建模相邻 horizon 的连续修正关系，参数和计算量远小于完整 Transformer。

### 5.5 受约束输出

最终输出：

$$Y^{final}=Y^{base}+\alpha\odot R^{memory}+\beta.$$

其中 `alpha=sigmoid(head)`，控制 memory 使用强度；`beta=0.25\,\sigma_\Delta\tanh(head)`，只用于补偿小幅聚合偏差，不能绕过 memory 独立预测。无效候选位置强制 `alpha=0, beta=0`。

## 6. 损失函数

正式训练只使用：

$$\mathcal{L}_{forecast}=MaskedMAE(Y^{final},Y).$$

删除 `blend loss` 和 `risk loss` 的反向传播。训练/验证日志仍记录：MAE、RMSE、15/30/45/60 分钟 MAE、helpful rate、risk Spearman、attention entropy、memory residual norm、additive residual norm。

## 7. 参数与时间预算

目标参数量：

| 部分 | 参数预算 |
|---|---:|
| Base Reliability Encoder | 60k–80k |
| Candidate token + Set Attention | 55k–75k |
| Horizon Mixer + heads | 65k–80k |
| 总计 | 190k–235k |

要求总参数不超过 250k，尽量接近原校正器规模；单 epoch 不超过 5 分钟。实现中 K≤5、H=12，不增加检索次数，不解冻 base/backbone。

## 8. 工程接口

优先在 `stanchor/models/downstream.py` 增加新校正器类，并通过 `target.correction_variant` 或新的 downstream mode 选择；旧 `StructuredErrorCorrector` 保留用于历史 checkpoint 复现，不直接覆盖。`compute_downstream_loss` 增加 `loss_variant="forecast_only"` 的明确路径。

## 9. 验证矩阵与判据

固定 Graph WaveNet、STGCN、seed=42、冻结同一个 base、同一个新 HN-OffsetDecay Bank、同一 candidate protocol、batch size 和 optimizer。先做单 batch 反向验收，再做每个下游 3–5 epoch 小实验。

保留条件：平均 MAE 比当前 residual-additive forecast-only 基线至少下降 0.01，且单 epoch≤5分钟；若 MAE 持平但 helpful rate 和 attention entropy 显著改善，作为诊断增强保留；若无改善，删除 Horizon Mixer，回退到两模块纯 residual-additive；若仍无改善，停止继续增加结构复杂度。

## 10. 未来信息边界

训练时真实 future 只用于最终 forecast loss 和离线诊断 target；候选排序、候选 attention、Base Reliability Encoder 输入和部署前向均不读取 query 真实 future。固定 base 参数 `requires_grad=False`，但不对 memory 或校正器输入做无必要 detach，以保证最终 forecast 梯度能够训练新模块。

## 11. 实施顺序

1. 写入新模块和配置开关；
2. 增加 forecast-only loss；
3. 单 batch shape、finite、反向和冻结 base 检查；
4. 参数量和单步时间检查；
5. 运行 Graph WaveNet/STGCN 3–5 epoch；
6. 清理 smoke 产物；
7. 根据判据决定保留、简化或回退。