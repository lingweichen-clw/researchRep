# Base-as-Candidate 统一残差混合校准器完整技术方案

**文档版本**: v2.0  
**日期**: 2026-08-27  
**作者**: 时空数据挖掘研究组  

---

## 1. 方案目标与动机

### 1.1 当前问题

现有校准器架构采用两阶段决策：

$$\text{候选池} \xrightarrow{\text{Attention}} \text{Memory} \xrightarrow{\alpha\text{-gate}} \text{最终预测}$$

存在以下问题：

1. **功能重叠**：Attention 学习"哪些候选有用"，$\alpha$ 又学习"Memory 整体可信度"，两者语义重叠
2. **梯度瓶颈**：Attention 的梯度需要经过 $\alpha$ 才能传递到最终 loss，若 $\alpha$ 学会压小 Memory，则 Attention 梯度被削弱
3. **论文难讲**：审稿人会质疑"既然有 Attention 了，为什么还需要 $\alpha$？"
4. **逻辑不清**：Base 预测作为"默认选项"的地位不明确

### 1.2 解决方案

**将 Base 预测显式建模为第 $K+1$ 个候选**，在统一的 attention 框架下进行选择：

$$\text{候选池} = \{\text{候选}_1, \ldots, \text{候选}_K, \text{Base}\}$$

$$\hat{Y} = Y^{\text{base}} + \sum_{i=1}^{K+1} \pi_i R_i$$

其中 Base 的残差 $R_{K+1} = 0$，因此当 $\pi_{K+1}=1$ 时，输出严格等于 Base。

### 1.3 预期优势

1. **逻辑统一**：单一决策点，无功能重叠
2. **梯度直接**：Attention 直接接收 forecast loss 梯度
3. **论文清晰**：标准的 retrieval-augmented 范式
4. **自然回退**：无有效候选时自动使用 Base

---

## 2. 符号定义

| 符号 | 含义 | 维度 |
|------|------|------|
| $B$ | Batch size | - |
| $H$ | Horizon（预测步长） | 12 |
| $N$ | 节点数 | 207 (METR-LA) |
| $K$ | Top-K 历史候选数 | 5 |
| $C$ | 特征通道数 | 1 |
| $T$ | 历史时间步 | 12 |
| $X^{\text{hist}}$ | 历史输入序列 | $[B, T, N, C]$ |
| $Y^{\text{base}}$ | Base 预测（冻结） | $[B, H, N, C]$ |
| $Y^{\text{cand}}$ | 候选 future | $[B, H, N, K, C]$ |
| $R$ | 残差向量 | $[B, H, N, K, C]$ |
| $\pi$ | Attention 权重 | $[B, H, N, K+1]$ |
| $D$ | Token 隐藏维度 | 256 |
| $D_s$ | State 编码维度 | 256 |

---

## 3. 核心架构设计

### 3.1 统一候选表示

#### 历史候选残差（$k=1,\ldots,K$）

$$R_{q,h,n,k} = \widetilde{Y}^{\text{cand}}_{q,h,n,k} - Y^{\text{base}}_{q,h,n}$$

其中：
- $\widetilde{Y}^{\text{cand}}$：经过 OffsetDecay 校正的候选 future
- $R_{q,h,n,k} \in \mathbb{R}^C$：第 $k$ 个候选相对 Base 的残差

#### Base 候选残差（$k=K+1$）

$$R_{q,h,n,K+1} = \mathbf{0} \in \mathbb{R}^C$$

Base 作为零残差候选，当获得全部权重时，输出严格等于 Base。

### 3.2 Token 编码

#### 历史候选 Token（$k=1,\ldots,K$）

输入特征向量：

$$\mathbf{f}^{\text{cand}}_{q,h,n,k} = [\Delta, |\Delta|, s^{\text{key}}, -d^{\text{level}}, p_h]$$

其中：
- $\Delta = R_{q,h,n,k}$：候选残差 $[C]$
- $|\Delta|$：残差绝对值 $[C]$
- $s^{\text{key}}$：Key 相似度 (标量)
- $d^{\text{level}}$：Level 距离 (标量)
- $p_h = h / (H-1)$：Horizon 位置 (标量)

总维度：$2C + 3 = 5$ (METR-LA, $C=1$)

编码为统一 Token：

$$\mathbf{t}^{\text{cand}}_{q,h,n,k} = \text{CandidateEncoder}(\mathbf{f}^{\text{cand}}_{q,h,n,k}) + \mathbf{e}^{\text{type}}_{\text{cand}}$$

其中：
- $\text{CandidateEncoder}$：2层 MLP，输出维度 $D=256$
- $\mathbf{e}^{\text{type}}_{\text{cand}} \in \mathbb{R}^D$：可学习的候选类型 embedding

#### Base Token（$k=K+1$）

输入特征向量：

$$\mathbf{f}^{\text{base}}_{q,h,n} = [\mathbf{0}_{2C}, r^{\text{base}}_{q,h,n}, \sigma^{\text{ctx}}_{q,n}, p_h]$$

其中：
- $\mathbf{0}_{2C}$：零残差占位（与历史候选对齐）
- $r^{\text{base}}_{q,h,n}$：Base 风险评估 (标量)
- $\sigma^{\text{ctx}}_{q,n}$：Context 波动性 (标量)
- $p_h$：Horizon 位置 (标量)

编码为统一 Token：

$$\mathbf{t}^{\text{base}}_{q,h,n} = \text{CandidateEncoder}(\mathbf{f}^{\text{base}}_{q,h,n}) + \mathbf{e}^{\text{type}}_{\text{base}}$$

其中 $\mathbf{e}^{\text{type}}_{\text{base}} \in \mathbb{R}^D$ 是可学习的 Base 类型 embedding。

### 3.3 State 编码

从历史序列和 Base 预测中编码查询状态：

#### 历史归一化

$$\tilde{X}_{q,t,n,c} = \frac{X_{q,t,n,c} - \mu^{\text{ctx}}_{q,n}}{\sigma^{\text{ctx}}_{q,n} + \epsilon}$$

其中 $\mu^{\text{ctx}}_{q,n}, \sigma^{\text{ctx}}_{q,n}$ 是节点级历史统计量。

#### State 编码器

$$\mathbf{z}_{q,n} = \text{StateEncoder}([\text{Flatten}(\tilde{X}_{q,:,n,:}), \text{Flatten}(Y^{\text{base}}_{q,:,n,:})])$$

其中：
- 输入维度：$(T \cdot C + H \cdot C) = 24$ (METR-LA)
- $\text{StateEncoder}$：2层 MLP
- 输出维度：$D_s = 256$

#### Base 风险预测

$$r^{\text{base}}_{q,h,n} = \text{softplus}(\text{RiskProbe}(\mathbf{z}_{q,n}))$$

其中 $\text{RiskProbe}: \mathbb{R}^{D_s} \rightarrow \mathbb{R}^H$。

### 3.4 统一 Attention

#### Query 投影

$$\mathbf{q}_{q,h,n} = \text{QueryProj}(\mathbf{z}_{q,n}) \in \mathbb{R}^D$$

扩展到所有 horizon：$\mathbf{q}_{q,h,n}$ 对所有 $h$ 相同。

#### Key 投影

$$\mathbf{k}_{q,h,n,i} = \text{KeyProj}(\mathbf{t}_{q,h,n,i}) \in \mathbb{R}^D, \quad i=1,\ldots,K+1$$

#### Logits 计算

$$\ell_{q,h,n,i} = \frac{\mathbf{q}_{q,h,n}^{\top} \mathbf{k}_{q,h,n,i}}{\sqrt{D}}$$

Base 候选增加可学习偏置：

$$\ell_{q,h,n,K+1} \leftarrow \ell_{q,h,n,K+1} + b_{\text{base}}$$

其中 $b_{\text{base}}$ 初始化为 1.0。

#### Mask 处理

无效候选的 logit 设为 $-\infty$：

$$\ell_{q,h,n,i} \leftarrow \begin{cases}
\ell_{q,h,n,i}, & \text{if valid} \\
-\infty, & \text{otherwise}
\end{cases}$$

Base 候选的 mask 永远为 1。

#### Softmax

$$\pi_{q,h,n,i} = \frac{\exp(\ell_{q,h,n,i})}{\sum_{j=1}^{K+1} \exp(\ell_{q,h,n,j})}, \quad i=1,\ldots,K+1$$

### 3.5 最终预测

残差聚合：

$$R^{\text{agg}}_{q,h,n} = \sum_{k=1}^{K+1} \pi_{q,h,n,k} R_{q,h,n,k}$$

因为 $R_{q,h,n,K+1} = 0$，所以：

$$R^{\text{agg}}_{q,h,n} = \sum_{k=1}^{K} \pi_{q,h,n,k} R_{q,h,n,k}$$

最终预测：

$$\boxed{\hat{Y}_{q,h,n} = Y^{\text{base}}_{q,h,n} + R^{\text{agg}}_{q,h,n}}$$

---

## 4. 候选质量监督

### 4.1 训练阶段 Teacher

使用真实 future 构造候选质量 teacher（**仅训练阶段**）：

#### 候选误差

$$e_{q,h,n,k} = \frac{1}{C} \sum_{c=1}^{C} |Y^{\text{cand}}_{q,h,n,k,c} - Y_{q,h,n,c}|, \quad k=1,\ldots,K$$

$$e_{q,h,n,K+1} = \frac{1}{C} \sum_{c=1}^{C} |Y^{\text{base}}_{q,h,n,c} - Y_{q,h,n,c}|$$

#### Query 内归一化

为避免将 METR-LA 的绝对数值尺度写入监督，对每个 query 的误差进行归一化：

$$\bar{e}_{q,h,n,k} = \frac{e_{q,h,n,k} - \min_j e_{q,h,n,j}}{\text{IQR}(e_{q,h,n,:}) + \epsilon}$$

其中 $\text{IQR}$ 是四分位距，$\epsilon = 10^{-6}$。

#### Teacher 分布

$$p^{\text{teacher}}_{q,h,n,k} = \frac{\exp(-\bar{e}_{q,h,n,k} / T)}{\sum_{j=1}^{K+1} \exp(-\bar{e}_{q,h,n,j} / T)}$$

其中温度 $T=0.2$。

### 4.2 候选质量损失

$$\mathcal{L}_{\text{quality}} = \sum_{k=1}^{K+1} p^{\text{teacher}}_k \left( \log p^{\text{teacher}}_k - \log \pi_k \right)$$

只在有效位置计算（至少一个候选有效）。

### 4.3 总损失

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forecast}} + \lambda_{\text{quality}} \cdot \mathcal{L}_{\text{quality}}$$

其中：
- $\mathcal{L}_{\text{forecast}} = \text{MaskedMAE}(\hat{Y}, Y)$
- $\lambda_{\text{quality}} = 0.05$

**重要**：验证和推理阶段不使用真实 future 计算 Attention。

---

## 5. 网络结构与参数量

### 5.1 模块组成

| 模块 | 输入维度 | 输出维度 | 参数量 |
|------|----------|----------|--------|
| StateEncoder | 24 | 256 | 72,448 |
| CandidateEncoder | 5 | 256 | 66,816 |
| QueryProj | 256 | 256 | 65,792 |
| KeyProj | 256 | 256 | 65,792 |
| RiskProbe | 256 | 12 | 3,084 |
| Type Embeddings | - | 256 | 512 |
| Base Logit Bias | - | 1 | 1 |
| **总计** | - | - | **274,701** |

### 5.2 超参数配置

```yaml
calibrator_arch: base_as_candidate
candidate_token_dim: 256          # Token 维度
candidate_state_dim: 256          # State 维度
candidate_attention_heads: 4      # Attention heads
base_logit_init_bias: 1.0        # Base 初始偏置
candidate_quality_weight: 0.05   # 质量监督权重
candidate_quality_temperature: 0.2  # Teacher 温度
```

---

## 6. 训练策略

### 6.1 初始化

1. **StateEncoder, CandidateEncoder**：Xavier uniform
2. **QueryProj, KeyProj**：Xavier uniform
3. **Type Embeddings**：$\mathcal{N}(0, 0.02)$
4. **Base Logit Bias**：1.0（可学习）
5. **RiskProbe**：Xavier uniform

### 6.2 优化器

- **算法**：Adam
- **学习率**：0.001
- **权重衰减**：0.0001
- **Scheduler**：StepLR (step_size=10, gamma=0.95)

### 6.3 训练协议

- **Batch size**：32
- **Epochs**：50
- **Early stopping**：关闭（完整训练）
- **Patience**：10（用于记录最佳 checkpoint）

---

## 7. Mask 与边界情况处理

### 7.1 候选 Mask

历史候选 $k=1,\ldots,K$ 的 mask 来自 Bank future payload：

$$m_{q,h,n,k} = \begin{cases}
1, & \text{if 候选有效} \\
0, & \text{otherwise}
\end{cases}$$

### 7.2 Base Mask

Base 候选的 mask **永远为 1**：

$$m_{q,h,n,K+1} = 1$$

### 7.3 无有效候选情况

当所有历史候选无效时：

$$m_{q,h,n,1} = \cdots = m_{q,h,n,K} = 0, \quad m_{q,h,n,K+1} = 1$$

Softmax 后：

$$\pi_{q,h,n,K+1} = 1, \quad \pi_{q,h,n,1} = \cdots = \pi_{q,h,n,K} = 0$$

最终输出：

$$\hat{Y}_{q,h,n} = Y^{\text{base}}_{q,h,n}$$

**不需要额外的特殊分支**，架构自动处理。

---

## 8. 与旧版架构对比

| 维度 | 旧版 (StructuredErrorCorrector + Alpha) | 新版 (Base-as-Candidate) |
|------|----------------------------------------|--------------------------|
| **决策阶段** | 两阶段（Attention → Alpha gate） | 单阶段（统一 Attention） |
| **参数量** | 22.4万 (256/128配置) | 27.5万 (256/256配置) |
| **梯度路径** | Attention → Memory → Alpha → Output | Attention → Output |
| **Base 地位** | 隐式（通过 $\alpha=0$ 回退） | 显式（第 K+1 个候选） |
| **无候选处理** | 需要 `memory_valid` 分支 | 自动回退到 Base |
| **论文解释** | 需要解释 Alpha 必要性 | 标准 retrieval-augmented |
| **功能重叠** | Attention 和 Alpha 都做选择 | 无重叠 |

---

## 9. 实验验证协议

### 9.1 Smoke 测试（必须通过）

#### 阶段 1：单 Batch 验证

1. **Forward 测试**：
   - 输入：模拟数据 (B=4, H=12, N=207, K=5, C=1)
   - 检查：输出 shape 正确 $[4, 12, 207, 1]$
   - 检查：所有输出 finite（无 NaN/Inf）

2. **Attention Shape 验证**：
   - 检查：`attention.shape == [4, 12, 207, 6]` (K+1=6)
   - 检查：每个位置 attention 和为 1

3. **无候选回退测试**：
   - 输入：所有候选 mask 为 0
   - 检查：输出严格等于 Base
   - 检查：Base attention 为 1

4. **Backward 测试**：
   - 运行 `loss.backward()`
   - 检查：所有参数有梯度
   - 检查：梯度 finite

#### 阶段 2：3 Epoch 快速训练

对 GWN、STGCN、STAEformer 各运行 3 epoch，记录：

| 指标 | 说明 |
|------|------|
| Val MAE | 验证集平均绝对误差 |
| 15/30/60 min MAE | 分 horizon 误差 |
| Base attention mean | Base 候选平均权重 |
| Base attention std | Base 权重标准差 |
| History attention mass | 历史候选总权重 $1 - \pi_{\text{base}}$ |
| Attention entropy | $-\sum_k \pi_k \log \pi_k / \log(K+1)$ |
| Top-1 mass | 最大候选权重 |
| Params (万) | 参数量（万） |
| Time (s/epoch) | 单轮训练时间 |

### 9.2 成功标准

#### 必须满足（否则回退）

- ❌ MAE 退化 > 0.02
- ❌ Base 权重恒定接近 0 或 1（塌缩）
- ❌ 出现 NaN/Inf
- ❌ 单轮时间增加 > 50%

#### 可以进入 50 轮（满足任一）

- ✅ 至少一个 backbone MAE 改善 ≥ 0.01 且其余不退化
- ✅ MAE 持平（$|\Delta| < 0.005$）且 Attention 行为合理
- ✅ Attention 与候选误差相关性提升

### 9.3 对比实验

| 版本 | 描述 | Val MAE | 参数量 |
|------|------|---------|--------|
| Base-only | 冻结 base，无 TGGE | 2.865 | 0 |
| 旧版 Alpha+Memory | StructuredErrorCorrector | 2.819 | 22.4万 |
| 新版 Base-as-candidate | 本方案 | ? | 27.5万 |

---

## 10. 代码接口

### 10.1 模型初始化

```python
from stanchor.models.downstream import CandidateSetHorizonCorrector

calibrator = CandidateSetHorizonCorrector(
    context_length=12,
    horizon=12,
    channels=1,
    hidden_dim=256,
    state_dim=256,
    attention_heads=4,
    base_logit_init_bias=1.0
)
```

### 10.2 Forward 调用

```python
final, historical_mass, contributions, learned_memory = calibrator(
    history=history,          # [B, T, N, C]
    base=base_prediction,     # [B, H, N, C]
    memory=None,              # 不使用
    features=None,            # 不使用
    memory_valid=None,        # 不使用
    risk_state=None,          # 内部计算
    candidates=candidates,    # NodeCandidates 对象
    aggregation=aggregation   # AggregationOutput 对象
)
```

### 10.3 配置文件

```yaml
target:
  downstream_mode: learned_topk_error_aware
  validation_correction_variant: base_as_candidate  # 新增
  candidate_token_dim: 256
  candidate_state_dim: 256
  candidate_attention_heads: 4
  base_logit_init_bias: 1.0
  candidate_quality_weight: 0.05
  candidate_quality_temperature: 0.2
```

---

## 11. 预期结果

### 11.1 定量预期

| 指标 | Base-only | 旧版 | 新版预期 |
|------|-----------|------|----------|
| Val MAE | 2.865 | 2.819 | 2.80-2.82 |
| Base attention | - | - | 0.2-0.4 |
| History mass | - | - | 0.6-0.8 |
| Attention entropy | - | - | 0.6-0.8 |

### 11.2 定性预期

1. **Base 使用率随场景变化**：
   - 稳定场景（Base 可靠）→ Base 权重高
   - 有历史先例 → 历史候选权重高

2. **Attention 与候选误差相关性**：
   - 目标：Spearman > -0.3
   - 当前旧版：-0.24

3. **无候选自动回退**：
   - 无有效候选时输出严格等于 Base
   - 不需要额外代码分支

---

## 12. 风险与限制

### 12.1 已知限制

1. **Oracle gap 仍存在**：
   - 当前 learned MAE ≈ 2.82
   - Oracle top-1 MAE = 1.66
   - Gap = 1.16（不可能全部消除）

2. **时空海市蜃楼**：
   - 历史相似 ≠ 未来相似
   - 新架构改善选择机制，但不能消除本质不确定性

3. **候选池覆盖度**：
   - Bank 只有 15,876 个事件
   - 某些罕见情况可能无匹配候选

### 12.2 不承诺的目标

本方案**不承诺**：
- ❌ 达到 oracle 性能
- ❌ 在所有场景都优于 Base
- ❌ 完全消除候选质量监督效果弱的问题

本方案**承诺**：
- ✅ 逻辑更清晰，论文更好讲
- ✅ 梯度路径更直接
- ✅ 参数量在合理范围（27.5万）
- ✅ 训练稳定，无 NaN/Inf

---

## 13. 论文表述建议

### 13.1 核心贡献

> We formulate the backbone forecast as a fallback candidate and perform horizon-aware residual mixture over the backbone and retrieved historical futures, allowing the model to jointly decide whether to retain the base forecast or apply retrieval-based corrections.

中文表述：

> 我们将下游 backbone 的基础预测显式建模为候选集合中的一个零残差候选，并通过 horizon-aware attention 在基础预测与历史事件候选之间进行统一选择，从而避免独立 memory gate 与候选 attention 的功能重叠。

### 13.2 避免的表述

❌ 不要写成：
- "Attention 自动知道哪个未来是正确的"
- "模型可以达到 oracle 性能"
- "完全解决了时空海市蜃楼问题"

✅ 应该写成：
- "Attention 根据 context、Base 风险和候选 residual 特征，学习选择更适合当前 query 和预测 horizon 的修正来源"
- "虽然 oracle 实验显示候选池中存在高质量 future，但 learned 模型与 oracle 之间仍有 gap，这部分来自时空演化的内在随机性"

---

## 14. 附录：参数量详细分解

### StateEncoder (72,448 参数)

```
Layer 1: Linear(24 → 256)     6,400
Layer 1 Bias: 256             256
Layer 2: Linear(256 → 256)    65,536
Layer 2 Bias: 256             256
```

### CandidateEncoder (66,816 参数)

```
Layer 1: Linear(5 → 256)      1,280
Layer 1 Bias: 256             256
Layer 2: Linear(256 → 256)    65,536
Layer 2 Bias: 256             256
```

### QueryProj (65,792 参数)

```
Weight: 256 × 256             65,536
Bias: 256                     256
```

### KeyProj (65,792 参数)

```
Weight: 256 × 256             65,536
Bias: 256                     256
```

### RiskProbe (3,084 参数)

```
Weight: 12 × 256              3,072
Bias: 12                      12
```

### Embeddings (513 参数)

```
candidate_type: 256           256
base_type: 256                256
base_logit_bias: 1            1
```

**总计：274,701 参数 (27.47 万)**

---

## 15. 参考文献

1. HN-OffsetDecay 预训练方案：`doc/诊断报告合集/E5-TGGE-HN-OffsetDecay-v1-最终CaseStudy报告.md`
2. 科研工作原则：`memory/research-work-principles.md`
3. 文档编写规范：`memory/documentation-writing-skill.md`

---

**文档结束**
