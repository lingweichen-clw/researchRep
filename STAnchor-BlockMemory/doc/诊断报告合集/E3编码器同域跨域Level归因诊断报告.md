# E3 编码器同域、跨域与 Level 归因诊断报告

## 1. 本轮实验回答的问题

本轮实验不改变 E3 的预训练目标，也不改变候选策略，只验证三个问题：

1. E3 source-pretrained encoder-selector 在 METR-LA 同域是否真的优于 random encoder-selector；
2. PEMS-BAY 上的失败是否由 `level_weight=0.25` 干扰 learned key 造成；
3. 是否应该把预训练候选强制限制为同 weekday-slot、严格历史候选。

所有结果均为完整 validation 诊断，未读取 test。

## 2. 实验对照

### 2.1 Source 与 random

- `source`：METR-LA E3 relation checkpoint，经过 31 个 epoch 的预训练；
- `random`：相同结构、相同 seed 42、训练步数为 0 的 encoder-selector。

两者均输出：

$$
\mathbf K\in\mathbb R^{B\times N\times D},
\qquad D=48,
$$

其中 $B$ 是事件数，$N$ 是节点数，$D$ 是节点 key 维度。

### 2.2 Level 对照

节点候选排序分数为：

$$
s_{b,n,r}
=
\cos(\mathbf k^{q}_{b,n},\mathbf k^{m}_{b,r,n})
+
\lambda_{\mathrm{level}}
\exp\left(-d^{\mathrm{level}}_{b,n,r}/\tau_{\mathrm{level}}\right).
$$

本轮比较：

- 默认：$\lambda_{\mathrm{level}}=0.25$；
- 纯 key：$\lambda_{\mathrm{level}}=0$。

`level_weight=0` 只删除最终节点排序分数中的显式 level 加分，不改变 encoder 输出的 key，也不改变候选集合。

## 3. METR-LA 同域结果

| Level weight | 模式 | learned Top-1 MAE | learned uniform Top-K MAE | learned weighted Top-K MAE |
|---:|---|---:|---:|---:|
| 0.25 | source | 4.265001 | 3.923117 | **3.732261** |
| 0.25 | random | 4.344191 | 3.979213 | 3.796042 |
| 0 | source | 4.299002 | 3.947811 | **3.786680** |
| 0 | random | 4.332677 | 3.979540 | 3.819926 |

预训练收益定义为：

$$
G_{\mathrm{pre}}
=
\frac{M_{\mathrm{random}}-M_{\mathrm{source}}}
{M_{\mathrm{random}}}
\times100\%.
$$

在 METR-LA 上：

- 默认 level：weighted Top-K source 比 random 好 `1.680%`；
- 纯 key：weighted Top-K source 比 random 好 `0.870%`。

因此，即使删除 level 项，E3 source 仍然优于 random。可以确认：

> **在当前 seed 42 对照中，E3 relation 预训练在源域学习到了优于该 random initialization 的 future 相似性表示。**

但 source 相对 random 的优势只有 `0.87%~1.68%`，当前也只有一个 random seed。因此该结果足以否定“E3 在同域完全不如 random”，尚不足以声称同域优势对随机初始化稳定。下一步需要低成本补充 random seeds 2024、2025，估计随机映射方差。

同时，source 的 weighted Top-K 从 `3.732261` 退化到 `3.786680`，说明 level 统计特征在源域对检索有辅助作用，但不是 E3 预训练有效的唯一来源。

## 4. PEMS-BAY 跨域结果

| Level weight | 模式 | learned Top-1 MAE | learned uniform Top-K MAE | learned weighted Top-K MAE |
|---:|---|---:|---:|---:|
| 0.25 | source | 2.530079 | 2.280555 | 2.209414 |
| 0.25 | random | **2.494819** | **2.233733** | **2.173856** |
| 0 | source | 2.679266 | 2.379090 | 2.333939 |
| 0 | random | **2.509838** | **2.241775** | **2.202655** |

PEMS-BAY 上 source 相对 random 的 weighted Top-K 预训练收益为：

$$
G_{\mathrm{pre}}^{\mathrm{PEMS}}
=
\frac{2.173856-2.209414}{2.173856}
\times100\%
=-1.636\%.
$$

纯 key 下则为：

$$
G_{\mathrm{pre}}^{\mathrm{PEMS,key}}
=
\frac{2.202655-2.333939}{2.202655}
\times100\%
=-5.960\%.
$$

也就是说，删除 level 项以后 source 不是变好，而是进一步落后 random。

### 4.1 Level 项的实际作用

对 source：

- weighted Top-K：`2.209414 -> 2.333939`，删除 level 后 MAE 增加 `0.124525`；
- uniform Top-K：`2.280555 -> 2.379090`，删除 level 后 MAE 增加 `0.098535`。

对 random：

- weighted Top-K：`2.173856 -> 2.202655`，删除 level 后 MAE 增加 `0.028799`；
- uniform Top-K：`2.233733 -> 2.241775`，删除 level 后 MAE 增加 `0.008043`。

level 项对 source 的帮助更大，但仍不足以使 source 超过 random。因此：

> **`level_weight` 不是当前跨域失败的根因；它反而部分补救了 source key 在 PEMS-BAY 上的域偏移。**

## 5. 三种结果的逻辑归因

| 同域 source vs random | PEMS-BAY source vs random | 结论 |
|---|---|---|
| source 更好 | source 更差 | relation loss 在同域有效，但表示跨域不稳定 |
| source 更好 | source 更差，且 level=0 更差 | 不是 level 项造成失败，不能靠删 level 修复 |
| random 在 PEMS 仍有效 | random 在 PEMS 仍有效 | 小候选池中随机映射保留了部分粗相似性，但不能解释 source 的跨域能力 |

因此当前最合理的结论是：

> **E3 解决了源域 future-relation 学习问题，但没有解决跨数据集的表示域偏移问题。**

可能的域偏移来源包括：

- METR-LA 与 PEMS-BAY 的节点语义和路网拓扑不同；
- 两个数据集的速度分布、拥堵强度和异常模式不同；
- source encoder 学到的时空组合关系依赖源域图结构；
- weekday/slot embedding 和一天模式在不同数据集上存在统计差异。

以上是基于结果的机制假设，不应在没有额外实验前写成已证实事实。

## 6. 对“同 weekday-slot 预训练”的结论

本轮结果不支持立即把预训练候选强制改成同 weekday-slot、严格历史集合。

理由是：

1. E3 在 METR-LA 同域上已经有效，说明当前广域候选确实能够学习未来相似性的共同特征；
2. 当前跨域失败发生在纯 key 和带 level 两种口径下，不能归因于候选范围过宽；
3. 强制同 weekday-slot 会缩小候选覆盖范围，可能让模型更依赖周期位置，削弱跨时段动态模式的学习；
4. 当前 PEMS-BAY 的同 weekday-slot 候选平均只有 `12.64` 个，直接把预训练关系限制在这个小集合，可能使预训练退化为周期桶内排序。

因此，原先的“E4 检索一致性候选收缩”暂缓，不作为下一步主实验。

## 7. 当前决策

### 保留

- E3 future relation loss；
- 广域、不强制同 weekday-slot 的预训练候选；
- `level_weight=0.25`，因为其在 source 和 target 上均有辅助作用；
- history Bank、节点 Top-K 和 horizon fusion。

### 暂停

- 立即实施同 weekday-slot-only 预训练；
- 当前 source checkpoint 的 PEMS-BAY 多 seed 和 test；
- 继续扫描多个 level weight。

### 当前失败点

- source-pretrained encoder-selector 尚未形成稳定的跨数据集迁移优势。

### 统计边界

- METR-LA source checkpoint 目前只与一个 random seed 比较；
- PEMS-BAY source/random 下游也只有 seed 42；
- 本报告给出机制方向判断，不给出统计显著性结论。

## 8. 证据文件

- PEMS-BAY source level=0：`artifacts/encoder_random_attribution_seed42/pemsbay_source_level0_val.json`
- PEMS-BAY random level=0：`artifacts/encoder_random_attribution_seed42/pemsbay_random_level0_val.json`
- METR-LA source level=0：`artifacts/encoder_random_attribution_seed42/metrla_source_level0_val.json`
- METR-LA source level=0.25：`artifacts/encoder_random_attribution_seed42/metrla_source_level025_val.json`
- METR-LA random level=0：`artifacts/encoder_random_attribution_seed42/metrla_random_level0_val.json`
- METR-LA random level=0.25：`artifacts/encoder_random_attribution_seed42/metrla_random_level025_val.json`
- PEMS-BAY source level=0.25：`artifacts/pemsbay_e3_transfer_diagnostics/retrieval_diagnostics_val.json`
- PEMS-BAY random level=0.25：`artifacts/pemsbay_e3_target_random_seed42/retrieval_diagnostics_val.json`
- 后台流水线：`artifacts/encoder_random_attribution_seed42/pipeline.log`

## 9. METR-LA Random 三种子稳定性补充

为判断 seed 42 下 `0.87%~1.68%` 的 source 优势是否只是随机映射波动，补充 random seeds 2024、2025。三套 random Bank 与 source Bank 的 future payload、掩码、weekday、slot、时间边界、sample id 和 level features 共 9 类文件 SHA-256 全部一致，只有 key 和 encoder fingerprint 不同。

### 9.1 三种子结果

| Random seed | learned Top-1 MAE | learned uniform Top-K MAE | learned weighted Top-K MAE |
|---:|---:|---:|---:|
| 42 | 4.344191 | 3.979213 | 3.796042 |
| 2024 | 4.337798 | 3.971383 | 3.781807 |
| 2025 | 4.347824 | 3.978459 | 3.789821 |
| Random 均值 | 4.343271 | 3.976352 | 3.789223 |
| Random 样本标准差 | 0.005076 | 0.004319 | 0.007136 |
| **E3 source** | **4.265001** | **3.923116** | **3.732261** |

三个 random seed 的样本均值与样本标准差分别定义为：

$$
\overline{M}_{\mathrm{rand}}
=
\frac{1}{3}\sum_{r=1}^{3}M^{(r)}_{\mathrm{rand}},
$$

$$
S_{\mathrm{rand}}
=
\sqrt{
\frac{1}{3-1}
\sum_{r=1}^{3}
\left(M^{(r)}_{\mathrm{rand}}-\overline{M}_{\mathrm{rand}}\right)^2
}.
$$

其中 $M^{(r)}_{\mathrm{rand}}$ 是第 $r$ 个 random initialization 的 validation MAE。

### 9.2 Source 相对 random 分布的位置

| 指标 | Source 相对 random 均值改善 | Source 是否优于全部 random seeds | 标准化间隔 $(\overline M_{rand}-M_{source})/S_{rand}$ |
|---|---:|---|---:|
| learned Top-1 | 1.802% | 是 | 15.42 |
| learned uniform Top-K | 1.339% | 是 | 12.33 |
| learned weighted Top-K | 1.503% | 是 | 7.98 |

`标准化间隔` 只表示 source 与这三个 random initialization 波动的相对距离，不是正式的假设检验统计量。由于 random 只有三个 seed、source 也只有一个预训练 seed，不能据此报告统计显著性。

### 9.3 更新后的结论

1. Source 在三项 learned 检索指标上都优于全部 random seeds；
2. random weighted Top-K 的标准差只有 `0.007136`，source 与 random 均值相差 `0.056963`；
3. 当前证据支持 E3 relation pretraining 在 METR-LA 同域具有稳定方向的非随机收益；
4. 但 weighted Top-K 相对收益只有 `1.50%`，仍属于较小机制收益；
5. PEMS-BAY 上 source 反而落后 random，说明当前编码器不是完全无效，而是学到的表示明显依赖源域，迁移性不足。

因此不立即扩大 encoder。下一步进入目标域自适应关系预训练，用同一结构和同一损失判断 source 初始化经过少量目标域校准后能否优于 target-random adapted。若仍不能，则再将问题升级为 encoder/预训练目标设计不足。

### 9.4 新增证据文件

- seed 2024：`artifacts/metrla_random_seed_stability/metrla_random_seed2024_level025_val.json`
- seed 2025：`artifacts/metrla_random_seed_stability/metrla_random_seed2025_level025_val.json`
- 正式流水线：`artifacts/metrla_random_seed_stability/pipeline.log`
