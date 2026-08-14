# PEMS-BAY E3 迁移 Target-Random 归因诊断报告

> **后续归因更新（2026-07-30）**：METR-LA 同域 trained-vs-random 与 PEMS-BAY `level_weight=0` 诊断已经完成。E3 在源域确实优于 random，但 PEMS-BAY 上的失败不是 level 项或候选宽度可以直接解释。最新综合结论见 `E3编码器同域跨域Level归因诊断报告.md`。

## 1. 诊断问题与实验边界

本报告只回答一个问题：

> 在相同 PEMS-BAY 数据、历史库、检索协议、下游结构和随机种子下，METR-LA E3 预训练得到的 encoder-selector，是否优于未经任何训练的同构 random encoder-selector？

对照组只改变 encoder-selector 参数来源：

- `source-pretrained`：METR-LA E3 relation checkpoint，epoch 31；
- `target-random`：相同结构、seed 42 初始化、训练步数为 0 的 checkpoint。

两组都在 PEMS-BAY 训练段重建 Bank，并只在 PEMS-BAY validation 上诊断和训练 `learned_topk_horizon`。本报告没有读取 PEMS-BAY test，也不能用单个 seed 判断统计显著性。

## 2. 对照完整性

两组 Bank 均包含：

- 25,327 个历史事件；
- 325 个节点；
- 288 步检索上下文；
- 12 步历史 future；
- 48 维节点 key。

两组 Bank 的以下八类文件 SHA-256 完全一致：

`future_values.npy`、`future_masks.npy`、`weekday.npy`、`slot.npy`、`context_start.npy`、`context_end.npy`、`future_end.npy`、`sample_id.npy`。

图指纹、scaler、数据范围和 future payload 也一致；只有 `event_keys.npy`、`node_keys.npy` 以及 encoder 指纹不同。因此，这是一组有效的单变量参数来源对照，不是数据范围不同造成的伪差异。

## 3. 检索层结果

### 3.1 指标含义

`retrieved future MAE` 直接比较检索得到的历史 future 与 query 真实 future。设 validation 中共有 $B$ 个 query，预测步数为 $H=12$，节点数为 $N=325$，通道数为 $C=1$，则：

$$
\operatorname{MemoryMAE}
=
\frac{
\sum_{b=1}^{B}\sum_{h=1}^{H}\sum_{n=1}^{N}\sum_{c=1}^{C}
M_{b,h,n,c}
\left|
\widehat{Y}^{\mathrm{mem}}_{b,h,n,c}-Y_{b,h,n,c}
\right|
}{
\sum_{b=1}^{B}\sum_{h=1}^{H}\sum_{n=1}^{N}\sum_{c=1}^{C}
M_{b,h,n,c}
}.
$$

其中：

- $Y\in\mathbb{R}^{B\times H\times N\times C}$ 是 query 的真实未来；
- $\widehat{Y}^{\mathrm{mem}}\in\mathbb{R}^{B\times H\times N\times C}$ 是检索到的 Top-K 历史 future 加权结果；
- $M\in\{0,1\}^{B\times H\times N\times C}$ 是真实 future 的有效观测掩码；
- 数值越低，说明检索到的历史模式越接近真实未来。

### 3.2 结果

| 历史来源 | Source-pretrained MAE | Target-random MAE | 更优者 |
|---|---:|---:|---|
| weekly mean | 2.681505 | 2.681505 | 完全一致 |
| raw L1 Top-1 | 2.479146 | **2.479092** | 实质持平 |
| raw L1 Top-K | 2.228216 | **2.228208** | 实质持平 |
| learned Top-1 | 2.530079 | **2.494819** | random |
| learned uniform Top-K | 2.280555 | **2.233733** | random |
| learned weighted Top-K | 2.209414 | **2.173856** | random |
| Oracle Top-1 | 1.434256 | 1.434256 | 完全一致 |

定义预训练相对收益：

$$
G_{\mathrm{pre}}
=
\frac{M_{\mathrm{random}}-M_{\mathrm{source}}}
{M_{\mathrm{random}}}
\times 100\%.
$$

其中 $M_{\mathrm{source}}$ 和 $M_{\mathrm{random}}$ 分别是 source-pretrained 与 target-random 的误差。$G_{\mathrm{pre}}>0$ 才表示预训练优于随机初始化。

对 learned weighted Top-K：

$$
G_{\mathrm{pre}}^{\mathrm{retrieval}}
=
\frac{2.173856-2.209414}{2.173856}\times100\%
=-1.636\%.
$$

所以在真正由 key 决定的三项 learned 检索中，source-pretrained 没有取得优势，random 反而全部略优。

## 4. Selector 行为

| 指标 | Source-pretrained | Target-random | 解释 |
|---|---:|---:|---|
| 原始合法候选数均值 | 12.641 | 12.641 | 日历和因果过滤后的历史事件数 |
| Top-1 weight 均值 | 0.2750 | 0.2507 | 最大候选聚合权重 |
| $K_{\mathrm{eff}}$ 均值 | 4.6387 | 4.7875 | 有效参与聚合的候选数 |
| learned Top-1 与 Oracle MAE 差距 | 1.0958 | **1.0606** | learned 排序距理想排序的距离 |

有效支持数定义为：

$$
K_{\mathrm{eff}}(b,n)
=
\frac{1}{\sum_{k=1}^{K}w_{b,n,k}^{2}},
$$

其中 $w_{b,n,k}$ 是 query $b$、节点 $n$ 对第 $k$ 个历史 future 的聚合权重，且 $\sum_k w_{b,n,k}=1$。当 $K=5$ 时，$K_{\mathrm{eff}}=1$ 表示几乎只使用一个候选，$K_{\mathrm{eff}}=5$ 表示五个候选接近等权。

source-pretrained 的权重更集中，但其 future MAE 更高。这说明“分数更尖锐”不等于“排序更准确”，当前 source key 的置信程度没有转化成更好的目标域候选。

还有一个直接的结构事实：当前 `event_top_r=32`，但每个 query 的合法候选最多只有 13 个。因此事件级 Top-R 没有删除任何候选，事件 key 在这组实验中不承担筛选作用；实际比较主要发生在节点级 Top-5 重排。

## 5. 下游预测结果

| 模式 | Validation MAE | RMSE | MAPE (%) | 最佳 epoch |
|---|---:|---:|---:|---:|
| Source-pretrained + horizon-only | 1.881369 | **4.107756** | **4.390351** | 34 |
| Target-random + horizon-only | **1.878035** | 4.114959 | 4.394232 | 34 |

对应的预训练相对收益为：

| 指标 | $G_{\mathrm{pre}}$ | 判断 |
|---|---:|---|
| MAE | -0.177% | random 略优 |
| RMSE | +0.175% | source 略优 |
| MAPE | +0.088% | source 略优 |

三项指标方向不一致，且相对差异都小于 0.2%。在只有一个 seed 的前提下，合理结论是两者在最终预测上功能性持平，而不是 source-pretrained 获胜。

### 5.1 不同预测距离

| 预测位置 | 模式 | MAE | RMSE | MAPE (%) |
|---|---|---:|---:|---:|
| 15 min | source | 1.498894 | **3.176010** | **3.225919** |
| 15 min | random | **1.498241** | 3.179396 | 3.230848 |
| 30 min | source | 1.965844 | **4.279348** | **4.560664** |
| 30 min | random | **1.962493** | 4.283602 | 4.565528 |
| 60 min | source | 2.359794 | **4.936744** | **5.848548** |
| 60 min | random | **2.353229** | 4.948632 | 5.853739 |

三个预测距离重复了同一模式：random 的 MAE 略低，source 的 RMSE/MAPE 略低，没有出现随 horizon 扩大的 source 预训练优势。

## 6. 同一 checkpoint 的分支归因

| 分支 | Source-pretrained MAE | Target-random MAE | 更优者 |
|---|---:|---:|---|
| base branch | **2.248062** | 2.260126 | source |
| memory branch | 2.209415 | **2.173857** | random |
| final fusion | 1.881369 | **1.878035** | 实质持平 |

memory 相对同 checkpoint base branch 的改善为：

- source-pretrained：1.72%；
- target-random：3.82%。

因此最终持平不是因为两组 memory 完全一样，而是 random memory 更好、source base branch 略好，经过联合训练和 horizon fusion 后相互抵消。最接近 encoder-selector 本职功能的 memory branch 仍然指向 random 更优。

## 7. 结果究竟说明什么

### 7.1 可以确认

1. 历史 memory 系统本身有效。random 组最终预测仍比其同 checkpoint base branch 降低 MAE 16.91%，说明日历过滤、历史 future 与 horizon fusion 确实提供了信息。
2. 当前收益不能归因于 METR-LA E3 预训练。预训练在检索 MAE 和最终 MAE 上都没有超过 random 对照。
3. 当前 source E3 并非完全没有学到东西。在 METR-LA 同域 validation 上，learned Top-K MAE 为 3.732261，优于 raw L1 Top-K 的 3.983865，相对改善 6.32%。问题更准确地说是：学到的关系没有迁移成 PEMS-BAY 优势。
4. Oracle Top-1 MAE 为 1.434256，明显低于 learned Top-1，说明历史候选中仍有大量可利用信息，主要瓶颈仍是排序，而不是历史库完全无效。

### 7.2 不能确认

1. 不能用一个 seed 声称 random 在统计上稳定优于 source；
2. 不能继续声称当前 E3 encoder-selector 已证明跨数据集迁移有效；
3. 不能把 confidence 已取得的下游收益解释为预训练收益。confidence 是另一个下游机制，仍可保留，但 selector 改动后必须重新训练；
4. 不能读取 test 或追加多 seed 来掩盖已经失败的归因门槛。

## 8. 最可能的原因

### 8.1 已由代码确认：预训练候选集与部署候选集不一致

E3 关系预训练的候选掩码为：

$$
M^{\mathrm{E3}}_{i,j,n}
=
\mathbb{I}(i\neq j)
\cdot
\mathbb{I}
\left(
t^{\mathrm{future\_end}}_i<t^{\mathrm{context\_start}}_j
\;\lor\;
t^{\mathrm{future\_end}}_j<t^{\mathrm{context\_start}}_i
\right)
\cdot
\mathbb{I}(i,j,n\text{ 的 future 有共同观测}).
$$

其中 $i,j\in\{1,\ldots,B\}$ 是同一训练 batch 中的事件，$n\in\{1,\ldots,N\}$ 是节点。该集合只要求事件不重叠，没有 weekday-slot 条件，而且对时间方向是对称的。

实际检索集合却是：

$$
\mathcal{C}^{\mathrm{deploy}}_i
=
\left\{
j\;\middle|\;
d_j=d_i,\;
s_j=s_i,\;
t^{\mathrm{future\_end}}_j<t^{\mathrm{context\_start}}_i
\right\},
$$

其中 $d_i$ 是 query 的 weekday，$s_i$ 是五分钟时间槽。部署阶段只允许同 weekday-slot 且严格位于 query 之前的历史事件。

这不是预测时的未来泄漏，但属于训练任务与部署检索空间不一致：模型在源域优化了大量部署时永远不会互相竞争的事件关系。

### 8.2 尚需最小实验确认：固定 level score 可能掩盖 key 的真实作用

节点重排分数为：

$$
s_{i,j,n}
=
\operatorname{cos}
\left(\mathbf{k}_{i,n},\mathbf{k}_{j,n}\right)
+
\lambda_{\ell}
\exp\left(
-\frac{
\left\|\boldsymbol{\ell}_{i,n}-\boldsymbol{\ell}_{j,n}\right\|_1/(4C)
}{\tau_{\ell}}
\right),
$$

其中：

- $\mathbf{k}_{i,n},\mathbf{k}_{j,n}\in\mathbb{R}^{D}$ 是 query 和候选的节点 key，当前 $D=48$；
- $\boldsymbol{\ell}_{i,n},\boldsymbol{\ell}_{j,n}\in\mathbb{R}^{4C}$ 是窗口 level 统计特征；
- 当前 $\lambda_{\ell}=0.25$，$\tau_{\ell}=1$。

source 与 random 的 level 特征构造相同，但 level 项会与两套不同的 key 排名共同决定 Top-K。仅凭现有结果还不能区分“预训练 key 本身不迁移”与“预训练 key 和固定 level 项组合后失配”。下一步应先将 $\lambda_{\ell}$ 设为 0 做一次不训练的诊断。

### 8.3 解释性假设：随机特征在小候选池中已经足够保留粗相似性

random encoder 并不是随机选择历史 future。它仍将 query 和 Bank 事件通过同一个固定随机映射投影到 48 维，并在只有约 12.64 个同日历候选中比较相似度。高维随机投影可能保留部分输入几何关系，再叠加固定 level 项后，已经足以形成可用排序。

该解释与结果一致，但目前没有独立实验直接证明，因此只能作为假设，不能写成确定机制。

## 9. 最终决策

按照预先固定的 target-random 规则，本轮结论为：

> **当前 E3 source-pretrained 迁移归因门槛未通过。停止宣称 METR-LA E3 预训练优于 target-random，也不进入当前 E3 的 PEMS-BAY 多 seed 或 test。**

同时保留两个独立事实：

- 历史 memory + horizon/confidence fusion 仍然有效；
- E3 relation objective 在 METR-LA 同域有效，但其跨数据集可迁移性尚未成立。

下一步不增加新编码器模块。先隔离 level 项；若仍失败，只修改关系损失的候选集合，使预训练任务与真实检索候选一致。具体规则见 `doc/优化方案合集/E4-检索一致性关系预训练改进方案.md`。

## 10. 证据文件

- source 检索诊断：`artifacts/pemsbay_e3_transfer_diagnostics/retrieval_diagnostics_val.json`
- random 检索诊断：`artifacts/pemsbay_e3_target_random_seed42/retrieval_diagnostics_val.json`
- source 下游诊断：`artifacts/pemsbay_e3_learned_topk_horizon_seed42/branch_diagnostics_val.json`
- random 下游诊断：`artifacts/pemsbay_e3_target_random_horizon_seed42/branch_diagnostics_val.json`
- source E3 同域诊断：`artifacts/metrla_e3_relation_relation_val_diagnostics.json`
- random checkpoint：`artifacts/pemsbay_e3_target_random_seed42/random_checkpoint.pt`
- random Bank：`artifacts/pemsbay_bank_target_random_seed42`
