# Offset-only Teacher 与 Payload 一致性消融方案

## 一、问题与现有证据

当前主线在两个位置使用 OffsetDecay：

1. 预训练 future-relation teacher 使用从 1 线性衰减到 0 的 level 系数；
2. 下游候选 future 使用相同的衰减系数，把历史候选映射到 Query 的当前 level。

现有完整 METR-LA validation 载荷诊断固定当前 OffsetDecay 预训练编码器，仅替换下游候选载荷，得到：

| Candidate payload | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Raw future | 3.433319 | 6.644278 | 9.478315 |
| Offset-only | 3.331687 | 6.341704 | 8.815815 |
| OffsetDecay | **3.221478** | **6.116368** | **8.643349** |

该结果证明当前编码器下 OffsetDecay payload 优于固定 Offset payload，但不能回答：若编码器本身使用 Offset-only teacher 预训练，一套 teacher/payload 语义一致的 Offset-only 系统是否更好。

本轮允许 Offset-only 推翻当前主线。实验口径在运行前固定，不因结果方向改变。

## 二、术语、公式与张量

### 2.1 输入与符号

- (B)：batch size；
- (T_r=288)：检索历史长度，对应 24 小时；
- (T=12)：下游预测历史长度，对应 1 小时；
- (H=12)：预测步数，对应未来 1 小时；
- (N=207)：METR-LA 节点数；
- (C=1)：速度通道数；
- (Y_i\in\mathbb{R}^{H\times N\times C})：历史事件 (i) 的真实 future；
- \(\ell_i\in\mathbb{R}^{N\times C}\)：事件 (i) 的 forecast context 最后一个有效 endpoint level。

### 2.2 OffsetDecay teacher

当前 teacher signature 为：

\[
\phi^{\mathrm{decay}}_{i,h,n}
=Y_{i,h,n}-\lambda_h\ell_{i,n},
\qquad
\lambda_h=1-\frac{h-1}{H-1}.
\]

近端完全消除当前 level，远端逐渐恢复候选的绝对 future level。

### 2.3 Offset-only teacher

本轮新增的简单 teacher 为：

\[
\phi^{\mathrm{off}}_{i,h,n}
=Y_{i,h,n}-\ell_{i,n},
\qquad h=1,\ldots,H.
\]

即 \(\lambda_h\equiv1\)，所有 horizon 都在相对当前 level 的坐标中比较。它不增加参数，不改变 encoder、hard-negative 采样或 reconstruction 分支。

### 2.4 Offset-only candidate payload

对 Query (q) 和检索候选 (j)：

\[
\widetilde Y^{\mathrm{off}}_{j,h,n}
=Y_{j,h,n}+(\ell_{q,n}-\ell_{j,n}).
\]

因此：

\[
\widetilde Y^{\mathrm{off}}_{j,h,n}-\ell_{q,n}
=Y_{j,h,n}-\ell_{j,n}
=\phi^{\mathrm{off}}_{j,h,n}.
\]

预训练检索语义与下游候选坐标严格一致。候选 mask、权重归一化、方差和无候选 Base fallback 与 OffsetDecay 路径相同。

## 三、未来信息边界

- Query encoder 输入只有 `retrieval_x: [B,288,N,1]` 及其日历和 observed mask；
- Query 真实 future 只在预训练 teacher、Router 训练标签和离线 validation 指标中使用；
- Bank 仅由目标数据训练历史前 70% 构建；
- 下游候选 future 是 Query 时刻之前已经发生的 Bank 历史，不是 Query future；
- `weekday_radius1_overlap` 可以与 Query 已观测 context 重叠，但候选 future 不越过 Query context end；
- Offset-only 不改变以上边界。

## 四、三格实验矩阵

| 组别 | Pretraining teacher | Candidate payload | 状态 | 解释 |
|---|---|---|---|---|
| A | OffsetDecay | OffsetDecay | 已完成 | 当前主模型 |
| B | Offset-only | OffsetDecay | 本轮新增 | 固定 payload，隔离 teacher decay |
| C | Offset-only | Offset-only | 本轮新增 | teacher/payload 一致的完整无 decay 系统 |

辅助证据 `OffsetDecay teacher + Offset-only payload` 已有完整 validation 的直接 Memory 诊断，但尚无 Router 下游；本轮不新增该 Router 训练，因为 A/B/C 已形成可归因的 L 形矩阵。

比较关系：

- A vs B：预训练 teacher 中 decay 的增量作用；
- B vs C：Offset-only encoder 下 payload decay 的增量作用；
- A vs C：两套语义一致系统的最终比较。

## 五、严格匹配约束

### 5.1 预训练

除 `relation_teacher_mode` 外完全匹配当前正式配置：METR-LA split/scaler、seed=42、模型初始化、288/12/12 时间契约、patch size 12、hidden 128、retrieval dim 64、4 层、4 heads、route Top-6、mask、hard-negative quantile、reconstruction weight 2、retrieval weight 1、Adam、学习率 0.001、weight decay 0.0001、batch 16、固定 50 epoch、关闭 early stopping。

不同 teacher 的 `val_total` 数值不直接比较，因为 teacher distance 分布不同；它只用于各自 checkpoint 选择和收敛检查。

### 5.2 Bank

Offset-only checkpoint 必须构建独立 Bank。Bank manifest 必须记录新的 encoder fingerprint；事件轴、future payload、mask、weekday/slot 和时间边界应与当前正式 Bank 对齐，只有 key 与 fingerprint 允许不同。

### 5.3 ARGCN 下游

B/C 共用同一 Offset-only checkpoint、Bank、ARGCN Base checkpoint、Router 架构与初始化、seed=42、physical MAE、batch 32、lr=0.0005、50 epoch、StepLR、关闭 early stopping、weekday-radius、event Top-32、node Top-12、level weight 0 和 frozen path cache。唯一差异是 `candidate_payload`。

## 六、CaseStudy 与指标

先报告完整 validation 聚合指标，再选择案例：

1. broad-causal 和 weekday-radius 的 Pair/Anchor Spearman、Kendall、Recall@1/5、NDCG@5；
2. 同一物理 query future 下的 Memory MAE、RMSE、MAPE 和 horizon-wise MAE；
3. teacher/student effective support、有效 anchor 数、候选数与缺失率；
4. strong-win、representative、failure 使用固定分位规则，不删除不利案例；
5. 预训练 loss 与物理 MAE 分开报告。

## 七、实现文件

- `stanchor/losses/pretraining.py`：新增 Offset-only signature，并接入 future-relation targets；
- `stanchor/config.py`：新增 `relation_teacher_mode=offset_only` 与 `candidate_payload=offset_only`；
- `stanchor/retrieval/strategies.py`：新增正式 Offset-only aggregation；
- `stanchor/engine/target.py`：按 checkpoint/config 解析并执行 Offset-only payload；
- `stanchor/diagnostics/retrieval_visualization.py`、`scripts/visualize_retrieval.py`：新增 Offset-only teacher 的检索排序与固定分位 CaseStudy；
- `tests/test_future_relation_loss.py`、`tests/test_candidate_ranking.py`、`tests/test_retrieval_strategies.py`：TDD 覆盖；
- `configs/metrla_e5_tgge_hn_offset_only_v1_transfer_hidden128_ffn2_b16.yaml`：正式预训练配置；
- `configs/ablation_offset_only_teacher_offset_decay_router_argcn.yaml`：B 组；
- `configs/ablation_offset_only_teacher_offset_only_router_argcn.yaml`：C 组；
- `scripts/run_offset_only_consistency_ablation.ps1`：预训练、建库、检索 CaseStudy、三载荷诊断和两组 ARGCN 的可恢复队列。

## 八、验证与停止条件

### 工程验证

1. Offset-only signature 在所有 horizon 精确减去 endpoint；
2. missing endpoint 回退和 mask 与当前 teacher 一致；
3. Offset-only aggregation 在所有 horizon 精确加固定 level offset；
4. 空候选仍返回无效 mask，使 Router 精确回退 Base；
5. checkpoint 保存并恢复 `candidate_payload=offset_only`；
6. 相关聚焦单测、完整相关测试和 one-batch smoke 通过后才启动正式预训练；
7. smoke 产物验证后立即删除，绝不作为正式证据。

### 科研决策

- 若 A 同时优于 B、C 超过 0.01 MAE，Keep OffsetDecay，停止扩展；
- 若 A 与 C 的绝对差异不超过 0.01 MAE，视为实用持平，保留已完成主线，但删除“teacher decay 必不可少”的表述；
- 若 C 优于 A 超过 0.01 MAE，Offset-only 有资格推翻当前主线；先补 GWN、第二种子和一个目标域检索诊断，再决定是否迁移全部实验；
- 若 B 优于 A 而 C 不优于 B，保留 `Offset-only teacher + OffsetDecay payload` 作为候选混合主线；
- 若检索指标改善但 ARGCN 不改善，不替换完整系统，只报告表示层结论。

本轮在完成 A/B/C 后停止，不预先扩展八骨干、多数据集或多种子。
