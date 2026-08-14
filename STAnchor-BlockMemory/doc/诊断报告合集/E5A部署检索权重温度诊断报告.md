# E5A 部署检索权重温度诊断报告

## 1. 实验动机

E5A 在 `RelaxedCalendar` 候选协议下已经能检索到比 random selector 更好的历史事件，但当前 Top-5 权重接近平均。需要验证一个具体问题：**selector 排出的好候选是否在聚合时被近似平均的权重稀释**。

本实验不训练新模型，只改变部署检索权重温度。若更尖锐的权重显著降低 pretrained memory 误差并扩大 pretrained 与 random 的差距，说明瓶颈在聚合；若低温度反而恶化，说明第一名候选不够可靠，多候选平滑是必要的，后续应停止调温度。

## 2. 特殊名词与实验边界

### 2.1 RelaxedCalendar

`RelaxedCalendar` 表示候选事件必须与 query 属于同一星期，并允许时间槽位于 query 槽的前一格、同一格或后一格。METR-LA 的采样间隔为 5 分钟，因此是同一星期、时间相差不超过 5 分钟的严格历史事件。候选 future 必须在 query context 开始前已经完整发生，不读取待预测的 query future。

### 2.2 Top-5 和检索权重温度

对 query 样本 (q)、节点 (n) 和合法历史候选 (j)，E5A selector 输出 key 相似度分数 (s_{qjn})。当前 `level_weight=0`，所以分数只来自 query key 与候选 key 的内积。先固定分数最高的 5 个候选：

\[
K_{q,n}=\operatorname{Top5}_{j}(s_{qjn}).
\]

检索权重温度 \(\tau_r\) 只把同一组 Top-5 分数转为权重：

\[
w_{qjn}(\tau_r)
=
\frac{\exp(s_{qjn}/\tau_r)}
{\sum_{l\in K_{q,n}}\exp(s_{qln}/\tau_r)}.
\]

输入是固定 Top-5 分数 `[B,N,5]`，输出是权重 `[B,N,5]`。温度越低，权重越集中于排名靠前的候选；它不改变候选 ID 和排序，也不读取 query future。

本实验包含两个端点：

- `UniformTop5`：5 个有效候选权重相同，用于表示完全平滑聚合；
- `HardTop1`：只保留第一名候选，用于表示最尖锐聚合。它是诊断上限，不是建议部署方法。

### 2.3 OffsetDecay 候选 payload

Bank 中保存的是已经发生的历史事件 **raw future** (Y_{j,h,n,c})，不是预先计算好的 OffsetDecay future。检索时才根据当前 query 和候选 context 生成 OffsetDecay 候选 payload。

对最近 12 个 context 步估计 query 端点 level \(\alpha_{q,n,c}\) 和候选端点 level \(\alpha_{j,n,c}\)。若最后一步有效，level 就是最后观测值；若最后一步缺失，则使用可见 context 拟合出的端点。第 \(h\) 个预测步的衰减系数为：

\[
\lambda_h=1-\frac{h-1}{H-1},\qquad h=1,\ldots,H,
\]

其中本实验 (H=12)。每个候选的 OffsetDecay payload 为：

\[
Z_{qjhnc}^{OD}
=Y_{j,h,n,c}+\lambda_h(\alpha_{q,n,c}-\alpha_{j,n,c}).
\]

近端预测完整执行 level 对齐，远端逐步回到 raw future。输入为 query history、候选历史 context 和 Bank raw future，输出为 `[B,H,N,5,C]`；推理不读取 query future。

### 2.4 Mask-aware 加权聚合

令 (M_{j,h,n,c}\in\{0,1\}) 表示候选 future 在该点是否有效。节点级权重 `[B,N,5]` 会广播到所有预测步和通道，并在每个预测点只对有效候选重新归一化：

\[
\widehat Y^{OD}_{q,h,n,c}
=
\frac{
\sum_{j\in K_{q,n}}w_{qjn}M_{j,h,n,c}Z^{OD}_{qjhnc}
}{
\sum_{j\in K_{q,n}}w_{qjn}M_{j,h,n,c}
}.
\]

输出 memory prediction 的形状是 `[B,H,N,C]`。若某个候选在某一 horizon 缺失，只在该点移除它，不会删除整个候选事件。

### 2.5 Effective Support

`Effective Support`（有效支持数）衡量权重相当于多少个候选共同参与：

\[
N_{eff}=\frac{1}{\sum_{j=1}^{5}w_j^2}.
\]

完全平均时 (N_{eff}=5)，HardTop1 时 (N_{eff}=1)。它描述权重集中度，不是预测误差。

## 3. 实验设置

固定以下条件：

- E5A pretrained checkpoint 和对应 Bank；
- seed 42 random checkpoint 和对应 Bank；
- 两个 Bank 的事件轴、raw future、mask 和日历元数据完全一致；
- `RelaxedCalendar`、Top-5、`level_weight=0` 和 OffsetDecay；
- 完整 METR-LA validation：2,993 个 query、94 batches；
- query future 只在检索与聚合完成后计算离线 MAE、RMSE 和 MAPE。

只扫描 `UniformTop5`、\(\tau_r=0.20,0.10,0.05,0.02\) 和 `HardTop1`。每个设置内部使用 pretrained/random 共同有效 mask；五个 soft/uniform 设置的共同 coverage 完全相同。

正式命令：

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/diagnose_retrieval_temperature.py --config configs/metrla_e5_offset_decay_relation_level0_v1.yaml --checkpoint artifacts/metrla_e5a_offset_decay_seed42/pretrain_best_relation.pt --bank artifacts/metrla_bank_e5a_offset_decay_relation_seed42 --random-checkpoint artifacts/metrla_e3_target_random_seed42/random_checkpoint.pt --random-bank artifacts/metrla_bank_e3_target_random_seed42 --output-dir artifacts/convergence/retrieval_temperature/e5a_relaxed_full_val --candidate-protocol relaxed_calendar
```

## 4. 完整 validation 结果

MAPE 单位为百分比。`Top-1 weight` 和 `Effective support` 列为 pretrained selector 的部署权重统计。

| 权重设置 | Pretrained MAE | Pretrained RMSE | Pretrained MAPE | Random MAE | Random RMSE | Random MAPE | Top-1 weight | Effective support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UniformTop5 | 3.479613 | **6.683316** | 9.289770 | 3.650927 | **6.795616** | 9.683890 | 0.200000 | 5.000000 |
| \(\tau_r=0.20\) | 3.472865 | 6.692139 | **9.279971** | **3.647909** | 6.813384 | **9.678882** | 0.211843 | 4.971736 |
| \(\tau_r=0.10\) | **3.472126** | 6.713090 | 9.285960 | 3.650737 | 6.840966 | 9.692773 | 0.224138 | 4.912362 |
| \(\tau_r=0.05\) | 3.481122 | 6.765632 | 9.319170 | 3.661508 | 6.895572 | 9.726010 | 0.248045 | 4.772391 |
| \(\tau_r=0.02\) | 3.526409 | 6.912209 | 9.440973 | 3.707781 | 7.046060 | 9.837003 | 0.310019 | 4.373742 |
| HardTop1 | 4.055775 | 7.841816 | 10.621824 | 4.240582 | 8.067128 | 10.921165 | 1.000000 | 1.000000 |

Soft/uniform 设置的共同评价 coverage 为 `99.270588%`；HardTop1 因第一名候选部分 future 点缺失，pretrained/random 共同 coverage 为 `94.824785%`。

### 4.1 当前设置与较平滑设置的逐步指标

| 预测时间 | \(\tau_r=0.10\) MAE | RMSE | MAPE | \(\tau_r=0.20\) MAE | RMSE | MAPE |
|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 2.489371 | 4.124170 | 5.807955 | 2.491676 | 4.115461 | 5.797756 |
| 10 min | 2.844464 | 5.045404 | 7.046028 | 2.845486 | 5.026223 | 7.030169 |
| 15 min | 3.080328 | 5.657871 | 7.865665 | 3.080544 | 5.633103 | 7.849802 |
| 20 min | 3.258486 | 6.102450 | 8.516540 | 3.259137 | 6.077425 | 8.503518 |
| 25 min | 3.404748 | 6.460366 | 9.063323 | 3.405730 | 6.434583 | 9.052320 |
| 30 min | 3.523650 | 6.756902 | 9.489580 | 3.524718 | 6.731339 | 9.481587 |
| 35 min | 3.634912 | 7.015142 | 9.884948 | 3.635302 | 6.989011 | 9.877518 |
| 40 min | 3.723806 | 7.241076 | 10.208502 | 3.724273 | 7.216267 | 10.205101 |
| 45 min | 3.810334 | 7.452466 | 10.510870 | 3.810935 | 7.430040 | 10.510877 |
| 50 min | 3.892238 | 7.655843 | 10.784217 | 3.892040 | 7.635626 | 10.785225 |
| 55 min | 3.969086 | 7.853692 | 11.035555 | 3.969574 | 7.836905 | 11.040328 |
| 60 min | 4.051091 | 8.045779 | 11.278558 | 4.051945 | 8.032458 | 11.285933 |

全部 6 个设置、两个 selector 和 12 个预测步的三项指标保存在 `artifacts/convergence/retrieval_temperature/e5a_relaxed_full_val/horizon_metrics.csv`。

## 5. 结果信号

1. **“低温度能解除好候选被平均稀释”的假设不成立。** 从 \(\tau_r=0.10\) 降到 0.05 后，pretrained MAE、RMSE、MAPE 分别恶化 `0.008995`、`0.052542`、`0.033211`；降到 0.02 后三项分别恶化 `0.054282`、`0.199119`、`0.155014`。
2. **HardTop1 明显失败。** 相比 \(\tau_r=0.10\)，其 pretrained MAE 增加 `0.583648`，说明第一名候选远未精确到可以单独承担预测。Top-5 平滑不是无意义稀释，而是在抵消单个历史事件的噪声和偶然偏差。
3. **当前 \(\tau_r=0.10\) 已处于 MAE 最优附近。** \(\tau_r=0.20\) 的 MAE 只差 `0.000738`，同时 RMSE 改善 `0.020951`、MAPE 改善 `0.005989`；差异太小，不足以支持为温度重新训练下游模型。
4. **E5A selector 的关系信号仍然存在。** 所有权重设置下 pretrained MAE 都优于相同设置的 random，差值为 `0.171314` 至 `0.184807`。这说明问题不是 selector 完全没有学习，而是继续强化 Top-1 权重不能把关系信号转化为更低的物理误差。
5. **部署权重稀释不是当前主要瓶颈。** E5A 在 \(\tau_r=0.10\) 下的 Top-1 weight 只有 `0.224138`、有效支持数为 `4.912362/5`，确实接近平均；但实验表明这种平滑恰好更稳健。下一步应回到候选关系几何和第一名可靠性，而不是继续调聚合温度。

## 6. 决策

- **保留** 当前 `search_temperature=0.10`，因为 MAE 是主指标且它在扫描中最低；
- **不启动** 新的下游温度训练，\(\tau_r=0.20\) 的变化不足 `0.001` MAE，不构成有效改进；
- **停止** 更低温度和 HardTop1 方向；
- 下一步只做一个收敛实验：比较当前 `AnchorMean` teacher 距离归一化与对称 `Pair Normalization` 的零训练关系诊断。只有后者同时改善 Spearman、Recall@5 和 oracle memory 误差，才考虑预训练 `E5A-Symmetric`，暂不增加多特征分支。

## 7. 结果文件

- `artifacts/convergence/retrieval_temperature/e5a_relaxed_full_val/metrics.json`：完整机器可读结果；
- `artifacts/convergence/retrieval_temperature/e5a_relaxed_full_val/summary.csv`：整体指标和权重统计；
- `artifacts/convergence/retrieval_temperature/e5a_relaxed_full_val/horizon_metrics.csv`：12 个预测步的 MAE、RMSE、MAPE。
