# E5A 候选协议归因诊断报告

## 1. 实验动机

本实验回答一个单变量问题：E5A 的关系学习和 memory 预测是否受“相同星期、相同时间槽”候选池过小的限制。

固定 E5A pretrained/random checkpoint、Bank、OffsetDecay、Top-5、validation split 和所有模型参数，只改变候选事件协议。query true future 只在候选排序完成后用于 Spearman、Recall@5、MAE、RMSE 和 MAPE，不参与候选生成或 key 排序。

## 2. 三种候选协议

### 2.1 ExactCalendar

`ExactCalendar`（精确日历候选）是当前部署基线。候选必须与 query 具有相同星期和相同 5 分钟时间槽，并满足：

\[
future\_end_j<context\_start_q.
\]

输入是 query 的星期、时间槽、context 起点和 Bank 元数据；输出因果历史事件 id。它不读取 query future。

### 2.2 RelaxedCalendar

`RelaxedCalendar`（放宽日历候选）保持星期相同，但允许候选时间槽位于 query 的前后一个 5 分钟槽，即 `slot - 1`、`slot`、`slot + 1`。候选仍满足严格因果条件。输入输出与 ExactCalendar 相同，不读取 query future。

### 2.3 BroadCausal

`BroadCausal`（宽因果候选）不限制星期和时间槽，只要求候选 future 在 query context 之前完全发生。完整 Bank 有 15,876 个合法事件，无法直接展开；因此按 Bank 时间轴均匀抽取最多 32 个事件。该抽样不依赖 pretrained/random key，两者使用完全相同的事件 id。BroadCausal 只作为诊断上限，不直接视为部署方法。

### 2.4 Top-K Saturation

`Top-K Saturation`（Top-K 饱和度）衡量 Top-5 占候选池的比例：

\[
S_q=\frac{\min(R_q,5)}{R_q},
\]

其中 \(R_q\) 是 query 的合法候选数。\(S_q\) 越接近 1，Top-5 越接近使用全部候选，selector 的筛选空间越小。它只使用候选数量，不读取 future。

## 3. 指标

- `Spearman`：KeyDistance 排名与 ODSignature future distance 排名的相关系数；越高表示 key-future 全局秩关系越一致。
- `Future-Neighbor Recall@5`：key 最近 5 个候选与 future 最近 5 个候选的交集比例。候选池大小变化时绝对值不可直接横向比较，重点看同一协议内 pretrained-random 差值。
- `MAE`：OffsetDecay memory 与真实 future 的平均绝对误差，单位为交通速度。
- `RMSE`：均方根误差，对大误差更敏感，单位为交通速度。
- `MAPE`：平均绝对百分比误差，单位为百分比。

## 4. 结果

| 协议 | 候选数均值 | Top-K 饱和度 | Spearman P/R | Spearman 增益 | Recall@5 P/R | MAE P/R | RMSE P/R | MAPE P/R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ExactCalendar | 8.014 | 0.631 | 0.318104 / 0.160026 | +0.158078 | 0.737649 / 0.697929 | 3.527230 / 3.667355 | 6.604164 / 6.786500 | 9.331257 / 9.723048 |
| RelaxedCalendar | 23.985 | 0.211 | 0.328968 / 0.135925 | +0.193043 | 0.330802 / 0.261578 | 3.472132 / 3.650739 | 6.713107 / 6.840967 | 9.285972 / 9.692777 |
| BroadCausal | 32.000 | 0.156 | 0.674464 / 0.139631 | +0.534833 | 0.387089 / 0.218435 | 3.512407 / 4.787420 | 6.678037 / 8.365744 | 9.818750 / 13.001227 |

P/R 表示 pretrained/random。三组均使用完整 2,993 个 validation query、94 个 batch。

## 5. 信号解释

1. **ExactCalendar 确实限制了筛选空间。** Top-5 平均占候选池 63.1%，而 RelaxedCalendar 只有 21.1%。扩大到相邻时间槽后，pretrained Spearman、Spearman 增益和 MAE/MAPE 都改善。
2. **宽候选显著暴露了 E5A 的关系能力。** BroadCausal 下 pretrained Spearman 达到 0.674464，而 random 只有 0.139631，说明 E5A 并非没有学到关系；严格日历池隐藏了一部分表示能力。
3. **关系更强不等于物理预测一定最好。** BroadCausal 的 pretrained Spearman 最高，但 MAE 3.512407 略差于 RelaxedCalendar 的 3.472132，MAPE 也差于 ExactCalendar。这说明宽候选包含日历或运行状态不匹配事件，ODSignature 排序能力虽强，Top-5 payload 仍可能存在分布偏移。
4. **Recall@5 绝对值下降不代表退化。** Exact 候选平均只有 8 个，随机重合概率天然较高；Relaxed/Broad 候选更多，Recall@5 任务更难。应比较同一协议内差值：Exact +0.039720、Relaxed +0.069224、Broad +0.168654。

## 6. 决策

- `Keep ExactCalendar`：保留为原始部署基线。
- `Keep RelaxedCalendar`：作为下一阶段最小部署候选。它不增加模型参数，只把同星期的时间容差扩大到前后 5 分钟，并取得最低 pretrained MAE 和 MAPE。
- `Diagnostic only BroadCausal`：保留为表示能力上限和候选瓶颈证据，不直接用于部署。其 random 预测明显恶化，表明完全移除日历约束引入了大量无关事件。
- `Stop architecture expansion`：本轮不引入趋势分解、confidence 或新 backbone。下一步先验证 RelaxedCalendar 在下游融合中的 MAE/RMSE/MAPE，以及多 seed 稳定性。

## 7. 产物

- ExactCalendar：`artifacts/convergence/visualization/e5a/`
- RelaxedCalendar：`artifacts/convergence/candidate_protocol/e5a_relaxed_calendar/`
- BroadCausal：`artifacts/convergence/candidate_protocol/e5a_broad_causal/`
- 实现入口：`scripts/visualize_retrieval.py --candidate-protocol ...`

两个新分支目录均包含 `metrics.json`、`cases.json`、`alignment_bins.csv` 和三张非空 PNG，错误日志为空。
