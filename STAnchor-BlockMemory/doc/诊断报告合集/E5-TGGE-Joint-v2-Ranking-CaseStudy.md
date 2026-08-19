# E5 TGGE Joint v2 检索排序 Case Study

## 1. 结论先行

本报告分析当前已经完成的 26 轮 **Joint v2** 预训练。Joint v2 使用
`Future-Relation loss` 与 masked reconstruction 的联合目标，其中关系损失
权重为 `0.1`；它不是尚未完成的 Relation-only 实验。

结论分成两个层面：

1. 在与预训练目标一致的 `pretrain_broad_causal` 候选协议下，Joint v2
   相对同配置随机编码器明显学到了 future relation。Anchor-wise Spearman 为
   `0.5687`，随机对照为 `0.1412`；Anchor-wise Kendall 为 `0.4275`，随机
   对照为 `0.0984`；NDCG@5 为 `0.4406`，随机对照为 `0.2644`。
2. 与旧的纯 Attention Latent48 v1 在同一 broad-causal 协议比较时，TGGE
   不是排序指标的明确赢家，而是以 `303,727` 个参数近似保留了 v1 的能力。
   v1 的 Spearman/Kendall 略高，TGGE 的 Recall@1/NDCG@5 略高，Memory MAE
   两者接近。TGGE 的参数量比 v1 的 `391,836` 少约 `22.5%`，但当前路由实现
   仍使每轮训练更慢，因此不能把参数减少表述成速度提升。

因此当前论文表述应是：**TGGE 是一个低参数、future-relation 对齐能力接近
Attention v1 的编码器候选；它尚未证明在排序或下游预测上显著优于 Attention
v1 或 CC-FGDA。** Relation-only 完成后需要单独报告，不能用本报告的 Joint
结果替代。

## 2. 实验对象与边界

### 2.1 Joint v2

Joint v2 的 checkpoint 为：

`artifacts/metrla_e5_tgge_latent48_v2_seed42/pretrain_best.pt`

该 checkpoint 在第 26 轮保存，当前记录为：

| 项目 | 数值 |
|---|---:|
| 总参数量 | 303,727 |
| Encoder 参数量 | 260,467 |
| Route attention 参数量 | 25,827 |
| Retrieval head 参数量 | 16,928 |
| Reconstruction head 参数量 | 972 |
| Epoch 26 validation total | 0.439572 |
| Epoch 26 validation reconstruction | 0.222973 |
| Epoch 26 validation relation | 2.165991 |

训练日志显示，Joint v2 每轮大约需要 `18.9` 分钟训练和 `1.7` 分钟验证，
合计约 `20.6` 分钟。验证每轮执行一次时，验证约占总时间的 `8%`；因此把
验证改为每两轮一次可以节省约 `4%` 的平均墙钟时间，但不会改变训练目标。

### 2.2 Anchor 与排序输入

一个 **anchor** 固定为一个 query-node 对 `(q,n)`。该 anchor 有一个候选
集合 `C(q,n)`，候选数量由协议决定。对每个候选，key distance 越小表示
编码器认为越相似；OffsetDecay teacher distance 越小表示未来轨迹越相似。

所有正式排名指标只在同一个 anchor 的候选内部计算，不把不同 query 或不同
节点的距离展平后混合。

真实 query future 只用于离线 teacher distance、Memory 误差、确定性案例选择
和图像绘制。它不进入 query key、候选构造、Top-R/Top-K 选择或 Memory 聚合，
因此不构成推理阶段 future leakage。

## 3. 候选协议

### 3.1 `pretrain_broad_causal`（主协议）

该协议复现关系预训练的候选边界：候选事件的 future 必须在 query context
开始之前结束，但不施加 weekday/slot 日历过滤。每个 query 使用相同的、与
模型无关的因果事件轴，并按事件时间做确定性采样，最多保留 `event_top_r=32`
个事件。

完整 validation 中候选数量均值为 `31.46`，范围为 `22--32`。因此 Recall@1
的随机期望约为 `0.0318`，Recall@5 的随机期望约为 `0.1592`。这两个期望值
来自真实候选数量，不使用旧的约 8 候选池假设。

### 3.2 `exact_calendar`（部署侧协议）

该协议保留同 weekday、同 slot 和严格因果约束，候选数量均值为 `8.014`，
范围为 `5--9`。它用于检查 future relation 在部署日历筛选后是否仍然有用，
但不能单独作为预训练价值的证明，因为预训练 relation teacher 并没有使用
这个窄日历集合构造所有 pair。

## 4. 指标定义

- **Anchor-wise Spearman**：对每个 `(q,n)` 分别计算 key ranking 与 teacher
  ranking 的 Spearman 相关，再对 eligible anchors 求平均；它衡量局部完整
  名次差异。
- **Anchor-wise Kendall**：在同一 anchor 内逐对比较候选顺序，计算
  `(concordant-discordant)/(strict pairs)`；并列对不进入分母，便于解释为
  两两顺序的一致程度。
- **Recall@1**：key 排名第一名是否与 teacher 第一名相同。候选数为 `M` 时，
  未学习的随机期望约为 `1/M`。
- **NDCG@5**：使用 `exp(-d_OD/tau)` 作为 teacher graded relevance，按 key
  排名计算前五名的折损累计增益，再除以 teacher 理想排序的增益。它同时考虑
  Top-5 内部顺序和候选重要性。
- **Recall@5**：只作为辅助集合覆盖率。它不区分 Top-5 内部顺序，不能单独
  支撑“第一候选选对”的结论。
- **Memory MAE**：将检索候选的 OffsetDecay payload 聚合为未来预测后，在
  物理速度单位上计算平均绝对误差。它是最终 payload 效果，不等价于排序指标。

## 5. 主协议结果：pretrain_broad_causal

### 5.1 Joint v2 与 matched random

| 指标 | Joint v2 | Matched random | 差值 |
|---|---:|---:|---:|
| Anchor-wise Spearman | 0.5687 | 0.1412 | +0.4275 |
| Anchor-wise Kendall | 0.4275 | 0.0984 | +0.3291 |
| Recall@1 | 0.1190 | 0.0558 | +0.0632 |
| NDCG@5 | 0.4406 | 0.2644 | +0.1762 |
| Recall@5（辅助） | 0.3855 | 0.2181 | +0.1674 |
| Memory MAE | 3.5718 | 4.8051 | -1.2333 |

Joint v2 相对随机控制在四个主排序指标上均有提升，且 Memory MAE 同方向
下降。这支持“Joint v2 的 key 空间包含可检测的 future relation”这一有限
机制结论，但仍然是单 seed validation 结果，不能写成统计显著性结论。

### 5.2 与 Attention v1 的同协议比较

Attention v1 使用旧的纯 Attention Latent48 encoder、相同的 relation teacher、
相同的 `event_top_r=32` broad-causal 候选和相同的随机初始化协议。

| 模型 | 参数量 | Anchor Spearman | Kendall | Recall@1 | NDCG@5 | Memory MAE |
|---|---:|---:|---:|---:|---:|---:|
| Attention v1 | 391,836 | 0.5729 | 0.4314 | 0.1177 | 0.4375 | 3.5563 |
| TGGE Joint v2 | 303,727 | 0.5687 | 0.4275 | 0.1190 | 0.4406 | 3.5718 |

相对于各自 matched random 的增益为：

| 模型 | Spearman 增益 | Kendall 增益 | Recall@1 增益 | NDCG@5 增益 |
|---|---:|---:|---:|---:|
| Attention v1 | +0.4419 | +0.3398 | +0.0629 | +0.1722 |
| TGGE Joint v2 | +0.4275 | +0.3291 | +0.0632 | +0.1762 |

两者差距很小，没有证据证明 TGGE 在 broad-causal 排序上超过 Attention v1。
TGGE 的实际贡献是用约 `22.5%` 更少参数保留了相近的局部排序和 payload 误差。
但 Attention v1 单轮训练约 `10.8` 分钟，TGGE Joint v2 约 `20.6` 分钟；当前
实现中的稀疏 route 仍有明显 Python/候选处理开销，需单独向量化后再讨论效率。

## 6. 部署侧结果：exact_calendar

TGGE Joint v2 在约 8 个候选的部署侧协议上得到：

| 指标 | TGGE Joint v2 | Matched random |
|---|---:|---:|
| Anchor-wise Spearman | 0.3041 | 0.1877 |
| Anchor-wise Kendall | 0.2372 | 0.1444 |
| Recall@1 | 22.09% | 18.54% |
| NDCG@5 | 0.6246 | 0.5720 |
| Recall@5（辅助） | 0.7302 | 0.6957 |
| Memory MAE | 3.5407 | 3.6735 |

与你之前的同协议汇总相比：

| 模型 | Anchor Spearman | Kendall tau-b | Recall@1 | NDCG@5 |
|---|---:|---:|---:|---:|
| Latent48 | 0.3056 | 0.2386 | 22.22% | 0.6265 |
| CC-FGDA | **0.3195** | **0.2497** | **22.67%** | **0.6326** |
| Matched random | 0.1901 | 0.1465 | 18.75% | 0.5738 |
| TGGE Joint v2 | 0.3041 | 0.2372 | 22.09% | 0.6246 |

这张表的方向很清楚：TGGE Joint v2 与纯 Latent48 基本持平，但当前没有
超过 CC-FGDA。由于随机控制和候选池实现存在微小运行版本差异，表中的旧汇总
与本次 exact-calendar 完整运行应分别标注来源，不能把小数点后三位的差异写成
显著提升。

## 7. 可视化证据

### 7.1 Joint v2 训练曲线

![Joint v2 training history](../../artifacts/convergence/visualization/tgge_joint/joint_v2_training_history.png)

该图显示 epoch 1--26 的 train/validation total、reconstruction/relation
分量和 teacher/student effective support。第 18 轮以后 validation total 与
relation loss 的下降已经很小，说明继续训练可能进入平台期；Relation-only
尚未完成，不能据此判断 mask reconstruction 是否应被删除。

### 7.2 预训练对齐主协议

![Broad causal key future alignment](../../artifacts/convergence/visualization/tgge_joint/pretrain_broad_causal/key_future_alignment.png)

![Broad causal local ranking](../../artifacts/convergence/visualization/tgge_joint/pretrain_broad_causal/ranking_metrics.png)

![Broad causal deterministic cases](../../artifacts/convergence/visualization/tgge_joint/pretrain_broad_causal/deterministic_top5_cases.png)

主协议的 decile 曲线从近 key 到远 key 总体上升，说明 key distance 与
OffsetDecay future distance 具有方向一致性。案例图包含 strong-win、
representative 和 failure 三类固定分位点，不是手工挑选成功样本。

### 7.3 部署侧协议

![Exact calendar key future alignment](../../artifacts/convergence/visualization/tgge_joint/exact_calendar/key_future_alignment.png)

![Exact calendar local ranking](../../artifacts/convergence/visualization/tgge_joint/exact_calendar/ranking_metrics.png)

![Exact calendar deterministic cases](../../artifacts/convergence/visualization/tgge_joint/exact_calendar/deterministic_top5_cases.png)

部署侧候选池较小，Recall@5 的绝对值较高是候选数量造成的，必须结合
Anchor-wise Kendall、NDCG@5 和 Memory MAE 解读。

## 8. 复现产物

- Joint v2 broad-causal：`artifacts/convergence/visualization/tgge_joint/pretrain_broad_causal/`
- Joint v2 exact-calendar：`artifacts/convergence/visualization/tgge_joint/exact_calendar/`
- Attention v1 broad-causal：`artifacts/convergence/visualization/attention_v1/pretrain_broad_causal/`
- Joint v2 checkpoint：`artifacts/metrla_e5_tgge_latent48_v2_seed42/pretrain_best.pt`
- Joint v2 random checkpoint：`artifacts/metrla_e5_tgge_latent48_v2_random_seed42.pt`
- Anchor-wise 指标实现：[retrieval_visualization.py](../../stanchor/diagnostics/retrieval_visualization.py)
- 训练曲线实现：[pretraining_curves.py](../../stanchor/diagnostics/pretraining_curves.py)

每个完整诊断目录包含 `metrics.json`、`cases.json`、`alignment_bins.csv`、
`ranking_metrics.csv` 和非空 PNG。所有运行均为完整 validation，没有使用
`--max-batches`，并在 `metrics.json` 中保存了 future-information boundary。

## 9. Keep / Remove / Next

- **Keep**：TGGE 作为低参数 Joint encoder 候选；保留 broad-causal 主协议和
  exact-calendar 部署侧协议双重报告。
- **Do not claim**：当前不能声称 TGGE 超过 Attention v1 或 CC-FGDA；只能声称
  在排序和 Memory MAE 上近似保持，同时减少约 `22.5%` 参数。
- **Next**：完成 Relation-only checkpoint 后，在两个协议下复用相同的随机控制、
  Anchor-wise Spearman/Kendall、Recall@1、NDCG@5 和 Memory MAE；该实验决定
  mask reconstruction 是保留、删除还是仅作辅助正则。
- **Efficiency**：把 route candidate 选择和 Bank candidate gather 向量化，
  然后重新测每轮时间；目前参数减少没有带来训练加速。
- **Reliability**：Relation-only 结论确定后，再补至少 3 个 seed；在此之前不
  扩大“模型无关泛化”论文主张。
