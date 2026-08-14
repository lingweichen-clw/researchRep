# E3 Future-Relation 软对比预训练实验步骤

## 1. 实验目的与边界

本实验只替换预训练阶段的检索监督目标，验证 Future-Relation 软对比是否能让历史 context 的节点 key 更符合未来轨迹相似性。

固定不变的内容：METR-LA 数据划分、288 步 retrieval context、12 步 forecast context、12 步 future、patch size=12、96 维 encoder、3 层时空编码器、稀疏图注意力、掩码重建、batch size=16、seed=42、Bank 候选过滤和检索诊断协议。

唯一变化是：

- E2：按 context/future 分位数构建正样本和海市蜃楼 hard negative；
- E3：对每个节点学习未来距离诱导的完整候选排序分布。

因此，E3 的结果只能与 E2 day-96 基线在相同 validation 检索协议下比较；不能用 test 结果选 checkpoint 或调温度。

## 2. 环境与数据检查

在 PowerShell 中执行：

```powershell
conda activate research
cd D:\projects\researchProjects\TrafficRobustST\STAnchor-BlockMemory

python -c "from stanchor.config import load_config; c=load_config('configs/metrla_e3_relation_v1.yaml'); print(c.pretrain.retrieval_loss_mode, c.data.encoder_context_length, c.model.hidden_dim)"
```

预期输出包含：`relation 288 96`。该命令只校验 YAML 和配置约束，不创建日志或模型产物。

需要的 Python 包与 E2 相同：`torch`、`numpy`、`pandas`、`h5py`、`PyYAML`。本次 E3 没有新增第三方依赖。

## 3. E3 中的新增日志

每个 epoch 会打印并写入 `pretrain_metrics.jsonl`：

- `val_retrieval`：验证集上的 \(\mathcal L_{\mathrm{rel}}\)，即 future teacher 分布与 key student 分布之间的交叉熵；越低表示 batch 内的 key 排名越接近 future 排名，但它不是最终检索 MAE。
- `val_anchors`：满足至少两个合法历史候选的 query-node 数。一个候选无法形成排序，因此被排除。
- `val_relation_candidates`：这些有效 query-node 对应的合法 query-candidate 节点对数。
- `val_teacher_keff`：未来 teacher 分布的有效支持数，\(K_{\mathrm{eff}}=1/\sum_j q_j^2\)。接近 1 表示 teacher 接近单一候选，较大表示多个未来模式都相近。
- `val_student_keff`：key 相似度 softmax 后的有效支持数，公式同上，但将 \(q_j\) 换为 student 概率 \(p_j\)。
- `val_positive_pairs`、`val_hard_negatives`：E3 中固定为 0，因为 E3 不再构造二值正样本或 hard negative；这不是训练异常。

若 `val_anchors` 长期接近 0，或 `val_teacher_keff`/`val_student_keff` 出现 NaN/Inf，应停止正式训练并检查时间边界、future 缺失掩码和 batch 排列，而不是直接调整模型结构。

## 4. 三轮 Pilot

Pilot 只检查张量流、梯度、候选支持数和损失趋势，不用于对外汇报或与 E2 做结论性比较。它使用独立目录，避免污染正式 E3 运行目录。

```powershell
python scripts/pretrain.py `
  --config configs/metrla_e3_relation_v1.yaml `
  --epochs 3 `
  --run-name metrla_e3_relation_pilot_seed42
```

检查 `artifacts/metrla_e3_relation_pilot_seed42/pretrain.log`：

1. `retrieval_loss=relation`；
2. `val_anchors > 0`；
3. `val_relation_candidates > 0`；
4. `val_teacher_keff` 与 `val_student_keff` 均为有限正数；
5. 没有 `NaN`、`Inf`、`skipped(train/val)` 持续接近全部 batch 的情况；
6. 同时生成 `pretrain_best.pt` 和 `pretrain_best_relation.pt`。

记录上述检查结果后，删除 `artifacts/metrla_e3_relation_pilot_seed42`。Pilot 仅是工程门槛，不保留为正式实验日志。

## 5. 正式 E3 预训练

Pilot 通过后执行一次完整单 seed 训练：

```powershell
python scripts/pretrain.py --config configs/metrla_e3_relation_v1.yaml
```

正式产物目录：

```text
artifacts/metrla_e3_relation_seed42/
  pretrain.log
  pretrain_metrics.jsonl
  pretrain_best.pt
  pretrain_best_relation.pt
```

两个 checkpoint 的含义：

- `pretrain_best.pt`：验证总损失 \(\mathcal L_{\mathrm{mask}}+0.1\mathcal L_{\mathrm{rel}}\) 最低；
- `pretrain_best_relation.pt`：验证 `val_retrieval`，即 \(\mathcal L_{\mathrm{rel}}\) 最低。

两者都只能进入下一步 validation 检索诊断，不能预先假定其中一个更好。

## 6. 为两个候选 checkpoint 分别建 Bank

```powershell
python scripts/build_bank.py `
  --config configs/metrla_e3_relation_v1.yaml `
  --checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best.pt `
  --output-dir artifacts/metrla_bank_e3_relation_total `
  --dataset-name METR-LA

python scripts/build_bank.py `
  --config configs/metrla_e3_relation_v1.yaml `
  --checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt `
  --output-dir artifacts/metrla_bank_e3_relation_relation `
  --dataset-name METR-LA
```

每个 Bank 仅包含训练段历史事件的 context key、历史 future、掩码、level 特征和时间元数据。future teacher 分布不写入 Bank，推理时不会读取待预测 future。

## 7. Validation 检索诊断与 checkpoint 选择

```powershell
python scripts/diagnose_retrieval.py `
  --config configs/metrla_e3_relation_v1.yaml `
  --checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best.pt `
  --bank artifacts/metrla_bank_e3_relation_total `
  --split val `
  --output artifacts/metrla_e3_relation_total_val_diagnostics.json

python scripts/diagnose_retrieval.py `
  --config configs/metrla_e3_relation_v1.yaml `
  --checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e3_relation_relation `
  --split val `
  --output artifacts/metrla_e3_relation_relation_val_diagnostics.json
```

从两个 validation JSON 中选择一个 checkpoint/Bank 对。选择依据依次为：

1. `learned_topk_future_mae` 越低越好；
2. `learned_top1_future_mae` 越低越好；
3. `future_ndcg_at_k` 越高越好；
4. 若前三项存在轻微冲突，优先保留 Top-K MAE 更低且 NDCG 更高的组合。

当前 E2 day-96 validation 参照值为：

| 指标 | E2 值 |
|---|---:|
| learned Top-1 future MAE | 4.2631 |
| learned weighted Top-K future MAE | 3.7600 |
| raw-L1 Top-K future MAE | 3.9892 |
| Top-1 Oracle gap | 1.4918 |

E3 明确保留的门槛：Top-1 或 weighted Top-K 至少一项优于 E2，另一项退化不超过 0.03，并且 Future NDCG@K 提升。若 weighted Top-K MAE 不低于 3.80，或 Top-1 MAE 不低于 4.31，或 key/teacher 分布坍缩，则删除 E3，不进入下游。

不允许使用 test split 选择这两个 checkpoint，也不允许根据下游 test 结果回头选择 Bank。

## 8. 条件性下游实验

只有第 7 节通过后，才对胜出的 checkpoint/Bank 运行下游。假设胜出的是 `pretrain_best_relation.pt`：

```powershell
python scripts/train_downstream.py `
  --config configs/metrla_e3_relation_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e3_relation_relation `
  --mode learned_topk_horizon `
  --seed 42 `
  --run-name metrla_e3_relation_horizon_seed42
```

再使用相同 checkpoint/Bank 训练置信度与融合：

```powershell
python scripts/train_downstream.py `
  --config configs/metrla_e3_relation_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e3_relation_relation `
  --mode learned_topk_confidence `
  --seed 42 `
  --run-name metrla_e3_relation_confidence_seed42
```

若胜出的是 `pretrain_best.pt`，只把上述命令中的 checkpoint、Bank 和 `run-name` 对应改为 `total`。不要将两个 checkpoint 的 encoder、Bank 或下游 checkpoint 混用。

## 9. 正式结论的最小证据

E3 首轮完成后，应保存：正式 `pretrain.log`、`pretrain_metrics.jsonl`、两个 validation retrieval diagnostics JSON、选择理由，以及通过门槛后对应的下游日志和 15/30/60 分钟 MAE、RMSE、MAPE。

单 seed 只能作为机制筛选证据。只有 E3 在 validation 检索门槛和一条下游路径均通过后，才值得补三 seed，而不是在失败方案上扩展实验数量。
