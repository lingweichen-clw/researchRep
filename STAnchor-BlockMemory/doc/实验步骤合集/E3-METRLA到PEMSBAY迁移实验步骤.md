# E3 METR-LA 到 PEMS-BAY 迁移实验步骤

## 1. 唯一实验问题

本阶段只回答：

> 在 METR-LA 上预训练并冻结的 E3 encoder-selector，能否在 PEMS-BAY 重建目标 Bank 后，稳定改善只使用最近 12 步的轻量预测 backbone？

本阶段不加入 adapter、不微调 encoder、不修改检索损失，也不研究反向迁移。

## 2. 固定协议

- 源预训练 checkpoint：`artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt`
- 目标数据：PEMS-BAY，速度变量，5 分钟采样；
- 目标配置：`configs/pemsbay_e3_transfer_v1.yaml`；
- 目标 Bank：`artifacts/pemsbay_bank_from_metrla_e3_relation`；
- Bank 事件数：25,327；
- 下游校准事件数：10,855；
- validation 事件数：4,912；
- test 事件数：10,125；
- seed：42、2024、2025；
- checkpoint 选择只依据 validation MAE；
- 所有模式和 seed 冻结前禁止读取 test。

## 3. 已完成的进入门槛

PEMS-BAY validation 检索诊断结果：

| 方法 | retrieved future MAE |
|---|---:|
| weekly mean | 2.681505 |
| raw-L1 Top-K | 2.228216 |
| learned weighted Top-K | **2.209414** |
| Oracle Top-1 | 1.434256 |

learned weighted Top-K 同时优于 weekly mean 和 raw-L1 Top-K，因此允许进入下游实验。该结论只说明检索出的历史 future 本身有价值，不等价于最终融合预测已经获益。

## 4. 三个下游模式

### 4.1 `base_only`

只使用最近 12 步输入训练轻量 backbone，不使用历史 Bank。它回答“短上下文本身能做到什么程度”。

### 4.2 `learned_topk_horizon`

使用冻结 E3 encoder-selector 检索并聚合 Top-K 历史 future，再通过 horizon-only fusion 与 backbone 预测融合。它与 `base_only` 的差异是是否加入检索 memory。

### 4.3 `learned_topk_confidence`

在相同 learned Top-K memory 上增加节点级、预测步级 confidence。它与 `learned_topk_horizon` 的差异只有 confidence 是否参与融合。

先在 seed 42 上运行三个模式。只有单 seed 方向符合预期，才扩展到 seed 2024 和 2025。

## 5. Seed 42 命令

从 `STAnchor-BlockMemory` 目录执行：

```powershell
python scripts/train_downstream.py `
  --config configs/pemsbay_e3_transfer_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt `
  --bank artifacts/pemsbay_bank_from_metrla_e3_relation `
  --mode base_only `
  --seed 42 `
  --run-name pemsbay_e3_base_only_seed42
```

```powershell
python scripts/train_downstream.py `
  --config configs/pemsbay_e3_transfer_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt `
  --bank artifacts/pemsbay_bank_from_metrla_e3_relation `
  --mode learned_topk_horizon `
  --seed 42 `
  --run-name pemsbay_e3_learned_topk_horizon_seed42
```

```powershell
python scripts/train_downstream.py `
  --config configs/pemsbay_e3_transfer_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt `
  --bank artifacts/pemsbay_bank_from_metrla_e3_relation `
  --mode learned_topk_confidence `
  --seed 42 `
  --run-name pemsbay_e3_learned_topk_confidence_seed42
```

## 6. Seed 42 决策

设三种模式的最优 validation MAE 分别为 \(M_{\mathrm{base}}\)、\(M_{\mathrm{mem}}\) 和 \(M_{\mathrm{conf}}\)。memory 相对改善为：

\[
G_{\mathrm{mem}}
=
\frac{M_{\mathrm{base}}-M_{\mathrm{mem}}}{M_{\mathrm{base}}}\times100\%.
\]

confidence 相对改善为：

\[
G_{\mathrm{conf}}
=
\frac{M_{\mathrm{mem}}-M_{\mathrm{conf}}}{M_{\mathrm{mem}}}\times100\%.
\]

决策规则：

- 若 \(M_{\mathrm{mem}} < M_{\mathrm{base}}\)，保留冻结迁移 memory，扩展三 seed；
- 若 \(M_{\mathrm{conf}} < M_{\mathrm{mem}}\)，保留 confidence，扩展三 seed；
- 若 memory 没有改善 backbone，停止扩展 seed，不新增模块，先检查融合是否忽略 memory；
- 若 confidence 没有改善 horizon-only，PEMS-BAY 主结果移除 confidence，不为它新增网络。

## 7. Target-random 对照

三模式 seed 42 通过后，再建立相同架构、相同随机种子但未经预训练的 frozen random encoder 和对应 PEMS-BAY Bank。比较 source-pretrained 与 target-random 的 `learned_topk_horizon`，以隔离“预训练表示”本身的贡献。

target-random 与 source-pretrained 必须保持以下变量一致：

- encoder 结构；
- retrieval head 结构；
- PEMS-BAY scaler 和图；
- Bank 时间范围；
- candidate pool、Top-R、Top-K 和温度；
- 下游 backbone、fusion、训练预算和 seed。

该对照所需的随机 checkpoint 生成命令在 seed 42 三模式通过后再加入，避免在主门槛失败时继续扩张实验。

## 8. 多随机种子和 test 边界

只有保留的模式运行 seed 42、2024、2025，并报告均值与样本标准差：

\[
\bar{x}=\frac{1}{3}\sum_{s=1}^{3}x_s,
\qquad
\operatorname{Std}(x)=
\sqrt{\frac{1}{2}\sum_{s=1}^{3}(x_s-\bar{x})^2}.
\]

三 seed validation 完成并冻结最终模式后，每个冻结 checkpoint 只允许执行一次 test。不得根据 test 更换模式、seed、checkpoint 或超参数。

## 9. 当前执行状态（2026-07-28）

已完成：

- PEMS-BAY 数据、图、时间轴和节点顺序审查；
- 正式 E3 目标 Bank 构建；
- validation 检索迁移诊断；
- seed 42 `base_only`；
- seed 42 `learned_topk_horizon`；
- horizon-only validation 分支诊断；
- seed 42 `learned_topk_confidence`；
- confidence validation 分支、校准与四分位诊断。

当前结果：

| 模式 | Validation MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| base-only | 2.171461 | 5.097139 | 5.232578 |
| learned Top-K + horizon fusion | 1.881370 | 4.107756 | 4.390352 |
| **learned Top-K + confidence** | **1.804609** | **4.037364** | **4.192822** |

horizon-only 相对 base-only 的 MAE 改善为 13.36%；confidence 又相对 horizon-only 改善 4.08%。confidence 的 AUROC 为 0.5977，AUPRC 为 0.5844，高于 0.4674 的正样本率基准；confidence 从最低到最高四分位时，memory gain 从 -0.9548 单调上升到 +1.2601。因此 seed 42 同时通过预测收益和机制语义门槛，暂时保留 confidence。

下一步不修改模型，只完成 target-random seed 42 对照，以隔离 METR-LA 预训练表示的贡献。详细执行与决策规则见 `doc/实验步骤合集/PEMS-BAY-E3迁移Confidence后续实验计划.md`。在 target-random 和多 seed validation 完成前，继续禁止读取 PEMS-BAY test。
