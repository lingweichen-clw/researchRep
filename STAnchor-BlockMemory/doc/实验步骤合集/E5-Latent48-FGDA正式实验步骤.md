# E5 Latent48 + FGDA 正式实验步骤

> 本文件是当前版本的唯一正式执行顺序。`CFDP profile` 仅保留为历史诊断，不在本轮主线训练；已经停止的 profile 下游半程目录不删除。

## 1. 名词与实验边界

**Latent48**：纯 48 维历史 latent key。输入是历史窗口，输出是 `[B,N,48]` 的 L2 归一化 key；`profile_dim=0`，不构造 CFDP profile，也不使用 profile loss。

**FGDA**：`Future-Guided Dynamics Adapter`（未来关系引导的动态适配器）。它只从历史的一阶差分和静态图提取小残差：`Local-FGDA` 使用本节点差分，`LocalGraph-FGDA` 额外聚合有效非自环邻居差分。future 只在 source-train 的 `OffsetDecay relation teacher` 中提供梯度，部署检索不读取 query future。

**OffsetDecay relation teacher**：训练期用历史 endpoint 对候选 future 做 horizon 衰减的 level 对齐，再把 pairwise future 距离转成 teacher 分布；它只读 source-train future，输出给 relation loss。部署时不运行 teacher。

**Bank**：由训练历史事件构成的只读 memory。Bank 保存历史事件的 48 维 key、raw future payload、endpoint/level 和时间索引。query 只能用当前 history 生成 key，在因果候选集合中排序；OffsetDecay 只对已经发生的 Bank raw future 做 query endpoint 对齐。

**no-confidence 下游**：`learned_topk_offset_decay_horizon`。下游 backbone 预测 base future，检索 memory 用 OffsetDecay 生成修正 future，但不训练 confidence/risk 融合头，用于单独判断检索器价值。

**ErrorAware 校准**：`learned_topk_error_aware`。在 no-confidence memory 之上，用 history 和 detached base prediction 估计基础误差，再学习一个沿 `memory-base` 方向的可解释 residual blend；真实 future 只在训练标签和离线诊断中使用。

## 2. 严格顺序

每一阶段都有明确决策；未达到决策条件时停止，不进入后续昂贵实验。

### R0：本机契约验证

在当前仓库目录执行：

```powershell
$py = (Get-Command python).Source
& $py -m unittest discover -s tests -v
& $py -m compileall -q stanchor scripts tests
```

预期：`170` 项测试全部通过，compileall 无输出。失败时只修复接口，不启动实验机训练。

### R1：实验机训练三个 288 步 encoder

实验机必须使用包含真实 METR-LA 数据、图文件和 `research` 环境的仓库。以下三条命令按顺序执行，每条都使用新的 run name，不覆盖旧目录：

```powershell
$py = (Get-Command python).Source

& $py scripts/pretrain.py `
  --config configs/metrla_e5_final_latent48_global288_v1.yaml `
  --run-name metrla_e5_final_latent48_global288_seed42

& $py scripts/pretrain.py `
  --config configs/metrla_e5_final_latent48_local_fgda_global288_v1.yaml `
  --run-name metrla_e5_final_latent48_local_fgda_global288_seed42

& $py scripts/pretrain.py `
  --config configs/metrla_e5_final_latent48_local_graph_fgda_global288_v1.yaml `
  --run-name metrla_e5_final_latent48_local_graph_fgda_global288_seed42
```

每个 run 必须保存 `pretrain_best_relation.pt` 和 `pretrain.log`。日志中应出现 `adapter=none/local/local_graph`、`val_adapter_valid`、`val_adapter_gate`、`val_adapter_spatial_gate` 和 `val_adapter_ratio`；这些字段分别表示动态有效位置比例、融合门控均值、图门控均值和 `||gR||/||Z||` 相对贡献。

### R2：构建三个同协议 Bank

每条命令只读取已经完成的 relation checkpoint。`--dataset-name` 只写 Bank 元数据，不改变候选协议。

```powershell
& $py scripts/build_bank.py `
  --config configs/metrla_e5_final_latent48_global288_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_latent48_global288_seed42/pretrain_best_relation.pt `
  --dataset-name METR-LA `
  --output-dir artifacts/metrla_bank_e5_final_latent48_global288_seed42

& $py scripts/build_bank.py `
  --config configs/metrla_e5_final_latent48_local_fgda_global288_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_latent48_local_fgda_global288_seed42/pretrain_best_relation.pt `
  --dataset-name METR-LA `
  --output-dir artifacts/metrla_bank_e5_final_latent48_local_fgda_global288_seed42

& $py scripts/build_bank.py `
  --config configs/metrla_e5_final_latent48_local_graph_fgda_global288_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_latent48_local_graph_fgda_global288_seed42/pretrain_best_relation.pt `
  --dataset-name METR-LA `
  --output-dir artifacts/metrla_bank_e5_final_latent48_local_graph_fgda_global288_seed42
```

验收：三个 manifest 的 `retrieval_dim=48`、`profile_dim=0`、`latent_dim=0`，Bank 保存的是 raw future payload，而不是 profile 或 offset payload。

### R3：固定 Bank 的检索诊断

`diagnose_retrieval.py` 只用 validation history 和 Bank 做部署式检索；validation future 只作为离线 MAE 标签，不参与 key 或候选排序。

```powershell
& $py scripts/diagnose_retrieval.py `
  --config configs/metrla_e5_final_latent48_global288_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_latent48_global288_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_final_latent48_global288_seed42 `
  --split val `
  --output artifacts/metrla_e5_final_latent48_global288_seed42/retrieval_val.json

& $py scripts/diagnose_retrieval.py `
  --config configs/metrla_e5_final_latent48_local_fgda_global288_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_latent48_local_fgda_global288_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_final_latent48_local_fgda_global288_seed42 `
  --split val `
  --output artifacts/metrla_e5_final_latent48_local_fgda_global288_seed42/retrieval_val.json

& $py scripts/diagnose_retrieval.py `
  --config configs/metrla_e5_final_latent48_local_graph_fgda_global288_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_latent48_local_graph_fgda_global288_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_final_latent48_local_graph_fgda_global288_seed42 `
  --split val `
  --output artifacts/metrla_e5_final_latent48_local_graph_fgda_global288_seed42/retrieval_val.json
```

重点记录：OD relation Spearman、Recall@5、Top-5 memory MAE/RMSE/MAPE、horizon-wise MAE、有效候选数和 adapter contribution。LocalGraph 还要记录图门控均值，判断图分支是否被使用。

### R4：三个版本的 no-confidence 下游门槛

先只测 future-guided retrieval 是否有效，不加入 ErrorAware 校准。三条命令的 `--mode learned_topk_offset_decay_horizon` 会覆盖配置中的下游模式；`--level-weight 0` 固定只评估 key/relation，不让 level reranking 混入归因。

```powershell
& $py scripts/train_downstream.py `
  --config configs/metrla_e5_final_latent48_global288_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e5_final_latent48_global288_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_final_latent48_global288_seed42 `
  --mode learned_topk_offset_decay_horizon `
  --level-weight 0 `
  --run-name metrla_e5_final_latent48_global288_downstream_seed42

& $py scripts/train_downstream.py `
  --config configs/metrla_e5_final_latent48_local_fgda_global288_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e5_final_latent48_local_fgda_global288_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_final_latent48_local_fgda_global288_seed42 `
  --mode learned_topk_offset_decay_horizon `
  --level-weight 0 `
  --run-name metrla_e5_final_latent48_local_fgda_global288_downstream_seed42

& $py scripts/train_downstream.py `
  --config configs/metrla_e5_final_latent48_local_graph_fgda_global288_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e5_final_latent48_local_graph_fgda_global288_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_final_latent48_local_graph_fgda_global288_seed42 `
  --mode learned_topk_offset_decay_horizon `
  --level-weight 0 `
  --run-name metrla_e5_final_latent48_local_graph_fgda_global288_downstream_seed42
```

R4 的决策：先比较三者的 no-confidence MAE/RMSE/MAPE 和 horizon-wise MAE，再按优化方案中的门槛选择 `Latent48`、`Local-FGDA` 或 `LocalGraph-FGDA` 进入 R5。若两个 FGDA 版本都不能超过纯 Latent48 和已有 seed 波动，删除 FGDA，不继续加模块。

### R5：只对 R4 胜者训练 ErrorAware 校准

将 `<WINNER>` 替换为 R4 胜者对应的三个字符串之一：

- `latent48_global288`
- `latent48_local_fgda_global288`
- `latent48_local_graph_fgda_global288`

以下命令使用同一个纯 Latent48 配置族，`--mode learned_topk_error_aware` 开启 ErrorAware 校准；`--base-warmup-epochs 5` 和 `--calibrator-warmup-epochs 5` 分别训练基础 backbone 和校准器 warm-up。校准器不改变 Bank 和 retrieval checkpoint。

```powershell
& $py scripts/train_downstream.py `
  --config configs/metrla_e5_final_<WINNER>_v1.yaml `
  --pretrained-checkpoint artifacts/<WINNER>_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_final_<WINNER>_seed42 `
  --mode learned_topk_error_aware `
  --level-weight 0 `
  --base-warmup-epochs 5 `
  --calibrator-warmup-epochs 5 `
  --run-name metrla_e5_final_<WINNER>_error_aware_seed42
```

比较 R4 的 no-confidence 结果、旧 `learned_topk_confidence`（只作为已有实现对照）和 ErrorAware 的 MAE/RMSE/MAPE、horizon-wise MAE、risk MAE/Spearman、blend target MAE 及 memory coverage。ErrorAware 若不优于 no-confidence 或旧 confidence，停止校准结构扩展，保留简单版本。

### R6：随机初始化归因

只对 R4/R5 胜者做随机对照，避免在三个未通过版本上重复昂贵实验。随机 checkpoint 不训练任何参数：

```powershell
& $py scripts/init_random_checkpoint.py `
  --config configs/metrla_e5_final_<WINNER>_v1.yaml `
  --output artifacts/metrla_e5_final_<WINNER>_random_seed42/checkpoint.pt `
  --seed 42
```

然后用该 checkpoint 构建同样大小的 Bank，重复 R3 和 R4 的诊断。只有 pretrained 相对 random 同时改善 relation/MAE 和下游指标，才能把收益归因于预训练，而不是星期-时间槽候选协议。

### R7：多 seed、可视化与跨数据集

R4/R5 通过后才运行 seed `2024/2025`、E2/E3/E5 的简单 key-distance/未来 MAE 可视化，以及 PEMS-BAY 的 target-local Bank transfer。失败时不增加新分支；先回到 R4 的 keep/remove 决策。

## 3. 不允许的操作

- 不恢复 profile head 作为主线，不把 CFDP 预测值写入 Bank key。
- 不用 query future 生成 key、候选集合或部署 confidence。
- 不在 R4 前启动 ErrorAware、PIR 复杂校准或多特征分解。
- 不覆盖已有 `artifacts` 目录，不删除 profile 半程日志；smoke/debug 产物不得写入正式结果表。

