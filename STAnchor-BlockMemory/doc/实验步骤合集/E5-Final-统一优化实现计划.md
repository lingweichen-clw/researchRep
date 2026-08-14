# E5-Final 统一优化实现计划与下一步实验步骤

> 本计划对应《E5-Final：可泛化未来语义检索与风险感知校准统一优化方案》。所有正式训练前先完成 R0；R0 任一项失败时停止昂贵训练并修复接口。

## 目标与边界

实现三条可独立消融的机制：

1. `SymNormTeacher`：对 OffsetDecay future relation 做对称几何均值归一化，只在 source-train teacher 中读取 future。
2. `CanonicalFutureDynamicsProfile (CFDP)`：将 future 变成固定 12 点、事件自身尺度归一化的无量纲 profile，由 history-only head 预测并与 36 维 latent key 拼接为 48 维 key。
3. `ErrorAwareAdditiveFusion`：冻结检索器和 Bank，在下游用基础风险、候选共识和 memory-base 差异学习单一 residual blend 权重。

不修改图数据、切分、train-only scaler、Top-5、search temperature、raw future Bank payload 和 OffsetDecay 公式；不加入趋势分解、FFT/DWT、新 backbone 或 PIR Local Revision Transformer。

## 执行顺序

### R0：契约测试与泄漏检查

**文件：**

- Create: `tests/test_e5_final_contracts.py`
- Create: `stanchor/retrieval/semantic_profile.py`
- Modify: `stanchor/losses/pretraining.py`
- Modify: `stanchor/bank/schema.py`
- Modify: `stanchor/bank/builder.py`
- Modify: `stanchor/bank/storage.py`

**步骤：**

- [x] 已测试：H=12 时 profile resampling 恒等；平移/正比例缩放保持 profile；SymNorm 输出严格对称；profile/latent/total key 为 12/36/48 且 cosine 可拆解；Bank schema v1/v2 互拒；缺少 query future 的推理接口输出不变。
- [x] 已实现 `resample_future_profile`、`build_cfdp_teacher`、`symmetric_geometric_mean_normalize` 和 key composition 的最小纯函数。
- [x] 已扩展 `BankManifest` 为显式 schema v2 layout 字段，v1 仍可读取旧实验，v2 仅接受 `canonical_profile_latent`。
- [x] 已在 Bank writer/reader 中保存并校验 schema v2 manifest；旧 Bank 不得静默加载。
- [x] 已运行全量回归；截至 2026-08-12，141 项测试和 `compileall` 全部通过。

**决策：** R0 全绿才进入 R1；失败则只修复契约，不训练。

### R1：SymNormTeacher

**文件：**

- Modify: `stanchor/config.py`
- Modify: `stanchor/losses/pretraining.py`
- Modify: `stanchor/engine/pretrainer.py`
- Create: `configs/metrla_e5_final_symnorm_v1.yaml`

**步骤：**

- [x] 已新增 `relation_distance_normalization=symmetric_geometric_mean` 与配置校验。
- [x] 已在 relation teacher 中保持 `ODSignature = future - lambda * endpoint`，仅替换 AnchorMean 距离归一化为 SymNorm；teacher 在 no-grad 中构造。
- [x] 诊断接口已支持对称性、teacher/student effective support、Recall@5、Spearman 和 Oracle Top-5 MAE。
- [x] 一批次前向/反向 smoke 已通过。
- [ ] 正式运行 R1-A，并与 AnchorMean 同口径比较。

**决策：** 训练后 relation 或 no-confidence memory 至少一项超过 seed 波动改善才保留，否则回退 AnchorMean。

### R2：CFDP semantic key

**文件：**

- Modify: `stanchor/config.py`
- Modify: `stanchor/models/retrieval_head.py`
- Modify: `stanchor/models/pretraining.py`
- Modify: `stanchor/losses/pretraining.py`
- Modify: `stanchor/bank/builder.py`
- Modify: `stanchor/retrieval/retriever.py`
- Modify: `stanchor/diagnostics/retrieval.py`
- Create: `configs/metrla_e5_final_sym_profile_v1.yaml`
- Create: `configs/pemsbay_e5_final_transfer_v1.yaml`

**步骤：**

- [x] 已将 retrieval head 拆成 12 维 profile head 与 36 维 latent head，并保持总 key 48 维。
- [x] 已使用事件自身 endpoint/mean/std 构造 mask-aware CFDP teacher；profile loss 为 mask-aware SmoothL1，权重固定 0.1。
- [x] 已保存 profile/latent layout 到 Bank v2 manifest，并实现两部分 similarity 诊断。
- [ ] 运行 source pretrained-vs-random、METR-LA source 与 PEMS-BAY transfer，对比 profile MAE、Spearman、Recall@5、OffsetDecay MAE/RMSE/MAPE 和成本。

**决策：** 只有跨域 pretrained-vs-random gap 和检索/下游指标共同改善才保留 profile key；只改善辅助 profile loss 则删除 profile key。

### R3：profile weight 三点消融

**步骤：** 固定同一 checkpoint、Bank、候选协议和 seed，仅比较 `gamma=0/0.25/1`。

**决策：** `gamma=0` 最好则删除 profile；不进行连续网格搜索或可学习 gate。

### C1：固定检索的校准实验

**文件：**

- Modify: `stanchor/config.py`
- Modify: `stanchor/models/downstream.py`
- Modify: `stanchor/losses/downstream.py`
- Modify: `stanchor/engine/target.py`
- Create: `configs/metrla_e5_final_calibrator_v1.yaml`

**步骤：**

- [x] 已实现 `PredictedBaseRisk`：history 与 detached base prediction 输入，Huber error target，输出 `[B,H,N,1]`。
- [x] 已实现 `ErrorAwareAdditiveFusion` 内部的 grouped additive 结构：10 个独立 `1->8->1` shape functions，初始权重 0.1，返回每项 logit contribution。
- [x] 已用 `w_star=clip(<Y-Y_base,Y_mem-Y_base>/(||Y_mem-Y_base||^2+eps),0,1)` 监督 residual blend；新模式不再使用旧的 horizon-limit x confidence 双门控。
- [x] 已实现 base warm-up、calibrator warm-up、joint fine-tuning 三阶段训练；retrieval/Bank 全程冻结。calibrator warm-up 中被冻结的 backbone 同时保持 eval 模式，避免 Dropout 让校准器的输入随机抖动。
- [ ] 依次比较 base-only、horizon-only、旧 confidence、旧融合+Risk、完整 ErrorAwareAdditiveFusion。

**决策：** 完整校准器若不优于当前 confidence，停止校准结构扩展并保留简单版本。

### C2/C3：必要性与泛化验证

- [ ] C2 依次移除 risk、profile/latent 分解、direction agreement、blend loss、risk loss，并以等参数 MLP 做对照。
- [ ] C3 在轻量 MLP、ST-SSDL 和另一种下游 backbone 上运行 seed 42/2024/2025；最终 test 仅运行一次。
- [ ] 记录 MAE/RMSE/MAPE、每个 horizon、Risk MAE/Spearman/R2、Blend target MAE、Helpfulness AUROC、参数量、epoch 时间、延迟、显存和 Bank 大小。

## 特殊名词速查（本文件自包含）

- **SymNormTeacher**：对训练期 OffsetDecay future 距离按 anchor 与 candidate 两侧的平均距离做几何均值归一化。输入是 source-train future、各事件 endpoint 和 pair mask，输出 `[B,B,N]` 无量纲 teacher distance；只在 teacher 的 `no_grad` 分支读 future，部署不运行。
- **CFDP（CanonicalFutureDynamicsProfile）**：把每个事件 future 映射到固定 12 个相对时间位置，并用事件自己的 context endpoint、mean、std 归一化。输出 `[B,12,N,1]` 的无量纲动态轮廓；训练时作为 history-only profile head 的 teacher，推理时不构造 query future profile。
- **Bank v2**：使用 `12-D profile + 36-D latent = 48-D` 新 key 布局的历史 memory bank。Bank 保存 raw historical future，OffsetDecay 仍在部署时对 payload 做 level 对齐；v1 Bank 与 v2 encoder 互相拒绝。
- **PredictedBaseRisk**：下游只用 query history 和 detached base prediction 估计基础预测的 Huber 误差 `[B,H,N,1]`。真实 future 只在训练标签和离线诊断中使用。
- **ErrorAwareAdditiveFusion**：把 10 个可部署特征分别映射成 logit contribution，再 sigmoid 得到单一 residual blend weight。该 weight 表示沿 `memory-base` 修正方向的幅度，不是 memory helpfulness 概率，因此不用 Brier/ECE 评价它。
- **base warm-up / calibrator warm-up / joint fine-tuning**：分别训练基础 backbone、冻结 backbone 训练 risk/fusion、再以 backbone 学习率为校准器的 0.1 联合微调；retrieval encoder 与 Bank 全程冻结。

## 严格实验顺序与命令

以下命令中的 `METR-LA` 和 `PEMS-BAY` 数据路径必须先在相应 yaml 中确认；每一步的输出目录必须使用新的正式 run name，不能覆盖旧实验。

### 0. R0 工程契约（必须先通过）

```powershell
C:/Users/31396/.conda/envs/research/python.exe -m unittest discover -s tests -v
C:/Users/31396/.conda/envs/research/python.exe -m compileall -q stanchor scripts tests
```

Keep/Stop：当前 141 项测试和 compileall 必须全绿才进入训练；任何 shape、Bank schema、SymNorm 对称性或 future leakage 测试失败都停止。

### 1. R1-A：只验证 SymNormTeacher

**动机：** 先判断 teacher 几何修正本身是否比 AnchorMean 更能表达对称 future relation，不把 profile head 或新校准器的收益混进来。

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/pretrain.py `
  --config configs/metrla_e5_final_symnorm_v1.yaml `
  --run-name metrla_e5_final_symnorm_seed42
```

当前工作区已确认旧 AnchorMean checkpoint `artifacts/metrla_e5a_offset_decay_seed42/pretrain_best_relation.pt` 能由新对照配置严格加载，且其 retrieval fingerprint 与 `artifacts/metrla_bank_e5a_offset_decay_relation_seed42` 完全一致，因此不重复训练 AnchorMean。上面的命令只训练新增 SymNorm。两份配置均关闭 CFDP；除 teacher 归一化、Bank 输出目录和 run name 外保持一致。报告 teacher asymmetry、teacher/student effective support、Spearman、Recall@5、Oracle Top-5 MAE 和无 confidence OffsetDecay memory MAE/RMSE/MAPE。

若换到没有上述旧正式产物的实验机，才运行以下 AnchorMean 补跑命令：

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/pretrain.py `
  --config configs/metrla_e5_final_anchor_mean_v1.yaml `
  --run-name metrla_e5_final_anchor_mean_seed42
```

分别构建两组 Bank 并运行同口径 `teacher metric diagnostic`（teacher 度量诊断）。该诊断在共享的 causal candidate pool 上比较 key distance 与 AnchorMean/SymNorm future distance，并用 OffsetDecay payload 计算 Top-1/Top-5 物理预测误差。Bank 保存历史事件的 raw future payload；OffsetDecay 使用 query endpoint 对 raw future 做 level 对齐。query future 只用于离线 teacher relation 和 MAE/RMSE/MAPE，不参与 query 编码、候选生成、key distance 或部署排序。

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/build_bank.py `
  --config configs/metrla_e5_final_symnorm_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_symnorm_seed42/pretrain_best_relation.pt `
  --dataset-name METR-LA `
  --output-dir artifacts/metrla_bank_e5_final_symnorm_seed42

C:/Users/31396/.conda/envs/research/python.exe scripts/diagnose_teacher_metrics.py `
  --config configs/metrla_e5_final_anchor_mean_v1.yaml `
  --checkpoint artifacts/metrla_e5a_offset_decay_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5a_offset_decay_relation_seed42 `
  --split val `
  --candidate-protocol relaxed_calendar `
  --output-dir artifacts/metrla_e5_final_r1_anchor_mean_reference/teacher_metric_val

C:/Users/31396/.conda/envs/research/python.exe scripts/diagnose_teacher_metrics.py `
  --config configs/metrla_e5_final_symnorm_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_symnorm_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_final_symnorm_seed42 `
  --split val `
  --candidate-protocol relaxed_calendar `
  --output-dir artifacts/metrla_e5_final_symnorm_seed42/teacher_metric_val
```

Keep：SymNorm 在训练后 Recall@5 不低于 AnchorMean，且 relation 或 no-confidence memory 超过 seed 波动改善。否则 Remove：回退 AnchorMean，仍可单独测试 CFDP。

### 2. R2：CFDP profile + latent key

**动机：** 验证可泛化的未来语义是否真正改善检索关系，而不是只降低一个辅助 profile loss。

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/pretrain.py `
  --config configs/metrla_e5_final_sym_profile_v1.yaml `
  --run-name metrla_e5_final_cfdp_seed42
```

随机对照必须使用相同配置初始化，不训练：

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/init_random_checkpoint.py `
  --config configs/metrla_e5_final_sym_profile_v1.yaml `
  --output artifacts/metrla_e5_final_cfdp_random_seed42/checkpoint.pt `
  --seed 42
```

R2 预训练完成后，先运行 `CFDP semantic diagnostic`（CFDP 语义诊断）。它在 validation split 上比较 history-only profile head 预测的 12 维无量纲未来动态轮廓与真实 future 构造的 teacher profile，并检查 profile/total key 的两两关系是否接近真实 future relation。输入是 validation history、冻结 checkpoint 和仅作离线标签的 validation future；输出是 profile MAE、profile cosine、relation Spearman 与 Recall@K。validation future 只参与离线评估，不参与 query 编码、候选排序、Bank 构建或部署推理。

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/diagnose_cfdp.py `
  --config configs/metrla_e5_final_sym_profile_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_cfdp_seed42/pretrain_best_relation.pt `
  --split val `
  --output artifacts/metrla_e5_final_cfdp_seed42/cfdp_diagnostic_val.json
```

Keep：CFDP 不仅要降低 profile MAE，还应改善 profile/total relation Spearman 或 Recall@5，并在后续 pretrained-vs-random retrieval 指标中产生收益。若只学会重建 profile、但关系与检索不改善，则 Remove：删除 profile 分支并保留纯 latent key。

### 3. R2-B：构建 target-local causal Bank v2

**动机：** 检验同一个 source-pretrained history-only encoder 在目标域重建 Bank 后是否仍能共享无量纲 future semantics。Bank 只写 target train 中已经完整发生的历史事件及 raw future。

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/build_bank.py `
  --config configs/metrla_e5_final_sym_profile_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_cfdp_seed42/pretrain_best_relation.pt `
  --dataset-name METR-LA `
  --output-dir artifacts/metrla_bank_e5_final_cfdp_seed42

C:/Users/31396/.conda/envs/research/python.exe scripts/build_bank.py `
  --config configs/pemsbay_e5_final_transfer_v1.yaml `
  --checkpoint artifacts/metrla_e5_final_cfdp_seed42/pretrain_best_relation.pt `
  --dataset-name PEMS-BAY `
  --output-dir artifacts/pemsbay_bank_e5_final_cfdp_seed42
```

### 4. C1：ErrorAwareAdditiveFusion 下游校准

**动机：** PIR 的可借鉴部分是预测基础模型误差；本实验验证 risk signal 与 future-guided OffsetDecay memory 的修正幅度是否比旧 confidence 更有效。新模式仍通过历史 Bank 检索 future，PIR 不替代 future-guided 核心。

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/train_downstream.py `
  --config configs/metrla_e5_final_calibrator_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e5_final_cfdp_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e5_final_cfdp_seed42
```

PEMS-BAY 跨数据集迁移必须使用 PEMS-BAY 图、scaler、Bank 和校准配置：

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/train_downstream.py `
  --config configs/pemsbay_e5_final_calibrator_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e5_final_cfdp_seed42/pretrain_best_relation.pt `
  --bank artifacts/pemsbay_bank_e5_final_cfdp_seed42
```

严格对照顺序：`base_only` -> `learned_topk_offset_decay_horizon` -> 旧 `learned_topk_confidence` -> 新 `learned_topk_error_aware`。每个设置使用独立 run name、相同 retrieval checkpoint、Bank、候选协议和 seed。

### 5. C1 诊断

```powershell
C:/Users/31396/.conda/envs/research/python.exe scripts/diagnose_downstream.py `
  --config configs/metrla_e5_final_calibrator_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_e5_final_cfdp_seed42/pretrain_best_relation.pt `
  --downstream-checkpoint artifacts/metrla_e5_final_calibrator_seed42/downstream_best.pt `
  --bank artifacts/metrla_bank_e5_final_cfdp_seed42 `
  --split val `
  --output artifacts/metrla_e5_final_calibrator_seed42/downstream_diagnostic.json
```

报告三组结果：

1. MAE、RMSE、MAPE 和 15/30/60 分钟 horizon-wise 指标；
2. `Risk MAE`、`Risk Spearman`、`Risk R2`；
3. `Blend target MAE`、权重四分位真实 memory gain、helpfulness AUROC/AUPRC、10 个 additive contribution 分布。

### 6. R3/C2/C3 只有前一步通过才执行

- R3：固定 checkpoint 和 Bank，只比较 `gamma=0/0.25/1`；若纯 latent 最好，删除 profile。
- C2：依次去掉 risk、profile/latent 分解、direction agreement、blend loss、risk loss，并用等参数 MLP 替换 additive fusion。
- C3：在轻量 MLP、ST-SSDL 和另一种 backbone 上运行 seed `42/2024/2025`，最终 test 只运行一次。

每一步必须写 Keep/Remove/Stop 结论；没有决策后果的实验不执行。

正式训练命令必须在 R0 和 smoke 通过后，使用方案合集中的版本化 config、checkpoint 和 Bank 路径执行；smoke 输出不得作为论文结果。
