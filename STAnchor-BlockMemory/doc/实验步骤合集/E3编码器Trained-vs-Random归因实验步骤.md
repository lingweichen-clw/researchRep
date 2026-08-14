# E3 编码器 Trained-vs-Random 归因实验步骤

## 1. 实验问题

本轮不修改 E3 预训练目标和候选策略，只回答两个问题：

1. 固定 `level_weight=0.25` 是否掩盖了 learned key 的真实能力？
2. E3 encoder-selector 在 METR-LA 同域是否优于同构、同 seed、零训练步的 random encoder-selector？

所有诊断只读取 validation，不读取 test，不重新训练下游模型。

## 2. 单变量约束

Source 与 random 对照保持以下内容相同：

- encoder 和 retrieval head 结构；
- seed 42；
- 数据切分、scaler、图、Bank 事件范围和 future payload；
- weekday-slot 与因果过滤；
- event Top-R、node Top-K、search temperature；
- validation query 和指标实现。

唯一变化是 checkpoint 参数是否经过 E3 预训练。`level_weight=0` 实验则只删除节点重排中的显式 level 加分，不改变 key、Bank 或候选集合。

## 3. 正式输出

后台流水线输出到：

```text
artifacts/encoder_random_attribution_seed42/
```

正式结果文件：

| 文件 | 含义 |
|---|---|
| `pemsbay_source_level0_val.json` | PEMS-BAY source-pretrained 纯 key 检索 |
| `pemsbay_random_level0_val.json` | PEMS-BAY target-random 纯 key 检索 |
| `metrla_random_level025_val.json` | METR-LA random，保持默认 level 项 |
| `metrla_source_level025_val.json` | METR-LA source-pretrained，保持默认 level 项 |
| `metrla_source_level0_val.json` | METR-LA source-pretrained 纯 key 检索 |
| `metrla_random_level0_val.json` | METR-LA random 纯 key 检索 |

METR-LA random checkpoint 和 Bank 分别保存为：

```text
artifacts/metrla_e3_target_random_seed42/random_checkpoint.pt
artifacts/metrla_bank_e3_target_random_seed42/
```

## 4. 判断规则

- 若 source 在 METR-LA 默认和纯 key 两种口径均优于 random，而在 PEMS-BAY 不优于 random，则 E3 学习有效，主要问题是跨数据集迁移。
- 若 source 在 METR-LA 也不优于 random，则当前 relation objective 对检索 key 的独立贡献没有成立，应先修正损失或表示学习，而不是讨论迁移。
- 若 PEMS-BAY 纯 key 下 source 开始优于 random，则 `level_weight=0.25` 是主要干扰项；删除显式 level 加分后再验证一次 horizon-only。
- 若 PEMS-BAY 纯 key 仍持平，则不扫描 level 超参数，下一步只分析源域有效性和域偏移。

只有上述归因完成后，才决定是否修改候选策略。当前不实施同 weekday-slot-only 预训练。

## 5. 后台运行

流水线会串行执行所有 GPU 步骤，避免多个进程同时争抢显存。后台 PID 和启动信息保存在同一正式输出目录，运行日志为：

```text
artifacts/encoder_random_attribution_seed42/background.stdout.log
artifacts/encoder_random_attribution_seed42/background.stderr.log
artifacts/encoder_random_attribution_seed42/pipeline.log
```

本轮不使用 `--max-events` 或 `--max-batches`。所有结果均为完整 validation 诊断，不能将中途日志视为最终证据。

## 6. 本轮结果状态

本轮五份 validation JSON 已全部生成，后台流水线退出码为 0，stderr 为空。

结论：

- METR-LA 同域：source-pretrained 优于 random；
- PEMS-BAY 跨域：source-pretrained 不优于 random；
- PEMS-BAY `level_weight=0`：source 进一步退化；
- 不执行同 weekday-slot-only 预训练候选收缩；
- 下一步转入 `doc/优化方案合集/E4-目标域自适应关系预训练方案.md`。

## 7. Random seed 稳定性补充

为判断 source 相对 random 的 `0.87%~1.68%` 优势是否超过随机映射方差，补充 random seeds 2024、2025：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_metrla_random_seed_stability.ps1 `
  -Python C:\Users\31396\.conda\envs\research\python.exe
```

正式输出目录：

```text
artifacts/metrla_random_seed_stability/
```

该流水线不更新任何模型参数，只生成 random checkpoint、完整 Bank 和 validation 检索结果。

### 执行结果

- [x] random seed 2024 checkpoint；
- [x] random seed 2024 METR-LA Bank；
- [x] random seed 2024 validation diagnostic；
- [x] random seed 2025 checkpoint；
- [x] random seed 2025 METR-LA Bank；
- [x] random seed 2025 validation diagnostic；
- [x] 两套 Bank 的非 key payload 哈希审计；
- [x] source 与 random 三种子汇总。

Random weighted Top-K MAE 为 `3.796042 / 3.781807 / 3.789821`，均值 `3.789223`，样本标准差 `0.007136`。E3 source 为 `3.732261`，优于全部 random seeds，Stage 0 通过。
