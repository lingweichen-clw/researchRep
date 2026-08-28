# STAEformer 时间特征协议统一修复方案

## 1. 问题与证据

2026 年 8 月 28 日的 STAEformer + HorizonMixer 接入实验中，验证集最佳 MAE 为 3.294523，而历史 STAEformer base-only checkpoint 的最佳 MAE 为 2.799559。该退化不能直接归因于 HorizonMixer：Graph WaveNet 在相同 HorizonMixer 回退代码下已复现 2.828445，与历史 2.828433 基本一致。

进一步的同 checkpoint 前向检查显示，使用数据集真实 `slot/weekday` 时 Base 输出显著失真，而使用旧模型内部 fallback 时间编码时输出恢复到原有量级。因此，旧 checkpoint 与当前接入阶段使用了不同的时间特征语义。

## 2. 修复目标

STAEformer 的 base-only 训练、冻结 Base 推理和校准器训练必须使用完全一致的时间特征协议。正式主线采用 `calendar` 模式：

- `slot`：数据集提供的日内时间槽，范围为 `[0, slots_per_day-1]`；
- `weekday`：数据集提供的星期索引，范围为 `[0, 6]`；
- 两者在进入 STAEformer 前广播到 `[B,T,N]`；
- 不再让 base-only 和 posthoc 阶段隐式选择不同的 fallback 行为。

`fallback` 模式仅保留用于复现旧 checkpoint，不作为新的正式结果协议。

## 3. 代码调整

1. `TargetConfig` 增加 `staeformer_time_feature_mode`，允许 `calendar` 或 `fallback`。
2. `run_target_epoch` 根据该配置调用 STAEformer：
   - `calendar`：传入 `batch["slot"]` 和 `batch["weekday"]`；
   - `fallback`：不传时间特征，由模型内部生成旧式 fallback 编码。
3. 启动日志显式记录时间特征模式和来源。
4. `validation_formal_20260826_staeformer_baseline.yaml` 与 `formal_horizon_mixer_staeformer.yaml` 明确设置为 `calendar`。
5. HorizonMixer、HN-OffsetDecay v1 encoder、Bank、检索协议、缓存和损失函数不做改动。

## 4. 训练与验证顺序

先重新训练 STAEformer base-only 50 轮，禁止 early stopping。训练完成后，先做单 batch 对齐检查：同一 batch 下 base-only forward 与 posthoc frozen-base forward 的输出应一致（允许浮点误差）。随后再使用该新 base checkpoint 训练 HorizonMixer。

## 5. 结果判定

- 若新 base-only 与 posthoc 的 Base 输出一致，说明协议问题已修复；
- 若接入 HorizonMixer 后 MAE 回到与 base-only 同一量级，才继续评估校准器收益；
- 若 Base 输出已对齐但接入后仍明显退化，再单独诊断 HorizonMixer 或候选质量，不能把协议问题与校准器能力混为一谈。

## 6. 信息边界

`slot` 和 `weekday` 只来自当前 context 的时间索引，不使用真实 future。真实 future 仍只用于训练目标和离线评价，不进入部署阶段的时间特征输入。
