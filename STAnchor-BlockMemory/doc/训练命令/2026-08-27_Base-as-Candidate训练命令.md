# Base-as-Candidate 下游训练命令

**生成时间**: 2026-08-27  
**架构版本**: Base-as-Candidate v2.0  
**参数配置**: hidden_dim=256, state_dim=256, 总参数量=27.5万

---

## 🚀 快速开始

### 阶段1: 3-Epoch Smoke 验证（必须先完成）

#### GWN (Graph WaveNet)
```bash
cd d:/projects/researchProjects/TrafficRobustST/STAnchor-BlockMemory

conda activate research

python -m stanchor.training.downstream \
    --config configs/smoke_base_as_candidate_gwn_3epoch.yaml \
    --override "target.candidate_token_dim=256" \
    --override "target.candidate_state_dim=256"
```

**预期结果**：
- 3 epoch 耗时：~15-20分钟
- Val MAE：2.82-2.85（首轮可能不稳定）
- Base attention mean：0.2-0.4
- 无 NaN/Inf

---

#### STGCN (Spatio-Temporal Graph Convolutional Network)
```bash
python -m stanchor.training.downstream \
    --config configs/smoke_base_as_candidate_stgcn_3epoch.yaml \
    --override "target.candidate_token_dim=256" \
    --override "target.candidate_state_dim=256"
```

**预期结果**：
- 3 epoch 耗时：~12-18分钟
- Val MAE：2.83-2.86
- Base attention mean：0.2-0.4
- 无 NaN/Inf

---

#### STAEformer (Spatio-Temporal Adaptive Embedding Transformer)
```bash
python -m stanchor.training.downstream \
    --config configs/smoke_base_as_candidate_staeformer_3epoch.yaml \
    --override "target.candidate_token_dim=256" \
    --override "target.candidate_state_dim=256"
```

**预期结果**：
- 3 epoch 耗时：~18-25分钟
- Val MAE：2.78-2.82（STAEformer base 较强）
- Base attention mean：0.2-0.4
- 无 NaN/Inf

---

## ✅ Smoke 验证通过标准

**必须满足（否则不进入50轮）**：
- ❌ MAE 退化 > 0.02（相对 base-only）
- ❌ Base 权重恒定接近 0 或 1
- ❌ 出现 NaN/Inf
- ❌ 单轮时间增加 > 50%

**可以进入50轮（满足任一）**：
- ✅ 至少一个 backbone MAE 改善 ≥ 0.01
- ✅ MAE 持平（|Δ| < 0.005）且 Attention 行为合理

---

## 🎯 阶段2: 50-Epoch 正式训练

**只有 Smoke 验证通过后才执行！**

### 创建正式配置文件

需要创建三个新配置文件（基于smoke配置）：

1. `configs/formal_20260827_base_as_candidate_gwn_50epoch.yaml`
2. `configs/formal_20260827_base_as_candidate_stgcn_50epoch.yaml`
3. `configs/formal_20260827_base_as_candidate_staeformer_50epoch.yaml`

**关键修改**：
```yaml
target:
  epochs: 50  # 改为50
  candidate_token_dim: 256  # 新架构
  candidate_state_dim: 256  # 新架构
  candidate_quality_weight: 0.05
  candidate_quality_temperature: 0.2
  base_logit_init_bias: 1.0
  validation_correction_variant: base_as_candidate  # 明确指定
```

### 正式训练命令

#### GWN 50-Epoch
```bash
python -m stanchor.training.downstream \
    --config configs/formal_20260827_base_as_candidate_gwn_50epoch.yaml
```

**预期耗时**：~5-6小时（CUDA:0）

---

#### STGCN 50-Epoch
```bash
python -m stanchor.training.downstream \
    --config configs/formal_20260827_base_as_candidate_stgcn_50epoch.yaml
```

**预期耗时**：~4-5小时（CUDA:0）

---

#### STAEformer 50-Epoch
```bash
python -m stanchor.training.downstream \
    --config configs/formal_20260827_base_as_candidate_staeformer_50epoch.yaml
```

**预期耗时**：~6-8小时（CUDA:0）

---

## 📊 训练监控

### 关键指标

训练日志中重点关注：

| 指标 | 含义 | 预期范围 |
|------|------|----------|
| `val_mae` | 验证集MAE | 2.80-2.85 |
| `val_rmse` | 验证集RMSE | 5.2-5.5 |
| `val_mae_15` | 15分钟MAE | 2.50-2.60 |
| `val_mae_30` | 30分钟MAE | 2.80-2.90 |
| `val_mae_60` | 60分钟MAE | 3.20-3.30 |
| `base_attention_mean` | Base使用率 | 0.2-0.4 |
| `historical_mass_mean` | 历史候选总权重 | 0.6-0.8 |
| `attention_entropy` | Attention熵 | 0.6-0.8 |

### 异常信号

**立即停止训练（出现以下任一）**：
- ❌ Loss 变为 NaN/Inf
- ❌ Base attention 持续 > 0.95（历史候选完全不用）
- ❌ Base attention 持续 < 0.05（Base完全不用，过拟合候选）
- ❌ Val MAE 突然跳跃 > 0.5

---

## 🧹 训练后清理

**训练完成后必须清理**：

```bash
# 删除smoke测试产物
rm -rf artifacts/convergence/smoke_*

# 只保留正式训练结果
ls -d artifacts/convergence/formal_20260827_*
```

**保留**：
- 最佳 checkpoint (`best_model.pt`)
- 训练日志 (`downstream.log`)
- 最终指标 (`final_metrics.json`)

**删除**：
- Smoke 测试 artifacts
- 失败启动的空日志
- 临时调试文件

---

## 📈 结果对比

训练完成后，对比三个模型：

| Backbone | Base-only MAE | Base-as-candidate MAE | Δ MAE | 参数量 |
|----------|---------------|----------------------|--------|--------|
| GWN | 2.865 | ? | ? | 27.5万 |
| STGCN | 2.901 | ? | ? | 27.5万 |
| STAEformer | 2.800 | ? | ? | 27.5万 |

**成功标准**：
- 至少两个模型改善 ≥ 0.01
- 没有模型退化 > 0.02
- Base使用率在合理范围（0.2-0.4）

---

## 🐛 常见问题

### Q1: `candidate_token_dim` 参数未识别
**A**: 确保使用 `--override` 参数传递新配置：
```bash
--override "target.candidate_token_dim=256"
```

### Q2: CUDA Out of Memory
**A**: 减小 batch size：
```bash
--override "target.batch_size=16"
```

### Q3: 训练速度很慢
**A**: 检查是否启用了缓存：
```yaml
target:
  frozen_path_cache: true  # 必须开启
```

### Q4: Base attention 始终为1
**A**: 检查候选是否有效：
```python
# 在训练日志中查找
grep "candidate_valid_ratio" downstream.log
```
如果 < 0.5，说明候选池质量差。

---

## 📝 实验记录模板

每次训练完成后，记录到 `doc/实验日志/2026-08-27_Base-as-Candidate实验.md`：

```markdown
## 实验：Base-as-Candidate v2.0

**日期**: 2026-08-27
**配置**: hidden_dim=256, state_dim=256
**参数量**: 27.5万

### GWN
- Base-only MAE: 2.865
- Base-as-candidate MAE: X.XXX
- Δ MAE: ±X.XXX
- Base attention: X.XX
- 训练时间: X小时

### STGCN
- Base-only MAE: 2.901
- Base-as-candidate MAE: X.XXX
- Δ MAE: ±X.XXX
- Base attention: X.XX
- 训练时间: X小时

### STAEformer
- Base-only MAE: 2.800
- Base-as-candidate MAE: X.XXX
- Δ MAE: ±X.XXX
- Base attention: X.XX
- 训练时间: X小时

### 结论
[是否成功 / 分析原因 / 下一步计划]
```

---

## 🎯 最终交付物

训练全部完成后，产出：

1. ✅ 三个 backbone 的最佳 checkpoint
2. ✅ 完整的训练日志
3. ✅ 对比实验表格
4. ✅ Attention 可视化（如果需要）
5. ✅ 实验报告文档

---

**注意事项**：
- 先完成 Smoke 验证，再启动50轮
- 监控训练过程，异常立即停止
- 训练完成后清理 smoke 产物
- 记录完整的实验结果

**开始训练前检查清单**：
- [ ] Conda 环境已激活（research）
- [ ] CUDA 设备可用
- [ ] Bank 文件存在（`artifacts/case_bank_hn_offset_decay_v1_seed42`）
- [ ] 磁盘空间充足（至少 10GB）
- [ ] Smoke 测试已通过

---

**Good Luck!** 🚀
