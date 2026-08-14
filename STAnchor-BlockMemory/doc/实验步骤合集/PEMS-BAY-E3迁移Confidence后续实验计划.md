# PEMS-BAY E3 迁移 Confidence 后续实验计划

## 1. 当前已确定的结论

seed 42 validation 已完成：

| 模式 | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| base-only | 2.171461 | 5.097139 | 5.232578 |
| learned Top-K + horizon-only | 1.881370 | 4.107756 | 4.390352 |
| **learned Top-K + confidence** | **1.804609** | **4.037364** | **4.192822** |

confidence 相对 horizon-only 降低 MAE 4.08%，并且 confidence 分桶的 memory gain 具有单调语义。因此当前决策是：**保留 confidence 候选，冻结现有结构和超参数，不继续增加模块。**

## 2. 下一项唯一实验问题

下一项只回答：

> METR-LA 预训练的 E3 encoder-selector 是否优于相同架构、相同初始化种子但未经预训练的 random encoder-selector？

这一步不比较 confidence，不读 test，也不修改检索、backbone 或 fusion。使用 `learned_topk_horizon` 是为了排除 confidence 网络带来的额外变量。

## 3. Target-random 对照准备

可复现 random checkpoint 生成脚本及正式实验已经完成：

1. 使用 `configs/pemsbay_e3_transfer_v1.yaml` 中与 E3 完全相同的 encoder 和 retrieval head 结构；
2. 固定 seed 42 后初始化参数，但不执行任何预训练更新；
3. 保存模型结构指纹、随机种子和 state dict，避免 Bank 与 checkpoint 不匹配；
4. 使用现有 `build_bank.py` 在相同 PEMS-BAY memory 时间段重建 random Bank；
5. 先执行 Bank 完整性、时间因果性和候选数量检查；
6. 只在 validation 上执行 retrieval diagnostic；
7. 使用 random Bank 训练 seed 42 `learned_topk_horizon`。

除 encoder 参数来源外，以下变量必须完全一致：

- PEMS-BAY scaler、图和数据划分；
- Bank 事件范围、future payload 和时间槽过滤；
- event Top-R、node Top-K、温度和聚合方式；
- 下游 backbone、horizon fusion、训练轮数、早停规则和 seed；
- checkpoint 只按 validation MAE 选择。

## 4. 比较指标与决策

记录 source-pretrained 与 target-random 的两个层级指标：

| 层级 | 指标 | 作用 |
|---|---|---|
| 检索 | learned retrieved future MAE | 判断 learned key 是否找到了更接近真实未来的历史块 |
| 下游 | horizon-only validation MAE/RMSE/MAPE | 判断检索表示是否真正改善最终预测 |

定义预训练相对收益：

\[
G_{\mathrm{pretrain}}
=
\frac{M_{\mathrm{random}}-M_{\mathrm{source}}}
{M_{\mathrm{random}}}
\times100\%,
\]

其中 $M_{\mathrm{source}}$ 是当前 source-pretrained horizon-only validation MAE，即 1.881370；$M_{\mathrm{random}}$ 是待测 target-random horizon-only validation MAE。

决策规则：

- 若 source-pretrained 的检索 MAE和下游 MAE均低于 target-random，则保留“预训练表示有效”的候选结论，并进入多 seed；
- 若 target-random 在两个层级均不差于 source-pretrained，则停止声称收益来自预训练，优先将方法收缩为非预训练历史检索模块；
- 若检索和下游结论相反，则只追加 seed 2024 的同一对照检查稳定性，不新增网络；
- 不因 target-random 结果重新调当前 E3 超参数。

## 5. Target-random 之后的条件计划

只有 target-random 门槛通过，才运行 PEMS-BAY seed 2024 和 2025：

1. `base_only`；
2. `learned_topk_horizon`；
3. `learned_topk_confidence`。

三 seed 汇总 validation 的均值、样本标准差和逐 seed 配对差值。confidence 还要汇总 AUROC、AUPRC、Brier、ECE 以及四分位 memory gain，确认 seed 42 的单调语义可以复现。

若 confidence 在 3 个 seed 中至少 2 个优于配对 horizon-only，且平均 MAE 更低，则保留 confidence；否则最终主模型退回 horizon-only。该规则在读取 test 前固定，不再根据结果增加筛选条件。

## 6. Test 边界

当前禁止运行 PEMS-BAY test。只有以下条件全部满足后才允许一次性评估：

- target-random 归因结论完成；
- 三 seed validation 完成；
- 最终模式已冻结；
- 不再修改网络、损失、超参数和 checkpoint 选择规则。

最终 test 报告 MAE、RMSE、MAPE，以及 15、30、60 分钟三组指标；不得依据 test 返回修改模型。

## 7. 执行顺序

- [x] seed 42 base-only；
- [x] seed 42 horizon-only；
- [x] seed 42 confidence；
- [x] seed 42 confidence validation 机制诊断；
- [x] 实现并验证 deterministic random checkpoint 生成；
- [x] 构建和诊断 target-random PEMS-BAY Bank；
- [x] 运行 target-random seed 42 horizon-only；
- [x] 根据预先固定规则作 keep/remove/stop 决策；
- [x] 当前 E3 未通过 random 归因门槛，停止其 PEMS-BAY 多 seed；
- [x] 当前 E3 不读取 PEMS-BAY test；
- [ ] 按 E4 方案先执行 `level_weight=0` 的零训练归因诊断。

## 8. Target-random 最终结论

正式 validation 结果如下：

| 层级 | Source-pretrained | Target-random | 判断 |
|---|---:|---:|---|
| learned Top-K future MAE | 2.209414 | **2.173856** | random 更低 |
| horizon-only MAE | 1.881369 | **1.878035** | 功能性持平，random 略低 |
| horizon-only RMSE | **4.107756** | 4.114959 | source 略低 |
| horizon-only MAPE | **4.390351** | 4.394232 | source 略低 |

按照第 4 节预先固定的规则，target-random 在检索和下游 MAE 上均不差于 source-pretrained，因此当前 E3 不进入多 seed，也不能继续声称收益来自 source pretraining。

完整依据见：

- `doc/诊断报告合集/PEMS-BAY-E3迁移Target-Random归因诊断报告.md`；
- `doc/优化方案合集/E4-检索一致性关系预训练改进方案.md`。
