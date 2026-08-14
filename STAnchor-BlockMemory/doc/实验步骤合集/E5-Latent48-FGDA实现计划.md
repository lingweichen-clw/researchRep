# E5 Latent48 + FGDA 实现计划

> 本计划实现已经确认的 A 方案。执行时采用测试驱动：每个行为先由失败测试定义，再写最小实现。本文所有特殊名称均在首次出现处给出含义、输入、计算、输出和 future 使用边界。

## 目标

将 E5 检索主线简化为纯 48 维隐表示 key，并加入一个轻量、可独立消融的 `FGDA`。`FGDA` 是 `Future-Guided Dynamics Adapter`（未来关系引导的动态适配器）：它只从历史序列的一阶变化和已有道路图中提取动态残差；训练时由既有 `OffsetDecay relation` future teacher 间接监督，推理时不读取 query future。

## 架构与张量契约

- 历史输入：`X in R^[B,T,N,C]`；Global288 配置中 `T=288, C=1`。
- 基础编码器输出：`Z in R^[B,P,N,D]`；`patch_size=12` 时 `P=24, D=96`。
- 相邻变化：`Delta X_t = X_t - X_(t-1)`。只有相邻两个位置都可见时该变化有效；其他位置填零。
- 局部动态：将每个 patch 的变化展平后线性投影为 `D_local in R^[B,P,N,D]`。
- 图动态：`LocalGraph-FGDA` 用已有静态邻接矩阵对有效邻居的 `D_local` 做行归一化加权聚合，得到 `D_graph in R^[B,P,N,D]`；`Local-FGDA` 不执行此步。
- 图门控：标量 `a in R^[B,P,N,1]` 决定每个节点、每个 patch 使用多少邻居变化。
- 瓶颈残差：`D_local + a D_graph` 经 `96 -> 16 -> 96` 的两层投影得到动态残差 `R`。
- 融合门控：标量 `g in R^[B,P,N,1]` 输出 `Z' = Z + g R`。
- 检索输出：共享 temporal pooling 后由普通 MLP 输出纯 `key in R^[B,N,48]`；不再包含 12 维 CFDP profile。

`W_up` 零初始化，两个门控的权重零初始化、偏置为负，因此训练开始时 `Z'=Z`，新增模块不会在初始化阶段破坏旧 encoder。FGDA 不新增 future target；future 只在 source-train 的 `OffsetDecay relation` teacher 中出现，且 teacher 在无梯度分支中构造。

## 文件边界

- 新建 `stanchor/models/dynamics_adapter.py`：差分构造、有效性掩码、稀疏图聚合、双标量门控和诊断输出。
- 修改 `stanchor/models/pretraining.py`：在基础 encoder 后分别处理 clean 与 masked 分支，并将 adapter 纳入 retrieval checkpoint/fingerprint。
- 修改 `stanchor/config.py`：增加 `none/local/local_graph` 三种模式、16 维瓶颈和门控初始偏置的校验。
- 修改 `stanchor/models/__init__.py`：导出 FGDA 类型。
- 修改 `stanchor/engine/pretrainer.py`：记录 adapter 参数量、有效率、门控均值与残差相对范数。
- 新建 `tests/test_dynamics_adapter.py`：覆盖形状、图聚合、缺失值、遮挡泄漏、初始化恒等和梯度。
- 新建三份配置：纯 `Latent48`、`Latent48 + Local-FGDA`、`Latent48 + LocalGraph-FGDA`。

## 测试驱动步骤

1. 写 adapter 不存在时必然失败的导入、形状和初始化恒等测试，并运行确认失败原因是缺少实现。
2. 写相邻任一点不可见时差分无效、修改被遮挡原值不改变 adapter 输出的测试，并确认失败。
3. 写 local 与 local-graph 输出、邻接有效性归一化测试，并确认失败。
4. 实现 `HistoryDynamicsAdapter` 的最小代码，使上述测试通过。
5. 写 `STAnchorPretrainModel` 的 disabled 精确旧行为、masked time block 零残差和 relation loss 梯度测试，并确认失败。
6. 将 adapter 接入 clean/masked 流程和 retrieval checkpoint，使集成测试通过。
7. 写配置合法性、配置文件加载、参数增量不超过 2% 和 checkpoint 严格加载测试，再实现配置与日志。
8. 运行 `unittest` 全量回归、`compileall`、一批次前后向和参数开销检查。

## 实验决策

- `Latent48` 是删除 profile 后的控制组，不使用 FGDA。
- `Local-FGDA` 只验证局部历史变化是否补足基础 encoder。
- `LocalGraph-FGDA` 只比 Local 多一个已有图上的有效邻居聚合，用于判断空间传播是否必要。
- Seed 42 同时达到：OD Spearman `+0.02`、Recall@5 `+1` 个百分点、memory MAE `-0.5%`、无 confidence 下游 MAE `-0.5%`、参数增量不超过 `2%`、推理延迟增量不超过 `5%`，才进入多 seed 与 PEMS-BAY。
- Local 有效而 LocalGraph 无额外收益：保留 Local；LocalGraph 进一步有效：保留完整 FGDA；两者均无效：删除 FGDA，最终使用纯 Latent48。

