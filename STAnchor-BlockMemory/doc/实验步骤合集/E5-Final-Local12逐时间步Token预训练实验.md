# E5-Final Local12 逐时间步 Token 预训练实验

## 1. 实验目的

本实验只验证一个问题：检索编码器仅输入预测起点前最近 12 个五分钟时间步，并保留每个时间步的独立 token，能否比当前一天历史的小时级 patch 更准确地学习“key 距离接近对应 future 关系接近”。

`Local12-StepToken` 可译为“最近 12 步逐时间步 token”。它不是新的注意力模块。输入是最近一小时历史，形状为 `[B,12,N,1]`；每个五分钟时间步独立投影成一个 96 维 token，输出为 `[B,12,N,96]`；现有三层时空注意力编码器处理这 12 个 token，最后输出每个节点的 48 维 retrieval key，形状为 `[B,N,48]`。训练和推理时 encoder 都只读取历史，不读取 query future。

## 2. 与一天历史版本的唯一显式配置差异

| 配置项 | Global288-Patch12 | Local12-StepToken | 含义 |
|---|---:|---:|---|
| `retrieval_context_length` | 288 | 12 | encoder 读取的历史从一天改为最近一小时 |
| `patch_size` | 12 | 1 | 从每小时一个 token 改为每五分钟一个 token |
| temporal token 数 | 24 | 12 | `历史长度 / patch_size` |
| `time_mask_block_size` | 36 | 3 | 每个连续遮挡块都覆盖 3 个 token；新版本每块为连续 15 分钟 |

两版的 `time_mask_ratio=0.25` 均保持不变，用于控制每个样本总共遮挡约 25% 的时间 token。Global 的单个 token 表示一小时，所以每个连续块为 3 小时；Local 的单个 token 表示 5 分钟，所以每个连续块为 15 分钟。该变化保持“连续遮挡 3 个 token”的结构一致性，并让遮挡任务适应各自时间分辨率。

其他核心设置保持不变：`hidden_dim=96`、三层时空注意力、`retrieval_dim=48`、OffsetDecay future teacher、SymNorm 距离归一化、训练轮数、学习率和 batch size。

## 3. Token 构造

设预测起点前最后一个观测时刻为 `t0`，输入为：

\[
X^{local}_{b}
=
[x_{b,t_0-11},\ldots,x_{b,t_0}]
\in\mathbb R^{12\times N\times1}.
\]

第 `t` 个时间步独立形成一个 token：

\[
z_{b,t,n}
=
W_x\widetilde x_{b,t,n}
+W_l l_{b,n}
+e^{slot}_{b,t}
+e^{weekday}_{b,t},
\qquad
z_{b,t,n}\in\mathbb R^{96}.
\]

- `x_tilde`：用该样本、该节点最近 12 步可见值归一化后的速度；
- `l`：该 12 步窗口的均值、标准差、最后可见值和首末差；
- `e_slot`：一天中五分钟时间槽嵌入；
- `e_weekday`：星期嵌入；
- `W_x`、`W_l`：可训练线性投影。

这里没有对多个时间步求平均，也没有在进入注意力前合并相邻时间步。

## 4. Future teacher 与信息边界

`ODSignature` 是训练期用于定义 future relation 的部署对齐未来表示：

\[
S^{OD}_{i,h,n,c}
=
Y_{i,h,n,c}-\lambda_h\alpha_{i,n,c},
\qquad
\lambda_h=1-\frac{h-1}{H-1}.
\]

- `Y_i`：source-train 事件 `i` 的未来 12 步，只用于训练 teacher；
- `alpha_i`：预测起点前最后一个历史时间步的值；若该点缺失，使用最近 12 步可见值的均值回退；
- `lambda_h`：从第 1 个预测步的 1 线性衰减到第 12 个预测步的 0；
- 输出 `S_OD`：`[B,12,N,1]`，用于计算训练 batch 内事件两两 future 距离。

teacher 在 `torch.no_grad()` 中使用 source-train future。部署 query 编码、Bank 查询和候选排序都不能读取 query future。

## 5. 非重叠 pair 说明

`Non-overlap pair` 指两个训练事件的历史窗口和 future 不互相覆盖。排除重叠 pair 是为了防止事件 `j` 的历史直接包含事件 `i` 的 future，从而形成关系学习捷径。

Local12 的历史窗口比 Global288 短，因此合法 pair 数可能增加。这是 12 步数据契约的自然结果，但也意味着正式比较必须同时报告有效 anchor/pair 数，并在共同 validation query、相同候选协议下比较 pretrained 与同构 random，不能只比较预训练 loss。

## 6. 正式预训练命令

在 `STAnchor-BlockMemory` 目录运行：

```powershell
python scripts/pretrain.py `
  --config configs/metrla_e5_final_symnorm_local12_v1.yaml `
  --run-name metrla_e5_final_symnorm_local12_seed42
```

正式实验不得添加 `--max-batches` 或 `--epochs 1`。输出目录为：

```text
artifacts/metrla_e5_final_symnorm_local12_seed42/
```

## 7. 训练完成后的判断顺序

1. 使用 relation validation loss 选出的正式 checkpoint 构建 Local12 Bank。
2. 构建相同架构、相同 seed 的 random checkpoint 和 random Bank。
3. 在共同 validation query 上比较 pretrained 与 random 的 Spearman、Future-Neighbor Recall@5，以及 OffsetDecay memory 的 MAE、RMSE、MAPE。
4. 报告有效 pair、teacher/student effective support、参数量、单轮训练时间和 Bank 检索耗时。
5. `Keep`：Local12 在关系指标和 memory 误差上都扩大 pretrained-random 差距；`Remove`：只降低训练 loss，或不优于同构 random；`Stop`：跨域迁移明显下降且同域收益不足以补偿。

## 8. Local12 + CFDP：直接启用未来轮廓分支

`CFDP`（Canonical Future Dynamics Profile，可译为“规范化未来动态轮廓”）是训练期的可解释未来语义分支。它的目的不是替代未来引导检索，而是把 48 维 key 中的 12 维明确监督成可跨数据集比较的相对未来形状，其余 36 维继续学习难以人工定义的潜在关系。

给定 source-train 事件的未来 `Y`、未来观测掩码、预测前 12 步 context，CFDP 先把未来重采样到 12 个相对位置，再减去从 context endpoint 平滑过渡到 context mean 的参考水平，最后除以该事件 context 的标准差：

\[
G_{i,k,n,c}
=
\frac{
\overline Y_{i,k,n,c}
-\lambda_k\alpha_{i,n,c}
-(1-\lambda_k)m_{i,n,c}
}{\max(s_{i,n,c},\epsilon)}.
\]

- `i`：训练事件；`k`：12 个相对未来位置；`n`：传感器节点；`c`：变量通道。
- `Y_bar`：未来在固定 12 点相对时间网格上的值；当前预测 horizon 也是 12，因此这里等于原未来序列。
- `alpha`：预测起点前最后一个可见 context 值；若缺失则回退到 context 可见均值。
- `m`、`s`：该事件、该节点最近 12 步 context 的可见均值和标准差。
- `lambda_k`：从近端的 1 线性衰减到远端的 0，使参考水平从 endpoint 逐渐过渡到 context mean。
- `epsilon=0.1`：标准差下限，防止平稳序列除以接近 0 的尺度。
- 输出 `G`：无量纲的 `[B,12,N,1]` 未来动态轮廓。

CFDP teacher 只在 source-train 预训练时读取训练样本 future，并位于 `torch.no_grad()` 分支。推理时不构造 query future profile；历史编码器根据 query 的 `[B,12,N,1]` 历史预测 12 维 profile key，因此不存在部署期 future 泄漏。

最终节点 key 仍为 `[B,N,48]`：

\[
k_{i,n}
=
\operatorname{Norm}
\left[
\sqrt{0.25}\,\operatorname{Norm}(p_{i,n});
\sqrt{0.75}\,\operatorname{Norm}(z_{i,n})
\right],
\]

其中 `p` 为 12 维可解释 profile key，`z` 为 36 维 latent key，`Norm` 表示最后一维 L2 归一化，分号表示向量拼接。`profile_weight=0.25` 表示余弦相似度中 profile 分支占 25%，latent 分支占 75%。新增的 `profile_loss_weight=0.1` 监督历史编码器预测 CFDP；ODSignature + SymNorm relation teacher、12 步输入、候选协议和总 key 维度均保持不变。

无 Profile Local12 已经完成，因此现在直接启用 CFDP 不再混淆“输入长度变化”和“Profile 分支变化”。两版配置差异仅为 `profile_dim: 0 -> 12`、`latent_dim: 0 -> 36`、`profile_loss_weight: 0 -> 0.1` 以及独立输出名称。

### 8.1 实验机正式预训练命令

```powershell
python scripts/pretrain.py `
  --config configs/metrla_e5_final_sym_profile_local12_v1.yaml `
  --run-name metrla_e5_final_sym_profile_local12_seed42
```

不得添加 `--max-batches` 或缩短 epoch。关系预训练模式使用 `val_retrieval` 选择早停时机和 `pretrain_best_relation.pt`；`val_retrieval` 是 student key 关系分布拟合 future teacher 分布的验证损失，不是速度物理单位的 MAE。

## 9. 本机零训练诊断顺序

`零训练诊断` 指固定已有 pretrained checkpoint，不更新任何参数，只构建历史 Bank 并在 validation 上比较 pretrained 与相同架构 random。query future 只在候选排序完成后用于计算评价指标，不参与部署候选生成或 key 排序。

严格顺序如下：

1. 用无 Profile Local12 的 `pretrain_best_relation.pt` 构建完整 pretrained Bank。
2. 用相同配置和 seed 42 生成 `trained_steps=0` 的 random checkpoint。
3. 用 random checkpoint 和相同历史事件轴构建完整 random Bank。
4. 使用 `exact_calendar` 候选协议和 `level_weight=0` 跑完整 validation 可视化诊断。

`exact_calendar` 表示只从与 query 同星期、同五分钟时间槽且严格发生在 query 之前的 Bank 事件中选候选。`level_weight=0` 表示节点排序完全由 learned key 的余弦相似度决定，不混入 endpoint level 差异；这样回答的是“预训练 key 是否学到 future 关系”，而不是 level 修正是否有用。

诊断比较 Spearman、Future-Neighbor Recall@5、key-distance decile monotonicity，以及 OffsetDecay memory 的 MAE、RMSE、MAPE。Spearman 衡量 key 距离与 teacher future 距离的秩相关；Recall@5 衡量 key Top-5 与真实 future 距离 Top-5 的重合比例；decile monotonicity 检查 key 越近的十等分组是否对应更小的 future 距离；三项预测误差只评价检索聚合得到的 memory prediction，不是下游模型结果。

决策标准：pretrained 同时优于同构 random 的关系排序指标和 OffsetDecay memory 误差，才保留 Local12 并进一步比较 CFDP；若只改善其中一类，不启动下游训练，先定位表征和部署聚合之间的不一致。
