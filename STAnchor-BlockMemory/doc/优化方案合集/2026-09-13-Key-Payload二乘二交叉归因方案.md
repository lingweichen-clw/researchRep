# Key–Payload 二乘二交叉归因方案

## 1. 问题与目标

现有 Future Surprise 分层诊断发现：在真实未来明显偏离 persistence baseline（持久性基线，即把历史最后状态复制到全部未来步）的样本上，端到端 Offset-only 检索的 Memory MAE 高于 OffsetDecay；但 Offset-only 的 Key 排序指标没有同步崩溃，而且差距随预测 horizon 增大。这只能提示 payload 可能是主要来源，尚不能排除 Key 与 payload 的交互。

本实验用无需训练的 `2 种 Key × 2 种 payload` 完全交叉前向，把检索排序和历史 future 重构拆成两个独立因素。本实验不修改 checkpoint、不重建 Memory Bank、不训练校准器。

## 2. 术语、输入与张量

### 2.1 Key 因素

Key 是只由查询历史、calendar 与图结构生成的节点检索向量：

\[
K\in\mathbb{R}^{B\times N\times D_r},
\]

其中 `B` 为查询 batch，`N=207` 为 METR-LA 节点数，`D_r=64` 为检索维度。两个 Key 水平为：

- `K_OO`：Offset-only epoch 32 encoder 与其指纹严格匹配的 Bank Key；
- `K_OD`：OffsetDecay epoch 41 encoder 与其指纹严格匹配的 Bank Key。

每个 Key 水平独立完成节点候选 Top-K 与权重计算。跨 Key 比较的含义是比较两套训练完成的检索表征系统，而不是只替换查询侧 encoder；查询 Key 与 Bank Key 不能跨指纹混用。

### 2.2 Payload 因素

Payload 是检索到历史事件后，用于构造 Memory 预测的候选 future 值。设历史候选原始 future 为

\[
Y_{b,h,n,k}^{\mathrm{hist}},
\]

查询 context endpoint level 为 `L^q`，候选 context endpoint level 为 `L^k`，则：

\[
P_{\mathrm{OO}}
=Y^{\mathrm{hist}}+(L^q-L^k),
\]

\[
P_{\mathrm{OD},h}
=Y_h^{\mathrm{hist}}+\lambda_h(L^q-L^k),
\qquad
\lambda_h:1\rightarrow0.
\]

`P_OO` 在全部 12 个 horizon 保留完整 endpoint 对齐；`P_OD` 从短期完整对齐线性衰减到远期原始历史 future。Payload 不改变候选 ID、Top-K 或权重。

## 3. 四格实验

| 单元 | Key / 候选与权重 | Payload | 含义 |
|---|---|---|---|
| `K_OO×P_OO` | Offset-only encoder + matching Bank Key | Offset-only | 当前 Offset-only 一致路径 |
| `K_OO×P_OD` | 与上一格完全相同 | OffsetDecay | 固定 Offset-only Key，只改变 payload |
| `K_OD×P_OO` | OffsetDecay encoder + matching Bank Key | Offset-only | 固定 OffsetDecay Key，只改变 payload |
| `K_OD×P_OD` | 与上一格完全相同 | OffsetDecay | 当前 OffsetDecay 一致路径 |

同一 Key 行中的两格必须复用同一个 `NodeCandidates` 对象，因此候选 ID、分数、Top-K 和权重逐元素一致。两个 Bank 必须验证以下非 Key 内容逐元素相同：历史 future、future mask、level、weekday、slot、context/future 索引和 sample ID。

## 4. 归因量

令

\[
E_{k,p,s}=\operatorname{MAE}(K_k,P_p\mid s)
\]

表示 Key 水平 `k`、Payload 水平 `p` 在 Future Surprise 分层 `s` 上的物理单位 Memory MAE。

固定 Key 后的 payload 效应：

\[
\Delta_P(k,s)=E_{k,\mathrm{OO},s}-E_{k,\mathrm{OD},s}.
\]

正值表示改用 OffsetDecay payload 降低误差。

固定 payload 后的 Key 效应：

\[
\Delta_K(p,s)=E_{\mathrm{OO},p,s}-E_{\mathrm{OD},p,s}.
\]

正值表示 OffsetDecay Key 更好，负值表示 Offset-only Key 更好。

交互效应：

\[
I_s=\Delta_P(\mathrm{OO},s)-\Delta_P(\mathrm{OD},s).
\]

`I_s` 接近 0 表示 payload 效应基本不依赖 Key；绝对值较大表示 Key 与 payload 有明显耦合。

由于存在交互时按“先换 Key”或“先换 payload”会得到不同的顺序归因，额外报告对称的平均主效应：

\[
M_P(s)=\frac{\Delta_P(\mathrm{OO},s)+\Delta_P(\mathrm{OD},s)}{2},
\]

\[
M_K(s)=\frac{\Delta_K(\mathrm{OO},s)+\Delta_K(\mathrm{OD},s)}{2}.
\]

二者满足：

\[
M_P(s)+M_K(s)
=E_{\mathrm{OO},\mathrm{OO},s}-E_{\mathrm{OD},\mathrm{OD},s},
\]

因此可以把两个匹配主线之间的总差异做顺序无关的二因子归因。

## 5. 查询、分层与指标

- 与上一诊断完全相同的 1024 个 validation 查询；
- 候选协议固定为 `weekday_radius1_overlap`；
- Future Surprise 阈值由四格共同 target 统一计算；
- 分层为 `all`、`ordinary_80`、`surprise_top20`、`surprise_top5`；
- 四格使用共同有效掩码。

每格报告 MAE、RMSE、MAPE 与 12 步 horizon MAE。额外报告：

- 两种 Key 的 Top-K Jaccard：两个候选集合交集大小除以并集大小；
- 四类逐锚点 paired effect；
- 按完整 query 有放回重采样 5,000 次的 95% bootstrap 区间，保留查询内空间节点相关性。

该 bootstrap 只描述固定 seed 和固定验证集上的抽样不确定性，不替代多 seed 训练。

## 6. Future 信息边界

查询 future 不参与 Key 编码、候选池、Top-K、权重或 Memory 预测。它只在四格预测全部完成后用于计算 Future Surprise、分层和误差。Bank 中的 future 都来自因果历史事件。

## 7. 实现与资源约束

- 新增独立诊断模块、CLI 和单元测试，不改变训练、Bank 或下游接口；
- 使用 `C:\Users\31396\.conda\envs\research\python.exe`；
- 全程 `torch.inference_mode()`；
- 复用每个 Key 行的检索结果，预计峰值显存低于 1 GB，总运行时间约一分钟；
- 先跑少量 query smoke，确认四格形状、Bank 内容审计和配对统计，再运行正式 1024 查询。

## 8. 决策规则

- **Payload 主导**：两个 `\Delta_P` 在 top20/top5 都显著为正，且绝对值明显大于固定 payload 的 `\Delta_K`；保留 Key 研究结论，重新考虑 downstream payload 一致性要求。
- **Key 主导**：固定任一 payload 后 `\Delta_K` 都稳定较大，而 `\Delta_P` 较小；优先重新评估预训练 teacher 与 Key。
- **明显交互**：`I_s` 较大或 payload 效应在两种 Key 下方向相反；不能独立决定 Key 或 payload，需要保留匹配组合并做下游验证。
- **Stop 结构扩展**：该归因结束前，不新增 patch-level level 或其他 context 模块。

## 9. 正式结果（1024 个 validation 查询）

正式实验覆盖 206,399 个有效 `(query,node)` 锚点。两个对角单元分别以绝对差 `0` 和 `4.13×10^-7` 复现上一轮 `K_OO×P_OO` 与 `K_OD×P_OD` 结果。两个 Bank 的 11 个非 Key 文件 SHA-256 全部相同，checkpoint–Bank encoder fingerprint 也分别通过严格验证。

### 9.1 四格 Memory MAE

| Future Surprise 层 | `K_OO×P_OO` | `K_OO×P_OD` | `K_OD×P_OO` | `K_OD×P_OD` |
|---|---:|---:|---:|---:|
| all | 3.2862 | 3.3258 | 3.3145 | **3.1794** |
| ordinary_80 | 1.9301 | 2.0354 | **1.9020** | 1.9349 |
| surprise_top20 | 8.7234 | 8.4995 | 8.9777 | **8.1692** |
| surprise_top5 | 16.6740 | 15.0342 | 17.2004 | **14.6632** |

在普通层，完整 offset 略好；在 surprise 层，decay payload 明显更好。两个匹配主线的 top20/top5 差异分别为 0.5543 和 2.0109。

### 9.2 顺序无关的平均主效应

| 层 | 平均 payload 主效应 `M_P` | query-block 95% 区间 | 平均 Key 主效应 `M_K` | query-block 95% 区间 |
|---|---:|---:|---:|---:|
| all | +0.0477 | [0.0349, 0.0696] | +0.0591 | [0.0507, 0.0685] |
| ordinary_80 | -0.0691 | [-0.0798, -0.0557] | +0.0643 | [0.0562, 0.0744] |
| surprise_top20 | **+0.5162** | [0.4708, 0.5924] | +0.0380 | [0.0078, 0.0672] |
| surprise_top5 | **+2.0885** | [1.9482, 2.3312] | -0.0777 | [-0.1574, 0.0128] |

正 payload 主效应表示 OffsetDecay payload 更好；正 Key 主效应表示 OffsetDecay Key 更好。top20 中 payload 解释匹配主线总差异的约 93%，Key 约 7%；top5 中 payload 主效应甚至略大于总差异，而 Key 主效应接近 0 且区间跨 0。因此 **Future Surprise 长尾差距主要由 payload 决定，不是 Offset-only Key 性能坍缩。**

### 9.3 Key–Payload 交互

条件效应揭示了明显语义适配：

- 固定 `K_OO`，从 `P_OO` 换到 `P_OD`，top20/top5 分别改善 0.2239/1.6399；
- 固定 `K_OD`，相同 payload 切换分别改善 0.8085/2.5372；
- 固定 `P_OO` 时，`K_OO` 在 top20/top5 比 `K_OD` 好 0.2543/0.5264；
- 固定 `P_OD` 时，`K_OD` 反而比 `K_OO` 好 0.3303/0.3710。

top20/top5 的 difference-in-differences 分别为 -0.5846/-0.8974，对应 query-block 95% 区间 `[-0.6273,-0.5275]` 与 `[-1.0226,-0.7611]`。这说明 Key 确实适配了各自 teacher 的 future 语义，但该交互不改变平均归因：极端层的主要误差来源仍是持续使用完整 offset 的 payload。

两套 Key 的 Top-K Jaccard 从 ordinary_80 的 0.6273 降至 top20/top5 的 0.5451/0.5311，说明 surprise 样本上候选选择差异变大；然而其平均 Key 主效应远小于 payload 主效应，所以“选了不同候选”不等于“Key 是主要性能瓶颈”。

### 9.4 Horizon 证据

两种 payload 在 horizon 1 完全相同，因为 `λ_1=1`。在 surprise_top5 中：

- `K_OO` 行的 `P_OO-P_OD` MAE 差从 horizon 1 的 0 增长到 horizon 12 的 3.5759；
- `K_OD` 行的差从 0 增长到 horizon 12 的 5.3139。

该随 horizon 增大的规律与 endpoint offset 是否持续保留完全一致，进一步支持 payload 机制归因。

## 10. 结论与后续边界

1. Offset-only Key 没有被本实验否定，更没有新的坍缩证据；
2. “预训练 teacher 是 Offset-only，因此下游 payload 必须 Offset-only”不是硬约束，交叉前向在工程上和信息边界上都成立；
3. 但 Key 与 payload 存在稳定交互，不能把任意交叉组合直接宣布为最终主线；
4. 当前固定方案中，`K_OD×P_OD` 在全体、top20 和 top5 都最好，而 `K_OD×P_OO` 只在普通层最好；
5. Future Surprise 使用真实 future，不能在部署时直接作为 payload 路由条件。若继续研究自适应 offset 强度，只能使用历史 context 可获得的状态，并必须单独训练和验证；
6. 当前证据不支持增加 patch-level level。下一步应先判断现有 context 交互校准器能否消化 `P_OO` 的远期偏差，再决定是否保留 Offset-only 一致路径。

## 11. 成本与产物

- 正式前向与 5,000 次 query-block bootstrap：36.23 秒；
- 峰值 CUDA 显存：0.451 GB；
- 完整 JSON：`artifacts/diagnostic_key_payload_factorial_1024_seed42/key_payload_factorial.json`；
- 逐锚点数组：同目录 `factorial_anchor_values.npz`；
- 四格图：同目录 `key_payload_factorial_memory_mae.png`。
