# HN-OffsetDecay v2-Transfer：检索器有目的扩容与跨数据集微调计划

更新时间：2026-08-28  
文档定位：HN-OffsetDecay v2-Transfer 的研究与工程计划及当前执行契约。本文区分已实现主线和后续保留方案；未通过阶段性验证前，不启动正式长轮次训练。

> **当前执行版本（2026-08-28）**：本轮只实现有目的的容量扩展和 batch size 24 训练。horizon relation projection 与对应的新增 relation loss 已暂停并从代码路径删除，保留在本文末尾作为后续候选方案；HN-OffsetDecay 主监督、masked single-view one-forward、原有 route/index 加速和 domain adapter 保持不变。

---

## 1. 任务目标与当前基线

当前主线检索器为 **HN-OffsetDecay v1**。它使用单源时空预训练得到 node-level key，并在目标数据集上构建 Bank，供下游 TGGE 校准器检索历史事件。

当前匹配基线必须固定为：

- 预训练模型：HN-OffsetDecay v1；
- retrieval key：48 维；
- 候选协议：当前主线的周时间槽、事件候选和 node-level Top-K 配置；
- future 聚合：OffsetDecay；
- 下游校准器：Base-as-candidate 版本，当前约 27.6 万可训练参数；
- 训练目标：masked reconstruction 与 HN-OffsetDecay relation/retrieval 监督；
- Bank：由同一 encoder 在同一数据划分上构建；
- 训练、验证、测试和 target Bank 的时间边界保持不变。

本计划的目标不是简单增加层数，而是针对三个已观察到的薄弱点进行扩容：

1. context 相似度与 future 相似度之间仍存在偏差；
2. 一个全局 key 难以表达不同 forecast horizon 的关系差异；
3. 后续跨数据集微调缺少清晰的参数分层和可冻结接口。

目标参数规模为检索器约 60--80 万；略高于 70 万可以接受，但不以超过 100 万为目标。扩容后的模型称为 **HN-OffsetDecay v2-Transfer**。

当前执行参数契约为：`hidden_dim=96`、`encoder_layers=4`、`retrieval_dim=64`、`adapter_bottleneck_dim=96`、`pretrain.batch_size=24`。relation projection 不实例化，relation loss 权重不进入配置。实际参数量以启动日志中的 `parameter_counts` 为准；其中主编码器、64 维 retrieval head、可选 domain adapter 和 reconstruction head 分开统计。

---

## 2. 术语和信息边界

### 2.1 Context、Future 和事件

对一个样本，context 表示预测时可见的历史窗口：

\[
\mathbf X_q^{ctx}\in\mathbb R^{T\times N\times C},
\]

其中 \(T\) 为历史长度，\(N\) 为节点数，\(C\) 为变量通道数。紧接其后的真实未来为：

\[
\mathbf Y_q^{future}\in\mathbb R^{H\times N\times C}.
\]

一个 historical event 包含一段历史 context、其后 future 和时间元数据。Bank 保存历史事件的 key、future payload 及必要元数据，不是可训练神经网络参数。

### 2.2 Retrieval key

query key 是当前 context 经 encoder 与 retrieval head 得到的表示：

\[
\mathbf z_q\in\mathbb R^{N\times D_z},
\qquad D_z=64\text{（v2）}.
\]

Bank 中历史事件的 key 记为 \(\mathbf z_j\)。检索阶段只使用 context 计算 \(\mathbf z_q\)，不能使用真实 future。

### 2.3 Horizon relation representation

relation representation 是面向预测步 \(h\) 的辅助表示：

\[
\mathbf r_{q,h,n}\in\mathbb R^{d_r},
\qquad
\mathbf R_q\in\mathbb R^{H\times N\times d_r}.
\]

它用于训练和候选质量诊断，第一阶段不替代 Bank 中的主检索 key。\(\mathbf r\) 的训练监督可以使用 future 构造的 OffsetDecay 关系，但推理时不得读取真实 future。

### 2.4 跨数据集微调

跨数据集微调指在目标数据集训练集上更新指定参数子集，同时保持目标验证集和测试集严格隔离。它不等于把源域 Bank 直接搬到目标域：只要 encoder 更新，就必须用更新后的 encoder 在目标训练历史上重建 Bank。

---

## 3. 扩容的核心判断

当前结果表明，问题更像是“关系表达和候选条件化不足”，而不是 key 维度单纯过小。因此扩容必须同时服务于：

- 更充分地建模长 context 中的时间和空间交互；
- 保留候选 future 在不同 horizon 上的差异；
- 为 adapter-only、顶层解冻和低学习率全量微调提供明确边界。

不采用以下无目的改法：

- 只把所有 hidden dimension 等比例放大；
- 重新引入已经移除的 rank、profile semantic key 或旧 Bank；
- 用未来值参与部署阶段的 query key 或 candidate score；
- 同时修改 encoder、监督函数、Bank 协议和下游校准器后直接比较 MAE。

---

## 4. HN-OffsetDecay v2-Transfer 的结构

### 4.1 输入与 patch embedding

输入仍为：

\[
\mathbf X^{ctx}\in\mathbb R^{B\times T\times N\times C}.
\]

经过 temporal patch embedding 后得到：

\[
\mathbf U\in\mathbb R^{B\times P\times N\times D_h},
\qquad P=T/p.
\]

patch size、归一化、缺失值掩码和一次 masked forward 的工程优化保持不变，不在本计划中重新定义。

### 4.2 扩大的因子化时空编码器

v2 建议使用：

- hidden dimension：96 或 128；
- encoder layers：4 层；
- attention heads：4；
- FFN multiplier：2；
- 保留 temporal factor 和 sparse graph factor 的分解结构；
- 保留已有 route/index/cache 优化和图边稀疏计算。

编码器输出为：

\[
\mathbf H\in\mathbb R^{B\times P\times N\times D_h}.
\]

扩容的目的不是让网络直接记忆 future，而是提升以下能力：

1. 长窗口内的周期、突变和趋势组合；
2. 邻近节点与远程节点之间的非均匀传播；
3. 时空海市蜃楼样本中局部差异的保留。

### 4.3 节点级 pooling 与主 retrieval key

沿 patch 维进行可学习 pooling：

\[
\mathbf h_{q,n}
=
\sum_{p=1}^{P}a_{q,p,n}\mathbf H_{q,p,n},
\qquad
\sum_p a_{q,p,n}=1.
\]

得到：

\[
\mathbf h_q\in\mathbb R^{B\times N\times D_h}.
\]

主 key 由 retrieval head 产生：

\[
\mathbf z_{q,n}
=
\operatorname{Normalize}\left(f_z(\mathbf h_{q,n})\right),
\qquad
\mathbf z_{q,n}\in\mathbb R^{64}.
\]

这个 key 仍然是 Bank 检索的唯一主表示。v2 的 key 维度从 48 扩为 64，扩容后必须重建 Bank。

### 4.4 Horizon relation projection（后续保留，当前不启用）

曾计划在节点 hidden 上增加关系分支：

\[
\mathbf r_{q,h,n}
=
f_r\left(\mathbf h_{q,n}+\mathbf e_h\right),
\]

其中：

- \(\mathbf e_h\in\mathbb R^{D_h}\) 是第 \(h\) 个预测步的 horizon embedding；
- \(f_r\) 是两层 bottleneck projection；
- \(d_r\) 首选 16 或 32；
- 输出 \(\mathbf R_q\in\mathbb R^{B\times H\times N\times d_r}\)。

这条分支表达“当前 query 在第 \(h\) 个 horizon 上的关系状态”，不直接输出预测值，也不替代主 key。

本分支本轮不实例化 `relation_embedding`/`relation_mlp`，也不计算新增 horizon relation loss。单 batch 实测显示它会显著增加显存和耗时，因此 relation projection 仅作为后续保留方案；未来若恢复，必须使用新版本号、独立消融和单独的速度/显存验收。

### 4.5 可冻结 domain adapter

在 retrieval head 前增加可选 adapter：

\[
\mathbf h'_{q,n}
=
\mathbf h_{q,n}
+
W_{up}\,\operatorname{GELU}(W_{down}\mathbf h_{q,n}).
\]

adapter 的 bottleneck 首选 32。其参数单独命名、单独统计、单独保存，使后续可以实现：

- zero-shot：encoder 和 adapter 全部冻结；
- adapter-only：只训练 adapter；
- adapter + relation head：适配目标域关系；
- top-layer tuning：再解冻 encoder 顶层；
- full low-LR tuning：最后才进行全量微调。

adapter 默认不改变主 Bank key 的维度和接口，但 adapter 一旦参与 key 生成，目标 Bank 必须重新构建。

---

## 5. 参数预算与分层

实际参数量以代码统计为准，目标分解如下：

| 部分 | 目标作用 | 目标参数范围 |
|---|---|---:|
| patch embedding | 原始窗口到 token | 2--5 万 |
| 4 层时空 encoder | 时空上下文建模 | 40--55 万 |
| retrieval head | 64 维 node key | 5--10 万 |
| domain adapter | 跨域适配 | 1--5 万 |
| reconstruction head | masked reconstruction | 1--3 万 |
| **总计（不含保留 relation 分支）** |  | **约 50--70 万，实际以日志为准** |

必须分别输出：

\[
\#\theta_{total},
\quad
\#\theta_{retrieval},
\quad
\#\theta_{adapter}.
\]

其中 `retrieval_state_dict()` 必须明确包含会改变 key 的模块；仅用于 reconstruction 的参数不能被误计入 Bank fingerprint。

---

## 6. 监督目标

### 6.1 保留 HN-OffsetDecay 主监督

保持当前 v1 的主要关系教师和 hard-negative 定义，避免把容量变化与监督函数变化混在一起：

\[
L_{pretrain}
=
L_{reconstruction}
+
\lambda_{retrieval}L_{HN\text{-}OD}.
\]

OffsetDecay 关系只在训练阶段使用 future 构造，部署阶段不访问 future。

### 6.2 Horizon relation 辅助监督（后续保留，当前不执行）

对候选事件 \(j\)，训练阶段可以由 query 和 candidate future 构造 horizon-specific 教师距离：

\[
d^{OD}_{q,j,h,n}
=
\operatorname{OffsetDecayDistance}
\left(Y_{q,h,n},Y_{j,h,n}\right).
\]

关系分支产生 query-side 表示后，可用关系得分拟合归一化后的教师关系：

\[
s^{rel}_{q,j,h,n}
=
\operatorname{MLP}_{rel}
\left(
[\mathbf r_{q,h,n},\mathbf x^{cand}_{j,h,n}]
\right),
\]

\[
L_{horizon\text{-}rel}
=
\operatorname{SmoothL1}
\left(s^{rel}_{q,j,h,n},
\widetilde d^{OD}_{q,j,h,n}\right).
\]

该监督方案本轮不执行。第一版建议（仅作为后续保留计划）：

\[
\lambda_{horizon}=0.05.
\]

不建议在当前主线重新加入 relation projection 或 rank loss。只有在扩容版完成匹配速度和候选质量验证后，才可单独恢复该分支并测试局部排序损失。

### 6.3 总损失

\[
L
=
L_{reconstruction}
+
\lambda_{retrieval}L_{HN\text{-}OD}
+
\lambda_{horizon}L_{horizon\text{-}rel},
\qquad \lambda_{horizon}=0\ \text{（当前版本）}.
\]

当前阶段只改变模型容量和 batch size；mask 比例、数据切分、OffsetDecay 定义、hard-negative 策略和优化器保持 v1 一致。新增 relation branch 作为后续独立实验，不与当前结果混报。

---

## 7. 下游如何使用 relation 输出（后续保留）

当前版本没有 relation representation 输出，不写入 Bank，也不改变 Base-as-candidate 的 residual mixture 公式。以下接口仅记录未来恢复 relation projection 时的候选接入方式；当前下游仍只消费 64 维主 key 和既有候选特征。

如果第一阶段确认它有效，第二阶段才接入下游候选打分：

\[
g_{q,h,n,k}
=
f_{score}
\left[
\mathbf r^q_{q,h,n},
\Delta_{q,h,n,k},
|\Delta_{q,h,n,k}|,
s^{key}_{q,n,k},
-d^{level}_{q,n,k},
p_h
\right],
\]

\[
\ell_{q,h,n,k}
=
\ell^{base}_{q,h,n,k}
+
\lambda_{rel}g_{q,h,n,k}.
\]

然后在历史候选和 Base token 上统一 softmax。Base 的 residual 仍为零：

\[
\hat Y
=
Y^{base}
+
\sum_{k=1}^{K}\pi_k
\left(Y^{cand}_k-Y^{base}\right).
\]

此连接方式让 relation 分支影响“候选选择”，而不是直接生成一个绕过 Attention 的校正量。

---

## 8. 跨数据集微调协议

### Protocol A：Zero-shot

- 源域训练完成后冻结 encoder、relation head 和 adapter；
- 在目标训练历史上用冻结 encoder 编码并重建目标 Bank；
- 不更新任何参数；
- 评估候选质量和下游 Base-as-candidate。

回答的问题是：源域表示是否具有跨数据集泛化能力。

### Protocol B：Adapter + relation head

- 冻结主 encoder；
- 只训练 domain adapter 和 horizon relation head；
- 用目标训练集的 future 构造训练监督；
- 每次 adapter 改变 key 时重建目标 Bank。

这是首选微调协议，因为参数少、归因清晰、较不容易破坏源域结构。

### Protocol C：顶层 encoder 微调

- 保留 adapter 和 relation head 的可训练状态；
- 解冻 encoder 最后一层或最后一个时空 block；
- encoder 学习率设置为 head 的 0.1 倍；
- 仅在 Protocol B 有收益但仍存在明显域偏移时进行。

### Protocol D：全量低学习率微调

只有在 A--C 都表明目标域适配仍不足时执行：

\[
\eta_{encoder}=0.05\sim0.2\,\eta_{head}.
\]

每个协议保存独立 checkpoint，不能覆盖源域模型。

---

## 9. Bank、checkpoint 与未来信息边界

### 9.1 Bank 一致性规则

以下任一项变化后必须重建 Bank：

- encoder 参数；
- adapter 参数；
- retrieval head 参数；
- retrieval key 维度；
- 输入归一化或节点映射。

Bank metadata 至少记录：

- encoder fingerprint；
- retrieval key dimension；
- dataset name；
- node count；
- context/horizon length；
- normalization fingerprint；
- candidate protocol；
- seed。

### 9.2 Checkpoint contract

扩容模型 checkpoint 必须保存：

```text
model_state_dict
retrieval_state_dict
relation_head_state_dict
adapter_state_dict
config
normalizer
metrics
epoch
seed
encoder_fingerprint
```

旧 v1 checkpoint 不得静默加载到 v2。若结构不匹配，应明确报错并提示需要从 v2 配置重新预训练。

### 9.3 信息边界

训练阶段允许使用真实 future 构造：

- HN-OffsetDecay teacher；
- horizon relation teacher；
- hard-negative 标签。

验证和部署阶段的 query 编码、key 检索、candidate scoring 不得读取 query 的真实 future。真实 future 只能用于离线指标计算和 oracle 诊断。

---

## 10. 实施范围

预计需要修改的模块：

1. `stanchor/models/encoder.py`：支持 v2 hidden dimension、4 层和现有 route 配置；
2. `stanchor/models/pretraining.py`：导出 adapter 与 retrieval-state，保留单次 masked forward；
3. `stanchor/models/retrieval_head.py`：支持 64 维 key 与可选 domain adapter，不输出 horizon relation；
4. `stanchor/config.py`：严格校验当前 v2 维度与训练字段；
5. `stanchor/engine/pretrainer.py`：保留 HN-OffsetDecay 指标、batch 清理和无 early stopping 的完整训练；
6. `scripts/pretrain.py`：支持 v2 配置和无 early stopping 的完整训练；
7. `scripts/build_bank.py`：写入 encoder/key/config fingerprint；
8. 下游接口：第一阶段保持 Base-as-candidate 输入兼容，不提前改校准器预测公式。

以下内容保持不变：

- HN-OffsetDecay v1 的数据切分边界；
- 当前 masking 的单次前向优化；
- 索引检索和缓存加速策略；
- OffsetDecay future 聚合定义；
- Base-as-candidate 校准器主逻辑；
- 下游 Graph WaveNet、STGCN、STAEformer 和 ARGCN 的 backbone 实现。

---

## 11. 分阶段验证与停止标准

### Stage 0：结构与参数检查

必须确认：

- key shape 为 `[B,N,64]`；
- 总参数在约 50--70 万附近，具体以启动日志为准；
- adapter 和 encoder 参数可分别冻结；
- v1 checkpoint 不会静默错配；
- retrieval fingerprint 包含所有会改变 key 的参数。

### Stage 1：单 batch forward/backward

检查：

- masked 单次 forward 仍然有效；
- reconstruction 与 HN-OffsetDecay retrieval loss 均为有限值；
- 所有可训练参数梯度有限；
- 不访问 future 的 query 推理路径能独立运行；
- 显存峰值和 v1 相比没有异常增长。

### Stage 2：3--5 轮短训练

严格比较：

1. HN-OffsetDecay v1；
2. 当前 v2-Transfer（hidden 96、4 层、key 64、batch 24）；
3. 如需恢复 relation projection，必须另建独立实验，不得与本轮结果合并。

每组只改变一个主要因素，记录：

- reconstruction loss；
- HN-OffsetDecay loss；
- candidate Top-1/Top-5 future MAE；
- oracle Top-1 MAE；
- valid candidate ratio；
- key/future 关系指标；
- 单轮时间；
- CUDA 峰值显存。

### Stage 3：Bank 重建与下游短验证

使用每个模型自己的 Bank，在 GWN 和 STGCN 上做 1--3 轮下游验证，保持：

- 相同 Base checkpoint；
- 相同 candidate protocol；
- 相同校准器；
- 相同 seed 和训练 batch；
- 相同 loss 与 early-stopping 设置。

### Stage 4：跨数据集微调

在目标数据集上依次执行 Protocol A、B、C。只有当轻量 adapter 微调已经验证有效，才执行全量低学习率微调。

---

## 12. 成功、持平与失败标准

### 成功

满足以下至少两项：

- Top-5 candidate future MAE 相对 v1 改善至少 0.05；
- oracle Top-1 不退化；
- 下游至少一个 backbone 的验证 MAE 改善至少 0.01；
- horizon relation 对候选质量或下游 Attention 有可解释改善；
- adapter-only 微调在目标数据集优于 zero-shot；
- 单轮耗时增加不超过 2 倍，显存仍在设备预算内。

### 持平

如果最终 MAE 变化小于 0.005，但：

- 跨域 adapter 能稳定工作；
- 候选质量不退化；
- 训练和 Bank fingerprint 可复现；
- 扩容没有引入明显时间/显存问题；

则可以保留 v2 作为迁移增强版本，但不声称它全面提升排序。

### 失败与回退

出现以下任一情况时停止扩容方向并回退 v1：

- candidate Top-5 和下游 MAE 同时退化；
- 扩容后的候选 future MAE 不改善且训练成本明显增加；
- adapter 微调破坏源域性能且无法通过低学习率缓解；
- 新 Bank 与 encoder fingerprint 不一致；
- 单轮耗时或显存超过设备预算；
- 出现 NaN/Inf、未来泄漏或不可复现的候选集合。

---

## 13. 最终研究判断

检索器扩容是合理的，但扩容的科学目标必须明确为：

> 提升时空上下文表示、保留 horizon-specific relation，并为跨数据集参数高效微调建立可冻结的表示分层。

它不应被表述为“参数更多所以一定更强”，也不应承诺消除时空海市蜃楼造成的 oracle gap。v2 的正确验证路径是：

\[
\text{结构检查}
\rightarrow
\text{单 batch 梯度}
\rightarrow
\text{短程候选质量}
\rightarrow
\text{Bank 重建}
\rightarrow
\text{下游匹配验证}
\rightarrow
\text{跨域 adapter 微调}.
\]

只有当每一步都满足对应标准，才进入下一步；任何一步失败都应保留 v1 作为可复现主线。

## ����ʵ�ֲ��䣺horizon relation supervision

v2 �� relation �������ͨ�� `relation_loss_weight=0.1` ��������ʧ����ÿ�� horizon �� relation token��ʹ������ HN-OffsetDecay һ�µ� future teacher ����ڵ�����ֲ������� relation token �� cosine similarity �ֲ����㽻���أ�

\[
L_{h-rel}=\frac{1}{|\mathcal A|H}\sum_{q,h,n\in\mathcal A}
-\sum_j p^{teacher}_{q,j,n}\log p^{student}_{q,j,h,n}.
\]

����ʧΪ��

\[
L=2.0L_{recon}+1.0L_{HN-OD}+0.1L_{h-rel}.
\]

���� `L_HN-OD` ����������Ŀ�꣬`L_recon` �� masked reconstruction ����Ŀ�꣬`L_h-rel` �ǽ�С�� horizon relation �����ල��ѵ���׶� teacher ����ʹ����ʵ future ���죬��������׶β���ȡ future��
