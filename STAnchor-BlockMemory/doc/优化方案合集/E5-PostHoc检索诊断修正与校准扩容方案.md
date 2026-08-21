# E5 PostHoc 检索诊断修正与校准扩容方案

## 1. 目的

本方案把 STAnchor-BlockMemory 定义为一个面向已训练下游预测器的后置风险诊断与检索修正模块。下游预测器先独立训练并保存 checkpoint；随后固定下游预测器和预训练检索编码器，只训练结构化误差修正器 `StructuredErrorCorrector`。

该协议解决两个现有问题：

1. 当前 `base warm-up -> calibrator warm-up -> joint` 会在最后继续更新下游 backbone，无法把最终增益完全归因于检索修正。
2. 当前风险诊断与融合模块参数量很小，需要在固定 base 的前提下验证适度扩容是否有效；同时必须删除纯 Latent48 Bank 下重复的 profile/latent 相似度输入。

## 2. 模块与信息边界

给定历史输入

\[
X\in\mathbb R^{B\times T\times N\times C},
\]

其中 \(B\) 为 batch 大小，\(T=12\) 为下游历史长度，\(N\) 为节点数，\(C=1\) 为速度通道数。已训练下游模型输出

\[
\widehat Y^{\mathrm{base}}=f_{\theta^\star}(X),
\qquad
\widehat Y^{\mathrm{base}}\in\mathbb R^{B\times H\times N\times C},
\]

其中 \(H=12\)，且 \(\theta^\star\) 在 PostHoc 训练中固定。

冻结的 Global288 Latent48 检索编码器只读取 288 步可见历史，生成 48 维 L2 归一化 key。检索器从 Bank 中选出历史相似事件，并聚合这些已经发生事件的 future payload：

\[
\widehat Y^{\mathrm{mem}}
=
\sum_{j\in\mathcal R(q)}\pi_jY_j^{\mathrm{future}}.
\]

Bank payload 可以使用历史训练事件的 future；当前 query 的 future 不能进入 query key、候选选择、诊断特征或推理计算。

风险头根据历史和停止梯度的 base prediction 预测基础误差风险：

\[
\widehat r=g_\phi\!\left(X,\operatorname{StopGrad}(\widehat Y^{\mathrm{base}})\right).
\]

九个部署可用诊断特征经过逐特征 shape function 后得到融合权重：

\[
w
=
\sigma\!\left(b+\sum_{d=1}^{9}f_d(z_d)\right),
\qquad
w\in[0,1].
\]

最终修正为

\[
\widehat Y^{\mathrm{final}}
=
\widehat Y^{\mathrm{base}}
+w\left(\widehat Y^{\mathrm{mem}}-\widehat Y^{\mathrm{base}}\right).
\]

当 memory 无效时强制 \(w=0\)，最终输出严格回退到 base prediction。

九个特征依次为：predicted base risk、完整 Latent48 retrieval similarity、Top-1/Top-2 score margin、Top-5 effective support、payload dispersion、direction agreement、level match、memory/base disagreement 和 horizon position。旧实现中的 `profile_scores` 与 `latent_scores` 在纯 Latent48 Bank 下都回退到同一个 `shape_scores`，因此不能作为两个独立特征保留。

## 3. 训练协议

新增 `posthoc_frozen_base` 协议：

1. 加载已有 `base_only` downstream checkpoint。
2. 只加载其中 `backbone.*` 参数，严格检查张量名称和形状。
3. 记录加载后的 backbone 指纹。
4. 冻结 backbone、confidence head、旧 fusion 和检索编码器。
5. 只训练 `StructuredErrorCorrector`，其内部包含风险分支、证据分支、联合门控和可解释特征贡献。
6. checkpoint 保存 base 来源、base 指纹、训练协议和完整下游状态。
7. 训练结束再次校验 backbone 指纹，若发生变化立即报错。

该协议不再执行 base warm-up 和 joint stage；`target.epochs` 全部用于 calibrator-only 训练。

## 4. 参数容量对照

两个实验共用同一个 base checkpoint、Latent48 checkpoint、Bank、候选协议、seed 和训练数据。

| 版本 | `risk_hidden_dim` | `fusion_feature_hidden_dim` | 校准参数量 |
|---|---:|---:|---:|
| Structured Error Corrector | 256 | 128 | 224,142 |

扩容同时实现文档规定的风险分支、检索证据分支、256 维联合状态、256 维交互门控、horizon 输出头和逐特征 shape function；不增加新特征或新的 future target，因此仍可逐特征解释。

## 5. 公平性与判定

主比较为同一固定 base 上的：

1. `base_only`；
2. `Structured Error Corrector`。

必须报告整体及 15/30/45/60 分钟 MAE、RMSE、MAPE，risk Spearman、risk AUROC/AUPRC、fusion-weight 分位组 helpful rate、blend target 误差、参数量、每轮训练时间和推理开销。

保留 Structured Error Corrector 的条件是：相对固定 base 的 test MAE 有稳定改善，或风险排序指标在多个 seed 上一致改善，同时 RMSE 不出现系统性退化。

若 Structured Error Corrector 不能稳定超过固定 base，则不能把现有联合训练收益解释为即插即用修正收益；joint 版本只能作为联合微调上界。

旧 BaseCap/Wide 队列和旧配置已废弃，不得作为当前主模型入口；正式实验必须使用同一个 frozen base、Latent48 checkpoint、Bank 和 seed，加载当前 `StructuredErrorCorrector` 配置重新训练。
