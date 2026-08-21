# TGGE v3 Horizon-Aware Signed-Direction Error Corrector 优化方案

## 1. 文档目的

本文档定义当前 TGGE v3 主线针对下游预测平台期的新一版优化方案。本文档自包含：读者无需翻阅旧报告即可理解问题、术语、公式、代码改动、未来信息边界、训练协议、验证指标和保留/删除标准。

本方案只作用于后置诊断校正器和当前主检索接口，不重新训练 TGGE 编码器，不改变 Graph WaveNet 或 STGCN 的 backbone 结构，不改变 METR-LA 数据划分和 Memory Bank 候选协议。

## 2. 已知问题与证据

当前两个下游模型的 MAE 在约 2.84–2.86 附近进入平台期。线 A 反事实验证已经给出以下证据：

- Memory 单独优于 base 的位置比例约为 41%–42%，说明 Memory 不是处处有益，不能直接增大固定融合权重。
- Oracle binary gate 的 MAE 约为 2.19，oracle continuous alpha 的 MAE 约为 2.0，明显低于当前 learned gate 的约 2.84–2.86。
- 因此候选历史块包含可利用的信息，主要不确定性在于：校正器能否根据部署时可见证据判断当前 horizon 是否应该修正、修正方向是什么以及修正幅度多大。

## 3. 术语定义

### 3.1 Base prediction

Base prediction 是已经训练完成的下游预测器输出，记为

$$
Y^{base}\in\mathbb{R}^{B\times H\times N\times C}.
$$

其中 B 是 batch 大小，H=12 是未来 12 个 5 分钟步，N=207 是 METR-LA 节点数，C=1 是速度通道。PostHoc 训练期间 base 参数被冻结。

### 3.2 Candidate future

Candidate future 是从合法历史事件中检索得到的每个候选节点未来 payload，保留完整候选维度：

$$
Y^{cand}\in\mathbb{R}^{B\times H\times N\times K\times C},
$$

其中 K=5 是每个 query node 的候选数。候选 future 是历史事件已经发生后的历史 payload，不是当前 query 的真实未来。

### 3.3 Candidate weight

检索器根据 query key 与候选 key 的相似度生成非负权重 w(q,n,k)，在有效候选上归一化：

$$
\sum_k w_{q,n,k}=1.
$$

这些权重只由历史 query、Latent48 key、日历过滤和因果候选协议产生，不使用当前 query 的真实 future。

### 3.4 Correction offset

候选相对于 base 的修正偏移为：

$$
\Delta^{cand}_{q,h,n,k,c}=Y^{cand}_{q,h,n,k,c}-Y^{base}_{q,h,n,c}.
$$

它表示第 h 个 horizon 上，第 k 个历史候选相对于 base 的建议修正方向和幅度。

## 4. 核心改动

### 4.1 保留 candidate future，不提前压缩

旧流程先把候选 future 聚合成一个 memory prediction，再从聚合结果计算诊断特征。新流程在诊断阶段保留 Y(cand) 的 K 维，以便区分不同 horizon 的候选一致性和分歧。

### 4.2 Horizon-aware 偏移统计

定义候选偏移加权均值：

$$
\mu_{\Delta,q,h,n,c}=\sum_k w_{q,n,k}\Delta^{cand}_{q,h,n,k,c}.
$$

定义候选偏移方差：

$$
\sigma^2_{\Delta,q,h,n,c}=\sum_k w_{q,n,k}(\Delta^{cand}_{q,h,n,k,c}-\mu_{\Delta,q,h,n,c})^2.
$$

对通道取平均后得到：

- delta_mean_abs：|mu_delta| 的通道平均，表示当前 horizon 候选建议修正的平均幅度；
- delta_std：sqrt(sigma_delta^2) 的通道平均，表示候选之间修正幅度和方向的分歧程度。

这两个特征保留 horizon 维度，不再把同一个节点级标量复制给全部 12 个 horizon。

### 4.3 保留有符号方向

候选修正符号为：

$$
 s_{q,h,n,k,c}=\operatorname{sign}(\Delta^{cand}_{q,h,n,k,c}).
$$

有符号方向一致性为：

$$
 a^{signed}_{q,h,n,c}=\frac{\sum_k w_{q,n,k}s_{q,h,n,k,c}}{\sum_k w_{q,n,k}}.
$$

通道平均后得到 signed_direction：

- 正值：候选整体建议向上修正；
- 负值：候选整体建议向下修正；
- 接近 0：候选方向相互抵消或证据不一致。

同时保留 a(abs)=|a(signed)| 作为 direction_agreement，表示方向一致程度。新方案不再只保留绝对值。

## 5. 诊断器输入

新版 error-aware 诊断特征共 12 维：

1. predicted base risk；
2. retrieval similarity；
3. top-1/top-2 score margin；
4. effective support；
5. payload dispersion；
6. absolute direction agreement；
7. signed direction；
8. level match；
9. memory disagreement；
10. normalized horizon position；
11. delta_mean_abs；
12. delta_std。

输出特征形状为 [B,H,N,12]。其中第 7、11、12 项是本方案新增的主要 horizon-aware 证据；第 2、3、4、8 项仍是节点级静态上下文，用于辅助判断，不再被误认为完整 horizon-specific 证据。

## 6. 融合方式与信息边界

最终预测仍采用有界残差融合：

$$
Y^{final}=Y^{base}+\alpha\odot(Y^{memory}-Y^{base}),
$$

其中 alpha 属于 [0,1]，由 StructuredErrorCorrector 预测；Memory 无效时强制 alpha=0。

训练阶段可以使用训练样本真实 future 生成校正监督目标；验证和测试阶段真实 future 只用于计算评价指标。真实 future 不进入 query key、候选排序、候选权重、部署特征或模型前向。

## 7. Latent-only 检索接口清理

当前 TGGE v3 的 key 是完整 Latent48 表示，配置为 profile_dim=0、profile_weight=0.0。因此当前主检索路径：

- 直接使用完整 latent key 计算相似度；
- 不再执行 profile/latent 分块 cosine；
- NodeCandidates 不再暴露 profile_scores 和 latent_scores；
- 不再使用 profile_weight_override 参与主检索。

历史 profile 代码仅作为旧实验兼容内容保留，不属于本方案的主训练路径。

## 8. 训练协议与成本约束

固定条件：

- 数据集：METR-LA；
- 下游：Graph WaveNet、STGCN；
- seed：42；
- Bank：TGGE single-view v3 reconstruction2；
- event Top-R：32；
- node Top-K：5；
- candidate protocol：exact_calendar；
- level weight：0；
- batch size：32；
- optimizer：Adam；
- base：加载已训练 checkpoint，并在 PostHoc 阶段冻结；
- 新增模块：只训练 StructuredErrorCorrector。

新版校正器参数量实测为 224,817，与 TGGE 编码器约 300k 处于同量级。新增特征计算复杂度为 O(BHNKC)，不增加检索次数、不增加 backbone 参数。

每个正式验证阶段先跑 5 epoch。若单 epoch 超过 5 分钟，则优先删除 delta_std，保留 signed direction 和 delta mean；若仍超时，回退到仅增加 signed direction 的版本。

## 9. 验证矩阵

Graph WaveNet 和 STGCN 各运行 5 epoch，严格固定上面的 seed、Bank、候选协议、batch size、optimizer 和冻结 base。

必须记录：

- MAE：平均绝对误差；
- RMSE：均方根误差；
- helpful rate：Memory 融合后误差小于 base 的位置比例；
- risk Spearman：预测 base risk 与真实 base error 的 Spearman 秩相关；
- 每个 epoch 用时；
- 校正器参数量；
- 15、30、45、60 分钟 horizon-wise MAE/RMSE。

标准训练日志记录 MAE、RMSE、epoch time 和参数量；helpful rate 与 risk Spearman 必须通过额外 evaluation/diagnostic 步骤计算，不能从训练 loss 代替。

决策标准：

- 两个下游平均 MAE 相对旧版至少下降 0.02，且单 epoch 不超过 5 分钟：保留完整 12 维特征；
- MAE 基本不变但 helpful rate 或 risk Spearman 明显改善：保留为诊断增强，不宣称预测增益；
- 新增特征无任何指标改善：删除 delta_std；
- signed direction 也无效：回退到旧版 9 维特征，并停止继续扩大校正器。

## 10. 工程验收

正式训练前必须通过：

1. py_compile 或 compileall；
2. [B,H,N,12] 特征形状检查；
3. 单 batch 前向与反向；
4. loss、gate、prediction 全部 finite；
5. 冻结 base 参数梯度为 0；
6. checkpoint 保存和加载；
7. smoke 产物清理并确认路径不存在。

正式实验结束后只保留正式 run 目录中的日志、配置、指标和最佳 checkpoint；删除所有 smoke、debug、max_batches 和临时输出。

## 11. 当前方案结论

本方案的核心不是增加一个更大的 MLP，而是避免在诊断之前丢失候选 future 的 horizon 和候选维度信息，并恢复修正方向的符号。它针对线 A 暴露的主要缺陷：候选池有信息，但 learned gate 无法判断何时、向哪边、以多大幅度使用 Memory。

在 5 epoch 双下游验证完成前，本方案只能被称为待验证优化方案；不能提前声称 MAE 改善或检索器已经达到最终上限。
