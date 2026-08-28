# Event Key 暂停与部署检索 Future 有效性可视化方案

## 1. 目的

当前 METR-LA `exact_calendar` 协议先按相同 weekday、相同 5 分钟时间槽和严格因果约束筛选历史事件。完整验证集 2,993 个 query 的合法事件数平均为 8.014、最少 5、最多 9；当前 `event_top_r=32`，因此事件级 Top-R 不会实际删除候选。event key 又由 207 个节点 key 均值后归一化得到，可能抹平节点差异，且没有独立事件级预训练目标。

本方案完成两项工作：

1. 在下游部署检索路径中暂停 event key，直接对周时间槽与因果过滤后的全部合法事件执行节点级 Top-5；
2. 以完整验证集和固定规则可视化检索 future 的有效性，并将两类时空海市蜃楼由单对案例改为群体案例图。

## 2. 检索路径

合法事件集合定义为

\[
\mathcal E_q=\{j:\operatorname{weekday}(j)=\operatorname{weekday}(q),
\operatorname{slot}(j)=\operatorname{slot}(q),
t_j^{future}<t_q^{context}\}.
\]

暂停 event key 后，每个节点直接计算

\[
s_{q,n,j}=\cos(z_{q,n},z_{j,n}),\qquad j\in\mathcal E_q,
\]

并保留

\[
\mathcal C_{q,n}=\operatorname{TopK}_{j\in\mathcal E_q}s_{q,n,j},\qquad K=5.
\]

该改动不改变 node key、预训练损失、Bank future payload 或 OffsetDecay，因此不重新训练编码器、不重建 Bank。为兼容历史产物，Bank 中的 `event_keys.npy` 暂不删除，仅停止在当前下游主路径中读取它进行排序。

## 3. 部署检索 Future 有效性

候选排序只能使用 query history、calendar、历史 Bank key 和历史元数据。query 真实 future 仅在排序完成后用于验证指标和作图。

主要展示：

- 检索 Top-1 future 与 query future 的 MAE/RMSE；
- Top-5 OffsetDecay 聚合 future 与 query future 的 MAE/RMSE；
- Random key、Weekly mean 与 Oracle Top-1 对照；
- 12 个 horizon 的 MAE 曲线，判断相似性是否随预测距离衰减；
- learned 与 oracle gap，区分“候选池无价值”和“候选排序/聚合不足”。

## 4. 时空海市蜃楼群体图

类型 A 满足 context distance 不高于 P8、future distance 不低于 P92、key distance 不低于 P92；类型 B 满足 context distance 不低于 P92、future distance 不高于 P8、key distance 不高于 P8。

每类从满足阈值的候选中按固定排序规则选择 12 对，即 24 条 context/future 曲线，并在同一坐标系叠加。曲线使用低透明度展示群体分布，以中位数粗线展示趋势；横轴仅标记起点、中点和终点，减少高频小细节干扰。Key PCA 仅绘制目标样本点、连线、两组中心和置信椭圆，不绘制全 Bank 黑色背景点。

## 5. 验证与保留标准

- node-only 路径必须在合法事件数小于等于 `event_top_r` 时保持候选空间完整；
- smoke 验证需确认输出 shape、候选有效数和无 NaN/Inf；
- 完整图表必须基于全部验证 query 或固定规则样本集合，不使用人工挑选；
- 报告必须注明 future 信息边界，Oracle 只作为不可部署上界；
- 若 node-only 路径与当前数学语义一致，则作为后续主线；event key 字段仅为旧 Bank 兼容保留。
