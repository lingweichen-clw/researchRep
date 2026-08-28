# 候选质量监督与 Residual Aggregation 优化方案

## 目标

当前候选池包含高质量 future，但绝对 future 加权聚合会抹平候选差异，且校准器 attention 与候选真实误差相关性不足。因此本方案只改下游校准训练：保留检索器、E5 encoder、base/backbone 和 K=5，使用训练阶段真实 future 构造候选质量 teacher，监督 attention；推理阶段使用相对 base 的 residual 聚合。

## 张量与公式

候选 future 为 `Y_cand [B,H,N,K,C]`，base 预测为 `Y_base [B,H,N,C]`，attention 为 `a [B,H,N,K]`。候选 residual：

\[
R_{q,h,n,k}=Y^{cand}_{q,h,n,k}-Y^{base}_{q,h,n}.
\]

训练 teacher 使用训练 query 的真实 future（仅训练监督，不进入部署输入）：

\[
e_{q,h,n,k}=\frac1C\sum_c|Y^{cand}_{q,h,n,k,c}-Y_{q,h,n,c}|,
\quad
\pi_{q,h,n}=\operatorname{softmax}(-e_{q,h,n,k}/\tau_q).
\]

候选质量损失为 masked KL：

\[
L_{candidate}=\sum_k\pi_k(\log\pi_k-\log a_k).
\]

Residual memory：

\[
R^{memory}_{q,h,n}=\sum_k a_{q,h,n,k}R_{q,h,n,k},
\quad
Y^{memory}=Y^{base}+R^{memory}.
\]

最终预测仍为：

\[
\hat Y=Y^{base}+\alpha R^{memory}+\beta.
\]

## 实现约束

- base、预训练 encoder 和检索器保持冻结；
- 默认 `candidate_quality_weight=0.05`、`candidate_quality_temperature=0.1`；
- 当前 `forecast_only` 仍保留，新增 candidate loss 只在训练阶段叠加；
- 通过配置开关可回退为旧版；
- 不新增大模块，参数量不变，单轮时间目标不增加显著开销；
- 验证阶段不能使用 candidate quality teacher。

## 验证与停止规则

先做 1 batch 前反向和 3 轮 GWN/STGCN 验证，记录 MAE、RMSE、15/30/60 分钟 MAE、candidate loss、attention-error Spearman、单轮时间。若两个 backbone 至少一个改善 >=0.01 且无另一模型恶化 >0.01，则进入 50 轮；否则回退并停止该方向。