# ST-SSDL 复现实验与可视化问题分析

更新时间：2026-07-07

## 1. 分析范围

本文梳理当前 `TrafficRobustST` 中已经复现的 ST-SSDL 相关实验，重点回答四个问题：

1. 各个 ST-SSDL 训练/消融实验结果如何。
2. 每个可视化图表是怎么做的，横纵坐标是什么。
3. 每张图暴露了 ST-SSDL 的什么问题，依据是什么。
4. Figure 10-like 缺失值场景里 ground truth 近似直线是否有问题。

涉及的主要代码和结果文件：

```text
src/train.py
src/losses.py
src/data.py
src/preprocessing.py
src/models/stssdl_baseline.py
src/visualize_stssdl.py

log/metrla_stssdl_full/
log/metrla_stssdl_wo_lcon/
log/metrla_stssdl_wo_ldev/
log/metrla_stssdl_wo_lcon_ldev/
log/metrla_stssdl_wo_ssdl/
```

当前可视化只对 `log/metrla_stssdl_full` 生成了完整图表与 CSV。

## 2. 数据与模型流程复核

### 2.1 数据通道

当前 ST-SSDL 复现使用三通道输入：

```text
x[..., 0] = 当前交通值
x[..., 1] = time_in_day
x[..., 2] = history anchor
```

代码依据：

```text
src/data.py:28  prepare_x_y()
src/data.py:35  x0 = x[..., 0:1]
src/data.py:36  x_cov = x[..., 1:2]
src/data.py:37  x_his = x[..., 2:3]
```

历史锚点由训练段内相同 `weekday-slot` 的历史均值构造：

```text
src/preprocessing.py:51   build_history_anchor()
src/preprocessing.py:116  history_anchor = build_history_anchor(...)
src/preprocessing.py:117  data = np.stack([values, tod, history_anchor], axis=-1)
```

因此模型实际比较的是：

```text
Xc: 当前输入窗口，形状 (B,T,N,1)
Xa: 当前输入窗口时间对齐的历史均值锚点，形状 (B,T,N,1)
```

### 2.2 ST-SSDL 核心模块

`STSSDLBaseline` 的 prototype 查询逻辑：

```text
src/models/stssdl_baseline.py:139  query_prototypes()
src/models/stssdl_baseline.py:141  att_score = softmax(query @ prototypes.T)
src/models/stssdl_baseline.py:143  topk(att_score, k=2)
src/models/stssdl_baseline.py:147  return value, query, pos, neg, mask, att_score
```

当前/历史双分支的距离：

```text
src/models/stssdl_baseline.py:173  latent_dis = |q_t - q_a|_1
src/models/stssdl_baseline.py:174  prototype_dis = |p_t - p_a|_1
```

训练损失：

```text
src/losses.py:33  L_MAE
src/losses.py:35  L_Con = triplet_margin_loss(query_c, pos_c, neg_c)
src/losses.py:44  L_Dev = L1(latent_dis.detach(), prototype_dis)
src/losses.py:55  total = L_MAE + lambda_c L_Con + lambda_d L_Dev
```

## 3. 训练与消融结果

### 3.1 实验设置

五个 ST-SSDL 实验使用同一数据、同一主干、同一 seed：

```text
processed_dir = data/METRLA
seed = 999
batch_size = 128
epochs = 200
model = baseline
```

消融项：

| 实验 | use_ssdl | L_Con | L_Dev | 目的 |
|---|---:|---:|---:|---|
| `metrla_stssdl_full` | True | True | True | 完整 ST-SSDL |
| `metrla_stssdl_wo_lcon` | True | False | True | 去掉对比损失 |
| `metrla_stssdl_wo_ldev` | True | True | False | 去掉偏差距离一致性损失 |
| `metrla_stssdl_wo_lcon_ldev` | True | False | False | 保留 prototype 注入，但去掉两个辅助损失 |
| `metrla_stssdl_wo_ssdl` | False | False | False | 去掉整个 SSDL/prototype 分支 |

### 3.2 结果汇总

| 实验 | best val MAE | test MAE | test RMSE | test MAPE | test 15min MAE | test 30min MAE | test 60min MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 2.7456 | 2.9491 | 6.0493 | 8.12% | 2.6379 | 2.9852 | 3.3983 |
| w/o LCon | 2.7417 | 2.9503 | 6.0456 | 8.18% | 2.6406 | 2.9892 | 3.3959 |
| w/o LDev | 2.7507 | 2.9498 | 6.0487 | 8.14% | 2.6411 | 2.9878 | 3.3933 |
| w/o LCon+LDev | 2.7570 | 2.9638 | 6.0875 | 8.15% | 2.6541 | 3.0034 | 3.4102 |
| w/o SSDL | 2.7413 | 2.9579 | 6.0802 | 8.12% | 2.6398 | 2.9929 | 3.4143 |

日志依据：

```text
log/metrla_stssdl_full/train.log
log/metrla_stssdl_wo_lcon/train.log
log/metrla_stssdl_wo_ldev/train.log
log/metrla_stssdl_wo_lcon_ldev/train.log
log/metrla_stssdl_wo_ssdl/train.log
```

### 3.3 训练结果暴露的问题

#### 问题 1：辅助损失对预测指标的贡献很弱

完整模型 test MAE 是 `2.9491`。

去掉 `L_Con` 后 test MAE 是 `2.9503`，只差 `0.0012`。

去掉 `L_Dev` 后 test MAE 是 `2.9498`，只差 `0.0007`。

这说明：

```text
在当前复现设置下，L_Con 和 L_Dev 对最终预测误差的边际贡献非常小。
```

如果论文声称两个自监督损失是核心有效模块，那么我们的复现实验没有给出强支持。

#### 问题 2：去掉整个 SSDL 后预测也没有明显崩掉

`w/o SSDL` 的 test MAE 是 `2.9579`，比 full 只差 `0.0088`。

同时 `w/o SSDL` 的 best val MAE 是 `2.7413`，甚至略低于 full 的 `2.7456`。

这说明：

```text
预测能力主要可能来自 AGCRN encoder-decoder、time embedding、node embedding 和动态图主干，
而不是稳定来自 prototype-based SSDL 模块。
```

#### 问题 3：两个辅助损失同时去掉才有轻微退化

`w/o LCon+LDev` 的 test MAE 是 `2.9638`，比 full 高 `0.0147`。

这个结果说明 prototype value 注入或隐空间约束可能仍有一点正则作用，但幅度很小，不足以证明论文式 prototype space 已经学到了清晰的科学信息空间。

## 4. 可视化实验总览

当前完整可视化输出位于：

```text
log/metrla_stssdl_full/visualization/
```

图表文件：

```text
fig4_like_deviation_cases.png
fig5_like_pca_queries_prototypes.png
fig6_like_physical_prototype_patterns.png
fig7_like_low_deviation_stssdl.png
fig8_like_medium_deviation_stssdl.png
fig9_like_high_deviation_stssdl.png
fig10_like_missing_values_stssdl.png
fig_prototype_usage_hist.png
```

表格文件：

```text
metrics_by_deviation.csv
prototype_alignment_by_deviation.csv
prototype_pattern_summary.csv
prototype_usage.csv
```

可视化数据采样：

```text
split = test
max_samples = 4096
query sample count for Fig5 = 400
top prototype count for Fig5 = 7
```

## 5. Prototype Usage 图

文件：

```text
log/metrla_stssdl_full/visualization/figures/fig_prototype_usage_hist.png
log/metrla_stssdl_full/visualization/tables/prototype_usage.csv
```

### 5.1 图怎么做

统计所有测试样本前 4096 个样本、所有节点的 `mask_c`，即当前 query `Q^c` 被分配到的 positive prototype。

横纵坐标：

```text
x 轴：Prototype ID
y 轴：Assignments，被分配到该 prototype 的 (sample,node) 数量
```

### 5.2 结果

总 assignment 数：

```text
4096 samples * 207 nodes = 847872
```

主要 prototype 使用量：

| Prototype | Count | 占比 |
|---:|---:|---:|
| P10 | 637515 | 75.19% |
| P17 | 154090 | 18.17% |
| P13 | 31261 | 3.69% |
| P0 | 21780 | 2.57% |

有效 prototype 数：

```text
effective_proto_num = 2.159811
```

### 5.3 暴露的问题

这是最强的 collapse 证据。

ST-SSDL 设置了 `prototype_num=20`，但实际几乎只有 P10 和 P17 在工作：

```text
P10 + P17 = 93.36% assignments
```

结论：

```text
prototype space 没有形成 20 个均衡、细粒度、可解释的交通模式；
它退化成两三个主导原型的粗糙分配。
```

这就是我们之前说的 prototype assignment collapse / lazy mode。

## 6. Figure 4-like：不同偏差水平案例

文件：

```text
log/metrla_stssdl_full/visualization/figures/fig4_like_deviation_cases.png
```

代码入口：

```text
src/visualize_stssdl.py:310  _plot_figure4()
```

### 6.1 图怎么做

先计算每个样本的偏差：

```text
sample_deviation = mean_N(latent_dis)
latent_dis = |Q^c - Q^a|_1
```

按 `sample_deviation` 的 33% 和 66% 分位数划分：

```text
low    <= q33
medium in (q33, q66]
high   > q66
```

每个组取一个接近组内中位数的代表样本。

每个样本再选择：

```text
node_idx = argmax_N(prototype_dis)
```

也就是该样本中 prototype distance 最大的节点。

### 6.2 横纵坐标

上排时间序列子图：

```text
x 轴：时间步 0-23
      0-11 是输入窗口
      12-23 是预测窗口
y 轴：Traffic Speed
```

曲线含义：

```text
蓝线：Ground Truth，0-11 为 Xc，12-23 为未来真实 Y
绿虚线：History Anchor，0-11 为输入锚点，12-23 为 future-aligned anchor
红线：Prediction，只画 12-23
```

下排 prototype 示意子图：

```text
x 轴：局部 PCA 第 1 维
y 轴：局部 PCA 第 2 维
```

注意：每一列的 PCA 都只用该案例的四个点：

```text
q_c, q_a, p_c, p_a
```

所以三列下排图的 PCA 坐标不在同一全局空间，不能跨列直接比较距离方向。

### 6.3 图中现象

Low deviation：

```text
mask_c = 17
mask_a = 10
|qc-qa| = 2.94
|pc-pa| = 37.21
```

Medium deviation：

```text
mask_c = 13
mask_a = 10
|qc-qa| = 26.43
|pc-pa| = 13.80
```

High deviation：

```text
mask_c = 17
mask_a = 10
|qc-qa| = 25.49
|pc-pa| = 37.21
```

### 6.4 暴露的问题

#### 问题 1：low deviation 并没有分到同一 prototype

论文期待：

```text
low deviation: Q^c 和 Q^a 应该映射到同一或非常接近的 prototype
```

但当前 low case 是：

```text
mask_c=17, mask_a=10
```

说明：

```text
即使 latent deviation 很低，prototype assignment 也不稳定。
```

#### 问题 2：latent distance 与 prototype distance 不一致

low case：

```text
|qc-qa| = 2.94
|pc-pa| = 37.21
```

这正好违背 `L_Dev` 想要的目标：

```text
输入/query 空间偏差小，prototype 空间距离也应小。
```

但实际 prototype distance 很大。

#### 问题 3：high deviation 下预测没有展示论文式稳健性

high case 中，历史锚点在未来窗口持续上升到 50-60，但真实值在 20-40 附近波动。预测红线平滑上升，更像被历史锚点趋势牵引，不能紧贴真实未来。

这说明：

```text
高偏差场景下，ST-SSDL 并没有可靠地区分“可用历史锚点”和“误导性历史锚点”。
```

## 7. Figure 5-like：Query 与 Prototype PCA

文件：

```text
log/metrla_stssdl_full/visualization/figures/fig5_like_pca_queries_prototypes.png
```

代码入口：

```text
src/visualize_stssdl.py:407  _plot_figure5()
```

### 7.1 图怎么做

1. 展平所有 `(sample,node)` 的 `mask_c`。
2. 统计最常被选中的 top 7 prototypes。
3. 只保留分配到这 7 个 prototypes 的 `Q^c`。
4. 最多采样 400 个 query。
5. 将采样 query 与 top 7 prototypes 拼在一起做 PCA 到 2D。

### 7.2 横纵坐标

```text
x 轴：PC1
y 轴：PC2
```

点含义：

```text
小圆点：query Q^c
星形点：prototype P_k
颜色：query 被分配到的 positive prototype id
```

### 7.3 暴露的问题

图中蓝色 P10 query 占绝大多数，橙色 P17 次之，其它颜色很少。

这与 prototype usage 表一致：

```text
P10 = 75.19%
P17 = 18.17%
```

问题：

```text
query 没有围绕多个 prototype 形成均衡、清晰、可解释的簇。
```

更严重的是，一些 prototype 星号靠得很近，但 query 极少，说明很多 prototype 只是形式上存在，并没有承担有效模式划分。

## 8. Figure 6-like：Prototype 物理模式恢复

文件：

```text
log/metrla_stssdl_full/visualization/figures/fig6_like_physical_prototype_patterns.png
log/metrla_stssdl_full/visualization/tables/prototype_pattern_summary.csv
```

代码入口：

```text
src/visualize_stssdl.py:503  _plot_figure6()
```

### 8.1 图怎么做

对每个 prototype `P_k`：

1. 收集所有 `Q^c` 被分配到 `P_k` 的输入序列 `Xc`。
2. 对这些输入序列在样本维度求均值。
3. 绘制：

```text
mean_curve
mean_curve ± std_curve
```

只展示 assignment 数不少于 `pattern_min_count=100` 的 prototype。

### 8.2 横纵坐标

```text
x 轴：Input Step，0-11
y 轴：Traffic Speed
```

图中：

```text
绿线：被分配到该 prototype 的输入速度均值
浅绿色阴影：± standard deviation
```

### 8.3 表格结果

主要 prototype：

```text
P10: n=637515, increasing, start=56.95, end=60.34, std_mean=13.96
P17: n=154090, rapidly decreasing, start=14.64, end=6.63, std_mean=17.69
P13: n=31261, rapidly decreasing, start=36.16, end=18.19, std_mean=30.04
P0:  n=21780, rapidly decreasing, start=24.38, end=3.85, std_mean=25.40
```

### 8.4 暴露的问题

#### 问题 1：模式类别高度单一

除 P10 外，多数 prototype 被自动标为：

```text
rapidly decreasing
```

这说明 prototype 没有覆盖多样交通模式，例如恢复、稳定、缓慢上升、拥堵形成等。

#### 问题 2：prototype 内部方差很大

多个 prototype 的 `std_mean` 很高：

```text
P13 std_mean = 30.04
P0  std_mean = 25.40
P1  std_mean = 27.87
```

这说明同一个 prototype 内部混入了大量不同物理轨迹。

因此：

```text
prototype 在物理空间中并没有恢复出清晰、紧凑、可解释的模式。
```

## 9. Figure 7/8/9-like：低/中/高偏差预测案例

文件：

```text
fig7_like_low_deviation_stssdl.png
fig8_like_medium_deviation_stssdl.png
fig9_like_high_deviation_stssdl.png
```

代码入口：

```text
src/visualize_stssdl.py:565  _plot_figure7_9()
```

### 9.1 图怎么做

复用 Figure 4 的低/中/高偏差样本选择方式：

```text
sample_deviation = mean_N(|Q^c - Q^a|_1)
low / medium / high = 33% / 66% 分位数划分
node_idx = argmax_N(prototype_dis)
```

但图只画预测对比，不画下方 prototype PCA。

### 9.2 横纵坐标

```text
x 轴：时间步 0-23
      0-11 输入窗口
      12-23 预测窗口
y 轴：Traffic Speed
```

曲线含义：

```text
蓝线：Ground Truth
绿虚线：History Anchor
红线：Prediction
```

### 9.3 暴露的问题

从 `metrics_by_deviation.csv` 看，误差随 deviation 分组明显上升：

| Group | MAE | RMSE | MAPE | 60min MAE |
|---|---:|---:|---:|---:|
| low | 2.2478 | 4.2547 | 5.04% | 2.5001 |
| medium | 3.1801 | 6.5595 | 9.39% | 3.7498 |
| high | 3.8794 | 7.7514 | 12.66% | 4.4426 |

这说明：

```text
ST-SSDL 在 high deviation 样本上仍然明显更难预测。
```

特别是 high deviation 案例图中，历史锚点与真实未来存在明显错位，预测红线更平滑、更靠近锚点趋势，无法捕捉真实未来的大幅波动。

问题：

```text
当前模型知道样本有偏差，但没有可靠机制判断锚点是否可信。
```

这正是我们后续提出 anchor reliability / need-reliability gate 的动机。

## 10. Prototype Alignment 表

文件：

```text
log/metrla_stssdl_full/visualization/tables/prototype_alignment_by_deviation.csv
```

结果：

```text
same_proto_rate = 0.761465
prototype_switch_rate = 0.238535
mean_query_distance = 14.638906
mean_proto_distance = 8.121952
```

解释：

1. `same_proto_rate=76.15%` 不能直接说明原型对齐好，因为该比例很大程度上被 P10 的超高占用率支配。
2. `mean_query_distance=14.64` 明显高于 `mean_proto_distance=8.12`，说明 prototype distance 对真实 latent deviation 有压缩倾向。
3. Figure 4 的 low case 已经展示了局部反例：query distance 很小但 prototype distance 很大。

结论：

```text
距离一致性没有稳定成立；
prototype assignment 的同/不同更多受坍缩分布影响，而不是细粒度偏差语义。
```

## 11. Figure 10-like：缺失值场景预测

文件：

```text
log/metrla_stssdl_full/visualization/figures/fig10_like_missing_values_stssdl.png
```

代码入口：

```text
src/visualize_stssdl.py:591  _plot_figure10()
```

### 11.1 图怎么做

1. 从 medium deviation 组选择一个代表样本。
2. 使用该样本的 normalized `x_current_norm` 作为输入。
3. 随机选择 20% 的输入元素置 0：

```text
src/visualize_stssdl.py:603  missing_input.view(-1)[flat_indices] = 0.0
```

4. 用缺失输入重新跑模型。
5. 选择误差最大的节点并画图。

### 11.2 横纵坐标

理论上：

```text
x 轴：时间步 0-23
y 轴：Traffic Speed
```

曲线含义：

```text
绿虚线：History Anchor
蓝线：Ground Truth，0-11 为已知输入，12-23 为真实未来
粉线：Prediction (clean)
红线：Prediction (missing)
```

### 11.3 当前 Fig10 的严重问题

当前代码存在一个明确错误：

```text
src/visualize_stssdl.py:606  pred_missing = output["prediction"][0, :, :, 0].cpu().numpy()
```

这里的 `output["prediction"]` 仍在 normalized space，没有经过：

```text
scaler.inverse_transform(...)
```

但图中的 `target`、`x_current`、`anchor` 都是原始 Traffic Speed 量纲。

因此当前红线：

```text
Prediction (missing)
```

不是 Traffic Speed，而是 normalized prediction。

这就是为什么图中红线在 `y≈1` 附近，而其它曲线在 `y≈70` 附近。

### 11.4 节点选择也被这个错误污染

当前代码选择节点：

```text
src/visualize_stssdl.py:607
node_idx = argmax(mean(abs(pred_missing - target)))
```

但这里：

```text
pred_missing 是 normalized
target 是 raw speed
```

二者量纲不一致。

所以节点选择会偏向目标速度很高的节点，而不是缺失输入真正导致预测退化最大的节点。

我复现该逻辑后得到：

```text
sample_idx = 3510
bug_node = 96
```

该节点数据：

```text
x_current_raw = [0, 0, 0, 0, 70.0, 69.889, 70.0, 70.0, 69.571, 69.556, 69.75, 70.0]
target_raw    = [69.571, 70.0, 69.778, 69.375, 70.0, 69.778, 70.0, 69.857, 70.0, 69.875, 70.0, 69.889]
target_std    = 0.1888
```

所以蓝色 ground truth 近似直线的原因是：

```text
被错误节点选择逻辑选中的 node 96 在未来 12 步本来就几乎恒定在 70 附近。
```

这不一定说明 ground truth 数据本身错了，但说明这张图选了一个不适合作为缺失值鲁棒性展示的案例。

### 11.5 正确反归一化后会选到不同节点

如果先对 missing prediction 做反归一化，再计算缺失场景误差，最大误差节点变成：

```text
fixed_node = 5
```

该节点未来真实值：

```text
target_raw = [60.857, 60.5, 59.444, 54.75, 55.5, 28.333, 28.625, 31.0, 35.75, 25.5, 27.75, 49.556]
target_std = 14.1119
```

这才是更有意义的缺失值预测案例，因为未来存在明显变化。

### 11.6 对 Fig10 的结论

当前 `fig10_like_missing_values_stssdl.png` 有问题，不能作为 ST-SSDL 缺失值鲁棒性的证据。

问题包括：

1. `Prediction (missing)` 没有反归一化，红线量纲错误。
2. 节点选择使用 normalized prediction 与 raw target 比较，量纲错误。
3. 被选中的 node 96 未来真实值几乎恒定，导致 ground truth 看起来像直线。
4. 输入窗口前 4 步为 0，这可能是原始 METR-LA 缺失/异常编码，不是脚本主动制造的缺失；但图上没有标注清楚。

正确修复方式：

```python
pred_missing = scaler.inverse_transform(output["prediction"])[0, :, :, 0].cpu().numpy()
node_idx = int(np.argmax(np.abs(pred_missing - target).mean(axis=0)))
```

同时建议：

1. 在图中用 marker 标出被 mask 的输入点。
2. 不要选未来几乎恒定的节点做 Fig10。
3. 同时画 clean prediction 和 missing prediction 的 MAE 差值。
4. 把缺失输入填充值从 0 改成更明确的策略，例如历史锚点值或 train mean，并单独消融。

## 12. 综合判断：ST-SSDL 在当前复现中暴露的问题

### 12.1 预测指标上：SSDL 模块贡献有限

消融实验表明，去掉 `L_Con` 或 `L_Dev` 后 test MAE 几乎不变；去掉整个 SSDL 后也只小幅退化。

结论：

```text
ST-SSDL 的预测能力主要来自时空预测主干，而不是强依赖 prototype-based self-supervised deviation learning。
```

### 12.2 表征空间上：prototype collapse 明显

20 个 prototype 的有效使用数只有 `2.16`，P10 和 P17 占据 `93.36%` assignment。

结论：

```text
所谓 scientific information space 没有形成多样、稳定、细粒度的交通模式划分。
```

### 12.3 解释性上：低/中/高偏差叙事不稳定

Figure 4 中：

```text
low deviation 也出现 mask_c != mask_a；
latent distance 和 prototype distance 局部严重不一致；
high deviation 预测没有贴近真实未来。
```

结论：

```text
ST-SSDL 的 prototype 距离不能稳定解释 current-history deviation。
```

### 12.4 物理模式上：prototype pattern 不够清晰

Figure 6 中：

```text
大多数 prototype 都是 decreasing；
很多 prototype 的标准差阴影极宽；
高频 P10 内部也包含很宽的速度分布。
```

结论：

```text
prototype 不具备足够强的物理可解释性。
```

### 12.5 鲁棒性可视化上：Fig10 当前无效

Figure 10-like 当前存在量纲错误和节点选择错误，不能用于说明 ST-SSDL 对缺失输入鲁棒。

## 13. 对我们后续工作的启发

当前复现实验支持以下研究切入点：

1. ST-SSDL 的历史锚点思想是有价值的，但 single mean anchor 可能过弱。
2. ST-SSDL 的 learnable prototype 离散空间存在 assignment collapse 风险。
3. 只依靠 `L_Con + L_Dev` 并不能保证 prototype 空间有可解释物理语义。
4. 高偏差样本仍明显更难预测，需要显式判断历史锚点是否可靠。
5. 缺失值鲁棒性必须用正确量纲、正确节点选择和明确 mask 标注重新评估。

因此，当前 DCD-ST / DHA-DCD-ST 方向是合理的：

```text
删除不稳定 prototype；
保留 current-anchor 对比思想；
把历史锚点从单一均值扩展为更可靠的分布式上下文；
再引入 anchor reliability，避免高偏差场景盲目信任历史锚点。
```
