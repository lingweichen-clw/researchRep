# DLinear 官方训练对齐与 ST-SSDL 完整 Base-only 接入方案

> 文档版本：2026-09-10
> 适用范围：METR-LA 主线新增下游基线，不改检索器 / Bank / Router
> 对应代码：stanchor/models/baseline/dlinear.py、stanchor/models/baseline/st_ssdl.py、stanchor/losses/downstream.py、stanchor/config.py、stanchor/engine/target.py

## 1. 这次改什么，不改什么

这次只做两件已经批准的事：

1. 把 DLinear 的 Base-only 训练配置对齐到 LTSF-Linear 官方可迁移核心，并去掉适配器里不该启用的 1/T 初始化。
2. 把完整 ST-SSDL 作为第三个新的 Base-only backbone 接入，而不是继续使用现有 ARGCN 骨架。

明确不改：

- HN-OffsetDecay v2 检索编码器、Bank、weekday_radius1_overlap、Top-12、frozen path cache、Router 结构；
- GWN / STGCN / STAEformer / ARGCN / ST-Norm 的 Base-only 与 Router 协议；
- DLinear 的 Router 侧协议：继续 physical MAE、lr=5e-4、50 epoch。

## 2. 术语

- Base-only：只训练下游 backbone，不检索、不经过 Router。
- Router / 后验校准：冻结 Base checkpoint，用历史候选做残差修正。
- 物理损失空间：先反标准化，再计算训练损失；报告指标始终是反标准化后的 MAE / RMSE / MAPE。
- 标准化损失空间：在 scaler 变换后的数值上计算训练损失；报告指标仍然反标准化。
- x_his：ST-SSDL 的周期原型输入。它是训练集上每个节点、每个 (weekday, slot) 的速度均值，不是真实过去窗口，也不是 query 的未来标签。
- Teacher forcing：解码器训练时按课程学习概率把真实未来速度当作下一步输入。这是训练技巧，不是测试泄漏；推理和 Router 冻结前向必须关闭。

## 3. DLinear 对齐范围

结构继续跟 LTSF-Linear 官方 DLinear：

- moving_avg kernel = 25，两端复制 padding；
- individual=False，节点共享 Linear；
- 去掉当前适配器中的 W=1/T 常数初始化。官方源码把这两行注释掉了，实际使用 PyTorch Linear 默认初始化。PIR 版才启用 1/T。

Base-only 训练对齐官方可迁移核心，但数据协议仍是 METR-LA 的 12 到 12：

| 项 | 官方 LTSF-Linear | 本次 Base-only |
|---|---|---|
| 优化器 | Adam | Adam |
| 学习率 | 1e-4 | 1e-4 |
| 损失 | 标准化 MSE | 标准化 MSE |
| batch | 32 | 32 |
| weight decay | 默认 0 | 0 |
| early stopping | patience=3 | 开启，patience=3 |
| epoch | 10 | 上限 100 |
| 学习率衰减 | type1，每轮减半 | 不搬。官方只有 10 轮；搬到 100 轮会把学习率打到无意义。本次保持恒定 1e-4，由 early stopping 收束 |
| lookback | 336 | 不搬，继续 12 |
| traffic 脚本 lr=0.05 | 针对长序列 traffic 脚本 | 不搬 |

验证和报告仍然用反标准化 MAE / RMSE / MAPE。Router 配置不改。

实现上新增可选 forecast_loss_name in {mae, mse}，默认 mae，避免其他模型回归。

## 4. ST-SSDL 完整接入

### 4.1 官方 METR-LA 结构

完整模型，不是 ARGCN。官方 METR-LA 设置为：

- use_STE=True
- rnn_units=128，rnn_layers=1，cheb_k=3
- prototype_num=20，prototype_dim=64
- input_embedding_dim=3，node_embedding_dim=25，tod_embed_dim=20，adaptive_embedding_dim=0
- 图支持：symadj，即 D^{-1/2} A D^{-1/2}
- 12 到 12

训练损失：

L = L_MAE_physical + 0.01 L_triplet + 1.0 L_deviation

其中 triplet 作用在当前分支的 (query, pos, neg)，deviation 是当前/历史 query 距离与正原型距离的 L1。官方会对 query 和 query 距离 detach，本次保持这一梯度边界。

优化器：Adam，lr=0.01，eps=1e-3，weight_decay=0。调度：MultiStepLR，gamma=0.1。正式 Base-only 配置按官方数据集口径执行，但遵守本项目“最多 100 epoch”的复现成本上限：METR-LA 使用 batch=128、epoch=100、patience=30、milestones=[50,70]；PEMS-BAY 使用 batch=64、epoch=100、patience=30、milestones=[10,70]、cl_decay_steps=8000，并采用该数据集专属的 input/node embedding 维度 10/20。PEMS-BAY 官方第 150 轮衰减点因超过训练上限而裁掉。Router 自身仍保持已经定稿的 physical MAE、batch=32、lr=5e-4、50 epoch；它只同步冻结 Base 所需的结构字段。

### 4.2 数据流与形状

Query 下游输入 x 形状为 [B, T, N, 1]，T=H=12。

时间协变量由数据集已有的 slot / weekday 构造，不读未来速度。未来 TOD 只由 context 末尾递推：

y_tod[b,h] = (slot[b,T-1] + 1 + h) mod S_day

周期原型表只用 train 段观测拟合。空桶回退到该节点 train 均值，再经同一个 scaler 标准化。因此 x_his 不依赖 y。

### 4.3 未来信息边界

可以接入，但必须守住：

1. x_his 只来自 train 段周期均值，验证/测试也只查这张表。
2. y_cov 只是未来 TOD，不是未来速度。
3. Base-only 训练允许官方 curriculum teacher forcing，因为它只用训练标签作为解码器输入，推理时关闭。
4. 评估、Router 冻结前向、缓存 Base 路径时：labels=None，课程学习不生效。
5. Router 只消费 Base 的预测 [B,H,N,1]，不读 ST-SSDL 内部 hidden、prototype 或动态图。

官方 embedding 索引用了 CPU LongTensor，本次改成与输入同设备的整数 index，避免 CUDA 索引错误；语义不变。

### 4.4 前向

1. STE：把 x 投影到 3 维，拼接 TOD 嵌入和节点嵌入，得到 [B,T,N,48]。
2. 当前分支与历史分支共享 encoder，各自取最后时刻 hidden，再查询 20 个 prototype。
3. 用 [h_t, v_t, h_a, v_a] 生成动态图，逐步解码 12 步。
4. 输出预测 [B,H,N,1]。训练期额外返回 contrastive / deviation 所需张量。

## 5. 文件与接口

- stanchor/models/baseline/dlinear.py：删除 1/T 初始化。
- stanchor/losses/downstream.py：新增 masked MSE，forecast_loss_name 默认 mae。
- stanchor/config.py：新增损失名、optimizer eps、multi-step scheduler、ST-SSDL 超参。
- stanchor/models/baseline/st_ssdl.py：完整模型适配器。
- stanchor/data/dataset.py：train-only 周槽均值表。
- stanchor/engine/target.py：工厂、协变量前向、辅助损失、history table 绑定。
- configs/formal_baseonly_dlinear.yaml：官方训练协议。
- configs/formal_baseonly_st_ssdl.yaml：完整 ST-SSDL Base-only。
- configs/formal_base_as_candidate_st_ssdl.yaml：Router 配置，损失仍走主线 physical MAE。

## 6. 验证

1. DLinear 权重不再全是 1/T；默认 MAE 不回归；DLinear Base-only 配置断言为 normalized MSE。
2. ST-SSDL 输出形状 [B,12,N,1]，有限，可反传。
3. 无泄漏：labels=None 时输出不依赖 y；x_his 只依赖 weekday/slot 与 train 表。
4. checkpoint round-trip。
5. 定向 pytest 后全量 pytest。本机 8G 不做 1-batch 时长测试。

## 7. Keep / Remove / Stop

- Keep：DLinear 官方初始化 + 标准化 MSE Base-only；完整 ST-SSDL + 周期原型 + 物理 MAE 与辅助损失。
- Remove：DLinear 适配器中的 1/T 初始化。
- Stop：不把官方 336 步 lookback、traffic lr=0.05、DLinear type1 每轮减半搬进 METR-LA 12 到 12。

## 8. 新增实验收口审计

正式运行 24 组新增基线实验前，只修正以下与新增实验直接相关的问题：

1. ST-SSDL 的 `symadj` 必须保留原始邻接矩阵已有的自环，再计算 $D^{-1/2}AD^{-1/2}$；不得复用会删除自环的旧图辅助函数默认语义。
2. ST-SSDL 的 decoder 动态图严格复现官方 `softmax(sigmoid(EE^T))`，不能替换为 ReLU。
3. raw-L1 消融在同一 `weekday_radius1_overlap` 合法候选池内，用 288 步原始归一化 context 的 L1 距离选 Top-12；选中后直接读取候选 future，不再经过 OffsetDecay，也不再执行检索编码器。Router 所需的 retrieval-key 输入使用同形状零向量，避免训练后的 query key 污染非学习检索对照。Bank 只承担合法时间池、历史 context 和 future 的存储职责。
4. `posthoc_frozen_base` 每个训练 epoch 虽会把整网请求切到 train mode，但冻结模块必须立即恢复 eval mode。测试必须覆盖 ST-Norm 的运行统计不更新以及两次冻结前向完全一致。
5. DCRNN Base-only 的 METR-LA 与 PEMS-BAY batch 均改为官方 64；ST-Norm 和 DLinear 保持已经核实的 8 与 32。

验收条件：新增 Base 的 forward/backward、单次 optimizer step、checkpoint round-trip、冻结 Base 确定性、Router 全梯度覆盖、配置成对一致性、raw-L1 无 OffsetDecay、PowerShell 队列语法和全量 pytest 全部通过。任何一项失败都暂停正式训练。
