# 新基线 METR-LA / PEMS-BAY 对照实验队列

> 文档版本：2026-09-10
> 范围：ST-Norm / DLinear / ST-SSDL / DCRNN 的 6 组对照，以及原四模型剩余 random-Bank 消融
> 不改：HN-OffsetDecay v2、weekday_radius1_overlap、Top-12、frozen path cache、Router 结构

## 1. 这次做什么

用户要的是每个 backbone 六组实验：

- METR-LA：Base-only、random 检索对照、训练后检索对照
- PEMS-BAY：Base-only、source 检索对照、finetune 检索对照

当前新接入且还需要这套完整矩阵的 backbone 是 4 个：

1. ST-Norm
2. DLinear
3. ST-SSDL
4. DCRNN

因此实验机正式队列是 4 x 6 = 24 组训练。原四模型 GWN / STGCN / STAEformer / ARGCN 的 METR-LA trained 与 PEMS-BAY source/finetune 已经跑完，不再整包重训。

本机另补原四模型里还没做完的 random-Bank 对照：STAEformer、GWN、ARGCN。STGCN 已经完成，队列会跳过。

## 2. 术语

- Base-only：只训练下游 backbone，不检索、不经过 Router。
- trained Router：冻结 Base，使用 METR-LA 上训练好的 HN-OffsetDecay v2 编码器与对应 Bank。
- random Router：同一套 Router，但 Bank 来自未训练随机编码器，用来检验检索表示是否真有贡献。
- source Router：目标域 PEMS-BAY 上直接使用源域编码器建 Bank，不微调。
- finetuned Router：目标域使用 T1 head-adapter 微调后的编码器建 Bank。
- weekday_radius1_overlap：部署检索协议。query 所在 weekday 以及前后各一天的同时段历史事件都可以进候选池，再按 key 取 node-level Top-12。
- frozen path cache：冻结 encoder / retrieval / backbone 路径，首轮后缓存，降低后续 epoch 成本。
- 物理损失空间：先反标准化再算训练损失；报告指标始终是反标准化 MAE / RMSE / MAPE。

## 3. 训练协议

Router 侧统一：

- mode = learned_topk_error_aware
- protocol = weekday_radius1_overlap
- node_top_k = 12
- physical MAE
- lr = 5e-4，StepLR step=10，gamma=0.5
- 50 epoch，关闭 early stopping
- candidate_quality_weight = 0
- frozen_path_cache = true

Base-only 不统一成一套 lr/loss，沿用各模型自己的论文可迁移核心：

| Backbone | 数据集 | 损失 | lr | epoch | early stop |
|---|---|---|---|---|---|
| ST-Norm | METR-LA / PEMS-BAY | physical MAE | 1e-3 | 50 | 关 |
| DLinear | METR-LA / PEMS-BAY | normalized MSE | 1e-4 | 100 | patience=3 |
| ST-SSDL | METR-LA / PEMS-BAY | physical MAE + 0.01 triplet + 1.0 deviation | 0.01 | 100 | patience=20 |
| DCRNN | METR-LA / PEMS-BAY | physical MAE | 0.01 | 100 | patience=50 |

PEMS-BAY 的 Base-only 只换数据路径和 run_name，不改模型超参。PEMS-BAY 检索 event_top_r=96，METR-LA 仍是 32。

## 4. Bank 生命周期

实验机没有现成 Bank。队列只在 artifacts/new_baseline_queue_banks/ 下建临时 Bank：

1. METR-LA 四个 Base-only
2. 建 METR-LA trained Bank，四个 trained Router，然后删该 Bank
3. 建 METR-LA random encoder 和 random Bank，四个 random Router，然后删该 Bank 和临时 random checkpoint
4. PEMS-BAY 四个 Base-only
5. 建 PEMS-BAY source Bank，四个 source Router，然后删该 Bank
6. 建 PEMS-BAY finetune Bank（复用已有 T1 checkpoint，不重做微调），四个 finetune Router，然后删该 Bank

删除函数拒绝删除队列根目录以外的路径，不会碰到正式 case_bank_* 或 cross_dataset_t1。

本机 random 队列复用已经存在的官方 random encoder / Bank，训练结束后不删除。

## 5. 命令

实验机：

```powershell
Set-Location -LiteralPath C:\Users\clw\projects\researchRep\STAnchor-BlockMemory
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_new_baselines_metrla_pemsbay_queue.ps1
```

本机剩余 random 对照：

```powershell
Set-Location -LiteralPath D:\projects\researchProjects\TrafficRobustST\STAnchor-BlockMemory
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_local_random_bank_remaining_queue.ps1
```

两条队列都使用 python -u，控制台打印 START/DONE，失败即停止。不要用 conda run。数据路径继续用相对路径 ../data/... ，本机父目录 TrafficRobustST 与实验机 researchRep 都能解析。

## 6. Keep / Remove / Stop

- Keep：4 个新基线的 24 组对照；本机补齐 GWN / STAEformer / ARGCN 的 random-Bank。
- Remove：实验机队列自建的临时 Bank，用完即删。
- Stop：不把原四模型的 METR-LA trained / PEMS-BAY source+finetune 再跑一遍；不把 DLinear/ST-SSDL/DCRNN 的 Base-only 强行改成 50 epoch physical MAE。

