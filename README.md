# TrafficRobustST

本项目当前默认训练的是和 `ST-SSDL/model_STSSDL/STSSDL.py` 对齐的原版 ST-SSDL baseline：

```text
ST-SSDL 历史偏差原型
  + 当前/历史双分支编码
  + 原型对比与偏差一致性损失
  + 偏差感知动态图解码
```

第一版区域增强模型仍保留在 `RegionAwareSTSSDL` 中，但需要显式添加 `--model region` 才会使用。

## 环境安装

目标环境：`conda research`，Python 3.10。

```powershell
conda activate research
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn tables
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

说明：PyTorch 当前没有常规稳定版 `cu132` wheel。CUDA 13.2 环境下通常安装 PyTorch 自带 CUDA runtime 的 `cu128` 包即可，前提是 NVIDIA 驱动足够新。

检查安装：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

## 数据预处理

完整生成 METR-LA 的 ST-SSDL 风格切分：

```powershell
conda activate research
python src\preprocessing.py --traffic-h5 data\METR-LA.h5 --output-dir data\METRLA
```

输出：

```text
data/METRLA/trainhis.npz
data/METRLA/valhis.npz
data/METRLA/testhis.npz
```

## 训练命令

原版 ST-SSDL baseline 训练：

```powershell
conda activate research
python -m src.train --model baseline --processed-dir data\METRLA --epochs 200 --patience 30 --batch-size 128 --device cuda:0 --rnn-units 128 --prototype-dim 64 --prototype-num 20 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --lamb-c 0.01 --lamb-d 1 --lr-scheduler multistep --lr-milestones 50,70 --lr-decay-ratio 0.1 --output-dir log --run-name metrla_stssdl_baseline
```

训练输出会写入：

```text
log/metrla_stssdl_baseline/train.log
log/metrla_stssdl_baseline/config.json
log/metrla_stssdl_baseline/best_model.pt
```

只验证前向和反传：

```powershell
python -m src.train --smoke-test --model baseline
```

## 每轮日志

训练时每个 epoch 会在控制台打印：

```text
epoch=1/100 batches=375 lr=0.001
train_loss=...
train_mae=...
contrastive=...
deviation=...
region=0.0000
graph_reg=0.0000
val_mae=...
val_rmse=...
val_mape=...
val_15min_mae=...
val_30min_mae=...
val_60min_mae=...
train_time=...s
val_time=...s
```

含义：

| 字段 | 含义 |
|---|---|
| `train_loss` | 总训练损失，baseline 包含预测误差、原型对比和偏差一致性 |
| `train_mae` | 反归一化后的训练 MAE，是最直观的训练误差 |
| `contrastive` | ST-SSDL 原型三元组对比损失 |
| `deviation` | 当前序列与历史锚点的 query 距离、prototype 距离一致性损失 |
| `region` | baseline 固定为 0；仅 `--model region` 使用 BCC 区域语义原型对比损失 |
| `graph_reg` | baseline 固定为 0；仅 `--model region` 使用图去噪正则 |
| `val_mae/rmse/mape` | 验证集整体指标 |
| `val_15min/30min/60min_*` | 第 3/6/12 个预测步的常用短期预测指标 |
| `train_time/val_time` | 当前 epoch 训练耗时和验证耗时 |

## 超参解释

### 运行和数据参数

| 参数 | 默认值 | 含义 | 调参建议 |
|---|---:|---|---|
| `--smoke-test` | False | 用随机小图跑一次前向、损失和反传 | 调试代码时使用，不训练真实数据 |
| `--model` | `baseline` | 选择模型架构，`baseline` 为原版 ST-SSDL，`region` 为第一版区域增强模型 | 做原版对照和论文消融时保持 `baseline` |
| `--generate-data` | False | 在训练前生成 METR-LA 切分 | 首次运行可用；完整实验建议单独先跑 `preprocessing.py` |
| `--traffic-h5` | `data/METR-LA.h5` | 原始 METR-LA h5 文件路径 | 换数据集时修改 |
| `--processed-dir` | `data/METRLA` | 预处理后 `trainhis/valhis/testhis.npz` 路径 | 训练时最常改 |
| `--adj-path` | `data/adj_mx.pkl` | 邻接矩阵文件路径 | 换数据集时修改 |
| `--dataset-name` | `METR-LA` | 数据集名称，用于 BCC 默认阈值选择 | PEMS04/PEMS07 等要改成对应名字 |
| `--adj-type` | `symadj` | 图归一化方式 | `symadj` 稳；有向扩散可试 `doubletransition` |
| `--seq-len` | 12 | 输入历史步数 | 5 分钟粒度下 12 表示过去 1 小时 |
| `--horizon` | 12 | 预测未来步数 | 12 表示未来 1 小时 |
| `--train-ratio` | 0.7 | 训练集比例 | 预处理时使用 |
| `--val-ratio` | 0.1 | 验证集比例 | 测试集比例为剩余部分 |
| `--max-windows` | None | 限制生成样本数 | 小样本调试用，例如 256 |
| `--output-dir` | `log` | 实验输出根目录 | 保存日志、配置和 best checkpoint |
| `--run-name` | None | 当前实验名 | 不填会自动生成时间戳名称 |
| `--device` | `cuda:0` | 训练设备 | 无 GPU 时用 `cpu` |
| `--seed` | 999 | 随机种子 | 复现实验要固定 |

### 训练参数

| 参数 | 默认值 | 含义 | 调参建议 |
|---|---:|---|---|
| `--epochs` | 3 | 训练轮数 | 正式训练可设 100 或 200 |
| `--patience` | 20 | 早停等待轮数 | val MAE 连续不提升则停止；设 0 关闭 |
| `--batch-size` | 64 | batch 大小 | 原版 METR-LA 复现建议显式设为 128 |
| `--num-workers` | 0 | DataLoader 进程数 | Windows 上 0 最稳 |
| `--max-batches` | None | 每轮最多训练多少 batch | 小闭环调试用，例如 1 |
| `--lr` | 0.001 | Adam 学习率 | 不稳定时降到 0.0005 |
| `--lr-scheduler` | `none` | 学习率调度方式 | 完整实验建议用 `multistep` |
| `--lr-milestones` | `40,70` | MultiStepLR 降学习率轮数 | 100 epoch 可用 `40,70`，80 epoch 可用 `30,55` |
| `--lr-decay-ratio` | 0.1 | 每次降学习率的倍率 | 常用 0.1 |
| `--weight-decay` | 0.0 | L2 正则 | 过拟合时可试 `1e-5` 或 `1e-4` |
| `--max-grad-norm` | 5.0 | 梯度裁剪阈值 | 防止 RNN/动态图训练梯度爆炸 |
| `--use-curriculum-learning` | True | 开启 decoder teacher forcing 课程学习 | 默认开启 |
| `--no-curriculum-learning` | False | 关闭课程学习 | 做消融时使用 |
| `--save-best` | True | 保存验证集 MAE 最优模型 | 默认开启 |
| `--no-save-best` | False | 不保存 best checkpoint | 只做快速调试时可用 |

### 模型结构参数

| 参数 | 默认值 | 含义 | 调参建议 |
|---|---:|---|---|
| `--rnn-units` | 128 | AGCRN 编码器/解码器隐藏维度 | 原版 METR-LA 配置为 128 |
| `--rnn-layers` | 1 | AGCRN 堆叠层数 | 1 最稳；2 层需更小学习率 |
| `--cheb-k` | 3 | Chebyshev 图卷积阶数 | 3 表示融合 0/1/2 跳邻域 |
| `--prototype-num` | 20 | ST-SSDL 原型数量 | 模式复杂可增大到 32/50 |
| `--prototype-dim` | 64 | 原型空间维度 | 通常与 `rnn-units` 接近 |
| `--input-embedding-dim` | 3 | 输入值投影维度 | METR-LA 轻量配置保留 3 |
| `--tod-embed-dim` | 20 | time-of-day 嵌入维度 | 时间周期强时可增大 |
| `--node-embedding-dim` | 25 | 节点 ID 嵌入维度 | 节点多时可适当增大 |
| `--adaptive-embedding-dim` | 0 | 自适应时空嵌入维度 | 原版 METR-LA 配置为 0 |

原版 METR-LA baseline 的参数量应为：

```text
646738 trainable parameters
```

### 损失权重参数

总损失为：

```text
L = L_mae
  + lamb_c * L_contrastive
  + lamb_d * L_deviation
  + lamb_region * L_region
  + lamb_graph * L_graph_reg
```

| 参数 | 默认值 | 含义 | 调参建议 |
|---|---:|---|---|
| `--lamb-c` | 0.01 | ST-SSDL 原型对比损失权重 | 原型不分化时增大；预测变差时减小 |
| `--lamb-d` | 1.0 | 偏差一致性损失权重 | 保持 ST-SSDL 主约束，通常不先动 |
| `--lamb-region` | 0.05 | BCC 区域语义对比损失权重 | 图区域表示弱时增大到 0.1 |
| `--lamb-graph` | 0.001 | 图去噪正则权重 | 图变化太大时增大 |
| `--use-region-loss` | True | 开启 BCC 区域对比损失 | 默认开启 |
| `--no-region-loss` | False | 关闭 BCC 区域对比损失 | 消融实验使用 |

### 图去噪参数

| 参数 | 默认值 | 含义 | 调参建议 |
|---|---:|---|---|
| `--graph-static-weight` | 0.15 | 去噪图中保留静态物理图的比例 | 越大越保守，越小越依赖动态图 |
| `--bcc-edge-threshold` | None | BCC 构图边阈值 | None 使用数据集默认；METR-LA 默认约 0.7 |
| `--use-graph-denoise` | True | decoder 使用 BCC 区域语义修正后的 `clean_support` | 默认开启 |
| `--no-graph-denoise` | False | decoder 直接使用动态邻接 `base_support` | 消融图去噪贡献时使用 |

## 推荐消融实验

以下命令用于判断第一版区域模块和图去噪模块分别贡献了多少。注意：这些命令必须显式使用 `--model region`，不属于原版 ST-SSDL baseline 消融。

完整模型：

```powershell
conda activate research
python -m src.train --model region --processed-dir data\METRLA --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --rnn-units 64 --prototype-dim 64 --prototype-num 20 --tod-embed-dim 20 --node-embedding-dim 25 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --output-dir log --run-name metrla_region_full_v2
```

关闭 BCC 区域对比损失：

```powershell
conda activate research
python -m src.train --model region --processed-dir data\METRLA --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --rnn-units 64 --prototype-dim 64 --prototype-num 20 --tod-embed-dim 20 --node-embedding-dim 25 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --no-region-loss --lamb-region 0 --output-dir log --run-name metrla_no_region_loss
```

关闭 BCC 图去噪：

```powershell
conda activate research
python -m src.train --model region --processed-dir data\METRLA --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --rnn-units 64 --prototype-dim 64 --prototype-num 20 --tod-embed-dim 20 --node-embedding-dim 25 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --no-graph-denoise --lamb-graph 0 --output-dir log --run-name metrla_no_graph_denoise
```

同时关闭 BCC 区域损失和图去噪：

```powershell
conda activate research
python -m src.train --model region --processed-dir data\METRLA --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --rnn-units 64 --prototype-dim 64 --prototype-num 20 --tod-embed-dim 20 --node-embedding-dim 25 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --no-region-loss --no-graph-denoise --lamb-region 0 --lamb-graph 0 --output-dir log --run-name metrla_no_darkfarseer
```
