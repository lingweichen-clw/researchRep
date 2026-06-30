# DCD-ST 训练说明

更新时间：2026-06-30

本目录存放 DCD-ST 第一版核心模型代码：

```text
DCD-ST/
  dcd_st.py
  deviation_decomposition.py
```

当前实现通过 `src/train.py --model dcd` 动态加载本目录下的模型文件。这样核心代码保留在 `DCD-ST/`，同时复用项目已有的数据加载、指标、训练日志和 checkpoint 保存逻辑。

## 1. 环境配置

实验环境固定使用：

```powershell
conda activate research
```

目标 Python 版本：

```text
Python 3.10
```

训练 DCD-ST 需要的核心依赖与原 TrafficRobustST baseline 一致：

```powershell
conda activate research
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn tables
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

可视化或论文解析阶段可额外安装：

```powershell
python -m pip install matplotlib pymupdf
```

检查环境：

```powershell
conda activate research
python -c "import torch, numpy, pandas, scipy, sklearn, tables; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

如果只在 CPU 上调试，把训练命令中的：

```text
--device cuda:0
```

改成：

```text
--device cpu
```

## 2. 数据准备

DCD-ST 复用当前项目的 ST-SSDL 风格数据切分，需要以下文件：

```text
data/METRLA/trainhis.npz
data/METRLA/valhis.npz
data/METRLA/testhis.npz
data/METRLA_data/adj_mx.pkl
```

如果还没有预处理文件，先运行：

```powershell
conda activate research
python src\preprocessing.py --traffic-h5 data\METRLA_data\METR-LA.h5 --output-dir data\METRLA
```

## 3. 快速验证

先跑随机小图 smoke test，验证模型加载、前向传播、loss 和反向传播：

```powershell
conda activate research
python -m src.train --smoke-test --model dcd --no-contrastive-loss --no-deviation-loss
```

期望输出类似：

```text
smoke ok: prediction=(2, 12, 8, 1) clean_support=(2, 8, 8) loss=... mae=...
```

建议同时确认原 baseline 没有被接线改动影响：

```powershell
python -m src.train --smoke-test --model baseline
```

## 4. DCD-ST 短训练

首次在真实 METR-LA 数据上调试，建议只跑少量 batch：

```powershell
conda activate research
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --epochs 3 --max-batches 20 --batch-size 64 --device cuda:0 --no-contrastive-loss --no-deviation-loss --run-name metrla_dcd_debug
```

输出位置：

```text
log/metrla_dcd_debug/train.log
log/metrla_dcd_debug/config.json
log/metrla_dcd_debug/best_model.pt
```

通过标准：

```text
train_loss 无 NaN
val_mae/val_rmse/val_mape 正常打印
best_model.pt 正常保存
```

## 5. DCD-ST 正式训练

第一版主实验命令：

```powershell
conda activate research
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --lr 0.001 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --rnn-units 128 --rnn-layers 1 --cheb-k 3 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --decomp-kernel-size 3 --dev-embed-dim 32 --gate-hidden-dim 128 --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_v1
```

说明：

```text
--no-contrastive-loss
--no-deviation-loss
```

这两个参数必须保留。DCD-ST 第一版已经删除 ST-SSDL 的 prototype 路径，不再使用原型三元组对比损失和 prototype distance consistency loss。

## 6. 稀疏门控训练

如果可视化或日志中发现 `g_dev` 大面积全开，可以加入很小的 gate sparse loss：

```powershell
conda activate research
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --lr 0.001 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --rnn-units 128 --rnn-layers 1 --cheb-k 3 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --decomp-kernel-size 3 --dev-embed-dim 32 --gate-hidden-dim 128 --gate-sparse-weight 0.0001 --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_v1_sparse
```

当前第一版 `gate_smooth_loss` 默认为 0，因为 `g_dev` 是节点级张量 `(B,N,R)`，不是逐时间步张量。后续如果实现 `(B,T,N,R)` 的时间门控，再启用：

```text
--gate-smooth-weight 0.0001
```

## 7. 推荐消融命令

### 7.1 去掉 temporal deviation norm

```powershell
conda activate research
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --no-temporal-deviation-norm --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_no_tnorm
```

### 7.2 去掉 spatial deviation norm

```powershell
conda activate research
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --no-spatial-deviation-norm --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_no_snorm
```

### 7.3 增大趋势分解窗口

```powershell
conda activate research
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --decomp-kernel-size 5 --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_k5
```

`--decomp-kernel-size` 必须是奇数，例如：

```text
3, 5, 7
```

METR-LA 默认输入窗口为 12，第一版建议先比较 `3` 和 `5`。

## 8. 对照实验命令

### 8.1 ST-SSDL full

```powershell
conda activate research
python -m src.train --model baseline --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --rnn-units 128 --prototype-dim 64 --prototype-num 20 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --lamb-c 0.01 --lamb-d 1 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --output-dir log --run-name metrla_stssdl_full_compare
```

### 8.2 ST-SSDL without SSDL

```powershell
conda activate research
python -m src.train --model baseline --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --rnn-units 128 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --no-ssdl --no-contrastive-loss --no-deviation-loss --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --output-dir log --run-name metrla_stssdl_no_ssdl_compare
```

这两个对照用于判断：

```text
DCD-ST 的收益来自连续偏差分解与门控，还是仅仅来自 AGCRN 主干。
```

## 9. 关键参数说明

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--model dcd` | - | 启用 `DCD-ST/dcd_st.py` 中的 DCDST 模型 |
| `--decomp-kernel-size` | 3 | 对 `Xc-Xa` 做 moving average trend 分解的窗口大小 |
| `--dev-embed-dim` | 32 | `z_dev` 偏差特征投影维度 |
| `--gate-hidden-dim` | 128 | deviation gate 内部 MLP 隐藏维度 |
| `--gate-sparse-weight` | 0.0 | gate 稀疏约束权重 |
| `--gate-smooth-weight` | 0.0 | 预留的时间平滑约束权重 |
| `--use-temporal-deviation-norm` | True | 启用窗口内时间偏差标准化 |
| `--no-temporal-deviation-norm` | False | 关闭时间偏差标准化 |
| `--use-spatial-deviation-norm` | True | 启用节点维空间偏差标准化 |
| `--no-spatial-deviation-norm` | False | 关闭空间偏差标准化 |

## 10. 日志字段

DCD-ST 训练日志除原有字段外，还会打印：

| 字段 | 含义 |
|---|---|
| `gate_sparse` | 当前 batch 的 `mean(g_dev)`，用于观察门控是否整体偏高 |
| `gate_smooth` | 当前第一版固定为 0，后续时间门控版本再启用 |

日志样式：

```text
epoch=1/100 batches=...
train_loss=...
train_mae=...
contrastive=0.0000
deviation=0.0000
region=0.0000
graph_reg=0.0000
gate_sparse=...
gate_smooth=0.0000
val_mae=...
val_rmse=...
val_mape=...
```

## 11. 当前实现状态

已验证命令：

```powershell
C:\Users\31396\.conda\envs\research\python.exe -m src.train --smoke-test --model dcd --no-contrastive-loss --no-deviation-loss
```

验证输出：

```text
smoke ok: prediction=(2, 12, 8, 1) clean_support=(2, 8, 8) loss=0.3957 mae=0.3957
```

baseline smoke test 也已通过：

```powershell
C:\Users\31396\.conda\envs\research\python.exe -m src.train --smoke-test --model baseline
```

验证输出：

```text
smoke ok: prediction=(2, 12, 8, 1) clean_support=(2, 8, 8) loss=2.2447 mae=0.4714
```

