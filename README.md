# TrafficRobustST

本项目当前保留两条实验主线：

```text
1. ST-SSDL baseline
   原版双分支编码、prototype query、contrastive loss、deviation consistency loss。

2. DCD-ST
   在 ST-SSDL 主干上做减法，移除 prototype 路径，使用连续偏差分解与 deviation gate。
```

旧版非 DCD-ST 扩展已经移除；当前训练入口只支持：

```text
--model baseline
--model dcd
```

## 环境

目标环境：`conda research`，Python 3.10。

```powershell
conda activate research
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn tables
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

检查环境：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

## 数据预处理

METR-LA 预处理命令：

```powershell
conda activate research
python src\preprocessing.py --traffic-h5 data\METRLA_data\METR-LA.h5 --output-dir data\METRLA
```

训练需要：

```text
data/METRLA/trainhis.npz
data/METRLA/valhis.npz
data/METRLA/testhis.npz
data/METRLA_data/adj_mx.pkl
```

## 快速验证

验证 baseline：

```powershell
conda activate research
python -m src.train --smoke-test --model baseline
```

验证 DCD-ST：

```powershell
conda activate research
python -m src.train --smoke-test --model dcd --no-contrastive-loss --no-deviation-loss
```

期望输出形态：

```text
smoke ok: prediction=(2, 12, 8, 1) clean_support=(2, 8, 8) loss=... mae=...
```

## Baseline 训练

```powershell
conda activate research
python -m src.train --model baseline --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --rnn-units 128 --prototype-dim 64 --prototype-num 20 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --lamb-c 0.01 --lamb-d 1 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --output-dir log --run-name metrla_stssdl_full_compare
```

## DCD-ST 训练

DCD-ST 代码放在：

```text
DCD-ST/dcd_st.py
DCD-ST/deviation_decomposition.py
```

主实验命令：

```powershell
conda activate research
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --lr 0.001 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --rnn-units 128 --rnn-layers 1 --cheb-k 3 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --decomp-kernel-size 3 --dev-embed-dim 32 --gate-hidden-dim 128 --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_v1
```

短训练调试：

```powershell
conda activate research
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --epochs 3 --max-batches 20 --batch-size 64 --device cuda:0 --no-contrastive-loss --no-deviation-loss --run-name metrla_dcd_debug
```

如果 `g_dev` 出现大面积全开，可加入轻量稀疏约束：

```powershell
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --epochs 100 --patience 20 --batch-size 64 --device cuda:0 --gate-sparse-weight 0.0001 --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_v1_sparse
```

## 日志字段

每个 epoch 会打印：

```text
train_loss
train_mae
contrastive
deviation
gate_sparse
gate_smooth
val_mae / val_rmse / val_mape
val_15min / val_30min / val_60min
```

其中：

| 字段 | 含义 |
|---|---|
| `contrastive` | baseline 使用的 ST-SSDL prototype 三元组损失；DCD-ST 通常关闭 |
| `deviation` | baseline 使用的 prototype distance consistency；DCD-ST 通常关闭 |
| `gate_sparse` | DCD-ST 的 `mean(g_dev)`，用于观察偏差门控是否整体偏高 |
| `gate_smooth` | 预留的时间平滑项，当前第一版默认为 0 |

## 文档入口

```text
docs/DCD-ST.md          DCD-ST 第一版实现方案
DCD-ST/README.md        DCD-ST 环境、命令和消融说明
docs/实验思路说明.md    DCD-ST 研究思路说明
docs/消融实验方案.md    baseline 与新模型消融设计
docs/可视化实验方案.md  ST-SSDL 可视化实验方案
```
