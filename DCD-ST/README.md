# DCD-ST 训练命令

更新时间：2026-06-30

本文件只保留当前 DCD-ST 主线和 ST-SSDL baseline 对照命令。旧版非 DCD-ST 扩展已移除，当前训练入口只支持：

```text
--model dcd
--model baseline
```

所有命令默认在项目根目录运行：

```powershell
cd D:\projects\researchProjects\TrafficRobustST
conda activate research
```

## 1. 环境依赖

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn tables
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

检查环境：

```powershell
python -c "import torch, numpy, pandas, scipy, sklearn, tables; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## 2. 数据预处理

如果已经存在以下文件，可跳过本步：

```text
data/METRLA/trainhis.npz
data/METRLA/valhis.npz
data/METRLA/testhis.npz
data/METRLA_data/adj_mx.pkl
```

生成 METR-LA 切分：

```powershell
python src\preprocessing.py --traffic-h5 data\METRLA_data\METR-LA.h5 --output-dir data\METRLA
```

## 3. 快速验证

DCD-ST smoke test：

```powershell
python -m src.train --smoke-test --model dcd --no-contrastive-loss --no-deviation-loss
```

baseline smoke test：

```powershell
python -m src.train --smoke-test --model baseline
```

期望输出形态：

```text
smoke ok: prediction=(2, 12, 8, 1) clean_support=(2, 8, 8) loss=... mae=...
```

## 4. DCD-ST 短训练

用于确认真实数据、日志、checkpoint 保存链路都正常：

```powershell
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --adj-type symadj --seq-len 12 --horizon 12 --epochs 3 --patience 20 --batch-size 64 --num-workers 0 --max-batches 20 --device cuda:0 --seed 999 --lr 0.001 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --weight-decay 0 --max-grad-norm 5 --rnn-units 128 --rnn-layers 1 --cheb-k 3 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --decomp-kernel-size 3 --dev-embed-dim 32 --gate-hidden-dim 128 --gate-sparse-weight 0 --gate-smooth-weight 0 --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_debug
```

输出位置：

```text
log/metrla_dcd_debug/train.log
log/metrla_dcd_debug/config.json
log/metrla_dcd_debug/best_model.pt
```

## 5. DCD-ST 正式训练

第一版主实验命令：

```powershell
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --adj-type symadj --seq-len 12 --horizon 12 --epochs 100 --patience 20 --batch-size 64 --num-workers 0 --device cuda:0 --seed 999 --lr 0.001 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --weight-decay 0 --max-grad-norm 5 --rnn-units 128 --rnn-layers 1 --cheb-k 3 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --decomp-kernel-size 3 --dev-embed-dim 32 --gate-hidden-dim 128 --gate-sparse-weight 0 --gate-smooth-weight 0 --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_v1
```

说明：

```text
--no-contrastive-loss
--no-deviation-loss
```

DCD-ST 第一版不使用 ST-SSDL prototype 路径，因此正式训练建议保留这两个开关，让 loss 只包含预测误差和可选 gate 约束。

## 6. DCD-ST 稀疏门控训练

如果可视化或日志中发现 `g_dev` 长期偏高，可加入轻量 gate sparse loss：

```powershell
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --adj-type symadj --seq-len 12 --horizon 12 --epochs 100 --patience 20 --batch-size 64 --num-workers 0 --device cuda:0 --seed 999 --lr 0.001 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --weight-decay 0 --max-grad-norm 5 --rnn-units 128 --rnn-layers 1 --cheb-k 3 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --decomp-kernel-size 3 --dev-embed-dim 32 --gate-hidden-dim 128 --gate-sparse-weight 0.0001 --gate-smooth-weight 0 --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_v1_sparse
```

当前 `gate_smooth_loss` 固定为 0，因为第一版 `g_dev` 是节点级张量 `(B,N,R)`，不是逐时间步张量。

## 7. DCD-ST 消融命令

关闭 temporal deviation norm：

```powershell
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --adj-type symadj --seq-len 12 --horizon 12 --epochs 100 --patience 20 --batch-size 64 --num-workers 0 --device cuda:0 --seed 999 --lr 0.001 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --rnn-units 128 --rnn-layers 1 --cheb-k 3 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --decomp-kernel-size 3 --dev-embed-dim 32 --gate-hidden-dim 128 --no-temporal-deviation-norm --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_no_tnorm
```

关闭 spatial deviation norm：

```powershell
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --adj-type symadj --seq-len 12 --horizon 12 --epochs 100 --patience 20 --batch-size 64 --num-workers 0 --device cuda:0 --seed 999 --lr 0.001 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --rnn-units 128 --rnn-layers 1 --cheb-k 3 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --decomp-kernel-size 3 --dev-embed-dim 32 --gate-hidden-dim 128 --no-spatial-deviation-norm --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_no_snorm
```

增大趋势分解窗口：

```powershell
python -m src.train --model dcd --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --adj-type symadj --seq-len 12 --horizon 12 --epochs 100 --patience 20 --batch-size 64 --num-workers 0 --device cuda:0 --seed 999 --lr 0.001 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --rnn-units 128 --rnn-layers 1 --cheb-k 3 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --decomp-kernel-size 5 --dev-embed-dim 32 --gate-hidden-dim 128 --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_dcd_k5
```

`--decomp-kernel-size` 必须是奇数，建议第一轮只比较 `3` 和 `5`。

## 8. Baseline 对照命令

ST-SSDL full：

```powershell
python -m src.train --model baseline --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --adj-type symadj --seq-len 12 --horizon 12 --epochs 100 --patience 20 --batch-size 64 --num-workers 0 --device cuda:0 --seed 999 --lr 0.001 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --weight-decay 0 --max-grad-norm 5 --rnn-units 128 --rnn-layers 1 --cheb-k 3 --prototype-num 20 --prototype-dim 64 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --lamb-c 0.01 --lamb-d 1 --output-dir log --run-name metrla_stssdl_full_compare
```

ST-SSDL without SSDL：

```powershell
python -m src.train --model baseline --processed-dir data\METRLA --adj-path data\METRLA_data\adj_mx.pkl --adj-type symadj --seq-len 12 --horizon 12 --epochs 100 --patience 20 --batch-size 64 --num-workers 0 --device cuda:0 --seed 999 --lr 0.001 --lr-scheduler multistep --lr-milestones 40,70 --lr-decay-ratio 0.1 --weight-decay 0 --max-grad-norm 5 --rnn-units 128 --rnn-layers 1 --cheb-k 3 --input-embedding-dim 3 --tod-embed-dim 20 --node-embedding-dim 25 --adaptive-embedding-dim 0 --no-ssdl --no-contrastive-loss --no-deviation-loss --output-dir log --run-name metrla_stssdl_no_ssdl_compare
```

## 9. CPU 调试

如果当前机器没有可用 GPU，把命令中的：

```text
--device cuda:0
```

替换为：

```text
--device cpu
```

## 10. 日志字段

DCD-ST 训练日志会打印：

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

其中 `contrastive` 和 `deviation` 在 DCD-ST 主实验中应为 `0.0000`；`gate_sparse` 默认只是记录 `mean(g_dev)`，只有设置 `--gate-sparse-weight` 后才进入总损失。
