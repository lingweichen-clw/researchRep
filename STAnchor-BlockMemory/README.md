# STAnchor-BlockMemory

独立实现的单源时空预训练与历史块检索项目。设计依据见 `doc/计划方案.md`。

## 当前实现范围

v1 已实现：

- HDF 交通序列读取、严格时间切分、节点级训练段 scaler。
- 时间 patch mask 与整节点 mask，按 batch 解耦交替。
- mask-aware 水平统计，整节点遮挡使用 unknown-level token。
- 时间 Transformer + 图边稀疏空间注意力的共享 encoder。
- clean retrieval 分支、masked reconstruction 分支和 future-guided retrieval loss。
- `.npy + mmap` 只读 Bank、calendar 倒排索引和 schema manifest。
- 事件 top-`R` 粗检索、节点 top-`K` 精排、历史 future 聚合与候选方差。
- 轻量节点 MLP backbone、六特征 confidence 和 exact fallback fusion。
- 预训练、建库、目标训练、评估四阶段 CLI 与 checkpoint 契约。

v1 未实现可选 target adapter、FAISS 近似索引、多源联合预训练和跨物理量迁移。

## 环境

实验环境固定为 `research`，当前机器为 Python 3.10、PyTorch 2.11 + CUDA 12.8。

```powershell
conda activate research
cd D:\projects\researchProjects\TrafficRobustST\STAnchor-BlockMemory
python -m pip install -e . --no-deps
```

核心依赖已经写入 `pyproject.toml`：`torch`、`numpy`、`pandas`、`scipy`、`tables`、`PyYAML`。单元测试使用 Python 内置 `unittest`，不要求安装 pytest。

运行测试：

```powershell
python -m unittest discover -s tests -v
python -m compileall -q stanchor scripts tests
```

## 数据契约

原始文件必须是带 `DatetimeIndex` 的 pandas HDF，形状为 `(L,N)`；邻接矩阵使用交通预测项目常见的 `adj_mx.pkl`。模型内部统一轴顺序：

```text
x / y                 (B,T/H,N,C)
patch token            (B,P,N,D)
node key               (B,N,Dr)
event key              (B,Dr)
event candidates       (B,R)
node candidates        (B,N,K)
candidate future       (B,H,N,K,C)
memory/base/final      (B,H,N,C)
confidence/fusion      (B,H,N,1)
```

METR-LA 的现有路径已配置在 `configs/metrla_v1.yaml`。PEMS-BAY 模板在 `configs/pemsbay_transfer_v1.yaml`，运行前需要确认本机原始 HDF 与邻接矩阵路径。

## 正式实验顺序

以下命令从本目录运行。

### 1. 在源数据集预训练

```powershell
python scripts/pretrain.py --config configs/metrla_v1.yaml
```

输出：

```text
artifacts/metrla_pretrain_seed42/pretrain_best.pt
artifacts/metrla_pretrain_seed42/pretrain.log
artifacts/metrla_pretrain_seed42/pretrain_metrics.jsonl
```

启动后控制台会立即打印数据规模、张量契约、图边数、优化器配置，以及
`embedding / encoder / retrieval_head / reconstruction_head` 的参数量。相同内容同步写入
`pretrain.log`；逐 epoch 的结构化指标写入 `pretrain_metrics.jsonl`，供后续脚本画曲线。

### Patch1 时间注意力实验

为了检验 4-token 时间注意力是否受到粗粒度 patch 限制，新增独立配置：

```powershell
python scripts/pretrain.py --config configs/metrla_patch1_v1.yaml
```

该实验不覆盖原 patch3 基线：

```text
token:       (B,12,N,64)，每个 token 对应 1 个 5 分钟时间步
time mask:  每次连续遮挡 3 个 token，即仍遮挡 15 分钟
output:      artifacts/metrla_patch1_pretrain_seed42
```

`time_mask_block_size` 以原始时间步为单位，必须能够被 `patch_size` 整除。patch1 checkpoint
不能复用 patch3 Bank；后续必须使用同一个 `metrla_patch1_v1.yaml` 重新运行建库命令。

### 2. 用目标训练历史重建 Bank

以 `METR-LA -> PEMS-BAY` 为例：

```powershell
python scripts/build_bank.py `
  --config configs/pemsbay_transfer_v1.yaml `
  --checkpoint artifacts/metrla_pretrain_seed42/pretrain_best.pt `
  --dataset-name PEMS-BAY
```

Bank 输出目录由目标配置的 `bank.output_dir` 决定。为避免实验污染，程序拒绝覆盖已有 Bank；新实验请使用新的目录名。

### 3. 训练目标 backbone、confidence 与 fusion

```powershell
python scripts/train_downstream.py `
  --config configs/pemsbay_transfer_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_pretrain_seed42/pretrain_best.pt `
  --bank artifacts/pemsbay_bank_from_metrla_v1
```

下游训练会在控制台和 `downstream.log` 中打印 `backbone / confidence_head / fusion` 的
可训练参数量，并单独报告被冻结的预训练模型参数量。逐 epoch 指标保存到
`target_metrics.jsonl`。三个日志文件的职责如下：

```text
*.log          人类可读；与控制台同步，适合实时排错
*.jsonl        机器可读；每行一个 epoch，适合统计和绘图
*_best.pt      验证集最优 checkpoint
```

### 4. 测试集评估

```powershell
python scripts/evaluate.py `
  --config configs/pemsbay_transfer_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_pretrain_seed42/pretrain_best.pt `
  --downstream-checkpoint artifacts/metrla_to_pemsbay_v1/downstream_best.pt `
  --bank artifacts/pemsbay_bank_from_metrla_v1 `
  --split test
```

训练损失在目标数据集标准化尺度计算；报告的 MAE、RMSE、MAPE 和 horizon MAE 会先逆变换回原始物理尺度。

## 调试命令

真实 METR-LA one-batch 预训练：

```powershell
python scripts/pretrain.py --config configs/metrla_v1.yaml --run-name metrla_smoke --epochs 1 --max-batches 1
```

构建小型调试 Bank：

```powershell
python scripts/build_bank.py `
  --config configs/metrla_v1.yaml `
  --checkpoint artifacts/metrla_smoke/pretrain_best.pt `
  --output-dir artifacts/metrla_bank_debug `
  --dataset-name METR-LA `
  --max-events 2016
```

下游 one-batch：

```powershell
python scripts/train_downstream.py `
  --config configs/metrla_v1.yaml `
  --pretrained-checkpoint artifacts/metrla_smoke/pretrain_best.pt `
  --bank artifacts/metrla_bank_debug `
  --run-name metrla_downstream_smoke `
  --epochs 1 `
  --max-batches 1
```

`--max-events`、`--epochs` 和 `--max-batches` 只用于调试，不应出现在正式结果中。

## Transformer 实现说明

当前时间分支使用 PyTorch 官方 `nn.MultiheadAttention`，并不是手写的稠密注意力；空间分支
使用项目内的图边稀疏注意力，因为它需要显式接收 `edge_index: (2,E)` 和
`edge_weight: (E)`。当前 `research` 环境不依赖 Hugging Face `transformers`。

Hugging Face Transformer 可以作为可选时间编码后端，但不适合直接替换整个时空 encoder：

- 按节点处理时间 token 时，可将 `(B,P,N,D)` 变为 `(B*N,P,D)` 后送入支持
  `inputs_embeds` 的 Transformer，再恢复为 `(B,P,N,D)`。
- 空间分支仍应保留图边稀疏注意力；把 `P*N` 展平成一个稠密序列会产生
  `O((P*N)^2)` 注意力开销，同时丢失邻接边和边权约束。
- NLP Transformer 的预训练权重不会天然迁移到交通数值 patch。若只随机初始化其 encoder，
  它主要带来接口与依赖开销，不等价于获得语言模型的预训练知识。

因此 v1 保持 PyTorch 原生时间注意力作为默认实现。Hugging Face 后端只有在需要做编码器
消融或复用某个经过时序数据预训练的 checkpoint 时再加入更合理。

## Bank 存储

不使用数据库服务。Bank 采用事件轴严格对齐的 mmap 数组：

```text
manifest.json
event_keys.npy
node_keys.npy
future_values.npy
future_masks.npy
level_features.npy
weekday.npy / slot.npy
context_start.npy / context_end.npy / future_end.npy
sample_id.npy
calendar_offsets.npy / calendar_event_ids.npy
```

完整节点 key 和 future 保持在磁盘/CPU mmap；小型事件 key 常驻内存；每次只把 top-`R` 和 top-`K` 候选送入 GPU。manifest 会校验 encoder、目标图、节点数、维度和 scaler，避免旧 Bank 与新模型混用。

## 因果边界

- scaler 只拟合当前数据集训练段。
- context 或 future 完全没有有效观测的事件不进入 Dataset；其余缺失位置仍由 observed mask 控制。
- Bank 只由目标训练历史的 `D_mem` 构建。
- 候选必须满足 `candidate.future_end < query.context_start`。
- validation/test 不写入 Bank。
- 当前 query 的真实 future 只用于损失或评估，不参与检索和 confidence 输入。
- 没有合法候选或候选 future 无效时，fusion weight 精确为 0，最终输出严格等于 base。
