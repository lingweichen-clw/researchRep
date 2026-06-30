# DCD-ST 第一版实现方案

更新时间：2026-06-30

实验环境记录：

```text
conda activate research
Python 3.10
```

## 1. 实现目标

DCD-ST 的第一版目标不是重新发明一个大模型，而是在当前 ST-SSDL baseline 上做减法：

```text
保留：Xc / Xa 双分支历史锚点、AGCRN encoder-decoder、node embedding、time-of-day embedding、动态 support。
删除：learnable prototypes、Top-2 positive/negative prototype、triplet contrastive loss、prototype deviation loss。
新增：连续偏差分解、时空身份条件化门控、可视化友好的中间变量。
```

第一版先实现：

```text
DCD-ST-v1 = ST-SSDL backbone - prototype path + deviation decomposition gate
```

测试时校准先写接口与实现计划，放到第二阶段。原因是 ST-TTC 源码显示它是 normal test 与 `test_with_ttc` 的测试阶段切换机制，不适合作为第一版训练主体强行塞进主模型。

## 2. 已参考的源码与可借鉴点

### 2.1 当前 TrafficRobustST baseline

当前 ST-SSDL baseline 主要结构在：

```text
src/models/stssdl_baseline.py
```

关键位置：

| 代码位置 | 当前作用 | DCD-ST 处理 |
|---|---|---|
| `src/models/stssdl_baseline.py:103-151` | 构造 prototypes、`Wq`、query-prototype attention、Top-2 prototype | 第一版删除，不再依赖离散 prototype |
| `src/models/stssdl_baseline.py:153-157` | 双分支编码 `Xc` 与 `Xa`，输出 `Hc/Ha` | 保留，这是 ST-SSDL 最有价值的设计 |
| `src/models/stssdl_baseline.py:169-190` | 对 `Hc/Ha` 查询 prototype，计算 `latent_dis/prototype_dis` | 替换为偏差分解与门控 |
| `src/models/stssdl_baseline.py:191-193` | `h_de = concat(Hc, Vc)`，再用 `Hc/Vc/Ha/Va` 生成动态图 | 替换为 `H_de = Hc + g_dev * Delta_H` |
| `src/models/stssdl_baseline.py:199-205` | AGCRN decoder 自回归预测 | 保留 |
| `src/data.py:28-40` | 从 batch 中拆出 `x0/x_cov/x_his/y/y_cov` | 保留，DCD-ST 直接复用数据接口 |
| `src/losses.py:24-65` | MAE + contrastive + deviation + region/graph loss | 保留 MAE，prototype 两个 loss 对 DCD-ST 置零或关闭 |

### 2.2 STID 源码

参考目录：

```text
../STID/stid/arch/stid_arch.py
../STID/stid/arch/mlp.py
```

借鉴点：

| 代码位置 | 机制 | DCD-ST 用法 |
|---|---|---|
| `../STID/stid/arch/stid_arch.py:35-46` | `node_emb`、`time_in_day_emb`、`day_in_week_emb` | 用 node/time identity 构造 `z_id` |
| `../STID/stid/arch/stid_arch.py:94-109` | time-series embedding 与身份 embedding 拼接 | DCD-ST 的 gate 输入也采用拼接式上下文 |
| `../STID/stid/arch/mlp.py:5-25` | 1x1 Conv MLP + residual | 可复用为轻量 gate/projection 风格 |

STID 对我们的启发是：很多时空预测收益来自解决节点和时间不可区分问题。DCD-ST 不引入复杂新 backbone，只把身份信息用于控制偏差校正强度。

### 2.3 ST-Norm 源码

参考目录：

```text
../ST-Norm/models/Wavenet.py
../ST-TTC/small_scale_scenario/src/models/stnorm.py
```

借鉴点：

| 代码位置 | 机制 | DCD-ST 用法 |
|---|---|---|
| `../ST-Norm/models/Wavenet.py:9-16` | `SNorm` 在节点维做 spatial normalization | 对 `R = Xc - Xa` 构造空间偏差 `D_s` |
| `../ST-Norm/models/Wavenet.py:22-45` | `TNorm` 在时间/批维维护 temporal normalization | 第一版用更轻的窗口内 `TemporalDeviationNorm` |
| `../ST-Norm/models/Wavenet.py:146-153` | 原特征、TNorm、SNorm 拼接后送入卷积 | DCD-ST 拼接 `R_trend/R_residual/D_t/D_s` 形成 `z_dev` |

ST-Norm 对我们的启发是：不要只看 raw residual，要把时间高频扰动与空间局部扰动拆开。

### 2.4 STDN 源码

参考目录：

```text
../STDN/model.py
```

借鉴点：

| 代码位置 | 机制 | DCD-ST 用法 |
|---|---|---|
| `../STDN/model.py:168-195` | `Trend`、`Seasonal`、`Trend_Seasonal_Decomposition` | DCD-ST 对 `Xc-Xa` 做趋势/残差分解 |
| `../STDN/model.py:534-548` | 根据时间 embedding 与节点 embedding 构造动态图 | DCD-ST 继续用 hidden representation 生成动态 support |

STDN 对我们的启发是：交通偏差不应只有一个标量距离。持续性偏移和短期扰动要分开建模。

### 2.5 ST-TTC 源码

参考目录：

```text
../ST-TTC/small_scale_scenario/src/base/engine.py
../ST-TTC/large_scale_scenario/main.py
../ST-TTC/continual_learning_setting/src/trainer/default_trainer.py
```

借鉴点：

| 代码位置 | 机制 | DCD-ST 用法 |
|---|---|---|
| `../ST-TTC/small_scale_scenario/src/base/engine.py:209-280` | `evaluate_with_ttc` 在测试阶段引入 FRP 校准模块 | 第二阶段实现 `evaluate_with_dcd_ttc` |
| `../ST-TTC/large_scale_scenario/main.py:241-303` | `test_with_ttc` 冻结主模型，只优化轻量校准模块 | DCD-ST 不改主模型参数，只校准预测输出 |
| `../ST-TTC/continual_learning_setting/src/trainer/default_trainer.py:20-34` | `FRPlusModule` 对预测张量做可学习校正 | DCD-ST-v2 可换成 gate-guided residual calibration |
| `../ST-TTC/continual_learning_setting/src/trainer/default_trainer.py:283-335` | 测试时逐批更新校准器 | 第二阶段实现 streaming residual memory |

ST-TTC 对我们的启发是：鲁棒性不一定全部塞进训练阶段，测试阶段可以用轻量状态做校准。但第一版先不做，以保证 DCD-ST 主体干净稳定。

## 3. 第一版总体结构

原 ST-SSDL：

```text
Xc, Xa
  -> Encoder
  -> Hc, Ha
  -> query prototypes
  -> Vc, Va
  -> concat(Hc, Vc)
  -> dynamic support
  -> AGCRN decoder
  -> prediction
```

DCD-ST-v1：

```text
Xc, Xa
  -> Encoder
  -> Hc, Ha
  -> deviation decomposition on R = Xc - Xa
  -> z_dev
  -> z_id from node/time embedding
  -> g_dev, Delta_H
  -> H_de = Hc + g_dev * Delta_H
  -> dynamic support from [H_de, Ha, H_de-Ha, g_dev]
  -> AGCRN decoder
  -> prediction
```

核心变化：

```text
prototype value Vc 被连续偏差校正 Delta_H 替代。
prototype assignment 被 gate activation g_dev 替代。
prototype usage visualization 被 trend/residual/spatial/temporal/gate visualization 替代。
```

## 4. 新增文件设计

### 4.1 `DCD-ST/deviation_decomposition.py`

该文件只负责偏差特征提取，不包含预测主干。

建议包含四个类：

```python
class MovingAverageDecomposition(nn.Module):
    """Decompose residual sequence into trend and short-term residual."""


class TemporalDeviationNorm(nn.Module):
    """Normalize deviation along the temporal dimension inside each window."""


class SpatialDeviationNorm(nn.Module):
    """Normalize deviation across nodes or graph neighbors."""


class DeviationFeatureExtractor(nn.Module):
    """Return raw, trend, residual, temporal norm, spatial norm and z_dev."""
```

#### 输入输出

输入：

```text
x:      (B,T,N,1) 当前窗口
x_his:  (B,T,N,1) 历史锚点
support: optional static support, (N,N)
```

中间变量：

```text
R_raw      = x - x_his                  -> (B,T,N,1)
R_trend    = moving_avg_T(R_raw)        -> (B,T,N,1)
R_residual = R_raw - R_trend            -> (B,T,N,1)
D_t        = temporal_norm(R_raw)       -> (B,T,N,1)
D_s        = spatial_norm(R_raw)        -> (B,T,N,1)
```

第一版 `z_dev` 建议采用 8 维节点级统计：

```text
z_dev = [
  mean_T(R_trend),
  mean_T(abs(R_trend)),
  mean_T(R_residual),
  mean_T(abs(R_residual)),
  mean_T(D_t),
  mean_T(abs(D_t)),
  mean_T(D_s),
  mean_T(abs(D_s)),
] -> (B,N,8)
```

为什么不是只用 4 维：只用绝对值会丢失“速度高于历史”还是“速度低于历史”的方向信息。8 维仍然非常轻量，但能同时保留方向与强度。

#### Moving average 实现建议

推荐用 `AvgPool1d`，不要手写 Python 循环：

```text
输入 R_raw: (B,T,N,1)
reshape:   (B*N,1,T)
replicate pad on T
AvgPool1d(kernel_size=k, stride=1)
reshape:   (B,T,N,1)
```

默认参数：

```text
decomp_kernel_size = 3
```

`T=12` 的 METR-LA 窗口较短，第一版不建议使用过大 kernel。后续可消融 `3/5/7`。

#### TemporalDeviationNorm 实现建议

第一版使用窗口内标准化：

```text
mean_t = mean(R_raw, dim=1, keepdim=True)
std_t  = std(R_raw, dim=1, keepdim=True) + eps
D_t    = (R_raw - mean_t) / std_t
```

说明：ST-Norm 原实现有 running stats，但 DCD-ST 的输入是 `Xc-Xa` 的偏差窗口，且第一版需要避免引入新的训练状态，所以先不用 running mean/var。

#### SpatialDeviationNorm 实现建议

第一版先实现全图空间标准化：

```text
mean_s = mean(R_raw, dim=2, keepdim=True)
std_s  = std(R_raw, dim=2, keepdim=True) + eps
D_s    = (R_raw - mean_s) / std_s
```

第二版可升级为邻域空间标准化：

```text
neighbor_mean = support @ R_raw
D_s = (R_raw - neighbor_mean) / neighbor_std
```

第一版先用全图版本，原因是当前 `src/models/agcrn.py` 已经在 encoder/decoder 内使用 support，偏差提取层如果一开始就再引入邻域统计，会让实验归因变复杂。

### 4.2 `DCD-ST/dcd_st.py`

该文件实现新模型类：

```python
class DCDST(nn.Module):
    """Deviation-Calibrated Decomposition model built on ST-SSDL backbone."""
```

第一版建议不要继承 `STSSDLBaseline`，而是复制其必要结构并删除 prototype 分支。原因是继承后需要绕开 `prototype_dim/decoder_dim/query_prototypes`，可读性反而差。

#### 初始化参数

与 `STSSDLBaseline` 保持大部分一致：

```text
num_nodes
supports
input_dim = 1
output_dim = 1
horizon = 12
rnn_units = 128
rnn_layers = 1
cheb_k = 3
input_embedding_dim = 3
tod_embed_dim = 20
node_embedding_dim = 25
adaptive_embedding_dim = 0
tday = 288
cl_decay_steps = 2000
use_curriculum_learning = True
```

DCD-ST 新增：

```text
decomp_kernel_size = 3
dev_feature_dim = 8
dev_embed_dim = 32
gate_hidden_dim = 128
use_spatial_deviation_norm = True
use_temporal_deviation_norm = True
```

删除或忽略：

```text
prototype_num
prototype_dim
use_ssdl
```

#### 编码部分

完全复用 ST-SSDL 的 embedding 与 AGCRN encoder：

```text
Hc = Encoder(embed(Xc, x_cov)) -> (B,N,R)
Ha = Encoder(embed(Xa, x_cov)) -> (B,N,R)
```

对应现有代码：

```text
src/models/stssdl_baseline.py:116-137
src/models/stssdl_baseline.py:153-157
```

#### DeviationGate

建议在 `dcd_st.py` 中定义：

```python
class DeviationGate(nn.Module):
    def __init__(
        self,
        rnn_units: int,
        dev_feature_dim: int,
        dev_embed_dim: int,
        node_embedding_dim: int,
        tod_embed_dim: int,
        gate_hidden_dim: int,
    ):
        ...
```

输入：

```text
Hc:       (B,N,R)
Ha:       (B,N,R)
z_dev:    (B,N,8)
node_emb: (N,node_dim)
time_emb: (B,N,tod_dim)
```

计算：

```text
z_delta = Hc - Ha                              -> (B,N,R)
z_dev_e = Linear(z_dev)                        -> (B,N,E_dev)
z_id = concat(node_emb, time_emb)              -> (B,N,E_id)

g_dev = sigmoid(MLP([z_delta, z_dev_e, z_id])) -> (B,N,R)
Delta_H = MLP([z_delta, z_dev_e])              -> (B,N,R)
H_de = Hc + g_dev * Delta_H                    -> (B,N,R)
```

这里的 `g_dev` 不是额外预测结果，而是“偏差校正强度”。当当前窗口与历史锚点很接近时，模型可以让 `g_dev` 接近 0，使 `H_de` 退化为原 AGCRN hidden；当出现高偏差时，`g_dev` 打开，让 `Delta_H` 对 hidden 做校正。

#### 动态 support

ST-SSDL 当前使用：

```text
h_aug = concat(Hc, Vc, Ha, Va)
node_embeddings = hypernet(h_aug)
support = softmax(relu(node_embeddings @ node_embeddings^T))
```

DCD-ST-v1 使用：

```text
h_aug = concat(H_de, Ha, H_de-Ha, g_dev) -> (B,N,4R)
node_embeddings = Linear(4R, tod_embed_dim)
support = softmax(relu(node_embeddings @ node_embeddings^T))
```

如果想再轻一点，也可以用：

```text
h_aug = concat(H_de, Ha, H_de-Ha) -> (B,N,3R)
```

第一版建议保留 `g_dev`，因为动态图应知道哪些节点处于高偏差状态。

#### Decoder

Decoder 不变，但 `decoder_dim` 从：

```text
rnn_units + prototype_dim
```

改为：

```text
rnn_units
```

对应：

```text
decoder = ADCRNNDecoder(..., dim_out=rnn_units, num_support=1)
proj = Linear(rnn_units, output_dim)
```

#### forward 输出字典

为了兼容 `src/losses.py` 与可视化，输出建议包含：

```python
output = {
    "prediction": prediction,

    # compatibility with baseline loss
    "query": zero_query,
    "pos": zero_query,
    "neg": zero_query,
    "mask": zero_mask,
    "latent_dis": zero_node,
    "prototype_dis": zero_node,

    # DCD-ST
    "gate_sparse_loss": g_dev.mean(),
    "gate_smooth_loss": gate_smooth_loss,
    "clean_support": support,
}
```

当 `return_intermediates=True` 时额外返回：

```text
h_c
h_a
h_de
delta_h
g_dev
z_dev
r_raw
r_trend
r_residual
d_t
d_s
clean_support
```

这样后续 `src/visualize_dcd.py` 可以直接画：

```text
current vs anchor
trend/residual heatmap
temporal/spatial deviation heatmap
gate activation map
dynamic support map
```

## 5. 损失函数改造

当前 `src/losses.py` 保留：

```text
MAE
contrastive
deviation
gate_sparse
gate_smooth
```

第一版建议写法：

```python
@dataclass
class LossWeights:
    contrastive: float = 0.01
    deviation: float = 1.0
    gate_sparse: float = 0.0
    gate_smooth: float = 0.0
    use_contrastive: bool = True
    use_deviation: bool = True
```

新增：

```python
gate_sparse_loss = model_output.get("gate_sparse_loss", zero_loss)
gate_smooth_loss = model_output.get("gate_smooth_loss", zero_loss)

total = (
    mae_loss
    + weights.contrastive * contrastive_loss
    + weights.deviation * deviation_loss
    + weights.gate_sparse * gate_sparse_loss
    + weights.gate_smooth * gate_smooth_loss
)
```

第一版默认建议：

```text
gate_sparse_weight = 0.0
gate_smooth_weight = 0.0
```

也就是说先只用 MAE 训练，让实验先回答：

```text
不用 prototypes 和辅助 prototype loss，只靠连续偏差分解与门控，能否超过或接近 ST-SSDL？
```

若 gate 出现全开问题，再打开：

```text
gate_sparse_weight = 1e-4
```

若实现时间步级 gate，再打开：

```text
gate_smooth_weight = 1e-4
```

注意：当前第一版 `g_dev` 是节点级 `(B,N,R)`，不是时间步级 `(B,T,N,R)`，所以 `gate_smooth_loss` 可以先定义为 0。后续如果把 gate 改为逐时间步，再计算时间差分。

## 6. 训练入口改造

### 6.1 `src/models/__init__.py`

保留 baseline 注册：

```python
from .stssdl_baseline import STSSDLBaseline

__all__ = ["STSSDLBaseline"]
```

DCD-ST 代码放在 `DCD-ST/` 目录下，由 `src/train.py` 动态加载。

### 6.2 `src/train.py`

导入：

```python
from .models import STSSDLBaseline
```

`parse_args()` 中：

```python
parser.add_argument("--model", default="baseline", choices=["baseline", "dcd"])
parser.add_argument("--decomp-kernel-size", type=int, default=3)
parser.add_argument("--dev-embed-dim", type=int, default=32)
parser.add_argument("--gate-hidden-dim", type=int, default=128)
parser.add_argument("--gate-sparse-weight", type=float, default=0.0)
parser.add_argument("--gate-smooth-weight", type=float, default=0.0)
```

`build_model()` 中：

```python
if args.model == "dcd":
    return DCDST(
        num_nodes=num_nodes,
        supports=supports,
        horizon=args.horizon,
        rnn_units=args.rnn_units,
        rnn_layers=args.rnn_layers,
        cheb_k=args.cheb_k,
        input_embedding_dim=args.input_embedding_dim,
        tod_embed_dim=args.tod_embed_dim,
        node_embedding_dim=args.node_embedding_dim,
        adaptive_embedding_dim=args.adaptive_embedding_dim,
        use_curriculum_learning=args.use_curriculum_learning,
        decomp_kernel_size=args.decomp_kernel_size,
        dev_embed_dim=args.dev_embed_dim,
        gate_hidden_dim=args.gate_hidden_dim,
    ).to(device)
```

`LossWeights` 构造中新增：

```python
gate_sparse=args.gate_sparse_weight,
gate_smooth=args.gate_smooth_weight,
```

`running` 日志建议新增：

```text
gate_sparse
gate_smooth
```

但第一版可以先只把它们写入 loss dict，不强制打印；为了对比清晰，建议打印。

### 6.3 推荐训练命令

第一版主实验：

```powershell
conda activate research
python -m src.train --model dcd --run-name metrla_dcd_v1 --epochs 100 --batch-size 64 --no-contrastive-loss --no-deviation-loss
```

如果先 smoke test：

```powershell
conda activate research
python -m src.train --smoke-test --model dcd --no-contrastive-loss --no-deviation-loss
```

若观察到 `g_dev` 大面积全开，再跑：

```powershell
conda activate research
python -m src.train --model dcd --run-name metrla_dcd_v1_sparse --epochs 100 --batch-size 64 --no-contrastive-loss --no-deviation-loss --gate-sparse-weight 0.0001
```

## 7. 第一版实现顺序

### Step 1：新增偏差分解模块

文件：

```text
DCD-ST/deviation_decomposition.py
```

完成后先做 shape smoke：

```text
x, x_his: (2,12,8,1)
z_dev:    (2,8,8)
r_raw:    (2,12,8,1)
r_trend:  (2,12,8,1)
d_t:      (2,12,8,1)
d_s:      (2,12,8,1)
```

### Step 2：新增 DCDST 模型

文件：

```text
DCD-ST/dcd_st.py
```

必须保证：

```text
prediction:    (B,H,N,1)
clean_support: (B,N,N)
g_dev:         (B,N,R)
h_de:          (B,N,R)
```

兼容 baseline loss 的空张量：

```text
query/pos/neg 最后一维为 0
prototype_dis/latent_dis 为 (B,N) zero tensor
```

这样 `src/losses.py` 即使不开新 loss，也不会因为缺 key 报错。

### Step 3：注册模型与训练入口

文件：

```text
src/models/__init__.py
src/train.py
```

新增 `--model dcd` 和 gate/decomposition 参数。

### Step 4：扩展 loss

文件：

```text
src/losses.py
```

新增 gate loss 但默认权重为 0，不影响 baseline 模型。

### Step 5：最小验证

运行：

```powershell
conda activate research
python -m src.train --smoke-test --model dcd --no-contrastive-loss --no-deviation-loss
```

通过标准：

```text
prediction=(2, 12, 8, 1)
clean_support=(2, 8, 8)
loss/mae 可反向传播
```

### Step 6：短训练

运行：

```powershell
conda activate research
python -m src.train --model dcd --run-name metrla_dcd_debug --epochs 3 --max-batches 20 --no-contrastive-loss --no-deviation-loss
```

通过标准：

```text
train_loss 正常下降或无 NaN
val_mae 可计算
best_model.pt/config.json/train.log 正常保存
```

### Step 7：正式训练

运行：

```powershell
conda activate research
python -m src.train --model dcd --run-name metrla_dcd_v1 --epochs 100 --patience 20 --batch-size 64 --no-contrastive-loss --no-deviation-loss
```

## 8. 第一版消融实验

建议先做 5 个实验，不要一开始铺太多：

| 实验名 | 命令差异 | 验证问题 |
|---|---|---|
| `STSSDL-full` | 原 baseline | 原型完整版本表现 |
| `STSSDL-no-ssdl` | `--no-ssdl --no-contrastive-loss --no-deviation-loss` | AGCRN 主干自身强度 |
| `DCD-ST-v1` | `--model dcd` | 连续偏差分解能否替代 prototype |
| `DCD-ST-v1-sparse` | `--gate-sparse-weight 1e-4` | gate 稀疏约束是否必要 |
| `DCD-ST-v1-k5` | `--decomp-kernel-size 5` | 趋势窗口大小是否敏感 |

后续再扩展：

```text
w/o trend
w/o residual
w/o temporal norm
w/o spatial norm
w/o identity gate
w/ test-time calibration
```

## 9. 可视化接口设计

第一版模型要为后续可视化保留中间量。建议 DCDST forward 支持：

```python
return_intermediates: bool = False
```

当为 True，输出：

```text
r_raw
r_trend
r_residual
d_t
d_s
z_dev
g_dev
delta_h
h_c
h_a
h_de
clean_support
prediction
```

后续新增：

```text
src/visualize_dcd.py
```

替代原 prototype 可视化的图：

| 图 | 内容 | 对应解释 |
|---|---|---|
| Figure A | `Xc/Xa/y/prediction` 低中高偏差对比 | 保留 ST-SSDL 的历史锚点解释 |
| Figure B | `R_trend/R_residual` 热力图 | 趋势漂移 vs 突发扰动 |
| Figure C | `D_t/D_s` 热力图 | 时间高频 vs 空间局部异常 |
| Figure D | `g_dev` 节点热力图 | 哪些节点触发偏差校正 |
| Figure E | gate 分位数组 MAE | gate 是否真的捕捉困难样本 |

## 10. 第二阶段：测试时校准设计

ST-TTC 源码中 `evaluate_with_ttc/test_with_ttc` 的共同点是：

```text
主模型 eval，不更新主模型；
创建轻量校准模块；
在测试流中用近期样本更新校准模块；
输出校准后的 prediction。
```

DCD-ST-v2 不直接照搬 FRPlusModule，而是用更贴合本工作的 gate-guided residual memory：

```text
pred_base = model(...)
gate_score = mean_R(g_dev)                  -> (B,N,1)
residual_memory = EMA(residual_memory, recent_y - recent_pred)
pred_calibrated = pred_base + gamma * gate_score * residual_memory
```

建议新增：

```text
src/test_time_calibration.py
```

包含：

```python
class GateGuidedResidualCalibrator:
    def __init__(self, num_nodes, horizon, momentum=0.9, gamma=0.1):
        ...

    def calibrate(self, prediction, gate_score):
        ...

    def update(self, prediction, label):
        ...
```

训练入口暂不加入，后续在 evaluate 中新增：

```text
--eval-method norm
--eval-method dcd-ttc
--ttc-momentum
--ttc-gamma
```

这一阶段要小心数据泄漏：只能用测试流中已经观测到的历史 label 更新 residual memory，不能用当前预测窗口未来 label 先校准当前窗口。

## 11. 成功判定标准

DCD-ST-v1 成功不只看 overall MAE，还要看鲁棒性与解释性。

### 11.1 指标

必须比较：

```text
overall MAE/RMSE/MAPE
15/30/60min horizon metrics
low/medium/high deviation 分组 MAE
missing-ratio robustness
```

### 11.2 可视化

必须观察：

```text
高偏差样本的 g_dev 均值高于低偏差样本
R_trend 能显示持续偏移
R_residual 能显示局部突变
D_s 能突出空间异常节点
support 在高偏差样本下发生合理变化
```

### 11.3 与 ST-SSDL 的核心对照

如果出现以下结果，就可以支撑新工作的论点：

```text
ST-SSDL full 的 prototype usage 仍然坍缩；
DCD-ST 不依赖 prototype，因此没有 assignment collapse；
DCD-ST 在 high-deviation 或 missing scenario 下优于 ST-SSDL full / STSSDL-no-ssdl；
DCD-ST 的可视化能解释偏差来自趋势、残差、时间高频还是空间局部异常。
```

## 12. 第一版论文贡献表述

建议之后论文中这样表述：

1. We identify that prototype-based self-supervised deviation learning in ST-SSDL can suffer from assignment collapse, where the forecasting backbone remains useful but the claimed discrete scientific information space loses fine-grained interpretability.
2. We propose DCD-ST, a subtraction-oriented robust forecasting framework that removes learnable prototypes and replaces them with continuous deviation decomposition over current-anchor residuals.
3. We design a spatio-temporal identity-conditioned deviation gate, which adaptively injects trend/residual deviation corrections into the AGCRN hidden state without adding a heavy backbone.
4. We further prepare a gate-guided test-time calibration interface inspired by ST-TTC, enabling future robustness enhancement under distribution shift without retraining the full model.

## 13. 最小代码改动清单

第一版需要改动：

```text
新增 DCD-ST/deviation_decomposition.py
新增 DCD-ST/dcd_st.py
修改 src/models/__init__.py
修改 src/train.py
修改 src/losses.py
```

第一版不改动：

```text
src/models/stssdl_baseline.py
src/models/agcrn.py
src/data.py
src/metrics.py
src/preprocessing.py
```

这样可以保证原 ST-SSDL baseline 和 DCD-ST 两条线同时存在，便于公平消融。
