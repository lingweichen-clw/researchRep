# PEMS-BAY 数据与 E3 迁移检索诊断

## 1. 诊断目的

本报告回答两个问题：

1. `data/pemsBay_data` 中的 PEMS-BAY 数据和图是否满足当前工程的数据契约；
2. 在不使用 PEMS-BAY 标签更新 encoder 的前提下，METR-LA 预训练的 E3 encoder-selector 是否能迁移到 PEMS-BAY。

所有检索指标只在 PEMS-BAY validation 上计算。test 尚未用于模型选择或机制诊断。

## 2. 数据完整性结论

| 检查项 | 结果 | 判断 |
|---|---:|---|
| HDF 形状 | `52116 x 325` | 符合 PEMS-BAY 325 传感器规模 |
| 时间范围 | 2017-01-01 00:00 至 2017-06-30 23:55 | 正常 |
| 时间戳重复数 | 0 | 通过 |
| 时间戳是否递增 | 是 | 通过 |
| NaN 数 | 0 | 通过 |
| 零值数 | 521 | 按现有协议作为缺失值掩码 |
| 全路网零值时间行 | 0 | 通过 |
| 邻接矩阵形状 | `325 x 325` | 与 HDF 节点数一致 |
| 传感器 ID 数 | 325 | 通过 |
| HDF 列与图节点顺序 | 完全一致 | 通过 |
| 图正权边数（含自环） | 2694 | 通过 |

时间轴只有一处非 5 分钟间隔：

```text
2017-03-12 01:55 -> 2017-03-12 03:00，共 65 分钟
```

该位置对应夏令时跳时。为保持标准 benchmark 的原始口径，本实验不插值、不删除，也不重排时间轴。数据加载器仍按原始行序构造窗口，并依据时间戳计算 weekday-slot。

邻接矩阵是 protocol-0 pickle，文件中含 CRLF 行尾，直接使用通用 `pickle.load` 会报 `STRING opcode argument must be quoted`。项目的 `stanchor.data.graph._load_pickle` 只在普通反序列化失败时尝试将 CRLF 恢复为 LF，恢复后的节点 ID、映射和 `325 x 325` 数值矩阵均通过检查。因此当前文件可以由本项目稳定使用，不需要改写原始数据文件。

## 3. E3 目标数据张量契约

使用 `configs/pemsbay_e3_transfer_v1.yaml`：

| 对象 | 形状或数量 | 含义 |
|---|---:|---|
| 原始速度序列 | `52116 x 325 x 1` | 时间、节点、速度通道 |
| 短期预测输入 | `B x 12 x 325 x 1` | 最近 60 分钟 |
| 检索输入 | `B x 288 x 325 x 1` | 最近一天 |
| patch token | `B x 24 x 325 x 96` | 每 12 步形成一个 patch |
| node key | `B x 325 x 48` | 节点级检索表示 |
| event key | `B x 48` | 事件粗检索表示 |
| future | `B x 12 x 325 x 1` | 未来 60 分钟监督 |

数据集按时间切分后得到：

| split | 有效事件数 |
|---|---:|
| train | 36,182 |
| validation | 4,912 |
| test | 10,125 |

没有事件因 context 或 future 全部缺失而被删除。

## 4. 正式目标 Bank

正式 Bank 使用 train 有效事件的前 70%，共 25,327 个事件；剩余 10,855 个 train 事件用于下游校准。Bank 大小约 1.322 GiB。

完整性检查结果：

- 所有 key、future、mask、level feature 和索引数组均为有限值；
- 2016 个 weekday-slot 桶全部非空；
- 每个桶包含 11 至 13 个历史事件，平均 12.563 个；
- calibration、validation 和 test query 的因果合法候选均为 11 至 13 个，无候选率均为 0；
- `event_top_r=32` 大于最大合法候选数 13，不会截断候选池；
- event key 的平均二范数为 1.0，符合归一化 key 契约。

检索合法性使用：

\[
t^{\mathrm{future\_end}}_j < t^{\mathrm{context\_start}}_i,
\]

其中，(j) 是 Bank 中的候选历史事件，(i) 是当前 query。该约束表示候选事件的未来片段必须在 query 一天检索窗口开始之前已经完整发生，因此 Bank 不会把 query 可见边界之后的信息泄漏给模型。

## 5. Validation 检索结果

所有方法在相同有效位置上评估，覆盖率为 100%。MAE 定义为：

\[
\operatorname{MAE}
=
\frac{1}{|\Omega|}
\sum_{(b,h,n,c)\in\Omega}
\left|
\widehat{Y}_{b,h,n,c}-Y_{b,h,n,c}
\right|,
\]

其中，\(\Omega\) 是存在真实观测的 batch、预测步、节点和通道位置集合。

| 历史预测方法 | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| weekly mean | 2.681505 | 5.199785 | 6.524859 |
| raw-L1 Top-1 | 2.479146 | 5.339728 | 5.720166 |
| raw-L1 Top-K | 2.228216 | 4.729153 | 5.321456 |
| learned Top-1 | 2.530079 | 5.286253 | 5.719548 |
| learned uniform Top-K | 2.280555 | 4.697889 | 5.316481 |
| **learned weighted Top-K** | **2.209414** | **4.593030** | **5.123775** |
| Oracle Top-1 | 1.434256 | 3.256051 | 3.270244 |

相对 weekly mean 的 MAE 改善为：

\[
\operatorname{Gain}_{\mathrm{weekly}}
=
\frac{2.681505-2.209414}{2.681505}\times100\%
=17.61\%.
\]

learned weighted Top-K 比 raw-L1 Top-K 低 0.018802 MAE，约改善 0.84%。该优势较小，但方向为正。

## 6. 机制判断

当前证据支持继续做下游迁移，但不能夸大：

1. 源域 E3 key 在目标域没有失效，learned weighted Top-K 是所有非 Oracle 历史方法中最优；
2. learned Top-1 反而弱于 raw-L1 Top-1，说明迁移优势不主要来自选中单个最佳事件；
3. weighted Top-K 比 learned uniform Top-K 低 0.071141 MAE，说明 selector 分数在 Top-K 内部的相对加权具有价值；
4. learned Top-1 与 Oracle Top-1 仍相差 1.095822 MAE，候选池仍有大量未被 selector 利用的信息；
5. 因此下一步只验证现有 memory 是否能改善简单下游预测，不新增 encoder、adapter 或频域分支。

## 7. 证据文件

- 配置：`configs/pemsbay_e3_transfer_v1.yaml`
- 正式 Bank：`artifacts/pemsbay_bank_from_metrla_e3_relation`
- 检索诊断：`artifacts/pemsbay_e3_transfer_diagnostics/retrieval_diagnostics_val.json`
- 源 checkpoint：`artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt`

