# TrafficRobustST 实现计划

## 当前已实现模块

| 模块 | 文件 | 状态 | 说明 |
|---|---|---|---|
| METR-LA 预处理 | `src/preprocessing.py` | 已实现 | 读取 `METR-LA.h5`，生成 ST-SSDL 风格 `trainhis/valhis/testhis.npz` |
| 训练集历史锚点 | `src/preprocessing.py` | 已实现 | 只用训练切分统计 weekday-time 历史均值，避免验证/测试泄漏 |
| 数据加载与标准化 | `src/data.py` | 已实现 | 读取 npz，按训练集 value 通道做标准化，同时标准化 history anchor |
| 指标 | `src/metrics.py` | 已实现 | masked MAE/RMSE/MAPE，支持整体和 15/30/60 分钟 horizon |
| 复合损失 | `src/losses.py` | 已实现 | `MAE + contrastive + deviation + region + graph_reg` |
| AGCRN 编码器/解码器 | `src/models/agcrn.py` | 已实现 | 参考 ST-SSDL 的 AGCN、AGCRNCell、ADCRNNEncoder、ADCRNNDecoder |
| 可学习原型 | `src/models/prototype.py` | 已实现 | 参考 ST-SSDL 的 query/prototype attention、top-2 正负原型 |
| BCC 区域选择 | `src/models/graph_regions.py` | 已实现 | 参考 DarkFarseer 的 biconnected components 构造区域正样本 |
| 区域原型 | `src/models/graph_denoise.py` | 已实现 | 用 BCC positive mask 聚合节点隐藏状态为区域原型 |
| 软图去噪 | `src/models/graph_denoise.py` | 已实现 | 用节点表示相似度和区域原型相似度给动态图边降噪 |
| 主模型 | `src/models/region_stssdl.py` | 已实现 | `RegionAwareSTSSDL` 串联 ST-SSDL 双分支、原型、区域图去噪和 decoder |
| 训练入口 | `src/train.py` | 已实现 | 支持 smoke test、训练、验证、测试和逐轮控制台日志 |
| 实现说明 | `docs/首版实现说明.md` | 已实现 | 记录环境、命令、接口和已验证结果 |
| 修改方案 | `docs/鲁棒交通预测修改方案.md` | 已实现 | 记录多条可选研究路线和数据集选择 |

## 已验证内容

| 验证项 | 状态 | 命令/结果 |
|---|---|---|
| 语法检查 | 通过 | `python -m py_compile ...` |
| 随机小图 smoke test | 通过 | `prediction=(2, 12, 8, 1)`, `clean_support=(2, 8, 8)` |
| METR-LA 小样本预处理 | 通过 | `train=(179,12,207,3)`, `val=(26,12,207,3)`, `test=(51,12,207,3)` |
| 小闭环训练 | 通过 | 1 epoch、1 batch 可输出训练损失、MAE、验证和测试指标 |

## 后续优先完善模块

### 1. 完整实验训练流程

目标：从小样本调试切到完整 METR-LA。

待做：

| 任务 | 说明 |
|---|---|
| 生成完整 `data/METRLA` | 不使用 `--max-windows` |
| 增加模型保存 | 保存 best val MAE 的 checkpoint |
| 增加日志文件 | 控制台打印保留，同时写入 `log/*.txt` |
| 增加早停 | 根据 val MAE 设置 patience |
| 增加学习率调度 | 可参考 ST-SSDL 的 MultiStepLR |

### 2. 配置文件化

目标：减少命令行超参过长的问题。

待做：

| 任务 | 说明 |
|---|---|
| 新建 `configs/metrla_region_stssdl.yaml` | 保存数据路径、模型参数、损失权重 |
| `train.py` 支持 `--config` | 命令行参数覆盖 config |
| 记录最终实验配置 | 每次训练复制 config 到输出目录 |

### 3. 更完整的 Region Graph Denoising

目标：让 DarkFarseer 迁移更接近论文机制。

待做：

| 任务 | 说明 |
|---|---|
| fragile node 选择 | 根据低度节点、高偏差节点、高训练误差节点选择 anchor |
| hard edge drop 消融 | 当前是软降权，后续可加 bottom-k 丢边版本 |
| epoch-level 图缓存 | 降低每 batch 生成动态图的开销 |
| BCC 阈值搜索 | METR-LA、PEMS04、PEMS07 使用不同阈值 |

### 4. 实验与消融

目标：证明每个模块的贡献。

待做：

| 实验 | 说明 |
|---|---|
| Base ST-SSDL | 关闭 `region` 和 `graph_reg` |
| + Region loss | 只加 BCC 区域对比 |
| + Graph denoise | 加完整软图去噪 |
| 不同 `graph_static_weight` | 测 0.0/0.15/0.3/0.5 |
| 不同 `lamb_region` | 测 0.01/0.05/0.1 |
| horizon-wise 对比 | 15/30/60 分钟分别报告 |

### 5. 扩展数据集

目标：验证方案不只适用于 METR-LA。

待做：

| 数据集 | 优先级 | 说明 |
|---|---|---|
| PEMS04 | 高 | 有 3 个原始通道，适合作为第二主实验 |
| PEMS07 | 中高 | 节点数 883，更能体现图鲁棒性 |
| PEMS08 | 中 | 节点少但通道多，适合轻量消融 |
| PEMS-BAY | 中 | 与 METR-LA 同类单变量速度数据 |

### 6. 工程稳健性

目标：让代码更适合长期实验。

待做：

| 任务 | 说明 |
|---|---|
| 单元测试 | 测 `prepare_x_y`、adj loading、模型 forward shape |
| checkpoint resume | 支持断点续训 |
| AMP 混合精度 | CUDA 上加速训练 |
| 显存监控 | 每 epoch 输出 peak memory |
| 结果导出 | 保存预测值、真实值和 horizon metrics |

