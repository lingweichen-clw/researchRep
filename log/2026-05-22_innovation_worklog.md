# TrafficRobustST Worklog

- 2026-05-22 预备：确认项目根目录暂无 log 目录，计划先改造 `src/preprocessing.py`，把 ST-SSDL 风格预处理改成可分离训练统计、可导出的统一数据接口。
- 2026-05-22 第 1 步完成：已将 `src/preprocessing.py` 改为 split-aware 预处理，新增 `fit / transform / split_windows / export_npz_splits`，并将历史均值统计限制在训练切分内。
- 2026-05-22 第 1 步校验：对 `src/preprocessing.py` 做了静态错误检查，结果通过。
- 2026-05-22 第 2 步完成：已为 `src/preprocessing.py` 增加命令行入口，可直接导出 train/val/test 的 npz 切分文件。
- 2026-05-22 第 3 步完成：已在 `docs/交通预测架构方案.md` 中写入当前落地顺序，明确先数据底座、再 ST-SSDL、再 STEVE 和 DarkFarseer。
- 2026-05-22 第 4 步完成：新增 `src/stssdl_adapter.py`，实现 ST-SSDL 风格模型适配层，统一预处理输出与后续鲁棒模块接口。
- 2026-05-22 第 5 步完成：新增 `dataFlow.md`，记录当前每个模块输入输出、张量维度与端到端数据流约定。
- 2026-05-22 第 6 步完成：新增 `src/stssdl_pipeline.py`，将适配层与 ST-SSDL 风格主干壳接入，形成统一调用入口 `STSSDLUnifiedPipeline`。
- 2026-05-22 第 7 步完成：新增 `src/__init__.py` 导出核心模块，并更新 `dataFlow.md` 增补接入后 pipeline 的输入输出与维度。
- 2026-05-22 第 7 步校验：`src/stssdl_pipeline.py`、`src/__init__.py`、`dataFlow.md` 静态错误检查通过。
- 2026-05-22 第 8 步完成：新增 `model_explanation.md`，绘制当前计划网络图，并逐模块说明作用、输入输出和张量维度。
- 2026-05-22 第 9 步完成：用户撤销后，已按 `STSSDL_modules_explanation.md` 风格用中文重建 `model_explanation.md`，并补齐模块作用、输入输出和维度细节。
 - 2026-05-22 第 10 步完成：实现表示级（hidden-state）共享混杂提取器，并接入 pipeline。
	 - 文件：`src/confounder.py`（SharedConfounderExtractor，表示级实现）已添加。
	 - 文件：`src/confounder_steve.py`（序列级实现，参考 STEVE）已添加，作为备选实现。
	 - 修改：`src/stssdl_adapter.py`，将 confounder 接口调整为同时接受/返回 `x_target` 与 `x_his`，并将 `mask` 加入 forward kwargs。
	 - 修改：`src/stssdl_pipeline.py`，调整调用顺序：先由 backbone 编码得到隐藏态，再由 backbone 调用 `confounder_module` 在隐藏态上去混杂，之后进行原型查询与解码。
	 - 校验：已做静态导入与接口对齐检查，本地无语法错误。
