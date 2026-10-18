# 三任务数据的导出适配

本分支从固定上游 `e84abb09bb4d79110cebc376711e78a2a0d7567a` 保存此前已用于三任务 checkpoint-1750 部署流程的 `export_all.py` 适配。改动来源为 rust-shyper-isure 的 `scripts/rdk-s100/rdt/prepare-robotwin-exporter.py`，生成结果逐字节核对一致。

- 增加 `--seed`，固定 NumPy 和 Torch 随机源，便于复现抽样；这不等于保证所有 GPU 算子都确定性执行。
- 从 episode 的指令集合独立选择语言 embedding，避免把轨迹时间步误当指令编号。
- 使用同帧 `observations/qpos` 作为关节输入，替代原来的 `action` 标签。

输入沿用已使用的数据合同：每个 episode 的指令文件从0连续编号，文本与 embedding 一一对应。稀疏指令编号不属于这份历史适配的已验证范围。没有重新设计模型、量化方案或编译器，也没有修改运行时并发逻辑。

**这是校准数据语义与复现性的适配，会改变相对于原导出脚本的校准输入。它不属于“固定 HBM 下保持输出逐位不变”的运行时优化。** 两条研究分支独立保存，不能把它混入 CPU/BPU 重排的加速或等价性结论。

在仓库根目录执行轻量输入选择检查：

```bash
python3 -B tests/test_export_inputs.py
```

该检查从当前导出源码提取实际抽样函数和设种子语句，用 HDF5/Torch 替身验证轨迹帧超过指令数量时仍可抽样、文本与 embedding 对齐、关节来自 qpos、同 seed 重复一致；没有执行真实图片解码、ONNX 导出、校准或 HBM 编译。本次提交不产生新模型，历史产物身份继续由原部署项目的复现清单维护。
