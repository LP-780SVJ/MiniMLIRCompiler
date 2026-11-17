# Day 18 · Pass Pipeline 与构建接入

## 已实现的标准 MLIR 路线

[mini-opt.cpp](../../tools/mini-opt/mini-opt.cpp#L17) 使用 `PassPipelineRegistration` 注册 `mini-compile`。现有次序为 `func canonicalize → func AddReluFusion → module MiniToLinalg → func CSE → func canonicalize`。Module Pass 可以看到整份计算，函数级 Pass 由 `addNestedPass<func::FuncOp>` 运行。

```bash
code/MiniMLIRCompiler/build/bin/mini-opt input.mini.mlir \
  --pass-pipeline='builtin.module(mini-compile)'
```

## Day 21 规划的 Virtual NPU 路线

新增独立 `mini-npu-compile`，采用 `func canonicalize → module MiniToNPU → NPU IR Verification`。解析时验证输入，保留 PassManager 验证。转换输入中的存活计算须全部进入目标允许集合。未支持计算的 Pipeline 负例应将其结果连接到返回值，避免 canonicalize 消除死代码改变测试含义。直接测试 `mini-to-npu` 则检查所有输入 Op。新 Pipeline 先不启用 AddReluFusion，因为当前 NPU 仅映射 MatMul。[Lowering 契约](lowering-contract.md)

```bash
# Day 21 规划命令
code/MiniMLIRCompiler/build/bin/mini-opt weighted_matmul.mini.mlir \
  --pass-pipeline='builtin.module(mini-npu-compile)' -o weighted_matmul.npu.mlir
code/MiniMLIRCompiler/build/bin/mini-opt weighted_matmul.npu.mlir -o /dev/null
```

| 日期 | 工程入口 | 验收 |
| --- | --- | --- |
| Day 19 | 在 [MiniOps.td](../../include/Mini/MiniOps.td) 新增 MatMul ODS，在 [MiniOps.cpp](../../lib/Dialect/Mini/MiniOps.cpp) 新增 Verifier，新增 Importer 与测试。 | 复用 [Mini TableGen](../../include/Mini/CMakeLists.txt)，真实 ONNX 两输入 MatMul 经导入后能被 driver 解析。 |
| Day 20 | 新增 `include/NPU/`、`lib/Dialect/NPU/`，接入 [include CMake](../../include/CMakeLists.txt)、[lib CMake](../../lib/CMakeLists.txt) 和 [driver 目标](../../tools/mini-opt/CMakeLists.txt)，规划库名 `MLIRNPUDialect`。 | driver 注册后能解析并验证 `npu.matmul`，Importer 验证共享常量和多节点 SSA。 |
| Day 21 | 新增 `include/Mini/MiniNPULowering.h`、`lib/Conversion/MiniToNPU.cpp`，在 [Conversion CMake](../../lib/Conversion/CMakeLists.txt) 加独立 `MLIRMiniToNPU`，driver 链接并注册 Pass。 | 复用 [CTest](../../test/CMakeLists.txt) 的真实 `mini-opt` 目标执行正反例、回归及 ONNX→Mini→NPU 集成。 |
