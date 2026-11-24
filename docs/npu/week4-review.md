# Week 4 · MiniNPUCompiler 验收与复盘

Day 28 从干净的 Day 27 工作树 `3bbaa4520b189b6970b768f94b15e965a50bc8e9`
开始，基线 CTest **53/53**。在 Apple ARM64 上使用 LLVM/MLIR
`llvmorg-23.1.4`、Python 3.11、ONNX 1.23.2、NumPy 2.4.6 与
ONNX Runtime 1.32.0，完成 **62/62**。结果和代码保留在工作区。

## 从 MatMul 到可执行程序

| 日程 | 架构决策 | 实际验收或源码契约 |
| --- | --- | --- |
| Day 22 Tiling | 沿 M/N 划分结果区域，沿 K 划归约段。保留有效尾块尺寸和 SSA。 | Multi-Tile 分为 4 个输出块及 12 个计算 Tile；ONNX M 尾块实测 `32,32,1`。[Tiling](tiling-design.md) |
| Day 23 SRAM Planning | 用 A/B/C 的活跃区间确定安全复用，C 保留到 STORE，A/B 在对应计算后释放。 | 加权和多块实例的计划峰值分别为 192 B 和 12 KiB。[SRAM](sram-planning.md) |
| Day 24 DMA | 为完整 Tensor 绑定 DRAM，为局部 Tile 绑定 SRAM，以两种 Stride 表达字节搬运。 | 加权模型搬运 104 B，多块模型 112 KiB。[DMA](dma-contract.md) |
| Day 25 Scheduling | 先以真实读写区域建立依赖，再验证合法的全局序列和存储状态。 | Multi-Tile 有 44 个事件和 93 条直接依赖；Chain 验证 H 的完整写回。[调度](scheduling-contract.md) |
| Day 26 CodeGen | 将已验证的指令、权重字节及 ABI 写成固定三文件。 | 同模型、配置和输入独立编译两次，三文件逐字节相同。[ISA](isa-spec.md) · [ABI](program-abi.md) |
| Day 27 Simulator | 记录 SRAM 字节初始化和逻辑 Buffer 归属，按 K 顺序逐段 FP32 累加。 | 两份真实模型以及尾块、连续 MatMul、坏 Buffer 的运行时测试。[Simulator](simulator-contract.md) |
| Day 28 E2E | 从同一份 ONNX 与固定输入启动编译执行路径与独立 CPU 路径，保留可重放的日志、IR 和数值证据。 | 六个真实 ONNX 正例、分层错误注入、可复现性及完整 CTest。[架构](backend-architecture.md) · [复现](e2e-validation.md) |

## Day 28 实测

| 用例 | 输出块 / 计算 Tile / 指令 | DMA 字节 | SRAM 计划峰值（B） | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| [Small](../../build/day28-acceptance/small/small/validation_report.json) | 1 / 1 / 5 | 104 | 192 | `5.960464477539063e-08` |
| [Weighted](../../build/day28-acceptance/weighted/weighted/validation_report.json) | 1 / 1 / 5 | 104 | 192 | `0` |
| [Multi-Tile](../../build/day28-acceptance/multitile/multitile/validation_report.json) | 4 / 12 / 44 | 114688 | 12288 | `7.62939453125e-06` |
| [Two-MatMul](../../build/day28-acceptance/chain/chain/validation_report.json) | 2 / 2 / 10 | 256 | 256 | `4.76837158203125e-07` |
| [M 尾块](../../build/day28-acceptance/tails/tails/validation_report.json) | 3 / 3 / 15 | 4916 | 2304 | `1.430511474609375e-06` |
| [Y、H 多输出](../../build/day28-acceptance/multioutput/multioutput/validation_report.json) | 2 / 2 / 10 | 256 | 256 | 两项均为 `4.76837158203125e-07` |

六项输出均为 FP32，使用新生成的 `CPUExecutionProvider` 参考，
默认 `rtol=1e-4`、`atol=1e-5`，不匹配数均为零。计划与实际
执行的指令和 DMA 字节统计匹配。新的模型生成共享 Metadata 和
参考实现，另行重跑原有 `generate_models.py --all --with-reference`
也生成了全部六份 Day 20 参考。已有 NPU 多块编译结果可追踪到全部
六个阶段的 IR 与完整 [Trace](../../build/day28-acceptance/multitile/multitile/execution_trace.json)。

### 分层故障与可复现性

- 非法 K 由 Importer 报 `E_SHAPE_MISMATCH`，ReLU 由 MiniToNPU 报
  `failed to legalize operation 'mini.relu'`，8 KiB 的配置在 SRAM
  Planner 报当前计划需要 12,288 B。
- 越界 DMA 和单字节权重变更在 Loader 拒绝。删除 LOAD、重复 ZERO、
  提前 STORE、错误 Buffer ID 在 Runtime 拒绝。测试为篡改后的
  Program 重算摘要及序号，使语义错误能到达目标检查层。
- 单元素数值扰动使 Comparator 退出 1，统一报告为 `failed` 且
  `mismatched_elements=1`。坏输入与错误诊断保留在
  [验收输出目录](../../build/day28-acceptance)。
- 两次独立编译的 Bundle 及 Compiler Stats 逐字节相同。Reserved 设
  为 1 B 的加权模型，三个 64 B Buffer 的活跃峰值为 192 B，地址
  高水位为 256 B。提交的目标目录已经存在时，重新运行被拒绝，原报告保留。

## 实际 CMake 与回归命令

验收时将 ONNX Runtime 1.32.0 安装到 `/private/tmp/mini-day28-deps`，
由 CMake 配置读取，现有 `.venv` 未改。以下从 AI-Compiler 根目录运行。

```bash
day28_project="$PWD/code/MiniMLIRCompiler"
day28_python="$PWD/.venv/bin/python"
PYTHONPATH=/private/tmp/mini-day28-deps cmake -S "$day28_project" \
  -B "$day28_project/build" -DMINI_ONNX_PYTHON="$day28_python"
cmake --build "$day28_project/build" -j 4
ctest --test-dir "$day28_project/build" --output-on-failure -j 4
```

实际构建、回归和单独保留验收证据均退出 0。配置继续显示已有
LLVM CMake `CMP0219` 警告。带原命令、阶段 IR、Bundle、Trace、
输出 NPZ 及比较结果的完整证据位于 `build/day28-acceptance`。
按 [一键验证说明](e2e-validation.md) 可在新路径复现。

## 能力边界

第四周打通静态二维 FP32 MatMul 主线。当前量化、广播、动态 Shape、
更多 NPU 算子、异步 DMA、多单元以及真实设备执行需要新的设计与验收。
计划内存、执行条数和搬运字节用于检查数据路径。实机延迟、吞吐率和
Coral 路径属于独立的验证目标。
