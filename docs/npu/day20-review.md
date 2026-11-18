# Day 20 · NPU Compute IR、ONNX 权重与 CPU 参考

开发前 Git 工作区干净，HEAD 为 `5dfbd039e9f88e47932cccf01ef9f3c44c94e4d6`。LLVM/MLIR 使用 `llvmorg-23.1.4`，原有 17 项 CTest 均通过。

## 接口与实现

- [NPU ODS](../../include/NPU/NPUOps.td) 约束两个静态 FP32 Tensor 输入与一个同类结果，TableGen 生成 `mlir::npu::MatMulOp`。自定义 [Verifier](../../lib/Dialect/NPU/NPUOps.cpp) 检查 rank 2、正尺寸、K 一致以及结果为 `[M,N]`。NPU Compute Op 表示整个逻辑 Tensor 计算，不受原生 32 块大小限制。
- [NPU CMake](../../lib/Dialect/NPU/CMakeLists.txt) 在生成 `MLIRNPUOpsIncGen` 后构建 `MLIRNPUDialect`。[mini-opt](../../tools/mini-opt/mini-opt.cpp) 链接该库并把 NPU Dialect 登记到 Registry。
- [Importer](../../python/onnx_importer/importer.py) 对内嵌有限 FP32 权重检查存储格式、Shape、元素数量和 64 KiB 上限。用十六进制位模式输出 `arith.constant`。运行参数、常量和节点结果统一登记为 SSA 引用；多消费者只查找已有定义。Add/Mul 保持同形状，ReLU 保持输入类型。
- [参考生成器](../../python/e2e/generate_models.py) 和 [比较器](../../python/e2e/compare_outputs.py) 属于验证工具。每组 fixture 有 ONNX、输入 NPZ、参考 NPZ 和元数据。比较器检查输出结构并应用绝对与相对容差。

## 实际命令

在 AI-Compiler 根目录，使用现有构建目录和 Day 18 的隔离 ORT（1.32.0）：

```bash
mini_project="$PWD/code/MiniMLIRCompiler"
cmake -S "$mini_project" -B "$mini_project/build" \
  -DMINI_ONNX_PYTHON="$PWD/.venv/bin/python"
cmake --build "$mini_project/build" --target mini-opt -j 4
PYTHONPATH=/tmp/mininpu-day18-ort.q8GNkf .venv/bin/python \
  "$mini_project/python/e2e/generate_models.py" --all --with-reference
ctest --test-dir "$mini_project/build" --output-on-failure -j 4
```

配置、构建、参考生成与最终回归均退出 0。构建生成 NPU TableGen 文件、`libMLIRNPUDialect.a` 并重新链接 `mini-opt`。配置保留已有 `CMP0219` 警告。ORT 提示无法持久化 telemetry ID，使用内存 ID 后正常完成推理。

## 结果

| 验证 | 实际结果 |
| --- | --- |
| [NPU 正例](../../test/npu/valid.mlir) | 小矩阵、完整 `64×96 × 96×64` 和非 32 倍数均通过。 |
| [NPU 负例](../../test/npu/diagnostics.mlir) | 错误 K、Rank、DType、结果 Shape、零维和动态维单独运行均退出 1，组合诊断测试通过。 |
| [ONNX 导入](../../test/e2e/check_matmul_import.py) | 双 MatMul、固定权重 MatMul、MatMul→ReLU、MatMul→Add→ReLU、MatMul→Mul→ReLU 和多 Tile 输入经 `mini-opt` 验证。共享 W 只定义一次，两个 MatMul 使用同一个值，按指定的图输出顺序返回。 |
| 权重保真 | 对相同的 12 个 FP32 元素分别用 ONNX `raw_data` 与 `float_data` 导入。MLIR 解析后逐位一致，覆盖负零、最小正次正规数及非整数值。 |
| ONNX 拒绝 | 广播 Add、External Data、可覆盖权重、稀疏权重、超限、非有限及不支持的类型等退出 1，未发布输出。首次回归中截断权重的预期诊断写错，实际由 ONNX Checker 拒绝（3 字节，预期 48 字节）；已据实修正测试。 |
| [CPU 参考与比较](../../test/e2e/check_reference.py) | 六组模型从磁盘加载并用保存的输入重跑，均与保存的参考逐元素相同。比较器接受完全一致和容差内结果，拒绝单元素 +0.1、错误 Shape、DType、名称、NaN/Inf 和无效容差。 |
| CTest | 22/22 通过，包含原有 17 项。Day 19 的带权重拒绝用例改为真实成功用例，同时加入明确的权重拒绝边界。 |

六组数据位于 [test/e2e](../../test/e2e)：`matmul`、`weighted_matmul`、`matmul_relu`、`matmul_add_relu`、`matmul_multitile`、`matmul_chain`。权重版 MatMul 的 Y 为 `[[0,1.5,3,-3.5],[5,-3,-5,9]]`，Add→ReLU 版本为 `[[0,1.75,3.5,0],[6,0,0,10.75]]`。多 Tile 参考输出为 FP32 `[64,64]`，实测范围约 `[-37.3109,42.5762]`。

## Day 21 交接

`weighted_matmul.onnx` 用于转换正例。Day 21 以 `mlir::mini::MatMulOp` 匹配源操作，复用操作数、结果类型和现有 `arith.constant`，通过生成的 `mlir::npu::MatMulOp::create` 构造目标操作并替换源 SSA 结果。合法化目标还须决定如何处理模块、函数、常量以及没有对应转换的存活 Mini 算子。带 ReLU 的图可用于验证后者。当前没有运行 MiniToNPU、Tiling 或 Simulator；参考值来自独立的 ONNX Runtime CPU。
