# Day 19 · ONNX MatMul 前端验收

开发基线为 `dbefa2130af140e6b95a7e226de5b4685095edd9`，开始时 Git 工作区干净，原有 CTest 12/12 通过。构建复用本地 LLVM/MLIR `llvmorg-23.1.4`。ONNX 测试使用仓库 `.venv` 的 Python 3.11.15 和 ONNX 1.23.2。

## 代码与分层

| 入口 | 责任 |
| --- | --- |
| [MiniOps.td](../../include/Mini/MiniOps.td)、[MiniOps.cpp](../../lib/Dialect/Mini/MiniOps.cpp) | 定义 `mini.matmul` 的两输入、一结果、静态 FP32 和验证规则。TableGen 生成 `MatMulOp`、访问器、Builder 和类型约束。已有 Mini dialect 注册链自动包含新增操作。 |
| [importer.py](../../python/onnx_importer/importer.py) | 装载二进制 `ModelProto`，检查受支持范围，运行 Checker 和严格 Shape Inference，按 `graph.node` 的拓扑顺序将名称转换为 SSA 引用，最后按 `graph.output` 顺序发射返回值。 |
| [generate_matmul_models.py](../../python/e2e/generate_matmul_models.py) | 生成并检查单 MatMul 和连续双 MatMul 的真实 ONNX 文件。连续模型的 `H` 没有预填 `value_info`，用来验证 Shape Inference。 |
| [test/CMakeLists.txt](../../test/CMakeLists.txt)、[check_matmul_import.py](../../test/e2e/check_matmul_import.py) | 使用 CTest 验证 IR 正反例、ONNX 导入结果和错误诊断。通过 `MINI_ONNX_PYTHON` 显式选择 Python 环境。 |

今天完成的是模型导入和 Graph IR 合法性。`M, N` 描述输出范围，`K` 是每个输出元素的归约范围。`[33,37] × [37,35] → [33,35]` 在 Graph IR 合法，后端将来需为逻辑维度超过原生块的计算安排分块和余块。`mini.matmul` 的结果是逻辑 Tensor SSA 值，存储和调度由后续层决定。

`type_map` 中可以提前知道 `H` 的类型，但第二个节点只能从 `value_map` 读取由第一个节点定义的 `H`。同理，`Y` 的类型正确并不能保证函数正确返回了计算结果，因此集成测试同时核对被返回的 SSA。

## 复现命令

从 AI-Compiler 根目录运行：

```bash
mini_project="$PWD/code/MiniMLIRCompiler"
cmake -S "$mini_project" -B "$mini_project/build" \
  -DMINI_ONNX_PYTHON="$PWD/.venv/bin/python"
cmake --build "$mini_project/build" --target mini-opt -j 4
.venv/bin/python "$mini_project/python/e2e/generate_matmul_models.py"
ctest --test-dir "$mini_project/build" --output-on-failure -j 4

mini_results="$(mktemp -d /tmp/mini-day19.XXXXXX)"
for model in matmul matmul_chain; do
  .venv/bin/python "$mini_project/python/onnx_importer/importer.py" \
    --input "$mini_project/test/e2e/models/$model.onnx" \
    --output "$mini_results/$model.mini.mlir"
  "$mini_project/build/bin/mini-opt" "$mini_results/$model.mini.mlir"
done
```

配置、增量构建和模型生成退出码均为 0。配置时 LLVM 的 `HandleLLVMOptions.cmake` 仍给出 `CMP0219` policy 警告，生成和构建成功完成。

## 实测结果

| 验证项 | 实际结果 |
| --- | --- |
| [合法 Mini IR](../../test/matmul/valid.mlir) | `[2,3] × [3,4] → [2,4]` 与 `[33,37] × [37,35] → [33,35]` 均通过。 |
| [非法 Mini IR](../../test/matmul/diagnostics.mlir) | 单独解析每个负例均退出 1。K 不匹配、错误结果 Shape、rank 3、FP16、动态维、缺少操作数和零维都有预期诊断。组合 `--verify-diagnostics` 测试通过。 |
| `matmul.onnx` | 导入、`mini-opt` 均退出 0。函数两参数，唯一 `mini.matmul` 的结果为 `tensor<2x4xf32>`，返回该结果。 |
| `matmul_chain.onnx` | 导入、`mini-opt` 均退出 0。先得到 `H: tensor<2x4xf32>`，再用它与第三个参数得到 `Y: tensor<2x5xf32>`。另测试了按 `Y, H` 顺序返回两个结果。 |
| ONNX 拒绝用例 | 15 个模型均退出 1，且没有创建输出。FP16、BF16、Batch、动态维、零维、未知算子等被项目契约拒绝；未定义输出和重复定义由 ONNX Checker 拒绝；K、输出或 ValueInfo 冲突由严格 Shape Inference 拒绝。Day 18 权重模型报 `E_INITIALIZER_UNSUPPORTED`。失败时已有输出文件保持原内容。 |
| CTest 回归 | 17/17 通过，含原有 12 项和新增 5 项。 |
| MatMul 使用 `mini-compile` | 退出 1，报告 `failed to legalize operation 'mini.matmul'`。MatMul 可解析与验证，但现有 MiniToLinalg 没有此操作的转换规则。 |

## 下一阶段

Day 20 在已有 `type_map` 与 `value_map` 上接入常量。`W` 的 SSA 定义将由 Initializer 生成，后续 MatMul 查找 Operand 的逻辑相同。NPU Dialect 单独建模，Day 21 的 MiniToNPU 消费相同的 Mini IR。当前 CLI 只支持无权重、静态二维 FP32 的标准域 Opset 18 MatMul，不支持广播、批量矩阵乘法、自定义域或控制流。今天验证的是图结构与 IR 合法性；数值正确性需有可执行的后端与独立参考输出。
