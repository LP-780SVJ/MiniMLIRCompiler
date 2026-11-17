# Day 18 · 参考数据与数值验证

ONNX Runtime CPU 使用与未来 Virtual NPU 相同的 ONNX 模型和固定输入，生成独立参考。Simulator 的实际输出由它独立执行产生，比较器消费两边的产物以发现布局、权重加载、K 累加和写回错误。[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)

## Fixture 文件与身份

一个逻辑测试包包含模型、固定输入、参考输出和元数据。当前在 `test/e2e/` 按类别存放，以 `matmul_relu` 为共同文件名。

| 文件 | 内容 |
| --- | --- |
| [models/matmul_relu.onnx](../../test/e2e/models/matmul_relu.onnx) | ONNX IR 9、标准域 Opset 18。运行参数 `X[2,3]`，内嵌权重 `W[3,4]`，`H = MatMul(X,W)`，`Y = Relu(H)`。 |
| [inputs/matmul_relu.npz](../../test/e2e/inputs/matmul_relu.npz) | 以 Graph Input 名为键，`X` 为固定字面值。 |
| [reference/matmul_relu.npz](../../test/e2e/reference/matmul_relu.npz) | 以 Graph Output 名为键的 ONNX Runtime CPU 输出 `Y`。 |
| [models/matmul_relu.metadata.json](../../test/e2e/models/matmul_relu.metadata.json) | schema、模型与输入 SHA256、IR 与 Opset、名称顺序、Shape、dtype、版本、参考 provider、目标及容差。输入为字面值，因此 seed 为 null。 |

`test/e2e/results/` 预留后续的 Mini/NPU IR、`actual_outputs.npz` 与比较报告。NPZ 用 `allow_pickle=False` 加载，依据 Graph 和元数据的名称与顺序对齐输出。

| 验收层 | 证据 |
| --- | --- |
| ONNX 模型 | Checker 通过，Shape Inference 有所需类型，输入、权重和节点依赖符合预期。 |
| Importer | 真实模型生成 Mini IR，`mini-opt` 可重新解析，签名、常量数据、Shape、输出顺序和 SSA 正确。 |
| NPU Lowering | 加权 MatMul 等支持的模型产出合法 NPU Compute IR，未支持的存活 Op 明确失败。 |
| 数值执行 | Simulator 从权重和固定输入生成实际输出，再与独立参考比较。 |
| 性能 | 单独记录 Tile 数、DMA 字节量、峰值 SRAM 等指标。 |

Day 20 的比较器检查名称集合、Shape、dtype 和 NaN/Inf，再对有限 FP32 使用 `abs(actual-reference) <= atol + rtol*abs(reference)`。候选值为 `rtol=1e-4`、`atol=1e-5`。报告最大绝对误差、采用有记录分母规则的最大相对误差、不匹配元素数量和通过状态。至少验证一致、单元素错误注入、Shape/DType/名称不匹配、NaN/Inf 拒绝。常量位模式保真用独立的精确测试。

## Day 18 实际执行

```bash
# 从 AI-Compiler 根目录执行。临时 ORT 路径为本次实际使用的目录。
PYTHONPATH=/tmp/mininpu-day18-ort.q8GNkf \
  .venv/bin/python code/MiniMLIRCompiler/python/e2e/generate_models.py --with-reference
cmake --build code/MiniMLIRCompiler/build --target mini-opt
ctest --test-dir code/MiniMLIRCompiler/build --output-on-failure
```

项目 `.venv` 已有 ONNX 1.23.2 和 NumPy 2.4.6，缺少 ONNX Runtime。本轮将 ONNX Runtime 1.32.0 的 macOS arm64 wheel 装到临时目录，生成时只为该命令设置 `PYTHONPATH`。复现时可同样用 `pip install --target <隔离目录> --no-deps onnxruntime==1.32.0`，基于已有 ONNX 和 NumPy 环境运行。[ONNX Runtime 1.32.0 wheel](https://pypi.org/project/onnxruntime/1.32.0/)

生成器退出 0，Checker 通过，推导 `H[2,4]`，图输出 `Y[2,4]`。CPUExecutionProvider 实测 `Y = [[0, 1.5, 3, 0], [5, 0, 0, 9]]`，dtype float32。Ninja 退出 0，报告无需重建。CTest 退出 0，12/12 通过。该 fixture 用于验证前端与参考数据，Day 21 的 NPU 正例使用单独的加权 MatMul。

从磁盘重新读取模型和 NPZ，核对两个 SHA256、节点连接、权重元素和类型，使用保存的输入第二次运行 CPU Session。结果与参考文件及手算矩阵逐元素一致。另在未加载临时 ORT 的环境重放生成器，确认只生成模型与输入时退出 0，显式要求生成参考时给出所缺依赖并退出 1。

用现有 `mini-opt` 单独解析十六进制浮点位模式的 Tensor `arith.constant`，退出 0。样例包含正数、负数、负零、最小正 FP32 次正规数和非整数。这确认了后续权重序列化可用的文本入口，真正的 ONNX Initializer 导入及位模式往返仍由 Day 20 验收。
