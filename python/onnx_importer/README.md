# ONNX Importer v1（Day 19）

从 AI-Compiler 根目录运行，使用已安装 `requirements.txt` 依赖的 `.venv`：

```bash
.venv/bin/python code/MiniMLIRCompiler/python/onnx_importer/importer.py \
  --input code/MiniMLIRCompiler/test/e2e/models/matmul.onnx \
  --output code/MiniMLIRCompiler/test/e2e/results/matmul.mini.mlir
code/MiniMLIRCompiler/build/bin/mini-opt \
  code/MiniMLIRCompiler/test/e2e/results/matmul.mini.mlir -o /dev/null
```

- `--input` 指向真实 ONNX ModelProto。首版从两个 Graph Input 的静态二维 FP32 MatMul 开始，输出 `func.func @main` 和 Generic Mini MLIR。
- `--output` 只在完整导入成功后发布。CLI 以非零值表示错误，诊断输出到 stderr，并提供节点位置和失败条件。
- `matmul_chain.onnx` 测试连续两个 MatMul 和中间结果推导。两个模型都可由 [generate_matmul_models.py](../e2e/generate_matmul_models.py) 重建。
- 目前 `matmul_relu.onnx` 在 Initializer 检查处退出，错误码为 `E_INITIALIZER_UNSUPPORTED`。Day 20 增加常量后，再扩展同一导入入口。

[generate_models.py](../e2e/generate_models.py) 可独立生成 Day 18 模型，加 `--with-reference` 才执行 ONNX Runtime。[导入约束与错误类型](../../docs/npu/onnx-import-contract.md)、[测试数据格式](../../docs/npu/validation-contract.md)
