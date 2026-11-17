# ONNX Importer v1（Day 18 接口）

Day 19 实现 `importer.py`。以下为从 AI-Compiler 根目录运行的计划接口：

```bash
.venv/bin/python code/MiniMLIRCompiler/python/onnx_importer/importer.py \
  --input code/MiniMLIRCompiler/test/e2e/models/matmul.onnx \
  --output code/MiniMLIRCompiler/test/e2e/results/matmul.mini.mlir
code/MiniMLIRCompiler/build/bin/mini-opt \
  code/MiniMLIRCompiler/test/e2e/results/matmul.mini.mlir -o /dev/null
```

- `--input` 指向真实 ONNX ModelProto。首版从两个 Graph Input 的静态二维 FP32 MatMul 开始，输出 `func.func @main` 和 Generic Mini MLIR。
- `--output` 只在完整导入成功后发布。CLI 以非零值表示错误，诊断输出到 stderr，并提供节点位置和失败条件。
- Day 20 增加 FP32 Initializer 和多节点支持后，用同一接口导入 `models/matmul_relu.onnx`，目标文件名为 `results/matmul_relu.mini.mlir`。

[generate_models.py](../e2e/generate_models.py) 可独立生成 Day 18 模型，加 `--with-reference` 才执行 ONNX Runtime。[导入约束与错误类型](../../docs/npu/onnx-import-contract.md)、[测试数据格式](../../docs/npu/validation-contract.md)
