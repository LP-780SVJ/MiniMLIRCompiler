# ONNX Importer（Day 20）

从 AI-Compiler 根目录运行，使用已安装 `requirements.txt` 依赖的 `.venv`：

```bash
.venv/bin/python code/MiniMLIRCompiler/python/onnx_importer/importer.py \
  --input code/MiniMLIRCompiler/test/e2e/models/matmul_add_relu.onnx \
  --output /tmp/matmul_add_relu.mini.mlir
code/MiniMLIRCompiler/build/bin/mini-opt \
  /tmp/matmul_add_relu.mini.mlir
```

- `--input` 指向 ONNX ModelProto。支持标准域 Opset 18、静态 FP32：二维 MatMul、同形状 Add/Mul、ReLU。输出 `func.func @main` 和 Mini IR。
- `--output` 只在完整导入成功后发布。CLI 以非零值表示错误，诊断输出到 stderr，并提供节点位置和失败条件。
- `matmul_chain.onnx` 测试连续两个 MatMul 和中间结果推导。两个模型都可由 [generate_matmul_models.py](../e2e/generate_matmul_models.py) 重建。
- `matmul_relu.onnx` 和 `matmul_add_relu.onnx` 的内嵌权重进入 `arith.constant`。单个常量上限 64 KiB，仅接受有限 FP32 数据。拒绝稀疏权重、External Data、可覆盖权重和广播。

[generate_models.py](../e2e/generate_models.py) 加 `--all` 生成六组数据，加 `--with-reference` 才执行 ONNX Runtime。[导入约束与错误类型](../../docs/npu/onnx-import-contract.md)、[测试数据格式](../../docs/npu/validation-contract.md)
