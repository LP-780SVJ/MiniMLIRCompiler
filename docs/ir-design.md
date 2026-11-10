
# MiniMLIRCompiler — IR Design

## 1. 项目目标

实现一个基于 MLIR 的小型 Tensor 编译器。

## 2. 支持范围

- mini.add
- mini.mul
- mini.relu
- FP32
- 静态 Shape

## 3. Dialect 设计

为什么需要 Mini Dialect？

Mini Dialect 与标准 linalg/arith Dialect
之间的语义边界是什么？

## 4. Operation 设计

### mini.add
- Operands:
- Results:
- Attributes:
- Traits:
- Verifier:

### mini.mul
- Operands:
- Results:
- Attributes:
- Traits:
- Verifier:

### mini.relu
- Operands:
- Results:
- Attributes:
- Traits:
- Verifier:

## 5. Shape Inference

哪些 Shape 能在 Operation 构建时确定？

哪些约束需要 Verifier 检查？

以后如何支持动态 Shape？

## 6. Optimization

### Canonicalization

### Add + Relu Fusion

匹配条件：

Rewrite 过程：

多使用者处理：

## 7. Lowering

Mini Dialect -> Standard MLIR Dialects

每个 Operation 的 Lowering 规则：

## 8. Pass Pipeline

依次描述每个 Pass 的职责和依赖。

## 9. 正确性测试

- 正常输入
- 错误 Shape
- 错误 Type
- 正例融合
- 反例不融合
- Lowering 后语义一致性

## 10. 未来扩展

- MatMul
- ONNX Frontend
- Dynamic Shape
- NPU Backend
