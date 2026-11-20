# Day 18 · 目标 Lowering 契约

两条目标路线消费同一套 Mini Graph IR。图层算子是否可表达、Pipeline 是否有规则、目标程序是否正确执行，分别由相应阶段验收。

## 已实现的 MiniToLinalg

[MiniToLinalg.cpp](../../lib/Conversion/MiniToLinalg.cpp#L157) 使用 `ConversionTarget` 与 `applyFullConversion`，标记 Mini Dialect 非法，为 Add、Mul、ReLU 和 AddReLU 注册 Pattern，输出标准 Tensor、Linalg 和 Arith 等 IR。Day 19 新增 `mini.matmul` 后，仍需另写对应的 Linalg Pattern 才能用原 Pipeline 转换它。未来的 Coral 桥接应以 `tensor.empty → linalg.fill(0) → linalg.matmul` 初始化累加目标。empty 提供类型和形状，fill 提供 MatMul 读取的初始值。

## Day 21 已实现的 MiniToNPU

- **接口。** [MiniToNPU.cpp](../../lib/Conversion/MiniToNPU.cpp) 实现 Module 级 Pass `mini-to-npu`。首版只注册 `OpConversionPattern<mini::MatMulOp>`，目标计算 Op 为 `npu.matmul`。
- **类型与 SSA。** 用 Pattern adaptor 提供的操作数、原结果类型和位置创建目标 Op，通过 rewriter 替换原结果的全部使用。保留计算语义、Shape、常量引用及函数签名。计算层使用同一 Tensor 类型，首版无需 TypeConverter。
- **合法性。** 明确允许 `builtin.module`、`func.func`、`func.return`、`npu.matmul` 和静态 FP32 Dense Tensor `arith.constant`。只给符合条件的常量实例赋予动态合法性，仍遍历模块和函数体。Mini Dialect 标为非法，其余未列出的操作无法取得合法性。首版没有 CPU fallback。`func.call` 和其他 Arith 运算不在允许集合。
- **失败。** `applyFullConversion` 失败时 `signalPassFailure()`，驱动返回非零，诊断定位未转换的源 Op。转换成功后用 Verifier 及重新解析确认 Type/Shape/SSA。[MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)

| 测试 | 预期 |
| --- | --- |
| 单个或两个相连的静态 FP32 MatMul | 全量转换并重解析通过，SSA 边保持正确。 |
| 带不可覆盖权重的 MatMul | 常量保留且使用者正确，函数参数只有运行时输入。 |
| MatMul → ReLU、Add → ReLU、无规则的 Mini Op | 目标合法化失败。 |
| `arith.addf` 等未列入目标的运算 | 即使其自身 MLIR 语法与类型正确，也应失败。 |
| K 不匹配、非法 rank / dtype / shape | Mini 或 NPU Verifier 拒绝，检查所在层的诊断。 |

`npu.matmul` 仍表示完整计算。Tiling 检查原生形状和边界策略，后续内存 Pass 决定常量布局、SRAM 地址及复用，DMA 和调度定义执行顺序。[IR 分层](ir-levels.md)
