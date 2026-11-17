# Day 18 · ONNX 导入契约

本页定义 Day 19–20 的 Python Importer。入口实际读取 `ModelProto`，输出可由 `mini-opt` 解析、验证的 Generic MLIR。[计划 CLI](../../python/onnx_importer/README.md)

## 输入域与支持范围

使用 ONNX 标准域（序列化域为 `""`）Opset 18、单个主 Graph、静态正尺寸 FP32 Tensor。形状必须已知且各维大于零。标量 Tensor 只允许在支持该 rank 的逐元素算子中使用。容器 `ir_version` 与 Opset 分别检查，容器版本遵循实际 ONNX Checker 的支持范围。Day 18 fixture 使用 IR 9。自定义域、控制流、模型本地函数、非受支持属性、Batch MatMul 和动态 Shape 超出首版契约。[ONNX IR 规范](https://onnx.ai/onnx/repo-docs/IR.html)

| ONNX 节点或定义 | 首版 Mini 输出与检查 | 首版 NPU 计算能力 |
| --- | --- | --- |
| MatMul | `mini.matmul`。两个输入和一个输出，rank 均为 2，`[M,K] × [K,N] → [M,N]`。 | Day 21 转换为 `npu.matmul`。 |
| Add / Mul | `mini.add` / `mini.mul`。输入与结果的完整 Tensor 类型相同。拒绝广播。 | 无对应转换。 |
| Relu | `mini.relu`。输入和结果的 Tensor 类型一致。数值测试先限定有限参考值。 | 无对应转换。 |
| 内嵌、不可覆盖的 FP32 Initializer | `arith.constant` 与 `DenseElementsAttr`，保留 Shape、元素顺序和数据。 | 计算 IR 阶段保留。 |

ONNX MatMul 有更广的秩与批量语义，ONNX Add 允许多向广播。Importer 应在源节点检查其具体使用方式。即使 ONNX Checker 接受 `[2,4] + [4]`，首版也要为该节点给出广播不支持的诊断。[MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html)、[Add](https://onnx.ai/onnx/operators/onnx__Add.html)

## 导入顺序与 SSA

1. 读取模型，关闭自动加载 External Data。检查标准 Opset、域和权重存储方式，再用 ONNX Checker 检查图结构。将 Checker 异常标记为模型阶段的诊断，并保留原错误和位置。
2. 运行 Shape Inference，合并 Graph Input、Initializer、ValueInfo 和 Graph Output 的类型信息。同名值的显式与推导类型须一致。对每个被使用的值检查 Tensor 类型、静态维度和 DType，并用算子自身规则检查结果关系。Shape Inference 提供信息，规则可能未推导出全部 Shape。[ONNX Shape Inference](https://onnx.ai/onnx/repo-docs/ShapeInference.html)
3. 按 `graph.input` 顺序创建函数参数，写入 Value Map。读取每个允许的 Initializer，一次性建立对应常量与映射。Value Map 的键是原 ONNX Tensor 名，值包含内部生成的 SSA 名和 Tensor 类型。
4. 按拓扑顺序处理 Node，查出所有输入的已有 SSA 定义，检查 arity、属性和类型，生成结果 SSA，并在成功创建后登记。遇到未定义输入或重复输出定义立即停止。内部 SSA 名如 `%arg0`、`%c0`、`%v0` 由 Importer 生成。原始名称用于查找与诊断。
5. 按 `graph.output` 顺序查找 SSA、核对其类型并生成 `func.return`。测试用现有 `mini-opt` 重新解析和验证。导入失败时只返回诊断，成功后才发布完整输出，避免留下看似可用的半份 IR。

以 Day 18 fixture 为例，`X → %arg0`，`W → %c0`，第一个 Node 写入 `H → %v0`，ReLU 读取同一个 `%v0`，再定义 `Y → %v1`。如果 `H` 有两个消费者，两处应引用同一个 SSA 值。多消费者共享和多输出顺序都是后续验收项。

## 权重与诊断

只在 Graph Input 中的值是运行参数。只在 Initializer 中的 FP32 内嵌 Tensor 是固定常量，可经 `onnx.numpy_helper.to_array` 读取。首版对单个 Initializer 采用 64 KiB 的原始数据上限。Graph Input 与 Initializer 同名时 ONNX 可表达可覆盖的默认值，首版拒绝该情况。External Data、非 FP32、超限、形状与元素数不符的常量均须诊断。静态 Shape 和 FP32 的检查独立于数据读取。

Dense 文本序列化须能往返保留有限 FP32 的位模式，包括负零和小数。不要依赖 NumPy 的默认字符串打印。可以使用固定宽度的浮点位编码，按形状生成嵌套 Dense 属性，再用当前 `mini-opt` 解析并比较还原的数据。特殊浮点数的算子输出另做语义测试。现有 MiniToLinalg ReLU 的有序比较对 NaN 和负零选择正零，当前 fixture 验证普通有限输入。[现有 ReLU 降级](../../lib/Conversion/MiniToLinalg.cpp#L23)

| 条件 | 预期诊断 |
| --- | --- |
| 不支持的算子、域、Opset、属性 | `E_UNSUPPORTED_OP`、`E_UNSUPPORTED_DOMAIN`、`E_UNSUPPORTED_OPSET`、`E_ATTRIBUTE_UNSUPPORTED` |
| FP16 等类型、Rank-3 MatMul、符号或缺失维度 | `E_UNSUPPORTED_DTYPE`、`E_UNSUPPORTED_RANK`、`E_DYNAMIC_SHAPE` |
| K 不匹配、输出类型冲突、非正尺寸、同形状约束失败 | `E_SHAPE_MISMATCH`。合法 ONNX 广播另用 `E_BROADCAST_UNSUPPORTED`。 |
| 可覆盖、External Data、超限或无法解码的权重 | `E_INITIALIZER_UNSUPPORTED` |
| 未定义输入或输出、重复定义、arity 不符 | `E_INVALID_GRAPH`。若 Checker 已先发现问题，保留其诊断。 |

诊断应包含模型路径、节点索引/名称、op_type、输入值及已知类型、失败约束。正例先验证两个运行时输入的 MatMul，再验证固定权重、共享 SSA 和有序返回。负例覆盖表中的每类边界。Day 19 初版遇到 Initializer 应明确拒绝，Day 20 扩展后用 Day 18 fixture 验证全部前端路径。
