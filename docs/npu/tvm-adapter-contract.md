# TVM → MLIR NPU 子图 Adapter 契约（Day 15）

> 这是未来接口的设计稿；仓库中没有 TVM Adapter、`mini.matmul` 或 NPU Runtime。本稿以 [TVM BYOC](https://tvm.apache.org/docs/arch/external_library_dispatch.html) 的外部子图路径为候选，并以仓库当前 [MiniOps.td](../../include/Mini/MiniOps.td) 的实际能力限定首版范围。

## 1. 放在哪里，交付什么

候选方案 B 的 Adapter **输入**是 TVM 已完成图优化/分区后的一段、边界确定的静态 Relax 计算子图及其函数签名，而不是整个 IRModule。**输出**是可被 MLIR 解析和验证的 `module { func.func ... }`，内部使用当前 Mini Dialect 支持的计算 Operation。边界上的每个输入、输出都必须有稳定的顺序、Tensor 类型与来源位置；Host 对 NPU 的调用和二进制 ABI 另行约定。

```text
TVM Relax 子图及签名
  → 支持能力与语义检查
  → 源 Value 到 MLIR Value 的映射
  → func.func + mini.*
  → MLIR 解析/Verifier
  → 现有 mini-compile（仅当前已支持的算子）
  → 将来的 NPU Lowering / Runtime 接口
```

这个位置保留已有 TVM Frontend 和图 Pass，允许把其余节点交由 TVM 原有路径处理。分区必须使用与 Adapter 相同的能力表：不满足合同的子图应留在 Host / 其他后端，或在编译期带定位信息失败；不能输出部分 NPU 程序后假装整图已成功编译。

## 2. 输入、输出和验证责任

| 信息 | 输入要求 | 目标映射与检查 |
| --- | --- | --- |
| 算子与属性 | 每个调用的语义、全部影响结果的属性、源位置 | 逐项支持检查；未知的语义属性拒绝，不能静默丢弃 |
| Shape / Rank | 每个边界值和中间值的推导结果 | 首版仅静态整数维；输入、输出、算子规则须相互一致 |
| DType | Tensor 元素类型和算子累积类型 | 首版 `f32`；不同精度或累积规则拒绝 |
| Layout | 逻辑维顺序及任何会影响语义的布局/转置信息 | 首版只接受无需重排的约定顺序；不可表达的变换拒绝 |
| 常量 | 内容、Shape、DType、是否为外部参数 | 首版只接收作为子图参数传入的权重/Bias；嵌入常量暂拒绝 |
| Value / 使用者 | 源函数参数、每个结果、所有 Use | 建立一对一 Value Map，保留 DAG 共用与多结果关系 |
| 函数签名 | 参数/结果个数、顺序、类型、调用约定 | 生成 `func.func`，检查返回类型和顺序；真实 Host/NPU ABI 待定 |
| 副作用 | 算子是否读写状态或依赖执行顺序 | 首版仅接受可按纯计算数据流表示的子图 |

MLIR Type 中的 `tensor<2x4xf32>` 描述逻辑形状和元素类型，**不自动定义** NPU 的物理排布、Stride、对齐、DMA 规则。未来若源操作要求特定 Layout，应在新 IR/属性中显式携带并验证，或在边界插入已证明等价的 Layout 转换。直接删掉布局属性会产生看似类型合法但数值错位的程序。

验证分三层：Adapter 在创建 Operation **前**检查源语义可表达性；MLIR Verifier 检查生成 IR 的结构和 Mini ODS/C++ 不变量；后续编译 Pass 检查当前 Lowering 能否完整转换。Verifier 只能验证它知道的合同，无法发现 Adapter 已经遗漏的广播、转置或调用约定。

## 3. 首版能力表与算子映射

| 来源计算 | 当前 Mini 目标 | 首版判定 | 理由 |
| --- | --- | --- | --- |
| 同形状 `f32` Add | `mini.add` | 满足特殊值和精度约定时可接受 | 两侧与结果完整 Type 一致 |
| 静态 `f32` ReLU | `mini.relu` | 须先核对 NaN / `-0.0` 语义 | Mini 当前规则是有序 `x > +0.0 ? x : +0.0` |
| `[2,4] + [4]` Bias Add | 无直接映射 | 拒绝 `E_BROADCAST_UNSUPPORTED` | Mini Add 不做 Broadcasting |
| 逐元素 Mul | `mini.mul` 已存在 | 可作为后续扩展，首版不承诺导入 | 源语义和特殊值需另行核对；它不是 MatMul |
| 2D MatMul | `mini.matmul` 计划中 | 当前拒绝 `E_TARGET_OP_UNAVAILABLE` | Day 19 才计划引入该 Operation |
| `mini.add_relu` | 当前已有内部融合 Op | 不作为源算子的默认直接映射 | 保留图 Pass 决定融合，需证明源子图与目标语义相同 |

未来 MatMul 的候选语义是 `A[M,K] × B[K,N] → C[M,N]`，首版仅二维、静态、`f32`、无 Batch、无隐式广播、无未经规范化的转置属性。`[2,3] × [3,4] → [2,4]` 在**形状合同**上合格，但当前仓库仍不能生成 `mini.matmul`；`[2,3] × [5,4]` 是 `E_SHAPE_MISMATCH`；三维 Batch、BF16、动态维分别是 Rank/DType/Shape 能力失败。来源中的 `Transpose(B)` 与某个 MatMul 的 `transpose_b` 属性都必须得到显式等价处理，不能只交换名称。

Bias 的 `[4]` 可在未来有明确广播 Operation 后显式扩成 `[2,4]`，或者由新的 `mini.bias_add` 表达；当前合同直接拒绝。即便将 Bias 作为函数参数传入，也不能解决广播问题。若该参数是不可变权重，Runtime 还须保证数据、生命周期和所有权；本稿未声称该 ABI 已实现。

ReLU 的符号行为要以**具体来源版本的语义与数值实验**核对，尤其 NaN、`-0.0`、无穷大和融合后舍入。Mini 的确切规则见 [MiniToLinalg.cpp](../../lib/Conversion/MiniToLinalg.cpp)。若来源不能保证相同规则，应在分区时拒绝或设计明确的语义修正，不能把 `max(x,0)` 的文字描述当成充分证明。

## 4. 保持 SSA 与失败原子性

以下只是算法草图，不依赖某个 TVM 版本的 Python/C++ API：

```text
importSubgraph(source):
  validateSignatureAndBoundary(source)
  validateAllOpsTypesShapesAttributesLayoutAndEffects(source)
  target = createModuleAndFunction(source.signature)
  valueMap = mapArguments(source.args, target.args)
  for op in dependencyOrder(source):
    operands = lookupAll(op.inputs, valueMap)
    results = createEquivalentMiniOp(op, operands)
    recordAllResults(op, results, valueMap)
  createReturn(lookupAll(source.outputs, valueMap))
  verify(target)
  return target
```

例如 `%sum = Add(%a,%b)` 同时被 ReLU 和另一个操作使用，两个 Use 必须读到**同一个**目标 `%sum`；若重建两次 Add，既可能重复计算，也可能在有副作用/特殊值约束时改语义。Value Map 的键应是源 IR 的值身份，不能依赖打印出的变量名。对无法映射的值立即诊断，不生成“半成功”的子图。先做完整预检、失败时丢弃临时目标模块；成功后才把产物交给下游。

## 5. 诊断合同与测试设计

以下 `E_*` 是**本项目拟定**的诊断码，不是 TVM 官方错误码。每条诊断要有源函数/算子位置、参数编号、实际与期望的 Shape/DType/属性、可行的处理方向。

| 代码 | 触发例子 | 归类 |
| --- | --- | --- |
| `E_UNSUPPORTED_OP` | 来源有未覆盖算子 | Adapter 能力不足 |
| `E_TARGET_OP_UNAVAILABLE` | 当前请求导入 MatMul | 目标 IR 能力不足 |
| `E_UNSUPPORTED_DTYPE` | BF16 / INT8 输入 | Adapter/目标能力不足 |
| `E_DYNAMIC_SHAPE` | `[?,4]` | 当前静态合同不支持 |
| `E_SHAPE_MISMATCH` | MatMul 的 K 维不同 | 源输入语义不合法 |
| `E_BROADCAST_UNSUPPORTED` | `[2,4]+[4]` | 源合法、当前 Mini 表示不足 |
| `E_LAYOUT_UNSUPPORTED` | 来源要求转置或特殊排布 | 无等价布局映射 |
| `E_ATTRIBUTE_UNSUPPORTED` | 未处理的 transpose/rounding 属性 | 语义属性无法保留 |
| `E_CONSTANT_UNSUPPORTED` | 子图内嵌权重常量 | 尚无常量导入策略 |
| `E_RELU_SEMANTICS` | 来源特殊值规则与 Mini 不同/未知 | 数值语义无法证明 |
| `E_SIDE_EFFECT` | 子图含状态更新 | 纯计算合同不支持 |
| `E_ABI_UNRESOLVED` | 边界张量的交付方式未定义 | 运行时集成未完成 |
| `E_MISSING_VALUE` | Use 找不到已定义值 | Adapter 实现错误或源 IR 非法 |

拟实施 Adapter 时的测试矩阵（**当前尚不能运行 Adapter 测试**）：

| 场景 | 预期 |
| --- | --- |
| `[2,4]` 同形状 Add；一个结果被两处使用 | 成功导入；一个定义、两个 Use；Verifier 通过 |
| `[2,4]` ReLU | 先核准特殊值规则，再成功导入并验证 |
| `[2,4]+[4]` | `E_BROADCAST_UNSUPPORTED`；没有目标产物 |
| `[?,4]`、BF16、未知布局、内嵌常量、副作用 Op | 对应明确诊断，不能静默降级 |
| `[2,3]×[3,4]` MatMul | 当前 `E_TARGET_OP_UNAVAILABLE`；Day 19 实现后重测 |
| 来源存在转置属性或 K 维不等 | 分别报告属性/Shape 问题 |
| 完整导入的 Add/ReLU 样例 | MLIR 解析/Verifier 通过；当前 `mini-compile` 输出无 Mini 计算 Op |
| 真实 TVM 与 NPU Runtime 接入之后 | 端到端与来源执行结果按明确定义的误差/特殊值规则比较 |

这些测试中的最后一项才是数值等价证据；单独通过 MLIR Verifier 或 `mini-compile` 只说明 IR 合法、Lowering 可完成。

## 6. 三种架构选择

| 方案 | 接口位置 | 初始改造成本 | 优化边界与运行时代价 | 何时适用 |
| --- | --- | --- | --- | --- |
| A. TVM Frontend → MLIR Graph → NPU | 整个高层图转入 MLIR | 高：需广泛覆盖 Relax 语义、控制流、动态 Shape 等 | MLIR 获得更大全图优化空间；要重建更多图编译和 Runtime 配合 | 计划长期把图优化主导权迁到 MLIR |
| B. TVM 图优化/分区 → MLIR NPU 子图 | 已确定边界的静态子图 | 较低的可控起点；仍要保证每个子图语义 | 跨分区融合受限；Host/NPU 数据传输、外部模块和 ABI 必须接通 | 已有 TVM 前端、需要渐进迁移 NPU 后端 |
| C. ONNX → MLIR → NPU | 绕开 TVM，直接导入模型 | 需建设/复用 ONNX Importer 与图优化；失去现有 TVM 投资 | 全链路可由 MLIR 统一设计；模型覆盖与 Runtime 工作量大 | 目标是独立工具链或 TVM 集成价值较低 |

**本教学项目选择 B 作为未来 Adapter 的首个候选**：第二周已有 Mini 图 IR、融合和到 Linalg 的 Lowering，可从小型、静态、`f32` 子图开始研究“TVM 前端 → MLIR NPU 后端”的边界。它不是对真实公司的普遍结论。若真实系统依赖大量跨子图融合、TVM 原有 Tensor Kernel 优化、动态 Shape 或完整 ONNX 覆盖，应该用实际模型集、编译覆盖率和运行时成本重新评估 A/B/C。

## 7. 接入前仍需决定

1. TVM 的具体版本、Relax/BYOC 入口及外部模块格式是什么？[官方 BYOC 文档](https://tvm.apache.org/docs/arch/external_library_dispatch.html)仅提供机制参考。
2. NPU 支持的 Shape、DType、Layout、指令、SRAM 和 DMA 约束是什么？这些约束何时反馈给图分区？
3. Host/NPU 边界的 Buffer 所有权、对齐、Stride、同步和错误传播如何定义？
4. 常量是嵌入编译产物、按函数参数传入，还是放入独立权重段？
5. 如何对每个算子证明语义等价，并建立端到端数值与性能基线？

这些是后续 Adapter、NPU Lowering 和 Runtime 的设计输入。Day 15 的交付是**可核查的接口与风险清单**，不是已经具备 TVM→NPU 的可执行链路。
