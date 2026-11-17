# Day 18 · 从图到命令的三层 IR

| 层 | 输入与输出 | 核心不变量 | 下游依赖 |
| --- | --- | --- | --- |
| Mini Graph IR | ONNX Graph → `func.func`、`arith.constant` 和 `mini.*` | SSA 定义和使用正确，Tensor Type/Shape 与算子语义一致。返回值遵循模型输出顺序。 | 图优化和两条目标转换。 |
| NPU Compute IR | MiniToNPU → `npu.matmul`、函数结构和常量 | 保留数学语义、函数签名、SSA 边与 Tensor 类型。可表达任意本契约内的完整静态二维 MatMul。 | Tiling 识别 M/N/K 与权重来源，为分块、地址和调度提供输入。 |
| NPU Command IR | 硬件 Lowering → Tile 级初始化、DMA、乘加、写回命令 | 显式记录 Tile 范围、字节地址/偏移、行跨度、Buffer 生命周期、依赖与 K 维累加。命令的访问和顺序满足已验证的资源计划。 | Virtual ISA 编码和执行。 |

## 跟踪一个带权重 MatMul

规划中的 Graph IR 用一个 `tensor<3x4xf32>` 常量承载不可覆盖的 `W`，以 `X: tensor<2x3xf32>` 为唯一函数参数。结果 `tensor<2x4xf32>` 表示逻辑值。MiniToNPU 更换计算 Op 为 `npu.matmul`，两输入与结果的 Tensor 类型保持一致。权重常量依旧是 SSA 定义，后续 Command Lowering 才决定它的权重段、DRAM 数据标识、SRAM 装载位置与时机。测试完整 NPU 转换时使用单独的 MatMul 例子。

对于 `[64,96] × [96,64]`，目标配置允许原生最大 `32×32×32`。NPU Compute IR 可先保留整个计算。经分块得到 4 个输出 Tile，每个 Tile 完成 3 次 K 块累加，输出 Tile 初始化一次。DMA 只能读取已准备的输入、权重或计算结果。SRAM 地址来自生命周期规划，布局和 stride 在命令化时具体化。对 `2×3×4` 这类小形状的边界执行语义由 Day 22 选定，Day 21 的 IR 验收与该策略独立。[Day 17 硬件契约](hardware-spec.md)

Graph Verifier 检查 MatMul 的类型、rank、K 和结果形状。NPU 的 Op Verifier 检查计算层的同类约束。目标原生块、对齐和峰值 SRAM 在 Tiling 与规划阶段检查。Simulator 在执行阶段还须检查地址边界、读写初始化状态和数值结果。每层应在掌握足够信息时给出对应诊断。
