# 从图到命令的 IR 分层

| 层 | 输入与输出 | 核心不变量 | 下游依赖 |
| --- | --- | --- | --- |
| Mini Graph IR | ONNX Graph → `func.func`、`arith.constant` 和 `mini.*` | SSA 定义和使用正确，Tensor Type/Shape 与算子语义一致。返回值遵循模型输出顺序。 | 图优化和两条目标转换。 |
| NPU Compute IR | MiniToNPU → `npu.matmul`、函数结构和常量 | 保留数学语义、函数签名、SSA 边与 Tensor 类型。可表达任意本契约内的完整静态二维 MatMul。 | Tiling 识别 M/N/K 与权重来源，为分块、地址和调度提供输入。 |
| NPU Tile Plan IR | `npu-tiling` → `npu.tiled_matmul` | 完整 Tensor SSA 加上 M/N/K 切分、输出块 ID 和 K 归约边界。 | SRAM Planner 按有效尺寸和累加生命周期建立 Buffer 计划。 |
| NPU SRAM Plan IR | `npu-sram-plan` → `npu.sram_planned_matmul` | 继续保留完整 Tensor 和 Tile，增加片上地址、活跃区间及每条 Tile 的 Buffer Binding。 | DMA Lowering 将 Tile 坐标与存储映射写成搬运命令。 |
| NPU Command IR | 硬件 Lowering → Tile 级初始化、DMA、乘加、写回命令 | 显式记录 Tile 范围、字节地址/偏移、行跨度、Buffer 生命周期、依赖与 K 维累加。命令的访问和顺序满足已验证的资源计划。 | Virtual ISA 编码和执行。 |

## 跟踪一个带权重 MatMul

已验证的加权 MatMul 用一个 `tensor<3x4xf32>` 常量承载不可覆盖的 `W`，以 `X: tensor<2x3xf32>` 为唯一函数参数。结果 `tensor<2x4xf32>` 表示逻辑值。MiniToNPU 更换计算 Op 为 `npu.matmul`，Tiling 随后附上计划，两输入与结果的 Tensor 类型始终一致。权重常量依旧是 SSA 定义，后续 Command Lowering 才决定它的权重段、DRAM 数据标识、SRAM 装载位置与时机。

对于 `[64,96] × [96,64]`，目标配置允许原生最大 `32×32×32`。经 Day 22 分块得到 4 个输出 Tile，每个 Tile 计划累加 3 个 K 块，输出 Tile 初始化一次。`2×3×4` 则直接使用有效尺寸。SRAM 地址来自生命周期规划，布局、stride 和 DMA 在后续阶段具体化。[Day 22 分块契约](tiling-design.md)

Graph Verifier 检查 MatMul 的类型、rank、K 和结果形状。NPU 的 Op Verifier 检查计算层的同类约束。目标原生块、对齐和峰值 SRAM 在 Tiling 与规划阶段检查。Simulator 在执行阶段还须检查地址边界、读写初始化状态和数值结果。每层应在掌握足够信息时给出对应诊断。
