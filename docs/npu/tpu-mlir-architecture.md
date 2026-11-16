# Day 16 · TPU-MLIR 的图到硬件编译

源码固定在 sophgo/tpu-mlir 提交 [81dc2a32acdfd5b9ed7f8461267d079d520db862](https://github.com/sophgo/tpu-mlir/commit/81dc2a32acdfd5b9ed7f8461267d079d520db862)。本页依据静态源码与官方手册。示例中的 Tile、地址和 IR 是教学推导。

## 1. 从模型到 BModel

```mermaid
flowchart LR
    Input[ONNX 等模型] --> Top[Top 图 IR]
    Top --> Mapping[TopToTpu 芯片与精度映射]
    Mapping --> Tpu[Tpu 硬件算子 IR]
    Tpu --> Local[LayerGroup 切片、调度和局部内存]
    Local --> Global[全局地址分配]
    Global --> Gen[后端 CodeGen]
    Gen --> Model[BModel]
```

图层描述 Tensor 计算、类型、形状、常量和数据依赖。硬件映射为算子选择目标表示与精度，后端再给出切片、调度、地址和设备指令。这是分层关系，具体 Pass 顺序会随目标与选项而变化。[Overall Design](https://www.tpumlir.org/developer_manual_en/04_over_design.html)

## 2. Top Dialect 与 MatMul 图优化

ONNX ModelProto 是模型交换结构，包含 Graph、节点名、Value 名称、类型信息、Initializer 和 Opset。TPU-MLIR 的 ONNX 导入器把节点转换为 Top Operation，以 MLIR SSA 值连接生产者与使用者。导入时的 `top.MatMul` 可以先产生未定秩结果，后续由 Shape 接口填入形状。

| 证据 | 路径和定位 | 结论 |
| --- | --- | --- |
| 导入 | [python/transform/OnnxConverter.py](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/python/transform/OnnxConverter.py#L1190-L1250) | `convert_gemm_op` 处理 MatMul/Gemm 的共享路径，构造 Top MatMul，缺省 Bias 使用 None。 |
| ODS | [include/tpu_mlir/Dialect/Top/IR/TopOps.td](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/include/tpu_mlir/Dialect/Top/IR/TopOps.td#L3935-L3995) | `Top_MatMulOp` 有 `input`、`right`、`bias` 三个 operand，一个 Tensor 结果。属性包括转置、`keep_dims`、融合 ReLU 和量化参数。 |
| Shape | [lib/Dialect/Top/Interfaces/MatMul.cpp](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Dialect/Top/Interfaces/MatMul.cpp#L288-L365) | 默认二维情况下，`[2,3] × [3,4]` 推导为 `[2,4]`。`setShapeOrVerify` 填入未定秩或动态结果的类型，对已有静态二维 Shape 检查一致性。另有转置和展平等分支。 |
| Canonicalizer | [lib/Dialect/Top/Canonicalize/MatMul.cpp](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Dialect/Top/Canonicalize/MatMul.cpp#L1116-L1129) | `MatMulOp::getCanonicalizationPatterns` 注册 `MatMulWithBias` 等规则。 |
| ReLU 融合 | [lib/Dialect/Top/Canonicalize/Relu.cpp](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Dialect/Top/Canonicalize/Relu.cpp#L15-L61) | `TopFuseRelu` 从 ReLU 沿 SSA 找生产者，检查单一使用、`SupportFuseRelu` 等条件，再更新激活属性。 |
| Pass 顺序 | [python/utils/mlir_shell.py](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/python/utils/mlir_shell.py#L508-L588) | Top 优化运行 Shape 推导和 canonicalization。部署的 Lowering 路径先设置芯片和模式，再运行 `processor-top-optimize`，随后转换到 Tpu。 |

`MatMulWithBias` 要检查右矩阵为权重、Bias 槽位为空、MatMul 只有一个使用者、该使用者是适用的 Add，并检查 Bias 元素数和 batch 等条件。若另一条边还读取原始 MatMul，直接把 Bias 写进生产者会改变这条边的值。图融合靠这些语义和依赖条件判断。处理器可以提前影响图改写规则，后续仍由 TopToTpu 映射计算。[Bias 规则](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Dialect/Top/Canonicalize/MatMul.cpp#L16-L73)、[ProcessorOptimizePass](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Dialect/Top/Transforms/ProcessorOptimize.cpp#L30-L54)

## 3. TopToTpu 的 BM1684X MatMul

| 步骤 | 代码 | 决定 |
| --- | --- | --- |
| 选 Pattern 集 | [lib/Conversion/TopToTpu/TopToTpuPass.cpp](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Conversion/TopToTpu/TopToTpuPass.cpp#L1917-L1929) | `ConvertTopToTpu::runOnOperation` 按芯片选择注册函数，再运行贪婪重写。 |
| 注册与匹配 | [LoweringBM1684X.cpp](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Conversion/TopToTpu/LoweringBM1684X.cpp#L113-L166)、[LoweringBM1684X.h](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/include/tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h#L66-L142) | 添加 `MatMulLowering`。类由宏声明，继承 `TopLowering<top::MatMulOp>`，因此匹配 Top MatMul。 |
| 选精度 | [include/tpu_mlir/Conversion/TopToTpu/TopLowering.h](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/include/tpu_mlir/Conversion/TopToTpu/TopLowering.h#L250-L352) | 普通路径按 module mode 和逐算子覆盖分派，已有量化模型另按结果 storage type 等条件分派。 |
| F32 替换 | [lib/Conversion/TopToTpu/BM1684X/MatMul.cpp](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Conversion/TopToTpu/BM1684X/MatMul.cpp#L15-L18)、[通用 helper](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/include/tpu_mlir/Conversion/TopToTpu/TopLowering.h#L429-L464) | `LoweringF32` 传入目标模板类型 `tpu::MatMulOp` 和五个 operand 槽位。helper 复用原三个值，给 `multi`、`buffer` 补 None，传递当前属性并用 `replaceOpWithNewOp` 接回 SSA 使用者。已有 F32 结果保持类型。 |
| BF16 与 INT8 | [BF16 实现](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Conversion/TopToTpu/BM1684X/MatMul.cpp#L973-L1052)、[INT8 实现](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Conversion/TopToTpu/BM1684X/MatMul.cpp#L391-L428) | BF16 处理结果类型与权重的精度，带低位权重的部分情况生成 `tpu.A16MatMul`。INT8 需要量化类型、scale 或 multiplier/shift、zero point 等信息。 |
| 不支持与回退 | [INT8 前置检查](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Conversion/TopToTpu/BM1684X/MatMul.cpp#L25-L62)、[F32 回退](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Conversion/TopToTpu/BM1684X/MatMul.cpp#L119-L123) | 批量 MatMul 的 Bias 元素数超过 N 时有明确的不支持分支。常量权重分支遇到 batch 大于 1 且输入 zero point 非零时会回退到 F32。 |

这段算子转换使用贪婪重写，并在 Pass 末尾遍历主函数检查仍未转成 Tpu 的 Operation。Weight、None、Input 和函数结构等有明确例外。MiniNPUCompiler 计划使用 MLIR Dialect Conversion 的 `ConversionTarget` 与完整转换来声明输出合法性。两种设计都需要处理「匹配不到规则时 IR 里留了什么」。[TPU-MLIR 末尾检查](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Conversion/TopToTpu/TopToTpuPass.cpp#L1948-L1963)

## 4. Tpu、LayerGroup 与内存

- **Tpu 表示目标计算。** `tpu.MatMul` 定义量化属性、临时 Buffer 槽位及 LocalGen 等接口。权重重排 Pass 根据目标硬件变换布局。[Tpu MatMul ODS](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/include/tpu_mlir/Dialect/Tpu/IR/TpuOps.td#L4742-L4828)、[WeightReorder](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Dialect/Tpu/Transforms/WeightReorder.cpp#L119)
- **Group 选相邻计算，Slice 选数据片，TimeStep 排计算和搬运，LMEM 分配排布片上地址。** 如果两个相邻算子能在同一组按切片连续执行，中间片可留在 LMEM，减少 GMEM 往返。切片大小会改变 Buffer 的尺寸，TimeStep 会改变它的生命周期，两者共同约束分配。官方手册的 BackwardH 用输出切片反推输入区域，属于其示例策略。[LayerGroup 手册](https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/tpu-mlir/developer_manual_en/html/10_layergroup.html)、[Pass 入口](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Dialect/Tpu/Transforms/LayerGroup.cpp#L70)、[TimeStepAssignmentPass](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.cpp#L599)、[LmemAllocator::assignLmemAddr](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Dialect/Tpu/Transforms/LayerGroup/LmemAllocator.cpp#L1033)
- **GMEM 分配处理跨组可见的值。** 官方手册用 Value、offset、size、ref count 说明分配。最后一次消费完成后才能复用相应范围，同时满足大小和对齐。固定提交的 Pass 入口使用目标分派，BM 路线由 `BMAddressAssign::assign` 为各 Value 建立 `TensorLive` 区间，再由 `GmemAllocator::assignGaddr` 按复用配置分配地址。手册的计数示意用于解释原则，具体实现依路径而异。[GMEM 手册](https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/tpu-mlir/developer_manual_en/html/11_memassign.html)、[AddressAssign](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Dialect/Tpu/Transforms/AddressAssign.cpp#L146)、[BMAddressAssign](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Dialect/Tpu/Transforms/AddressAssign/BMAddressAssign.cpp#L957-L1026)
- **CodeGen 将已规划的计算交给目标后端。** BM 路线的 Pass 调用 `BMCodegen::init/run/store`，按子模块生成并保存 BModel。后端的设备命令和模型封装在这一阶段产生。[Codegen.cpp](https://github.com/sophgo/tpu-mlir/blob/81dc2a32acdfd5b9ed7f8461267d079d520db862/lib/Dialect/Tpu/Transforms/Codegen.cpp#L22-L55)

### 两道手算

1. 对 `[128,96] × [96,64]`，按 32 分 M/N/K 得到 4、2、3 个块。输出有 8 个 Tile，总共执行 24 个块乘加。每个输出 Tile 初始化一次，再累加 3 个 K 块，最后写回。仅 A、B、C 的 FP32 原始数据占 `3 × 32 × 32 × 4 = 12 KiB`。
2. 若 A 存活于步骤 1–3，D 存活于步骤 6–8，且 A 的最后一次实际硬件访问已完成，两者可以使用同一个符合容量和对齐要求的地址范围。若 DMA 仍在读取 A，SSA 的最后一个消费者已经出现也不能据此覆盖对应地址。

## 5. 对应 MiniNPUCompiler 的哪个阶段

| TPU-MLIR 职责 | 方案 C 中的位置 |
| --- | --- |
| Top、Shape 与图优化 | ONNX 导入生成 Mini Graph IR，Verifier 和图 Pass 保证语义与依赖。 |
| TopToTpu、Tpu | `MiniToNPU` 产生 `npu.matmul`，封住首版静态二维 FP32 的支持边界。 |
| Slice、LMEM、TimeStep | NPU Tiling、SRAM Planning、DMA 和 Scheduling。首版重点是一个 MatMul 内的正确性。 |
| CodeGen、设备执行 | Command IR、Virtual ISA、Simulator 和独立 ONNX Runtime 数值对比。 |

Day 16 位于架构研究阶段。它为 Day 18 的 IR 和 Pass 接口、Day 21 的目标合法化以及第四周的片上存储与执行设计建立证据。首版统一 FP32，可以先用同一种数值语义把 M/N/K 分块、权重加载和结果写回验证清楚，再引入 BF16/INT8 的舍入和量化契约。Coral 的独立执行链见 [工具链调查](coral/toolchain-research.md)。
