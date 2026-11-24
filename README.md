# MiniMLIRCompiler

基于本地 LLVM/MLIR 23.1.4 的独立学习项目。Week 2
已实现 Mini Dialect、静态 FP32 Tensor 算子、Verifier、Add + Relu 融合，
以及四种算子到 Tensor/Linalg/Arith 的 Lowering。`mini-opt` 可通过
`mini-compile` 运行组合流水线，输出不含 Mini Operation 的标准 MLIR。

```text
输入 Mini IR → 解析与验证 → canonicalize → AddReluFusion
             → MiniToLinalg Full Conversion → CSE → canonicalize
             → 标准 Tensor / Linalg / Arith / Func IR
```

Week 3 Day 19 打通 `ONNX → Mini Graph IR`，支持从真实 ONNX 图导入静态
二维 FP32 MatMul。参见 [Importer 用法](python/onnx_importer/README.md) 和
[Day 19 验收](docs/npu/day19-review.md)。MatMul 可由 `mini-opt` 解析与验证，
当前 `mini-compile` 尚无对应的 Linalg 转换规则。

Day 20 支持内嵌 FP32 权重，以及 MatMul、同形状 Add/Mul、ReLU 的组合导入。
`mini-opt` 也可独立验证 `npu.matmul`。
CPU 参考输出、比较器及复现命令见 [Day 20 验收](docs/npu/day20-review.md)。

Day 21 打通 `ONNX → Mini Graph IR → NPU Compute IR`。使用独立的
`builtin.module(mini-npu-compile)` 流水线转换 MatMul，并保留函数签名、
FP32 Tensor 常量和 SSA 数据依赖。运行命令见 [Week 3 验收](docs/npu/week3-review.md)。

Day 22 的 `npu-tiling` 从 Virtual NPU JSON 读取原生上限，将完整 MatMul
转换为带静态计划的 `npu.tiled_matmul`，支持 M/N/K 尾块并保留 Tensor SSA。
见 [分块契约](docs/npu/tiling-design.md) 和 [Day 22 验收](docs/npu/day22-review.md)。

Day 23 的 `npu-sram-plan` 为 Tile 规划可复用的 SRAM 地址，生成
`npu.sram_planned_matmul`。它保留 Tensor SSA，并记录容量、对齐、活跃区间、
每条 Tile 的 A/B/C Binding 和使用量。见 [SRAM 规划](docs/npu/sram-planning.md)
和 [Day 23 验收](docs/npu/day23-review.md)。

Day 24 的 `npu-insert-dma` 为完整 Tensor 分配 DRAM 区间，并将 Tile 与
SRAM 绑定转成可验证的 ZERO、二维 DMA、MatMul 和写回命令。
见 [命令契约](docs/npu/dma-contract.md) 和 [Day 24 验收](docs/npu/day24-review.md)。

Day 25 的 `npu-schedule` 为原命令建立全局执行顺序和依赖，检查 SRAM
地址复用、K 累加和跨 MatMul 的完整写回。见 [调度契约](docs/npu/scheduling-contract.md)
和 [Day 25 验收](docs/npu/day25-review.md)。

Day 26 的 `mini-npu-codegen` 将已验证的 Scheduled MLIR 导出为
`program.json`、`weights.bin` 和 `manifest.json`。程序包可由独立的
Python 标准库加载器验证。见 [Virtual ISA v1](docs/npu/isa-spec.md)、
[程序 ABI](docs/npu/program-abi.md) 和 [Day 26 验收](docs/npu/day26-review.md)。

Day 27 的 Virtual NPU Simulator 从程序包加载权重和指令，在独立的 DRAM
和 SRAM 中解释四种 Opcode，并导出命名 FP32 NPZ。加权和多块 MatMul
均从 ONNX 重新编译并通过数值比较。见 [Simulator 契约](docs/npu/simulator-contract.md)
和 [Day 27 验收](docs/npu/day27-review.md)。

Day 28 的 [统一 E2E 入口](docs/npu/e2e-validation.md) 接收 ONNX 和输入 NPZ，
保存各阶段 IR，导出程序包，在 Simulator 执行并与独立 CPU 参考比较。
已验证 Small、Weighted、Multi-Tile、真实双 MatMul、尾块和多输出。
见 [后端架构](docs/npu/backend-architecture.md) 和 [第四周复盘](docs/npu/week4-review.md)。

## 构建与运行

在 `AI-Compiler` 根目录运行，复用已有 LLVM/MLIR 构建：

```bash
cmake -S code/MiniMLIRCompiler -B code/MiniMLIRCompiler/build -G Ninja \
  -DMLIR_DIR="$PWD/llvm-project/build/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/llvm-project/build/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build code/MiniMLIRCompiler/build -j 6
ctest --test-dir code/MiniMLIRCompiler/build --output-on-failure
```

解析基本计算图、运行完整流水线与回归测试：

```bash
code/MiniMLIRCompiler/build/bin/mini-opt \
  code/MiniMLIRCompiler/test/dialect/basic.mlir
code/MiniMLIRCompiler/build/bin/mini-opt \
  code/MiniMLIRCompiler/test/conversion/full-pipeline.mlir \
  --pass-pipeline='builtin.module(mini-compile)'
bash code/MiniMLIRCompiler/test/verifier/check-verifier.sh
bash code/MiniMLIRCompiler/test/transforms/check-fusion.sh
bash code/MiniMLIRCompiler/test/conversion/check-full-pipeline.sh
```

完整流水线先把 Add + Relu 融合成 `mini.add_relu`，再将四种 Mini
计算 Operation 转换成 `linalg.generic`。单独运行
`mini-fuse-add-relu` 或 `mini-to-linalg` 仍可用于观察各阶段。

## 当前 Operation 与合法性

| Operation | 语义 | Operand / Result |
| --- | --- | --- |
| `mini.ping` | Day 8 工程连通性测试 | 0 / 0 |
| `mini.add` | 逐元素加法 | 2 / 1 |
| `mini.mul` | 逐元素乘法 | 2 / 1 |
| `mini.relu` | 逐元素 ReLU | 1 / 1 |
| `mini.add_relu` | 融合的逐元素 Add + ReLU | 2 / 1 |
| `mini.matmul` | 二维矩阵乘法 `[M,K] × [K,N] → [M,N]` | 2 / 1 |

四个逐元素 Tensor 算子要求静态 Shape、FP32 元素类型，输入与结果的完整 Tensor
Type 一致。`mini.add` 不支持 Broadcasting；`mini.mul` 是逐元素乘法。
ODS 检查每个 Operand/Result 的基本类型；C++ Verifier 检查它们之间
的 Shape 与 Type 关系。当前 Lowering 将 ReLU 明确实现为有序
`x > +0.0f ? x : +0.0f`，因此 NaN 和 `-0.0f` 输出 `+0.0f`。
独立 ReLU 与融合 ReLU 共用这份标量实现。

## 文件与模块关系

```text
include/Mini/MiniDialect.td + MiniOps.td
  → mlir-tblgen → build/include/Mini/*.inc
  → lib/Dialect/Mini/ → MLIRMini
  → lib/Transforms/ → MLIRMiniTransforms
  → lib/Conversion/ → MLIRMiniToLinalg
  → tools/mini-opt/ → mini-opt
```

- `include/Mini/`：ODS 定义、生成 C++ 接口的入口及 Pass 注册声明。
- `lib/Dialect/Mini/`：Operation 注册与自定义 Verifier。
- `lib/Transforms/`：融合 Pattern 与函数级 Pass。
- `lib/Conversion/`：四种计算 Operation 到 Linalg 的转换与完整合法化。
- `tools/mini-opt/`：注册 Dialect、Pass、`mini-compile` Pipeline，并调用 `MlirOptMain`。
- `test/dialect/`、`test/verifier/`、`test/transforms/`、`test/conversion/`：分层回归测试。
- `build/`：独立构建产物，已由 `.gitignore` 排除。

`MiniOps.td` 引入 `MiniDialect.td`；`add_mlir_dialect(MiniOps mini)`
生成 Dialect 与 Operation 的 `.inc` 文件。`MiniDialect::initialize()`
使用生成的操作列表注册所有 Mini Operation。`MLIRMini` 保存 IR 定义
与验证；`MLIRMiniTransforms` 保存优化；`MLIRMiniToLinalg` 保存跨方言
转换；Driver 链接并组织三者。

融合 Pass 运行在 `func.func` 层级：从 Relu 的输入沿 SSA 定义边找到
Add，仅当 Add 结果只有当前 Relu 一个使用者且类型一致时，创建
`mini.add_relu`、替换 Relu 的结果并删除旧 Add。独立 Pass 让 Pipeline
可以决定是否启用融合。Canonicalization 是另一个可组合的通用 Pass。

Lowering Pass 在 `builtin.module` 层级使用 `applyFullConversion`。
`mini.add`、`mini.mul`、`mini.relu`、`mini.add_relu` 各有对应的
Conversion Pattern，生成 `tensor.empty`、`linalg.generic` 和 Region
内的标量 Arith 运算。`mini.ping` 能解析，但因没有 Lowering 规则而被
完整合法化拒绝。

## 测试与当前边界

Day 28 的 CTest **62/62 通过**，包含 Day 27 的 53 项与新增的 9 项 E2E。
从 AI-Compiler 根目录配置时，通过 `-DMINI_ONNX_PYTHON="$PWD/.venv/bin/python"`
启用 ONNX 导入和模拟器回归。完整 E2E 测试需要该 Python 环境可加载
ONNX Runtime，执行已编译的程序包只需要 Python 与 NumPy。

Day 13 的完整 CTest **12/12 通过**。测试覆盖四种独立 Lowering、
可融合与不可融合的计算图、SSA 结果连接、输出重新解析、重复运行、
不支持的 Mini Operation 与非法 Shape。逐日构建和排错记录见
`docs/development-log.md`。

CPU 路径输出标准 Tensor/Linalg/Arith IR，Bufferization、循环降级及
CPU 执行尚待实现。Virtual NPU 路径可以生成和执行独立程序包。
当前为功能级模拟，支持同步 DMA 和静态二维 FP32 MatMul。
