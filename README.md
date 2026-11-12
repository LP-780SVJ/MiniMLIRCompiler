# MiniMLIRCompiler

基于本地 LLVM/MLIR 23.1.4 的独立 C++ 学习项目。当前（Week 2 Day 13）
已实现 Mini Dialect、静态 FP32 Tensor 算子、Verifier、Add + Relu 融合，
以及到 Tensor/Linalg/Arith 的完整计算算子 Lowering。`mini-opt` 可通过
`mini-compile` 运行组合流水线，输出不含 Mini Operation 的标准 MLIR。

```text
输入 Mini IR → 解析与验证 → canonicalize → AddReluFusion
             → MiniToLinalg Full Conversion → CSE → canonicalize
             → 标准 Tensor / Linalg / Arith / Func IR
```

## 构建与运行

在 `AI-Compiler` 根目录运行，复用已有 LLVM/MLIR 构建：

```bash
cmake -S code/MiniMLIRCompiler -B code/MiniMLIRCompiler/build -G Ninja \
  -DMLIR_DIR="$PWD/llvm-project/build/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/llvm-project/build/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build code/MiniMLIRCompiler/build --target mini-opt -j 6
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

四个 Tensor 算子要求静态 Shape、FP32 元素类型，输入与结果的完整 Tensor
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

Day 13 的完整 CTest **12/12 通过**。测试覆盖四种独立 Lowering、
可融合与不可融合的计算图、SSA 结果连接、输出重新解析、重复运行、
不支持的 Mini Operation 与非法 Shape。逐日构建和排错记录见
`docs/development-log.md`。

目前输出是标准 Tensor/Linalg/Arith IR；尚未进行 Bufferization、循环
降级、LLVM Lowering 或 CPU 数值执行。结构测试不等于数值等价或性能
验证。
