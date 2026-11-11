# MiniMLIRCompiler

基于本地 LLVM/MLIR 23.1.4 的独立 C++ 学习项目。当前（Week 2 Day 11）
已实现 Mini Dialect、静态 FP32 Tensor 算子、Verifier 和 Add + Relu 融合
Pass。`mini-opt` 可以解析、验证、优化并打印 Mini IR。

```text
输入 .mlir → Mini Dialect 解析与验证
           → 可选：canonicalize / mini-fuse-add-relu
           → 输出 Mini IR
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

解析基本计算图、执行融合与单独运行测试脚本：

```bash
code/MiniMLIRCompiler/build/bin/mini-opt \
  code/MiniMLIRCompiler/test/dialect/basic.mlir
code/MiniMLIRCompiler/build/bin/mini-opt \
  code/MiniMLIRCompiler/test/transforms/add-relu-positive.mlir \
  --pass-pipeline='builtin.module(func.func(mini-fuse-add-relu))'
bash code/MiniMLIRCompiler/test/verifier/check-verifier.sh
bash code/MiniMLIRCompiler/test/transforms/check-fusion.sh
```

融合正例的输出应含一个 `mini.add_relu`，不再含原来的 `mini.add`
和 `mini.relu`。

## 当前 Operation 与合法性

| Operation | 语义 | Operand / Result |
| --- | --- | --- |
| `mini.ping` | Day 8 工程连通性测试 | 0 / 0 |
| `mini.add` | 逐元素加法 | 2 / 1 |
| `mini.mul` | 逐元素乘法 | 2 / 1 |
| `mini.relu` | 逐元素 `max(x, 0)` | 1 / 1 |
| `mini.add_relu` | 融合的逐元素 `max(lhs + rhs, 0)` | 2 / 1 |

四个 Tensor 算子要求静态 Shape、FP32 元素类型，输入与结果的完整 Tensor
Type 一致。`mini.add` 不支持 Broadcasting；`mini.mul` 是逐元素乘法。
ODS 检查每个 Operand/Result 的基本类型；C++ Verifier 检查它们之间
的 Shape 与 Type 关系。

## 文件与模块关系

```text
include/Mini/MiniDialect.td + MiniOps.td
  → mlir-tblgen → build/include/Mini/*.inc
  → lib/Dialect/Mini/ → MLIRMini
  → lib/Transforms/ → MLIRMiniTransforms
  → tools/mini-opt/ → mini-opt
```

- `include/Mini/`：ODS 定义、生成 C++ 接口的入口及 Pass 注册声明。
- `lib/Dialect/Mini/`：Operation 注册与自定义 Verifier。
- `lib/Transforms/`：融合 Pattern 与函数级 Pass。
- `tools/mini-opt/`：注册 Dialect、Pass，并调用 `MlirOptMain`。
- `test/dialect/`、`test/verifier/`、`test/transforms/`：IR、合法性与改写测试。
- `build/`：独立构建产物，已由 `.gitignore` 排除。

`MiniOps.td` 引入 `MiniDialect.td`；`add_mlir_dialect(MiniOps mini)`
生成 Dialect 与 Operation 的 `.inc` 文件。`MiniDialect::initialize()`
使用生成的操作列表注册所有 Mini Operation。`MLIRMini` 保存 IR 定义
与验证；`MLIRMiniTransforms` 保存优化；Driver 链接两者。

融合 Pass 运行在 `func.func` 层级：从 Relu 的输入沿 SSA 定义边找到
Add，仅当 Add 结果只有当前 Relu 一个使用者且类型一致时，创建
`mini.add_relu`、替换 Relu 的结果并删除旧 Add。独立 Pass 让 Pipeline
可以决定是否启用融合。Canonicalization 是另一个可组合的通用 Pass。

## 测试与当前边界

Day 11 的完整 CTest **10/10 通过**。融合测试覆盖正反例、多使用者、
重复运行、与 Canonicalization 组合、融合 Operation 自身的 Verifier，
以及 Day 10 非法 Shape 的回归。逐日构建和排错记录见
`docs/development-log.md`。

目前输出仍是 Mini IR；Mini 到标准 MLIR Dialect 的 Lowering 和 CPU
数值执行尚未实现。融合前后的数值等价需要在后续 Lowering 阶段验证。
