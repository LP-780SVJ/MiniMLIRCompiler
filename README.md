```markdown
MiniMLIRCompiler
│
├── Mini Dialect
│   ├── mini.add
│   ├── mini.mul
│   └── mini.relu
│
├── Optimization
│   ├── Canonicalization
│   └── AddReluFusion
│
├── Lowering
│   └── Mini → arith / linalg / tensor
│
├── Driver
│   ├── Dialect Registration
│   ├── Pass Registration
│   └── Pass Pipeline
│
└── Tests
    ├── IR Verification
    ├── Rewrite Tests
    └── Lowering Tests
```

## Day 8：独立工程骨架

本项目在 `llvm-project` 之外构建。Day 8 定义并注册 `mini` Dialect 和
零输入、零输出的 `mini.ping`，用于验证自定义 Operation 的解析与打印；
`mini.add`、`mini.mul`、`mini.relu` 将在 Day 9 加入。
`test/mini-ping.mlir` 验证自定义 IR；`test/valid.mlir` 验证基础 MLIR；
`--show-dialects` 验证 `mini` 已注册；未知 `mini` Operation 的诊断用于验证
Driver 没有开放未注册操作。

目录职责：

- `include/Mini/`：Dialect 的 ODS 定义、生成 C++ 声明的入口。
- `lib/`：Dialect 的 C++ 实现和静态库。
- `tools/`：`mini-opt` Driver，注册可解析的 Dialect。
- `test/`：CTest 验收输入与检查脚本。
- `build/`：独立构建产物，已加入 `.gitignore`。

在仓库根目录运行（沿用当前 LLVM 23.1.4 构建）：

```bash
cmake -S code/MiniMLIRCompiler -B code/MiniMLIRCompiler/build -G Ninja \
  -DMLIR_DIR="$PWD/llvm-project/build/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/llvm-project/build/lib/cmake/llvm"
cmake --build code/MiniMLIRCompiler/build -j 4
ctest --test-dir code/MiniMLIRCompiler/build --output-on-failure
code/MiniMLIRCompiler/build/bin/mini-opt --show-dialects
code/MiniMLIRCompiler/build/bin/mini-opt code/MiniMLIRCompiler/test/valid.mlir
```

`MiniOps.td` 引入 `MiniDialect.td`，由 `add_mlir_dialect(MiniOps mini)` 调用
`mlir-tblgen` 生成 `MiniOpsDialect.h.inc`、`MiniOpsDialect.cpp.inc` 和操作的
`.inc` 文件；`MiniDialect.cpp` 包含生成的 Dialect 实现并在 `initialize()`
中注册 `PingOp`。
`mini-opt` 把 `MiniDialect` 放入 `DialectRegistry`，`MlirOptMain` 使用该
Registry 解析输入并执行 MLIR 自带的验证。Day 9 添加 Operation 时，
将在 ODS 定义中声明操作，由生成的操作列表自动纳入 `initialize()` 注册。
