# MiniMLIRCompiler 开发日志

## Day 8：独立工程初始化

### 环境

- macOS ARM64（M4）；Apple Clang 17.0.0。
- CMake 4.4.5、Ninja 1.13.2。
- 沿用本地 LLVM/MLIR 23.1.4，CMake 包位于 `llvm-project/build/lib/cmake/mlir`。
- `llvm-project` 工作区未显示改动，处于 detached HEAD。

### 构建路径与模块

`CMakeLists.txt` 用 `find_package(MLIR CONFIG REQUIRED)` 取得 MLIR 提供的
目标、头文件目录和 CMake 函数。`include/Mini/CMakeLists.txt` 使用
`add_mlir_dialect(MiniOps mini)` 读取 `MiniOps.td`；后者引入
`MiniDialect.td`，因此一次 TableGen 链路同时生成 Dialect 和 Operation 的
`.inc` 文件。`MLIRMiniOpsIncGen` 是代码生成目标，不是 C++ 库。

`MiniDialect.cpp` 与 `MiniOps.cpp` 引入生成文件，编译成静态库 `MLIRMini`。
`mini-opt.cpp` 链接该库及所需 MLIR 库，通过 `DialectRegistry` 注册
`MiniDialect`，再交给 `MlirOptMain` 解析、验证和打印输入 IR。

### 验证记录

从 `AI-Compiler` 根目录执行：

```bash
cmake -S code/MiniMLIRCompiler -B code/MiniMLIRCompiler/build -G Ninja \
  -DMLIR_DIR="$PWD/llvm-project/build/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/llvm-project/build/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build code/MiniMLIRCompiler/build --target mini-opt -j 6
cmake --build code/MiniMLIRCompiler/build --target MLIRMiniOpsIncGen
ctest --test-dir code/MiniMLIRCompiler/build --output-on-failure
```

结果：配置、TableGen、`MLIRMini` 和 `mini-opt` 构建成功；CTest 4/4 通过。
`mini-opt --help` 正常显示帮助与已注册的 `mini` Dialect。
`mini-opt test/mini-ping.mlir`（使用项目内完整路径）能打印
`"mini.ping"() : () -> ()`。官方 `mlir-opt` 解析相同输入时因未注册
`mini` 而失败；`mini-opt` 解析 `invalid-op.mlir` 时因 `mini.unknown`
未注册而失败。这两种失败均为预期行为。

另在新的临时构建目录完成干净构建与 4/4 CTest，排除了旧生成文件的影响。

### 排错记录

1. 误把 `MLIR_DIR` 指向 `build/tools/mlir/cmake/modules/CMakeFiles`：那里有
   内部 `MLIRConfig.cmake`，却缺少它依赖的导出目标。改用
   `build/lib/cmake/mlir`；Day 8 教程中的查找命令也已纠正。
2. 生成的 `MiniOps.cpp.inc` 使用 `Builder` 与 `OpBuilder`，但
   `MiniOps.cpp` 只见到前向声明。显式引入 `mlir/IR/Builders.h` 后编译通过。
3. 原工程同时生成两套同一 Dialect 的 `.inc`，且 `MiniDialect.h` 引用
   旧的一套。保留 `add_mlir_dialect(MiniOps mini)` 单一链路，并让头文件
   引入 `MiniOpsDialect.h.inc`。干净构建发现并验证了这一问题。

### 当前边界与下一步

`mini.ping` 是零输入、零输出的工程连通性测试，不执行张量计算。
Day 9 将在相同 Dialect 下定义 `mini.add`、`mini.mul`、`mini.relu` 的
Operand、Result 和类型约束。当前还没有优化 Pass、Lowering 或 CPU 执行。
