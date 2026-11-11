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
Operand、Result 和类型约束。Day 8 当时尚无优化 Pass、Lowering 或 CPU 执行。

## Day 9：定义 Tensor Operation

在 `MiniOps.td` 中加入 `mini.add`、`mini.mul`、`mini.relu`。
Add/Mul 各有两个 Operand 和一个 Result；Relu 有一个 Operand 和一个
Result。ODS 使用 `StaticShapeTensorOf<[F32]>` 和 `Pure`，TableGen 生成类、
访问器与基本验证代码。`basic.mlir` 的输出保留 Add → Relu → Mul 的 SSA
Use-Def 关系；打印器重新编号 SSA 名称，不改变 IR 语义。

错误输入实验：`i32` Tensor、动态 Shape、错误 Operand 数量被拒绝；
不同静态 Shape 的 Add 输入当时仍可通过。前三类预期失败用例纳入 CTest，
当时完整测试为 7/7 通过。这一缺口留给 Day 10 的跨值 Verifier。

## Day 10：Verifier 与 IR 合法性

为 Add、Mul、Relu 设置 `hasVerifier = 1`。ODS 先检查每个值是否为
静态 FP32 Tensor；C++ Verifier 再检查输入与结果之间的 Shape 和完整
Tensor Type 关系。Add/Mul 共用二元张量规则，Relu 使用一元规则。

`test/verifier/` 包含合法算子、输入 Shape 错误、结果 Shape 错误、
错误元素类型、未知 Rank 与内联诊断检查。Day 9 的 `shape-mismatch.mlir`
从接受变为拒绝；Day 9 的合法 `basic.mlir` 仍通过。当时 CTest 为
9/9 通过。Verifier 在后续 Pass 和 Lowering 前建立明确的 IR 不变量。

## Day 11：Add + Relu Fusion

### 新 IR 与优化模块

`mini.add_relu(lhs, rhs)` 表示逐元素 `max(lhs + rhs, 0)`，与
`mini.add → mini.relu` 数学语义相同。它复用二元张量 Verifier，因而即使
直接作为输入 IR，也必须满足相同 Shape 和完整 Type 约束。

新增 `MLIRMiniTransforms` 库，存放 `FuseAddReluPattern` 和
`AddReluFusionPass`。Pattern 以 `ReluOp` 为根，通过其输入的
`getDefiningOp<AddOp>()` 查找上游 Add；要求 Add 结果只有一个使用者，
并检查替换前后的结果 Type。匹配成功后，先创建 AddRelu，再以融合结果
替换 Relu 的所有使用，最后删除已无使用者的 Add。多使用者时保持原图。

Pass 运行在 `func.func` 上，通过 `applyPatternsGreedily` 应用规则。
`mini-opt` 注册自定义 Pass 和 MLIR 通用变换 Pass；命令行可选择
`builtin.module(func.func(mini-fuse-add-relu))`，也可组合
`canonicalize,mini-fuse-add-relu,canonicalize`。

### 构建与测试记录

```bash
cmake --build code/MiniMLIRCompiler/build --target mini-opt -j 6
bash code/MiniMLIRCompiler/test/transforms/check-fusion.sh
ctest --test-dir code/MiniMLIRCompiler/build --output-on-failure
```

结果：TableGen 生成 `AddReluOp`，`MLIRMini`、`MLIRMiniTransforms` 和
`mini-opt` 构建成功；`--help` 显示 `mini-fuse-add-relu` 和
`canonicalize`。融合脚本验证直接构造的合法/非法 AddRelu、正例、
Mul → Relu 与单独 Relu 反例、多使用者、重复运行、组合 Pipeline，
以及 Day 10 非法 Shape 仍被拒绝。完整 CTest **10/10 通过**。

本机 LLVM 23.1.4 对 `rewriter.create<AddReluOp>` 给出弃用警告，
改为使用生成的 `AddReluOp::create(rewriter, ...)` 接口。

### 架构边界

Operation 定义融合后的合法 IR；Pattern 决定何时安全改写；Pass 决定
在哪个 IR 层级和 Pipeline 阶段运行。当前测试证明结构、SSA 和验证
行为，尚未验证数值执行或性能收益。后续需要为 `mini.add_relu` 以及
未融合的 Add、Mul、Relu 补齐 Lowering，保证两条路径都能继续编译。
