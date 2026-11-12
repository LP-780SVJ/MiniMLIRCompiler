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

## Day 12：Mini 到 Linalg 的基础 Lowering

新增 `MLIRMiniToLinalg` Conversion 库与 Module Pass `mini-to-linalg`。
同一 `BinaryOpLowering` 模板处理 `mini.add` 和 `mini.mul`：每个模式
创建 `tensor.empty` 作为输出目标，并以三个恒等 indexing map、两个
parallel iterator 构造 `linalg.generic`。Region 内分别使用
`arith.addf` / `arith.mulf` 计算标量元素，再由 `linalg.yield` 返回。
转换模式使用 adaptor 提供的 Operand，并让 rewriter 将原 Mini 结果的
SSA 使用者接到新 Linalg 结果。当前阶段保持 `tensor<...xf32>` 类型，
因此没有引入 TypeConverter。

`ConversionTarget` 允许 Arith、Linalg、Tensor、Func 与 Module；
Mini Dialect 整体标为非法，只对尚未 Lowering 的 `mini.relu` 和
`mini.add_relu` 给出合法例外。`applyPartialConversion` 允许这两种
操作暂存，却会拒绝没有转换规则的 `mini.ping`。这使“合法的输入 IR”
和“满足本阶段目标的 IR”成为两个分别可检验的条件。

配置后构建 `mini-opt` 成功。`check-lowering.sh` 覆盖 Add、Mul、
混合 IR、转换后重解析、类型与 SSA 返回值连接、恒等访问映射、parallel
迭代、非法 `mini.ping` 和输入 Shape 错误。完整 CTest **11/11 通过**。
`test/conversion/results/` 保存了转换前后与 Pass 调试输出，便于对照。

这是从自定义图级 Mini IR 向标准 Tensor/Linalg/Arith IR 的第一步。
当前输出仍是张量级结构化计算，尚无 Bufferization、循环降级、LLVM IR
或可执行代码；Day 13 需要继续 Lowering `mini.relu` 和
`mini.add_relu`，并对完整目标 IR 采用更严格的合法化检查。

## Day 13：完整 Lowering 与 Mini 编译流水线

### 新增的计算语义与转换

新增公共 `emitReluScalar`，用标量 `arith.constant + arith.cmpf ogt +
arith.select` 定义 ReLU：有序且大于正零时保留输入，其余情况选择
`+0.0f`。因此 NaN 与 `-0.0f` 都输出 `+0.0f`。独立
`mini.relu` 与融合的 `mini.add_relu` 使用同一函数，避免融合前后
出现不同的特殊值处理。它们分别 Lowering 为一个 `linalg.generic`；
Relu 有一个输入和一个输出，因此使用两个 indexing map；AddRelu 有
两个输入和一个输出，因此使用三个。AddRelu 在同一个 Region 内先
`arith.addf`，再比较、选择，保留融合后的结构化计算表示。

### 合法化与 Pipeline

`mini-to-linalg` 现在注册 Add、Mul、Relu、AddRelu 四条转换规则，
改用 `applyFullConversion`。目标允许 Module、Func、Tensor、Linalg、
Arith；整个 Mini Dialect 非法，已不再给 Relu/AddRelu 例外。
因此 `mini.ping` 虽然能解析和验证，因没有转换规则而被拒绝。

为两个自定义 Pass 提供 Factory，保留原有命令行注册能力。
`mini-opt` 注册 `mini-compile`：函数级 canonicalize → 函数级
AddReluFusion → 模块级 MiniToLinalg → 函数级 CSE → 函数级
canonicalize。Factory 让 Driver 组合 Pass，而不需要公开具体 Pass 类。
完整流水线在输出中不保留 Mini Operation。

### 构建与验收

`mini-opt` 构建成功，命令行显示 `mini-compile`。独立验证了 Relu、
AddRelu Lowering；集成验证了 Add/Mul、可融合的 Add→Relu、
不可融合的 Mul→Relu、非法 `mini.ping`、非法输入 Shape、转换后
重新解析和二次运行 Pipeline。测试还检查两个 Linalg 计算之间及
函数返回处的 SSA 数据流。旧 Day 12 回归脚本已更新为 Day 13 的
完整合法化预期。完整 CTest **12/12 通过**。

`test/conversion/results/` 保存了单独转换、融合后、完整转换后和逐
Pass 调试输出。完整 IR 已是标准 Tensor/Linalg/Arith/Func 表示，
仍未执行数值计算，也未进行 Bufferization、循环降级或机器码生成。
`arith.addf` 本身也支持 Tensor Operand；选择 Linalg 是为了在本项目
显式保留迭代空间、访问映射与标量 Region，供后续结构化优化使用。
