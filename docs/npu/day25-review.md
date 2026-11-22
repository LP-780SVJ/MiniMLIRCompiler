# Week 4 · Day 25 验收

开发基线为 `1741f8be72929f419cbcab4e69cfb259b297a8fd`，开工前工作区干净。LLVM/MLIR 为 `llvmorg-23.1.4`，原有 CTest 38/38 通过。

## 源码与职责

| 位置 | 职责 |
| --- | --- |
| [CommandEffects](../../lib/Dialect/NPU/CommandEffects.cpp)、[ScheduleDependencies](../../lib/Dialect/NPU/ScheduleDependencies.cpp) | 解释真实读写地址，按重叠区间建立 SRAM 冲突依赖、DRAM 生产者依赖与串行边界。 |
| [SchedulePlan.cpp](../../lib/Dialect/NPU/SchedulePlan.cpp)、[ScheduleState.cpp](../../lib/Dialect/NPU/ScheduleState.cpp) | 生成全局 ID，检查 DAG 与实际 Sequence；独立追踪 A/B 装载、C 归约、生存期、STORE 覆盖与 Tensor 完成状态。 |
| [SchedulePlanAttrs.cpp](../../lib/Dialect/NPU/SchedulePlanAttrs.cpp)、[NPUDialect.cpp](../../lib/Dialect/NPU/NPUDialect.cpp) | 编解码模块级 `npu.execution_schedule`，反序列化时重新验证整个调度。 |
| [NPUScheduler.cpp](../../lib/Backend/NPUScheduler.cpp)、[Backend CMake](../../lib/Backend/CMakeLists.txt)、[mini-opt](../../tools/mini-opt/mini-opt.cpp) | 注册独立 Pass，完整验证后提交属性；保留原 Command Records 和 Device Layout。 |
| [C++ 状态测试](../../test/npu/schedule/ScheduleStateTest.cpp)、[集成测试](../../test/npu/schedule/check_npu_schedule.py) | 分别验证底层状态机、完整调度与落盘属性，独立比较小型计划中的物理冲突依赖。 |

## 构建与复现

从 AI-Compiler 根目录运行，复用已有 CMake 构建。

```bash
day25_project="$PWD/code/MiniMLIRCompiler"
day25_opt="$day25_project/build/bin/mini-opt"
day25_results="$day25_project/test/e2e/results"
cmake -S "$day25_project" -B "$day25_project/build" \
  -DMINI_ONNX_PYTHON="$PWD/.venv/bin/python"
cmake --build "$day25_project/build" \
  --target mini-opt mini-sram-plan-test mini-command-plan-test mini-schedule-state-test -j 4
"$day25_opt" --help | rg -- 'npu-schedule'
for model in weighted_matmul matmul_multitile; do
  "$day25_opt" "$day25_results/$model.command.mlir" --npu-schedule --verify-each \
    -o "$day25_results/$model.scheduled.mlir"
  "$day25_opt" "$day25_results/$model.scheduled.mlir" -o /dev/null
done
ctest --test-dir "$day25_project/build" --output-on-failure -j 4
```

配置、构建、两个模型调度、重读及最终 CTest 均退出 0。`mini-opt --help`
显示 `npu-schedule`。配置仍显示已有 LLVM CMake `CMP0219` 警告。

## 实测调度

| 实际产物 | 命令 | 直接依赖 | 输出块 | SRAM 峰值 |
| --- | ---: | ---: | ---: | ---: |
| [加权 MatMul](../../test/e2e/results/weighted_matmul.scheduled.mlir) | 5 | 4 | 1 | 192 B |
| [多块 MatMul](../../test/e2e/results/matmul_multitile.scheduled.mlir) | 44 | 93 | 4 | 12 KiB |

多块模型仍按四组执行，分别占全局位置 `0–10`、`11–21`、`22–32`、`33–43`。
每组是 ZERO、三个 `LOAD A → LOAD B → MATMUL_TILE`、STORE。第一组实际记录如下：

| Sequence = ID | 命令 | K offset | 前驱 ID |
| ---: | --- | ---: | --- |
| 0 | ZERO | 0 | 无 |
| 1 | LOAD A | 0 | 无 |
| 2 | LOAD B | 0 | 无 |
| 3 | MATMUL_TILE | 0 | 0、1、2 |
| 4 | LOAD A | 32 | 1、3 |
| 5 | LOAD B | 32 | 2、3 |
| 6 | MATMUL_TILE | 32 | 3、4、5 |
| 7 | LOAD A | 64 | 4、6 |
| 8 | LOAD B | 64 | 5、6 |
| 9 | MATMUL_TILE | 64 | 6、7、8 |
| 10 | STORE | 64 | 9 |

A 地址 4096、B 地址 8192 复用；C 地址 0 持续有效直至事件 10。下一组
ZERO（ID 11）实际依赖 9、10，最后一条 STORE（ID 43）依赖 32、42。
真实模型重读及重复调度恒等，原命令、DRAM 布局、Tile、Tensor SSA 与权重
逐位保真；累计 DMA 数据量仍为 104 B / 112 KiB。

## 错误注入与回归

- M/N/K 尾块实测分别生成 15/15/11 条命令。直接比较小例子所有成对冲突，
  验证必要顺序可以在 DAG 中直接或传递得到。共享常量、重复输出 ID
  `[3,2,3]` 保持不变。双 64³ MatMul 的首个 `LOAD H`（ID 33）依赖
  生产者最终 STORE（ID 31），沿 STORE 链等待 ID 7、15、23、31 全部完成。
- 仅改调度里的 Sequence 而保留合法原命令，独立状态验证拒绝先算后搬、
  先算后清零、输入槽提前覆盖、过早 STORE、K 乱序，以及只写回 H 的
  3/4 区域就读取。删掉声明的依赖同时乱序，即使图仍无环也会失败。
- 重复 ID、Sequence 缺口、未知来源或前驱、缺失依赖、环和错误 schema
  均按预期拒绝。第一轮负例测试只因重复依赖诊断文本不同而失败，CLI
  已正确返回非零，修正诊断断言后通过。
- 直接 C++ 状态测试绕开 Day 24 顺序验证，注入重复 ZERO、C 提前释放、
  同时存活区间重叠、缺少 STORE 和部分中间结果读入等 10 个错误场景，
  全部得到对应失败诊断。
- 额外规模检查：256³ 分块生成 1664 条命令 / 3833 条直接依赖；
  `[512,1024]×[1024,640]` 生成 31360 条命令 / 72953 条直接依赖。
  两份计划均生成并重新解析通过。
- 完整 CTest **42/42**，包含原有 38 项与新增 4 项。

## Day 26 接口

[调度契约](scheduling-contract.md) 定义了全局 Sequence 和原命令的索引方式。CodeGen 将已验证的顺序与命令 payload 打包，同时引用原 DRAM 表和实际 Constant 字节。Day 25 不执行命令、不做周期估算，也不改变 Day 23/24 的地址计划。
