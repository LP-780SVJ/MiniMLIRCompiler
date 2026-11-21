# Week 4 · Day 24 验收

开发基线为 `7b545474b76111dd2b0cbb031d29d52bdb6fe06a`，Git 工作区干净，LLVM/MLIR 为 `llvmorg-23.1.4`，原有 CTest 34/34 通过。沿用真实 Day 23 的 SRAM IR。

## 修改与职责

| 位置 | 职责 |
| --- | --- |
| [DRAMLayout.h](../../include/NPU/DRAMLayout.h)、[DRAMLayout.cpp](../../lib/Dialect/NPU/DRAMLayout.cpp) | 生成模块级 Tensor ID、DRAM 字节区间、来源和有序 I/O 表。 |
| [CommandPlan.h](../../include/NPU/CommandPlan.h)、[CommandPlan.cpp](../../lib/Dialect/NPU/CommandPlan.cpp) | 强类型 payload、受检二维地址计算、串行事件生成和验证。 |
| [CommandPlanAttrs](../../lib/Dialect/NPU/CommandPlanAttrs.cpp)、[NPU ODS](../../include/NPU/NPUOps.td)、[Op Verifier](../../lib/Dialect/NPU/NPUOps.cpp)、[模块验证](../../lib/Dialect/NPU/NPUDialect.cpp) | 将布局与命令序列保存在 IR，验证字段、绑定、范围和重读的一致性。 |
| [DMAInsertion.cpp](../../lib/Backend/DMAInsertion.cpp)、[VirtualNPUProfile](../../lib/Dialect/NPU/VirtualNPUProfile.cpp)、[SRAMPlanner](../../lib/Backend/SRAMPlanner.cpp) | 共用 SRAM profile 验证，DMA Pass 先规划全模块再统一替换 Op。 |
| [Backend CMake](../../lib/Backend/CMakeLists.txt)、[mini-opt](../../tools/mini-opt/mini-opt.cpp)、[CTest](../../test/CMakeLists.txt) | 构建独立 `MLIRNPUDMAInsertion`，注册 `npu-insert-dma`、命令地址算术和集成测试。 |

## 实际执行

```bash
day24_project="$PWD/code/MiniMLIRCompiler"
day24_opt="$day24_project/build/bin/mini-opt"
day24_config="$day24_project/configs/virtual-npu-v1.json"
day24_results="$day24_project/test/e2e/results"
cmake -S "$day24_project" -B "$day24_project/build" \
  -DMINI_ONNX_PYTHON="$PWD/.venv/bin/python"
cmake --build "$day24_project/build" --target mini-opt mini-sram-plan-test mini-command-plan-test -j 4
"$day24_opt" --help | rg -- 'npu-insert-dma'
for model in weighted_matmul matmul_multitile; do
  "$day24_opt" "$day24_results/$model.sram.mlir" \
    --npu-insert-dma="config=$day24_config" --verify-each \
    -o "$day24_results/$model.command.mlir"
  "$day24_opt" "$day24_results/$model.command.mlir" -o /dev/null
done
ctest --test-dir "$day24_project/build" --output-on-failure -j 4
```

首次 C++ 编译退出 1：泛型列表推导混用了 `Value` 与 `OpResult`，且 `llvm::enumerate` 无法推导裸花括号初始化列表。显式使用 `Value` 和有类型的数组后编译、链接成功。CMake 仍输出已有的 `CMP0219` 警告。最终配置、构建、命令生成和重读均退出 0。

## 两个真实产物

| 模型 | DRAM A / B / C 基址（B） | DRAM 排他末端 | ZERO / LOAD / MATMUL / STORE | 有效 DMA 数据 |
| --- | --- | ---: | --- | ---: |
| [Weighted](../../test/e2e/results/weighted_matmul.command.mlir) | 0 / 64 / 128 | 160 B | 1 / 2 / 1 / 1，合计 5 | 104 B |
| [Multi-Tile](../../test/e2e/results/matmul_multitile.command.mlir) | 0 / 24576 / 49152 | 65536 B | 4 / 24 / 12 / 4，合计 44 | 114688 B = 112 KiB |

加权模型的两个 LOAD 分别搬 `2×12 B` 和 `3×16 B`，STORE 搬 `2×16 B`。多块模型每次搬运 4096 B，有 24 次读和 4 次写。该数据量包含重复加载的字节，不等于 Tensor 区间大小或硬件计数器实测。

| 多块事件 | 实际记录 | Rows / RowBytes | Source / Destination stride |
| --- | --- | --- | --- |
| 4，Tile 1 | A，Tensor 0 / Buffer 3，DRAM 128 → SRAM 4096 | 32 / 128 | 384 / 128 |
| 16，Tile 4 | B，Tensor 1 / Buffer 11，DRAM 32896 → SRAM 8192 | 32 / 128 | 256 / 128 |
| 43，Tile 11 | C，Tensor 2 / Buffer 21，SRAM 0 → DRAM 57472 | 32 / 128 | 128 / 256 |

## 验证与交接

- 独立 Python 检查逐条核对所有命令的地址、有效尺寸、绑定、事件与流量。覆盖 M/N/K 尾块、跨 MatMul 写回与回读、共享权重单 ID、输出顺序与重复输出 ID，以及从原 ONNX 到命令 IR 的权重逐位一致。
- 错误注入验证零行/行宽、短或不正确的 stride、未知或错误 Tensor/Buffer、跨所属 Tensor/Buffer、末行越界、i64 溢出、非法 Opcode/字段、漏 K 段、错误 ZERO、提前 STORE、错误尺寸、配置缺失/冲突和未规划输入，均按预期非零退出。模块内第二条 MatMul 预检失败时，两条原 Op 和模块属性均未被部分改写。
- 直接 C++ 算术测试检查最大合法 i64 末端和乘加溢出；完整 CTest **38/38**（基线 34 项、新增 4 项）。
- Day 25 的 [命令与事件契约](dma-contract.md) 已记录。尚未执行设备指令或模拟器，命令条数和流量不能代替数值验证。
