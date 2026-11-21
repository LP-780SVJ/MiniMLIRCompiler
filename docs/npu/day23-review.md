# Week 4 · Day 23 验收

开发基线为 `494dbcd56b1f07d4a37197fbd41e186aa7cc88ec`，开工前 Git 工作区干净。复用 LLVM/MLIR `llvmorg-23.1.4` 和已有 Python 环境，基线 CTest 30/30 通过。

## 源码与职责

| 位置 | 责任 |
| --- | --- |
| [MemoryPlan.h](../../include/NPU/MemoryPlan.h)、[MemoryPlan.cpp](../../lib/Dialect/NPU/MemoryPlan.cpp) | 从 Tile 的有效尺寸和逻辑事件推导需求，按活跃区间做 First-Fit，独立验证分配与统计。 |
| [MemoryPlanAttrs](../../lib/Dialect/NPU/MemoryPlanAttrs.cpp)、[NPU ODS](../../include/NPU/NPUOps.td)、[Op Verifier](../../lib/Dialect/NPU/NPUOps.cpp) | 把分配和 Tile Binding 持久化到 `npu.sram_planned_matmul`，重读时检查 Tile 覆盖与 SRAM 计划。 |
| [VirtualNPUProfile](../../lib/Dialect/NPU/VirtualNPUProfile.cpp)、[SRAMPlanner](../../lib/Backend/SRAMPlanner.cpp) | Tiling 和 SRAM Planner 共用目标与计算配置检查。新的 Module Pass 核对 SRAM 和 DMA 前提，先完成全部规划，再用 IRRewriter 保留 Tensor SSA。 |
| [Backend CMake](../../lib/Backend/CMakeLists.txt)、[mini-opt](../../tools/mini-opt/mini-opt.cpp) | 构建并链接独立的 `MLIRNPUSRAMPlanner`，复用已有 Dialect 与 PassManager。 |
| [CTest](../../test/CMakeLists.txt)、[C++ 分配器测试](../../test/npu/sram/MemoryPlanTest.cpp)、[IR 集成测试](../../test/npu/sram/check_sram_plan.py) | 检查地址、逻辑生存期、统计、失败诊断和真实 ONNX 权重逐位保真。 |

## 构建和复现

从 AI-Compiler 根目录运行，复用已有 CMake 构建：

```bash
day23_project="$PWD/code/MiniMLIRCompiler"
day23_opt="$day23_project/build/bin/mini-opt"
day23_config="$day23_project/configs/virtual-npu-v1.json"
day23_results="$day23_project/test/e2e/results"

cmake -S "$day23_project" -B "$day23_project/build" \
  -DMINI_ONNX_PYTHON="$PWD/.venv/bin/python"
cmake --build "$day23_project/build" --target mini-opt mini-sram-plan-test -j 4
"$day23_opt" --help | rg -- 'npu-sram-plan'
for model in weighted_matmul matmul_multitile; do
  "$day23_opt" "$day23_results/$model.tiled.mlir" \
    --npu-sram-plan="config=$day23_config" --verify-each \
    -o "$day23_results/$model.sram.mlir"
  "$day23_opt" "$day23_results/$model.sram.mlir" -o /dev/null
done
ctest --test-dir "$day23_project/build" --output-on-failure -j 4
```

配置、TableGen、编译、链接和两个模型的规划及重读均退出 0。
`mini-opt --help` 显示 `--npu-sram-plan`。配置时仍有已有的 LLVM
`CMP0219` 警告。

## 实测布局

| 模型 | Tile / 输出块 | 逻辑 Buffer 数 | 峰值活跃容量 | 排他地址高水位 |
| --- | --- | ---: | ---: | ---: |
| [Weighted MatMul](../../test/e2e/results/weighted_matmul.sram.mlir) | 1 / 1 | 3 | 192 B | 192 B |
| [Multi-Tile MatMul](../../test/e2e/results/matmul_multitile.sram.mlir) | 12 / 4 | 28 | 12,288 B | 12,288 B |

小模型实际布局是：

| Buffer ID / 角色 | 地址 | 原始容量 | 对齐后容量 | 活跃区间 |
| --- | ---: | ---: | ---: | --- |
| 0 / C | 0 | 32 B | 64 B | `[0,5)` |
| 1 / A | 64 | 24 B | 64 B | `[1,4)` |
| 2 / B | 128 | 48 B | 64 B | `[2,4)` |

多块模型的每个有效块均为 32³。其第一组的三个 Binding 分别为
`(A1,B2,C0)`、`(A3,B4,C0)`、`(A5,B6,C0)`（数字是 ID）。
C0 在地址 0 活到事件 11，三个 A 依次使用地址 4096，三个 B 依次使用
地址 8192。A 的区间为 `[1,4)`、`[4,7)`、`[7,10)`；B 的区间为
`[2,4)`、`[5,7)`、`[8,10)`。下一组从事件 11 开始，C7 再使用地址 0。
两个模型的 Planned IR 重新解析通过；测试逐位比较来自原 ONNX 的权重，
并核对原 Tensor 类型、操作数、返回值与 Tile Plan。

## 其他实测与失败证据

| 检查项 | 实际结果 |
| --- | --- |
| M/N/K 单轴尾块 | 均通过。M 和 N 尾场景各有 3 个输出块，峰值均为 2304 B；K 尾场景有 3 条记录、一个输出块，峰值 12 KiB。 |
| 64³ 同时驻留 | 使用替代原生配置，单块占 48 KiB。 |
| 256³ 同时驻留与分块 | 单块需求 786,432 B，超过 262,144 B，被拒绝。默认 32³ 上限下，512 条计算、64 个输出块，峰值 12 KiB。 |
| 连续 MatMul | 串行使用相同的地址空间；第二条仍读取第一条结果，两个返回值顺序不变。 |
| 高水位独立检查 | 保留 1 B 后，第一可用地址对齐到 64，峰值仍为 192 B，高水位为 256 B。把合法布局整体挪高 64 B 并更新高水位，Verifier 仍接受。 |
| 非法布局 | 对齐、边界、尺寸、部分或完全重叠、重复 ID、无效 Binding、错误统计、错误调度契约等注入均被拒绝。篡改 C 使其在写回前结束，诊断 `buffer lifetime does not match required events`；将下一段 A 放进存活 C 的地址，诊断 `simultaneously live buffers overlap`。 |
| 配置及阶段失败 | 缺配置、非法 JSON、大小溢出、错误模式、Tile/Profile 不匹配、未分块输入均被拒绝。8 KiB 计划满块时退出 1。第二条 MatMul 失败的 IR dump 显示第一条仍是原来的 `npu.tiled_matmul`。 |
| 容量与碎片区分 | 分配器独立测试中，活跃需求 256 B、Arena 256 B，但两个 64 B 空洞不连续，First-Fit 无法放下 128 B。将同一测试的 Arena 缩到 192 B，则直接诊断活跃容量下界不足。 |
| CTest | 34/34 通过，包含原有 30 项和新增 4 项。 |

## Day 24 交接

读取 [SRAM 契约](sram-planning.md) 中的 `tile_bindings`，在同一 MatMul 的
`allocations` 中取得 SRAM 地址。坐标来自 Tile Plan，SRAM 采用紧凑 Row-major，
DRAM 基址由后续生成。保留单缓冲的 LOAD、COMPUTE、STORE 顺序以及跨 K 的 C
存活约束。Day 23 只验证资源布局，尚未执行 DMA 或模拟器数值测试。
