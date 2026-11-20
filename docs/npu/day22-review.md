# Week 4 · Day 22 验收

基于 Day 21 commit `dfce2ce6f06982423a3e6ea34d427a1836dfcb2c`，沿用 LLVM/MLIR `llvmorg-23.1.4`。开工前有一处已有的 Week 3 验收文档空格改动；本次未修改该文件。基线 CTest 25/25 通过。

## 实现与职责

| 位置 | 职责 |
| --- | --- |
| [NPUOps.td](../../include/NPU/NPUOps.td)、[NPUOps.cpp](../../lib/Dialect/NPU/NPUOps.cpp) | 定义 `npu.tiled_matmul`，复用原有 MatMul 类型检查，再验证静态计划的结构与覆盖。 |
| [TilePlan.h](../../include/NPU/TilePlan.h)、[TilePlan.cpp](../../lib/Dialect/NPU/TilePlan.cpp) | 定义记录结构、安全计算块数与边界，提供无需 MLIR Op 的索引式计划。 |
| [TilePlanAttrs.h](../../include/NPU/TilePlanAttrs.h)、[TilePlanAttrs.cpp](../../lib/Dialect/NPU/TilePlanAttrs.cpp) | 编码与解码 MLIR 字典，供 Verifier 和后端复用。 |
| [NPUTiling.h](../../include/NPU/NPUTiling.h)、[NPUTiling.cpp](../../lib/Backend/NPUTiling.cpp) | 加载目标配置，检查阶段输入，收集 MatMul，用实际 MLIR Rewriter 替换旧结果。 |
| [Dialect CMake](../../lib/Dialect/NPU/CMakeLists.txt)、[Backend CMake](../../lib/Backend/CMakeLists.txt)、[mini-opt](../../tools/mini-opt/mini-opt.cpp) | 把共享计划逻辑编入 Dialect 库，新增并链接 `MLIRNPUTiling`，注册独立 Pass。 |
| [CTest](../../test/CMakeLists.txt)、[分块测试](../../test/npu/tiling/check_npu_tiling.py) | 验证手写 IR、Pass 输出、错误注入，以及真实 ONNX 到 Tile IR 的集成。 |

## 实际命令

从 AI-Compiler 根目录执行（复用已配置的 LLVM/MLIR）：

```bash
day22_project="$PWD/code/MiniMLIRCompiler"
day22_opt="$day22_project/build/bin/mini-opt"
day22_config="$day22_project/configs/virtual-npu-v1.json"
day22_results="$day22_project/test/e2e/results"
cmake -S "$day22_project" -B "$day22_project/build" \
  -DMINI_ONNX_PYTHON="$PWD/.venv/bin/python"
cmake --build "$day22_project/build" --target mini-opt -j 4
"$day22_opt" --help | rg -- 'npu-tiling'

for day22_model in weighted_matmul matmul_multitile; do
  if [ "$day22_model" = weighted_matmul ]; then
    day22_prefix="$day22_results/day22"
  else
    day22_prefix="$day22_project/build/day22-multitile"
  fi
  .venv/bin/python "$day22_project/python/onnx_importer/importer.py" \
    --input "$day22_project/test/e2e/models/$day22_model.onnx" --output "$day22_prefix.mini.mlir"
  "$day22_opt" "$day22_prefix.mini.mlir" --pass-pipeline='builtin.module(mini-npu-compile)' \
    -o "$day22_prefix.npu.mlir"
  "$day22_opt" "$day22_prefix.npu.mlir" --npu-tiling="config=$day22_config" \
    --verify-each -o "$day22_results/$day22_model.tiled.mlir"
  "$day22_opt" "$day22_results/$day22_model.tiled.mlir" -o /dev/null
done
ctest --test-dir "$day22_project/build" --output-on-failure -j 4
```

首次构建退出 1：新 Op 的固有属性使 TableGen 生成 `BytecodeOpInterface`，`NPUOps.h` 未直接包含声明。补头文件后成功构建两个库和 `mini-opt`，最终配置、增量构建、两个模型命令和 CTest 都退出 0。LLVM CMake 仍提示已有的 `CMP0219` 警告。

## 实测结果

| 正例 | 实际记录数 / 输出块数 | 有效边界与依赖 |
| --- | --- | --- |
| 小型 `[2,3]×[3,4]` | 1 / 1 | 有效尺寸 `(2,4,3)`，first 和 last 同为真 |
| 原生满块 | 1 / 1 | `(32,32,32)` |
| `[64,96]×[96,64]` | 12 / 4 | 每组 K offset 按 `0,32,64` |
| M 尾 `[65,7]×[7,9]` | 3 / 3 | M offset `0,32,64`；M size `32,32,1` |
| N 尾 `[7,9]×[9,65]` | 3 / 3 | N offset `0,32,64`；N size `32,32,1` |
| K 尾 `[32,65]×[65,32]` | 3 / 1 | K offset `0,32,64`；K size `32,32,1` |
| 连续 MatMul，多处使用 | 每条 1 / 1 | 第二条读取第一条的结果，`return %1, %0` 保持 |
| 替代配置 `(2,3,4)` | 27 / 9 | `(M,N,K)=(5,7,9)`；用原配置重新进入时拒绝 |

正例均核对原函数签名与 SSA，并由 C++ Verifier 和独立覆盖检查验证。两组真实 ONNX 的权重与输出 IR 中的 Dense 原始字节逐位一致，输出可重新解析。保存了 [加权 Mini IR](../../test/e2e/results/day22.mini.mlir)、[加权 Compute IR](../../test/e2e/results/day22.npu.mlir)、[加权 Tile IR](../../test/e2e/results/weighted_matmul.tiled.mlir) 和 [多块 Tile IR](../../test/e2e/results/matmul_multitile.tiled.mlir)。

下表按实测打印次序记录 12 条描述，每条 `(m_size,n_size,k_size)=(32,32,32)`：

| ID | `(m_offset,n_offset,k_offset)` | first | last |
| --- | --- | --- | --- |
| 0 | (0,0,0) | true | false |
| 0 | (0,0,32) | false | false |
| 0 | (0,0,64) | false | true |
| 1 | (0,32,0) | true | false |
| 1 | (0,32,32) | false | false |
| 1 | (0,32,64) | false | true |
| 2 | (32,0,0) | true | false |
| 2 | (32,0,32) | false | false |
| 2 | (32,0,64) | false | true |
| 3 | (32,32,0) | true | false |
| 3 | (32,32,32) | false | false |
| 3 | (32,32,64) | false | true |

直接运行的错误注入均退出 1：缺块或重复块、M/N/K 越界、K 空洞与重叠、错误 ID 与归约标记、字段损坏、配置缺失/无效、越过 65,536 条记录上限，以及残留 Mini Op。手写非法 K 和缺失 `tiles` 的测试核对了预期诊断。完整回归 **30/30**，包含基线 25 项及新增 5 项。

## 与通用 MLIR 分块的关系

LLVM/MLIR 23.1.4 的 `TilingInterface` 将迭代域、迭代器种类、局部计算构造和结果定位交给具体算子。Linalg 通过外部模型实现该接口，SCF 的 `tileUsingSCF` 再据此构造循环。我们的第一版只需要一个静态 MatMul 计划，使用相同的职责分离思路，将坐标计算与 MLIR 改写分开，计划留在纯 Tensor Op 上以供下游资源分析。

## 交接

Day 23 从 [计划契约](tiling-design.md) 读取 `TiledMatMulOp::getTiles()` 和 `decodeTileDesc`，并核对目标配置。地址、对齐后分配和生命周期仍待实现。Day 22 验证的是计划正确性，尚未生成执行命令或运行数值模拟；Day 20 的参考输出继续留待模拟器验收。
