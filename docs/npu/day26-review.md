# Week 4 · Day 26 验收

开发起点为 `c558bb8aa94128c220d1373ee25b03f1a8320f3f`（`week4 day25`），
工作区干净。Apple ARM64、LLVM/MLIR `llvmorg-23.1.4`，基线 CTest **42/42**，
本次开发保留在工作区。

## 后端代码生成

| 文件 | 责任 |
| --- | --- |
| [工具入口](../../tools/mini-npu-codegen/mini-npu-codegen.cpp) | 解析并验证 MLIR，驱动导出，在同级临时目录重读三文件，原子发布。 |
| [VirtualISA.h](../../include/NPU/VirtualISA.h)、[VirtualISA.cpp](../../lib/CodeGen/VirtualISA.cpp) | 复用 `verifySchedule`，按全局 Sequence 用 `(matmulIndex, localEventIndex)` 定位原 Record，逐字段编码四种指令。 |
| [ProgramPackage.h](../../include/NPU/ProgramPackage.h)、[ProgramManifest.cpp](../../lib/CodeGen/ProgramManifest.cpp) | 读取 Schedule 与 DRAM 表，匹配 Target，绑定有序 I/O，计算 SHA-256 并写 Manifest。 |
| [VirtualNPUProfile.cpp](../../lib/Dialect/NPU/VirtualNPUProfile.cpp) | 复用同一个 Profile 解析函数，对供解析的原始字节计算摘要。 |
| [WeightPacker.cpp](../../lib/CodeGen/WeightPacker.cpp) | 展开 Splat，按原 FP32 位模式写 Little-endian，生成对齐的条目映射。 |
| [独立加载器](../../python/runtime/validate_package.py) | 重读并检查固定文件名、格式、引用、文件完整性和访问范围。 |
| [权重单测](../../test/npu/codegen/WeightPackingTest.cpp)、[集成测试](../../test/npu/codegen/check_codegen.py) | 验证位级打包、真实命令 Round-trip、可复现性、拒绝坏输入及事务发布。 |

Day 25 输出已确定 `什么时候执行哪个事件`。Day 26 将各事件的执行
语义和启动数据编码为外部接口。代码生成由独立工具发出文件，调度和
SRAM 规划仍由各自的 Pass 负责。字段定义见 [ISA](isa-spec.md)
及 [Program ABI](program-abi.md)。

## 实际构建与验收

从 AI-Compiler 根目录运行。复现时用新的输出路径保留仓库中的验收包。

```bash
day26_project="$PWD/code/MiniMLIRCompiler"
day26_build="$day26_project/build"
day26_results="$day26_project/test/e2e/results"
day26_profile="$day26_project/configs/virtual-npu-v1.json"
cmake -S "$day26_project" -B "$day26_build" \
  -DMINI_ONNX_PYTHON="$PWD/.venv/bin/python"
cmake --build "$day26_build" -j 4
day26_out=$(mktemp -d "$day26_build/day26-export.XXXXXX")
for model in weighted_matmul matmul_multitile; do
  "$day26_build/bin/mini-opt" "$day26_results/$model.scheduled.mlir" -o /dev/null
  "$day26_build/bin/mini-npu-codegen" \
    --input "$day26_results/$model.scheduled.mlir" --target "$day26_profile" \
    --model-metadata "$day26_project/test/e2e/models/$model.metadata.json" \
    --out-dir "$day26_out/$model"
  python3 -I -S "$day26_project/python/runtime/validate_package.py" \
    --package "$day26_out/$model"
done
ctest --test-dir "$day26_build" --output-on-failure -j 4
```

配置、构建、两份 Scheduled IR 重读、导出、隔离加载和重导出文件
逐字节比较都退出 0。配置仍显示已有的 LLVM CMake `CMP0219` 警告。
原构建目录和新建的 Release 构建目录均通过完整 CTest **46/46**，
其中新增 4 项。

## 程序包实测

| 指标 | Weighted MatMul | Multi-Tile MatMul |
| --- | ---: | ---: |
| 指令数 | 5 | 44 |
| ZERO / LOAD / MATMUL / STORE | 1 / 2 / 1 / 1 | 4 / 24 / 12 / 4 |
| 全局 Sequence | 0–4 | 0–43 |
| 直接依赖 | 4 | 93 |
| DMA 有效搬运量 | 104 B | 114,688 B |
| DRAM Arena | 160 B | 65,536 B |
| SRAM 容量 | 262,144 B | 262,144 B |
| I/O | X → Y | X → Y |
| 权重 | 1 项，48 B | 1 项，24,576 B |

Weighted 实际指令。地址为各内存空间中的字节偏移，stride 按源和目的列出。

| Seq | Opcode | 实际操作 | 前驱 |
| ---: | --- | --- | --- |
| 0 | ZERO | SRAM 0 清零 2×4 | 无 |
| 1 | DMA_LOAD_2D | DRAM 0 → SRAM 64，2 行 × 12 B，stride 12/12 | 无 |
| 2 | DMA_LOAD_2D | DRAM 64 → SRAM 128，3 行 × 16 B，stride 16/16 | 无 |
| 3 | MATMUL_TILE | A=64、B=128、C=0，M/N/K=2/4/3，累加 | 0、1、2 |
| 4 | DMA_STORE_2D | SRAM 0 → DRAM 128，2 行 × 16 B，stride 16/16 | 3 |

Multi-Tile 的四个输出区域各执行 11 条指令。集成测试通过 ISA JSON
重读逐条对照完整 Command 与 Schedule，保留 DMA 的双 stride、
SRAM 复用和四组 K offset `0, 32, 64` 的累加顺序。

| 实际文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| [Weighted program.json](../../test/e2e/results/compiled_weighted/program.json) | 2036 | `f7bdd12f78f93deb88ef4dd9619f7afb1fabc28ecb9b18be112ff8df5155925b` |
| [Weighted weights.bin](../../test/e2e/results/compiled_weighted/weights.bin) | 48 | `4c13ffa392193294154aae9b1466fe8ffaf97b811205d0f919334e614ca0b514` |
| [Weighted manifest.json](../../test/e2e/results/compiled_weighted/manifest.json) | 1687 | `de523034cb2d9da7fa86a187aaa524043c6873b7167825bd154d501a7dd00b90` |
| [Multi-Tile program.json](../../test/e2e/results/compiled_multitile/program.json) | 18571 | `0c5cae6038d7f78cb503bf1b1aa0faa87d26e542afe2f47633b07fedac7ec7a2` |
| [Multi-Tile weights.bin](../../test/e2e/results/compiled_multitile/weights.bin) | 24576 | `ef1a2210d721fabd952767b51f6b470700d2d3814cf947e73d7379e8aa3db693` |
| [Multi-Tile manifest.json](../../test/e2e/results/compiled_multitile/manifest.json) | 1723 | `135e4ac1a73cbf12cdc4b8bf488db4fc8dd44b1a0897d050ea358feb76ac61e1` |

两份 Manifest 的 Target Profile SHA-256 均为
`86fc96185eebf9fb38506b80cfe0188538dc39329cbd933e8b3f6e11a5a16f9a`。

## 正例、负例与交接

- C++ 测试直接比较正负零、正负次正规数、带 payload 的 NaN、正负无穷、
  Splat 展开、多个常量的 Padding 和第二个常量的映射。将记录的大小改错
  后，打包器拒绝。
- 两个真实模型的权重字节与 MLIR 常量一致。共享 Splat 只写一份，
  连续 MatMul 的局部事件从 0 重新开始，保留第二组 LOAD 对前组 STORE
  的依赖，以及输出 Tensor ID `[3,2,3]`。双输入模型可加载零字节权重。
- 重新导出三个文件时逐字节相同。反转 Schedule Attribute 的记录存储顺序
  后导出仍相同。
- 坏 Scheduled IR 被 MLIR 或 CodeGen 拒绝，覆盖缺失或不一致的 Layout、
  Schedule，目标容量或原生 Tile 变化，以及 Metadata 形状、顺序、dtype
  错误。初次运行发现测试夹具的十六进制打印及诊断匹配需要调整，修正
  后全部通过。
- 独立加载器拒绝版本、Opcode、字段、重复 ID、Sequence、依赖、Tensor
  引用、I/O 遗漏、SRAM/DRAM 越界、Stride 与路径错误。修改一个字节可触发
  摘要失败。重新计算摘要后，结构破坏及未完整展开的真实 Splat 仍被拒绝。
- 测试已存在的输出目录、悬空符号链接、校验失败及发布时路径竞争，原路径
  保留且失败不遗留临时包。

Day 27 可直接使用 [加载接口](program-abi.md#day-27-加载流程) 建立内存、
绑定输入和解释四种 Opcode。Day 26 验收范围是程序编码、字节保真和加载
合法性。Simulator 执行、FP32 舍入行为和 ONNX Runtime 数值比较待后续交付。
导出和加载当前在内存中保留文件内容，规模过大时受主机内存限制。
