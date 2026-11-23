# Week 4 · Day 27 验收

开工时 HEAD 为 `91a96b74884571dc7c149b070f4455e8ef397221`（`week4 day26`），
工作区干净。Apple ARM64、LLVM/MLIR `llvmorg-23.1.4`、Python 3.11、
NumPy 2.4.6。原有 CTest **46/46**，完成后 **53/53**。

## 代码入口与职责

| 位置 | 职责 |
| --- | --- |
| [Day 26 加载器](../../python/runtime/validate_package.py) | 复用三文件 ABI、摘要、指令形状与地址检查。 |
| [tensor_io.py](../../runtime/virtual_npu/tensor_io.py) | 根据有序 I/O 与权重映射初始化 DRAM，导出命名 FP32 NPZ。 |
| [memory.py](../../runtime/virtual_npu/memory.py) | 维护物理字节与初始化标记，先检查全部源行，再执行二维同步 DMA。 |
| [matrix_unit.py](../../runtime/virtual_npu/matrix_unit.py) | 对 SRAM 中有效尺寸的 A、B、C，按 K 顺序分开执行 FP32 乘法和加法。 |
| [interpreter.py](../../runtime/virtual_npu/interpreter.py) | 顺序执行四种指令，检查活跃 Buffer、K 进度、STORE 区间与完整 Tensor 可用性。 |
| [run.py](../../runtime/virtual_npu/run.py) | CLI，写入逐指令 Trace 与输出，失败时给出阶段及指令位置。 |
| [CTest 注册](../../test/CMakeLists.txt) | 接入内存、解释器、双模型端到端、负例、尾块及连续 MatMul。 |

## 构建和完整复现

从 AI-Compiler 根目录运行，使用已有 LLVM/MLIR 配置：

```bash
day27_workspace="$PWD"
day27_project="$day27_workspace/code/MiniMLIRCompiler"
day27_python="$day27_workspace/.venv/bin/python"
cmake -S "$day27_project" -B "$day27_project/build" -DMINI_ONNX_PYTHON="$day27_python"
cmake --build "$day27_project/build" -j 4
ctest --test-dir "$day27_project/build" --output-on-failure -j 4

cd "$day27_project"
day27_out=$(mktemp -d "$PWD/build/day27.XXXXXX")
day27_target="$PWD/configs/virtual-npu-v1.json"
for model in weighted_matmul matmul_multitile; do
  "$day27_python" python/onnx_importer/importer.py \
    --input "test/e2e/models/$model.onnx" --output "$day27_out/$model.mini.mlir"
  build/bin/mini-opt "$day27_out/$model.mini.mlir" --verify-each \
    "--pass-pipeline=builtin.module(mini-npu-compile,npu-tiling{config=$day27_target},npu-sram-plan{config=$day27_target},npu-insert-dma{config=$day27_target},npu-schedule)" \
    -o "$day27_out/$model.scheduled.mlir"
  build/bin/mini-npu-codegen --input "$day27_out/$model.scheduled.mlir" \
    --target "$day27_target" --model-metadata "test/e2e/models/$model.metadata.json" \
    --out-dir "$day27_out/$model.bundle"
  "$day27_python" -B -m runtime.virtual_npu.run --bundle "$day27_out/$model.bundle" \
    --inputs "test/e2e/inputs/$model.npz" --outputs "$day27_out/$model.actual_outputs.npz" \
    --trace "$day27_out/$model.trace.json"
  "$day27_python" python/e2e/compare_outputs.py \
    --reference "test/e2e/reference/$model.npz" --actual "$day27_out/$model.actual_outputs.npz" \
    > "$day27_out/$model.comparison.json"
done
```

配置、构建、两份真实模型重新编译、执行与比较均退出 0。配置显示既有
LLVM CMake `CMP0219` 警告。集成测试还逐字节核对重新编译的三个文件和
仓库里的 Day 26 程序包。数值基准使用仓库中由 ONNX Runtime
CPUExecutionProvider 生成的参考 NPZ，容差为原有 `rtol=1e-4` 和
`atol=1e-5`。

## 真实执行结果

| 指标 | Weighted | Multi-Tile |
| --- | ---: | ---: |
| Y 的 Shape | `[2,4]` | `[64,64]` |
| 执行指令数 | 5 | 44 |
| ZERO / LOAD / MATMUL / STORE | 1 / 2 / 1 / 1 | 4 / 24 / 12 / 4 |
| 实际 DMA 字节 | 104 | 114,688 |
| 最大绝对误差 | 0 | `7.62939453125e-06` |
| 最大相对误差 | 0 | `0.0006840870652792231` |
| 不匹配元素 | 0 / 8 | 0 / 4,096 |

Weighted FP32 Y 实测为 `[[0, 1.5, 3, -3.5], [5, -3, -5, 9]]`。
多块模型按顺序写出 `(0,0)`、`(0,32)`、`(32,0)`、`(32,32)`
四块，每块的 K 起点为 `0,32,64`。第一个 C 从 Seq 0 的 ZERO 开始，
经 Seq 3、6、9 累加，Seq 10 写回。第二段 A 从 DRAM `128` 搬 32 行，
每行 128 B，源行跨度 384 B，写入 SRAM `4096`，目的行跨度 128 B。

| 模型 | 可复核的产物 |
| --- | --- |
| Weighted | [Trace](../../test/e2e/results/day27/weighted_matmul.trace.json) · [输出 NPZ](../../test/e2e/results/day27/weighted_matmul.actual_outputs.npz) · [比较报告](../../test/e2e/results/day27/weighted_matmul.comparison.json) |
| Multi-Tile | [Trace](../../test/e2e/results/day27/matmul_multitile.trace.json) · [输出 NPZ](../../test/e2e/results/day27/matmul_multitile.actual_outputs.npz) · [比较报告](../../test/e2e/results/day27/matmul_multitile.comparison.json) |

## 覆盖与剩余边界

- 内存测试覆盖读写、越界、失效、双 Stride 及失败时零部分写入。
  手工 2×3×2 指令测试精确得到 `[[58,64],[139,154]]`。另用构造数值
  验证 K 次序和两次舍入。
- 负例测试验证 22 个坏程序包或执行状态、3 个错误输入绑定。覆盖非法
  Schema、Opcode、边界、Stride、Tensor、权重截断，以及重签摘要后仍能
  在运行期间发现的漏 LOAD、漏 ZERO、错误 Buffer、错误 K、提前 STORE
  与写回缺块。漏掉第二段 K 的 A LOAD 后，Trace 在修改后程序的
  Seq 5、Tile 1 的 MATMUL 报告缺少活跃 A/B。
- 进阶测试执行 M、N、K 单轴尾块，最终有效尺寸为 1。
  连续 MatMul 测试完成 64 条指令，后组从 DRAM 读取前组完整写回的
  中间结果，两个输出名指向同一个 Tensor 时两项都保留。
- 本次 7 个新 CTest 测试组与 46 个原有测试均通过。模拟器是单矩阵单元、
  同步 DMA 的功能模型。验证覆盖当前静态二维 FP32 MatMul，周期、实机性能、
  Coral ISA 及其他算子的执行属于其他验证范围。

Day 28 可在相同 Bundle、固定输入和 [Simulator 契约](simulator-contract.md)
之上复用逐指令 Trace 和比较器。调试时可先按 Sequence 找到实际发生
差异的 Tile，再沿其 DRAM 窗口和 SRAM Buffer 回溯编译阶段。
