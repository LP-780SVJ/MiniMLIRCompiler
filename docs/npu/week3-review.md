# Week 3 · Day 21 集成验收

工作依据是 Week 3 总计划的 Day 21 章节；本地未找到单独的 `day21.md`。开发前工作区干净，HEAD 为 `6c93c8aa25c0e6f8fabb7eeb6109d4073dfeeede`，LLVM/MLIR 为 `llvmorg-23.1.4`，基线 CTest 22/22 通过。

## 第三周交付  

| 内容 | 结果 |
| --- | --- |
| Day 15–17 研究与硬件契约 | 记录 TVM、TPU-MLIR、Coral，与 [Virtual NPU v1](hardware-spec.md)。Coral 独立 MPACT G1 已通过；集成 G2/G3 未执行。 |
| Day 18–20 图与参考 | [导入契约](onnx-import-contract.md)、Mini/NPU MatMul、固定权重与多节点导入已实现。六组 CPU 参考与比较器见 [Day 20](day20-review.md)。 |
| Day 21 目标合法化 | [MiniToNPU](../../lib/Conversion/MiniToNPU.cpp) 与独立 NPU Pipeline 已打通 ONNX 到 NPU Compute IR。原 MiniToLinalg 路线回归通过。 |

## 实现与阶段接口

- [MiniNPULowering.h](../../include/Mini/MiniNPULowering.h) 暴露 Pass 工厂和注册入口；[Conversion CMake](../../lib/Conversion/CMakeLists.txt) 构建独立的 `MLIRMiniToNPU` 库。第一次增量配置因 LLVM 要求同目录源文件由目标申明而退出 1，给两目标标注 `PARTIAL_SOURCES_INTENDED` 后配置、编译和链接均成功。
- [MatMul Pattern](../../lib/Conversion/MiniToNPU.cpp) 从 adaptor 读取当前转换映射中的输入，复用位置和结果类型，创建 `npu.matmul` 并将旧结果的使用迁移到新结果。类型一致，首版不需要 TypeConverter。
- Module 级 `applyFullConversion` 只允许模块、函数、返回、目标 MatMul 与静态 FP32 Dense Tensor 常量。合法容器内的操作仍逐个检查。失败由 `signalPassFailure()` 传播，驱动退出非零。
- [mini-opt](../../tools/mini-opt/mini-opt.cpp) 注册独立的 `mini-to-npu` 和 `mini-npu-compile`。后者在转换前做函数级 canonicalize，由 MLIR 默认机制验证结果。

## 复现命令

在 AI-Compiler 根目录，用现有 CMake 构建目录：

```bash
mini_project="$PWD/code/MiniMLIRCompiler"
cmake -S "$mini_project" -B "$mini_project/build" \
  -DMINI_ONNX_PYTHON="$PWD/.venv/bin/python"
cmake --build "$mini_project/build" --target mini-opt -j 4

for model in weighted_matmul matmul_chain; do
  .venv/bin/python "$mini_project/python/onnx_importer/importer.py" \
    --input "$mini_project/test/e2e/models/$model.onnx" \
    --output "$mini_project/test/e2e/results/$model.mini.mlir"
  "$mini_project/build/bin/mini-opt" "$mini_project/test/e2e/results/$model.mini.mlir" \
    --pass-pipeline='builtin.module(mini-npu-compile)' \
    -o "$mini_project/test/e2e/results/$model.npu.mlir"
  "$mini_project/build/bin/mini-opt" "$mini_project/test/e2e/results/$model.npu.mlir" -o /dev/null
done
ctest --test-dir "$mini_project/build" --output-on-failure -j 4
```

## 实测

| 检查 | 结果 |
| --- | --- |
| [转换单测](../../test/npu/conversion/check-mini-to-npu.sh) | 两条相连 MatMul 转换通过，中间结果同时被下一条 MatMul 和 return 使用，返回顺序保持；已合法的输出可重新进入 Pass。 |
| 目标拒绝 | 无规则的 `mini.relu`、`arith.addf`、标量 `arith.constant` 和 `func.call` 被拒绝。 |
| 源 IR 拒绝 | 错误 K、BF16、动态 Shape 在源 IR 验证阶段被拒绝，未进入改写。 |
| [ONNX 正例](../../test/e2e/check_onnx_to_npu_ir.py) | 两输入单 MatMul、加权 MatMul、双 MatMul、多 Tile MatMul 的导入、转换、重解析均退出 0。转换前后核对函数、类型、SSA 与返回值；将原始 ONNX 权重与 NPU IR Dense bytes 逐位比较。 |
| ONNX 负例 | Add→ReLU 与 MatMul→ReLU 导入及验证退出 0，NPU Pipeline 分别因 `mini.add`、`mini.relu` 退出 1。合法 ONNX Sigmoid 在导入阶段以 `E_UNSUPPORTED_OP` 退出 1。均无失败阶段的输出文件。 |
| 回归 | 25/25，包括原有 22 项、新增 3 项。 |

可复查的输出：[加权 MatMul](../../test/e2e/results/weighted_matmul.npu.mlir)、[连续 MatMul](../../test/e2e/results/matmul_chain.npu.mlir)。

## 第四周交接

后端以已验证的 `npu.matmul` 为输入。它表示整个逻辑 MatMul，仍可从 Tensor Type 读出 M/N/K，从 SSA 边区分运行参数、常量和前序计算结果。对于现有多 Tile 数据 `[64,96] × [96,64]`，原生 `32×32×32` 块需要分别处理 2 个 M 块、2 个 N 块和 3 个 K 块。每个输出块先初始化，再累加 3 个 K 块贡献。地址、DMA 与相互依赖留给后续阶段。今天未执行 Tiling、Simulator 或 Coral 桥接，数值验收将在模拟器产出实际结果后使用 Day 20 的参考数据。
