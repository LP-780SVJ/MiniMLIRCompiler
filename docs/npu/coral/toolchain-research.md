# Day 16 · Coral Compiler、IREE Runtime 与 MPACT

调查日期 2026-11-16。以下基于固定提交的官方源码、依赖声明和本机版本探测。G1–G3 的结果待执行，证据要求见 [兼容性矩阵](isa-compatibility-matrix.md)。

## 1. 仓库与依赖版本

| 对象 | 固定版本 | 证据 |
| --- | --- | --- |
| Coral Compiler | `770dc9332dffed8ff91d8cface7e9d176227d912` | [官方提交](https://github.com/google-coral/coralnpu-compiler/commit/770dc9332dffed8ff91d8cface7e9d176227d912) |
| 独立 Coral MPACT | `704b28ce126debd2d951f9c3b36704080d75dc60` | [官方提交](https://github.com/google-coral/coralnpu-mpact/commit/704b28ce126debd2d951f9c3b36704080d75dc60) |
| IREE gitlink | `89ece87b89852100c0a4ac3d515e14842d090967` | [固定 third_party 目录](https://github.com/google-coral/coralnpu-compiler/tree/770dc9332dffed8ff91d8cface7e9d176227d912/third_party) |
| LLVM gitlink | `32d702efa29d0fdd02cff7b79dea8fdbbcf059d4` | [固定 third_party 目录](https://github.com/google-coral/coralnpu-compiler/tree/770dc9332dffed8ff91d8cface7e9d176227d912/third_party) |
| Coral 硬件 gitlink | `1aeb3f46c3f56b00d277a5285da7f6c7cf801ab4` | [固定 third_party 目录](https://github.com/google-coral/coralnpu-compiler/tree/770dc9332dffed8ff91d8cface7e9d176227d912/third_party) |
| 集成依赖 | Coral MPACT `704b28ce126debd2d951f9c3b36704080d75dc60`，MPACT-RiscV `4262467a57837aa5de0a6283340868dd06d1dfe7`，MPACT-Sim `9436377658889cd678101931f62ddba8a4d17b6d` | [Compiler MODULE.bazel](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/MODULE.bazel#L167-L213) |
| 独立依赖 | MPACT-RiscV `e4f1e9c1b243954ff8388fd39019248b2ae7341a`，MPACT-Sim `9c43949f80bef9978654473d9703ac29de30bc34` | [MPACT WORKSPACE](https://github.com/google-coral/coralnpu-mpact/blob/704b28ce126debd2d951f9c3b36704080d75dc60/WORKSPACE#L43-L57) |

集成路线与独立路线碰巧共享相同的 Coral MPACT 基础 SHA，但上游依赖与补丁不同。集成构建给 Coral MPACT 应用 M4 decoder 和向量取指操作数相关补丁，并给 MPACT-RiscV 应用 BF16 舍入、向量转换与 i16 MatMul 等补丁。LLVM 的 Zvt 补丁和 IREE 插件、外部驱动、CodeGen hook 等补丁也属于集成配置的一部分。已确认的是声明和补丁内容，实际应用状态留待独立构建目录复核。`--restore-first` 会恢复子模块工作树，使用前须确认外部 checkout 干净。[模块声明](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/MODULE.bazel#L167-L213)、[补丁脚本](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/scripts/patch-third_party.sh)、[LLVM Zvt 补丁](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/llvm-project-0001-add-zvt-support.patch)

## 2. 输入、编译、产物与执行

```mermaid
flowchart LR
    MLIR[标准 Linalg MLIR] --> Compiler[Coral IREE Compiler]
    Compiler --> Code[LLVM IR 与目标 ELF32]
    Code --> VMFB[VMFB 模块]
    VMFB --> Runtime[IREE Runtime 与 Coral HAL]
    Runtime --> Sim[MPACT 执行设备指令]
    Sim --> Result[输出 Tensor]
```

| 组件 | 源码入口与职责 |
| --- | --- |
| IREE 输入 | README 的 MatMul 用标准 `func`、`arith`、`tensor`、`linalg`。这份样例界定了已找到的输入证据。`mini.matmul` 若要接入，应先由我们转换成被 IREE 接受的标准 IR。[MatMul 示例](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md#L390-L428) |
| Coral Compiler | `compiler/Target/CoralNPUTarget.cpp` 中的 `iree_register_compiler_plugin_coralnpu` 注册插件。目标后端借用 IREE LLVMCPU 基础设施并配置 `riscv32`、`ilp32` 和目标 Features。Tiling 与 DTCM 规划的扩展也接入后端 Pipeline。[插件入口](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/compiler/Target/CoralNPUTarget.cpp#L142)、[目标配置](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/compiler/Target/CoralNPUTargetBackend.cpp#L292-L310)、[后端 Pipeline](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/compiler/Target/CoralNPUTargetBackend.cpp#L417-L435) |
| 设备产物 | LLVM IR 经目标后端编译和链接得到含 RV32 设备代码的 ELF。在源码的 embedded ELF 分支中，后端将字节封装为 HAL ExecutableBinary。VMFB 是 IREE Runtime 加载的模块，可包含 VM 字节码和这类设备可执行载荷。[ELF 封装](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/compiler/Target/CoralNPUTargetBackend.cpp#L994-L1037) |
| IREE Runtime / Coral HAL | Runtime 加载 VMFB，调用导出的 `matmul`，用 Tensor buffer 绑定输入输出。Coral 驱动校验 little-endian RISC-V ELF32，设备端 dispatch 将 image、dispatch state 交给执行后端。设备 ABI 还包括入口、绑定和运行状态，ELF 可被辨认只是其中一环。[ELF32 校验](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/runtime/driver/coralnpu_executable.c#L87-L119)、[dispatch 实现](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/runtime/driver/coralnpu_device.c#L94-L131) |
| MPACT | 官方集成的 `iree-run-module` 默认使用 MPACT 功能仿真后端。仿真器执行设备指令，Tensor 调用与结果导出由上层 Runtime 处理。[官方运行说明](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md#L420-L440)、[集成仿真构建](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/runtime/sim/BUILD.bazel#L23-L86) |

## 3. 目标 Features 与片上预算

`riscv32` 上默认启用 M、F、Zbb、Zve32f、Zvl128b、Zfbfmin、Zvfbfmin、Zvfbfwma，以及 Zvt 子集 `zvtbase`、`zvt8e`、`zvt16e`、`zvti8i32mm`、`zvtf16f32mm`、`zvtf32f32mm`。README 将矩阵扩展标为 Zvt v0.4。Feature 列表可以影响指令选择，实际生成了哪些指令要查看选定模型的目标代码。[Features 默认值](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/compiler/Target/CoralNPUTargetBackend.h#L48-L82)、[README](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md#L7-L12)

官方默认 DTCM 32 KiB，减去 4 KiB stack 和 4 KiB reserve，按 4 KiB 向下取整后得到 24 KiB 的 Tile 预算。默认每 operand 一个 Buffer。输入切片、输出暂存、归约累加和矩阵工作区都会消耗预算，布局与对齐也会改变实际开销。README 对 Zvt 的矩阵模型使用 16×16 维度描述，实际可用的布局和指令还要按 datatype 与子集确认。项目 Virtual NPU 的 32×32×32 和 256 KiB 属于独立的教学规格。[DTCM 与 Zvt 说明](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md#L268-L340)

## 4. 构建证据与主机风险

| 项目 | 官方版本或构建路线 | 本轮实际探测 |
| --- | --- | --- |
| 主机 | 建议在可复现的隔离环境中验证完整集成 | `uname -m` 为 arm64，`sw_vers` 为 macOS 26.5.2 |
| Bazel | Compiler `9.1.0`，独立 MPACT `8.6.0` | `command -v bazel bazelisk` 无输出 |
| 工具链 | Clang / LLD 19，Python 3.13，Bash ≥ 4 | 默认 Apple Clang 17.0.0、Python 3.15.2，系统 Bash 3.2.57，`command -v lld ld.lld` 无输出 |
| CMake / Ninja | CMake ≥ 3.26，官方有 Ninja 构建入口 | CMake 4.4.5、Ninja 1.13.2 |

Bazel 集成仿真的相关源文件按 Linux x86-64 条件选择，CMake 的 `runtime/sim/CMakeLists.txt` 也有 x86-64 条件。README 的 macOS ARM64 打包平台项只能说明存在配置。因此 Day 17 应分别验证 Mac 的编译器目标和完整 Coral HAL/MPACT 路线，后者优先考虑隔离的 Linux x86-64。官方 CMake 的 Compiler 目标可独立构建。Runtime 仍需 MPACT 库，可用预构建产物或按官方规则用 Bazel 构建。[Compiler 要求](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md#L58-L72)、[独立 Bazel 版本](https://github.com/google-coral/coralnpu-mpact/blob/704b28ce126debd2d951f9c3b36704080d75dc60/.bazelversion)、[Bazel 平台条件](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/runtime/sim/BUILD.bazel#L23-L86)、[CMake 平台条件](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/runtime/sim/CMakeLists.txt#L15-L55)、[CMake 的 Runtime 依赖](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md#L190-L205)

## 5. 官方 MatMul 与 Day 17 验收

官方样例先用 `linalg.fill` 将 `tensor.empty` 的目标初始化为零，再交给 `linalg.matmul` 累加。`tensor.empty` 提供形状与类型，MatMul 的初始累加值由 fill 决定。输入 A 为 `32×64`、全 1，B 为 `64×128`、全 2。每个输出元素累加 64 次 `1 × 2`，预期得到 `32×128` 全 128。以下为固定提交的官方命令，状态为**未执行**。[官方模型与运行步骤](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md#L390-L435)

```bash
bazel run --config=dev @iree_core//tools:iree-compile -- \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host \
  --iree-hal-target-device=coralnpu \
  "$(pwd)/model.mlir" -o "$(pwd)/model.vmfb"

bazel run --config=dev @iree_core//tools:iree-run-module -- \
  --device=coralnpu --module="$(pwd)/model.vmfb" --function=matmul \
  --input=32x64xf32=1.0 --input=64x128xf32=2.0
```

- **G0** 补齐拟使用主机上的完整工具链探测，固定 gitlinks，记录编译器、Runtime 和依赖的版本及补丁状态。
- **G1** 在独立环境尝试 `bazel build //sim:coralnpu_sim`、`bazel test //sim/test:coralnpu_decoder_test`，记录命令、退出码和首个失败点。若要证明 M4 指令支持，补 `//sim/test:coralnpu_m4_ops_test`。[独立构建说明](https://github.com/google-coral/coralnpu-mpact/blob/704b28ce126debd2d951f9c3b36704080d75dc60/README.md)、[测试目标](https://github.com/google-coral/coralnpu-mpact/blob/704b28ce126debd2d951f9c3b36704080d75dc60/sim/test/BUILD)
- **G2** 优先用固定版本的官方集成路线编译并运行上述 MatMul，保存退出码、Shape 和数值。编译时加 affinity profile，运行时可加 `--coralnpu_stats_format=json`，记录该 MatMul 确实派到了 Coral。因为编译命令同时启用 Host 和 Coral，仅有 VMFB 和正确输出无法确认分派。[分派报告与运行统计](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md#L249-L258)、[运行统计参数](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md#L505-L519)
- **G3** 待候选的 MatMul→Linalg 路线可用，用自研导出替换官方模型并重测。Coral 官方后端负责该路径的 DTCM、CodeGen 和 ISA。Virtual NPU 则执行本项目自己的 Command IR 和 ISA。

## 6. 本轮执行记录

主机探测使用了 `uname -m`、`sw_vers`、`cmake --version`、`ninja --version`、`clang --version`、`python3 --version`、`/bin/bash --version` 和 `command -v`。GitHub API 树查询遇到 403 限流，浅克隆遇到 TLS EOF，`git ls-remote` 重试超时。最终从公开仓库页面确认 SHA，从该 SHA 的树页面取得 gitlinks，并从固定提交读取源码。上述问题属于取证网络路径，未用于判断 Coral 构建或兼容性。
