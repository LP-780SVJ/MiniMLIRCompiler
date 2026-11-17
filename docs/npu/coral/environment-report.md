# Day 17 · Coral 环境与目标可行性

调查日期 2026-11-17。状态区分已确认的主机与源码事实、已执行的构建或测试、以及仍待验证的 ISA 和运行时兼容性。构建命令和实际结果见 [build-log.md](build-log.md)。

## G0 主机与工具

| 实际探测 | 结果 |
| --- | --- |
| `uname -sm`、`sw_vers`、`system_profiler SPHardwareDataType` | Darwin arm64，macOS 26.5.2 (25F84)，Apple M4，32 GB |
| `git --version`、`cmake --version`、`ninja --version` | Git 2.56.0，CMake 4.4.5，Ninja 1.13.2 |
| `clang --version`、`xcode-select -p`、`xcrun --show-sdk-path` | Apple Clang 17.0.0，使用 Command Line Tools 与其 macOS SDK |
| `python3 --version`、`.venv/bin/python --version`、`/bin/bash --version` | 默认 Python 3.15.2，项目 Python 3.11.15，系统 Bash 3.2.57 |
| `command -v bazel bazelisk lld ld.lld` 与临时 Bazel `--version` | 原 PATH 无上述工具。临时下载并校验了官方 ARM64 Bazel 8.6.0，版本探测成功。 |
| `/usr/libexec/java_home -V`、Bazel embedded JDK 的 `java -version` | 主机默认 Java 路径未识别到运行时，临时 Bazel 内置 OpenJDK 25 可执行。 |
| `docker info --format '{{.OSType}}/{{.Architecture}}'`、`limactl list` | Docker 客户端连不到当前 Colima socket，Lima 没有现成实例。当前没有可直接运行完整 Coral 实验的隔离 Linux 环境。 |
| `git -C llvm-project describe --tags --always --dirty` 与 `build/CMakeCache.txt` | 现有工程为 `llvmorg-23.1.4`，Ninja，CMake 从工作区 `llvm-project/build` 查找 LLVM 和 MLIR。 |

独立 MPACT 固定 Bazel 8.6.0，使用 `WORKSPACE`。Coral Compiler 固定 Bazel 9.1.0，README 对 Compiler 路线要求 Bash ≥4、Clang/LLD 19、Python 3.13、CMake ≥3.26。这些是两个实验路径的各自要求。[独立版本](https://github.com/google-coral/coralnpu-mpact/blob/704b28ce126debd2d951f9c3b36704080d75dc60/.bazelversion)、[编译器版本](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/.bazelversion)、[Compiler 前置条件](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md#L58-L72)

## G0 源码和锁定版本

| 组件 | 检查到的 SHA |
| --- | --- |
| Coral Compiler 固定源码 | `770dc9332dffed8ff91d8cface7e9d176227d912` |
| 独立 Coral MPACT 源码与 Compiler 的集成 MPACT 基础版本 | `704b28ce126debd2d951f9c3b36704080d75dc60` |
| Compiler IREE / LLVM gitlink | IREE `89ece87b89852100c0a4ac3d515e14842d090967`，LLVM `32d702efa29d0fdd02cff7b79dea8fdbbcf059d4` |
| Compiler Coral 硬件 gitlink | `1aeb3f46c3f56b00d277a5285da7f6c7cf801ab4` |
| 独立 MPACT-RiscV / MPACT-Sim | `e4f1e9c1b243954ff8388fd39019248b2ae7341a` / `9c43949f80bef9978654473d9703ac29de30bc34` |
| 集成 MPACT-RiscV / MPACT-Sim | `4262467a57837aa5de0a6283340868dd06d1dfe7` / `9436377658889cd678101931f62ddba8a4d17b6d` |

复用 Day 16 核对过的固定源码快照和 gitlink 证据。独立 G1 构建使用该版本自己的依赖与构建规则，集成路线的 IREE、LLVM 和 MPACT 补丁未应用到本次实验目录。集成清单声明了 LLVM Zvt、IREE 插件与 Runtime 接入、M4 decoder、BF16 舍入和 i16 MatMul 等补丁，必须随集成版本一起核对。[锁定子模块](https://github.com/google-coral/coralnpu-compiler/tree/770dc9332dffed8ff91d8cface7e9d176227d912/third_party)、[独立 WORKSPACE](https://github.com/google-coral/coralnpu-mpact/blob/704b28ce126debd2d951f9c3b36704080d75dc60/WORKSPACE#L43-L57)、[集成依赖与补丁](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/MODULE.bazel#L167-L213)、[IREE/LLVM 补丁脚本](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/scripts/patch-third_party.sh)

## Target 证据

- **Coral Compiler** 配置 `riscv32` 与 `ilp32`，启用 M、F、Zbb、Zve32f、Zvl128b、BF16 与多项 Zvt 子集。README 声明 Zvt v0.4。精确默认 Feature 字符串与固定版本记录在 [coral-target.json](../../../configs/coral-target.json)。[后端配置](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/compiler/Target/CoralNPUTargetBackend.h#L48-L82)、[目标三元组](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/compiler/Target/CoralNPUTargetBackend.cpp#L292-L310)
- **DTCM** 默认 32 KiB，扣除 4 KiB stack 和 4 KiB reserve，再以 4 KiB 向下对齐，得到 24 KiB Tile 预算。源码也通过 `TilingInterface` 区分 parallel 和 reduction，并对 Zvt 累加器 Shape 独立设限。因此即使 Byte Footprint 通过，还需要检查算子循环、矩阵块形状及所选指令路线。Footprint 代码按 64 B 对齐 Buffer 开销。[预算与循环分类](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/compiler/Transforms/CoralNPUTileSizeSelectionUtils.cpp#L43-L57)、[预算和累加器检查](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/compiler/Transforms/CoralNPUTileSizeSelectionUtils.cpp#L121-L178)、[Footprint](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/compiler/Transforms/CoralNPUTileSizeSelectionFootprint.cpp#L60-L78)
- **独立模拟器** 的 M3 有向量与浮点相关 decoder 依赖，M4 继承 M3 并引入 `riscv_zvt.isa`。源码中存在基础和 M4 模拟器可执行目标以及各自测试。`Apple M4` 指宿主 CPU，`Coral M4` 指模拟目标变体。[M3 ISA](https://github.com/google-coral/coralnpu-mpact/blob/704b28ce126debd2d951f9c3b36704080d75dc60/sim/coralnpu_m3.isa)、[M4 ISA](https://github.com/google-coral/coralnpu-mpact/blob/704b28ce126debd2d951f9c3b36704080d75dc60/sim/coralnpu_m4.isa#L20-L42)、[模拟器构建规则](https://github.com/google-coral/coralnpu-mpact/blob/704b28ce126debd2d951f9c3b36704080d75dc60/sim/BUILD)、[测试目标](https://github.com/google-coral/coralnpu-mpact/blob/704b28ce126debd2d951f9c3b36704080d75dc60/sim/test/BUILD)
- **集成平台** 的 Bazel 仿真源文件按 Linux x86-64 选入，CMake 对集成仿真依赖也有 x86-64 条件。独立 MPACT 的 macOS 构建结果与完整 Coral HAL 运行结果应分别登记。[Bazel 仿真规则](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/runtime/sim/BUILD.bazel#L23-L86)、[CMake 仿真规则](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/runtime/sim/CMakeLists.txt#L15-L55)

## G0–G3 与决策

- **G0 已完成**。固定源码、依赖、补丁声明、Feature、当前宿主条件和构建缺口已记录。
- **G1 通过**。独立 MPACT 的基础与 Coral M4 目标在本机生成 arm64 可执行文件，指定测试 5/5 通过。最终使用嵌入 JDK 与宿主、目标 macOS 最低版本 11.0 的命令行参数。[实际构建和测试日志](build-log.md)
- **G2 未执行**。完整路径需要隔离的 Linux x86-64、锁定并应用补丁的 Coral Compiler/IREE/MPACT 组合。实测要同时保存 Coral 的 MatMul 分派和 `32×128` 全 128 的结果。
- **G3 未执行**。依赖 G2 和本项目的标准 Linalg MatMul 导出。

**Coral 决策为 Conditional Go。** 已复现的本机独立仿真器可继续用于指令实验。完整集成待隔离 Linux x86-64 环境中通过 G2，再决定是否接入自研 Linalg 导出。第四周按 Virtual NPU 硬件规格推进。
