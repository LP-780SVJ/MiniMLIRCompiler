# Day 17 · 构建与测试记录

执行日期 2026-11-17，宿主为 Apple M4 / macOS 26.5.2 / arm64。虚拟 NPU 测试从 AI-COMPILER 根目录运行，Coral G1 在独立临时目录运行。

## Virtual NPU SRAM 容量检查

使用 `.venv/bin/python` 3.11.15 运行真实 CLI。以下为五组课程算例的调用形式：

```bash
CHECK=code/MiniMLIRCompiler/python/hardware/check_sram.py
CONFIG=code/MiniMLIRCompiler/configs/virtual-npu-v1.json
for size in 32 64 128 256; do
  .venv/bin/python "$CHECK" --config "$CONFIG" --m "$size" --n "$size" --k "$size"
done
.venv/bin/python "$CHECK" --config "$CONFIG" --m 128 --n 128 --k 128 --double-buffer-inputs
```

运行结果通过 Python 子进程断言退出码和关键输出。

| 案例 | 实际对齐后需求 | 容量通过 | 原生形状上限通过 | 退出码 |
| --- | ---: | --- | --- | ---: |
| 32³ 单缓冲 | 12288 B（12 KiB） | True | True | 0 |
| 64³ 单缓冲 | 49152 B（48 KiB） | True | False | 0 |
| 128³ 单缓冲 | 196608 B（192 KiB） | True | False | 0 |
| 256³ 单缓冲 | 786432 B（768 KiB） | False | False | 2 |
| 128³ 双输入缓冲 | 327680 B（320 KiB） | False | False | 2 |
| 非方形 M/N/K=3/7/5 | A 64 B、B 192 B、C 128 B，合计 384 B | True | True | 0 |
| 32³，可用预算正好 12288 B | 12288 B | True | True | 0 |
| 32³，可用预算少 1 B | 12288 B | False | True | 2 |

另外实际验证了负数、零维度、非整数、传入 Coral profile、保留量等于 SRAM 总量、对齐为零、JSON 损坏这七项，退出码均为 1，共 **15/15 检查符合预期**。两项边界预算和无效配置使用临时 JSON。

MiniMLIRCompiler 的 `cmake --build code/MiniMLIRCompiler/build --target mini-opt` 退出 0，Ninja 报告 `no work to do`。`ctest --test-dir code/MiniMLIRCompiler/build --output-on-failure` 退出 0，现有测试 **12/12 通过**。这些覆盖 IR 与已有 Pass 的回归。

## Coral G1 实验环境

- 源码来自 [独立 MPACT 固定提交 704b28ce126debd2d951f9c3b36704080d75dc60](https://github.com/google-coral/coralnpu-mpact/commit/704b28ce126debd2d951f9c3b36704080d75dc60) 的官方快照，工作目录 `/tmp/mininpu-day17/mpact-704b28c`。使用上游 `WORKSPACE` 锁定依赖。
- 下载 [官方 Bazel 8.6.0 ARM64](https://releases.bazel.build/8.6.0/release/bazel-8.6.0-darwin-arm64)，以 [官方 SHA256](https://releases.bazel.build/8.6.0/release/bazel-8.6.0-darwin-arm64.sha256) 验证为 `948a7186641f601c83344b63b88bc6943025585f2bb7f407e19cface5fe4aa3b`。临时二进制执行 `--version` 退出 0，输出 `bazel 8.6.0`。
- 复现命令中的 Bazel 路径设为 `/tmp/mininpu-day17/bazel-8.6.0-darwin-arm64`，工作目录固定为快照。所有轮次使用 `--batch --output_user_root=/tmp/mininpu-day17/bazel-user-root`。原始日志在该临时目录，关键命令、状态与诊断保留于本文。

## G1 初次构建和两轮诊断

首次在默认 Java 路径下分别执行下列目标，使用 `--jobs=4 --color=no --curses=no`，测试额外使用 `--test_output=errors`。

| Bazel 子命令与目标 | 结果 |
| --- | --- |
| `build //sim:coralnpu_sim` | 退出 1。MPACT-Sim 的 ANTLR genrule 找不到 Java Runtime。 |
| `build //sim:coralnpu_m4_sim` | 退出 1。同一个宿主 ANTLR 工具报错。 |
| `test //sim/test:coralnpu_decoder_test //sim/test:coralnpu_top_test` | 退出 1，0/2 实际执行。 |
| `test //sim/test:coralnpu_m4_state_test //sim/test:coralnpu_m4_encoding_test //sim/test:coralnpu_m4_config_test` | 到本轮 45 秒限额时仍在编译，发 SIGTERM，退出 143。 |

仅加 `--action_env=PATH=…` 后，合并的两个构建目标和五个测试仍各退出 1，ANTLR 仍报 `Unable to locate a Java Runtime`。给 `--host_action_env` 和 Bazel 进程同时提供 embedded JDK 的路径后，构建越过 ANTLR，进入 MPACT-Sim decoder C++ 编译。它使用 Bazel 默认的 `-mmacosx-version-min=10.11`，在 `instruction_set_visitor.cc:115` 首次报 `'path' is unavailable: introduced in macOS 10.15`。这轮两个构建目标和五个测试也各退出 1，测试实际执行 0/5。

## G1 使用明确的宿主和目标部署版本

通过 Bazel 命令行同时把宿主和目标的最低 macOS 版本设为 11.0。在当前实验目录，等价复现命令如下：

```bash
cd /tmp/mininpu-day17/mpact-704b28c
BAZEL=/tmp/mininpu-day17/bazel-8.6.0-darwin-arm64
BAZEL_ROOT=/tmp/mininpu-day17/bazel-user-root
JAVA_BIN="$BAZEL_ROOT/install/1d833823d96e93f4793a467c87470079/embedded_tools/jdk/bin"
BUILD_PATH="$JAVA_BIN:/usr/bin:/bin:/usr/sbin:/sbin"
FLAGS=("--action_env=PATH=$BUILD_PATH" "--host_action_env=PATH=$BUILD_PATH"
       --macos_minimum_os=11.0 --host_macos_minimum_os=11.0
       --jobs=8 --color=no --curses=no --verbose_failures)

PATH="$BUILD_PATH" "$BAZEL" --batch --output_user_root="$BAZEL_ROOT" \
  build //sim:coralnpu_sim //sim:coralnpu_m4_sim "${FLAGS[@]}"
PATH="$BUILD_PATH" "$BAZEL" --batch --output_user_root="$BAZEL_ROOT" \
  test //sim/test:coralnpu_decoder_test //sim/test:coralnpu_top_test \
  //sim/test:coralnpu_m4_state_test //sim/test:coralnpu_m4_encoding_test \
  //sim/test:coralnpu_m4_config_test --test_output=errors "${FLAGS[@]}"
```

`build` 退出 **0**，耗时 140.0 秒，两个目标均成功构建。对两份二进制运行 `file`，均报告 `Mach-O 64-bit executable arm64`。`test` 退出 **0**，耗时 8.6 秒，实际执行 **5/5 且全部通过**。

| 测试目标 | 实际状态 | 测试耗时 |
| --- | --- | ---: |
| `coralnpu_decoder_test` | PASSED | 0.6 s |
| `coralnpu_top_test` | PASSED | 1.1 s |
| `coralnpu_m4_state_test` | PASSED | 0.9 s |
| `coralnpu_m4_encoding_test` | PASSED | 1.4 s |
| `coralnpu_m4_config_test` | PASSED | 1.1 s |

最终原始日志为 `/tmp/mininpu-day17/mac11-build.log` 与 `/tmp/mininpu-day17/mac11-tests.log`。编译中报告了第三方警告，包括链接器忽略重复库，最终构建和测试均成功。G1 证据覆盖这些独立目标和指定测试。

## G2 / G3

两项均为 **Not Run**。Day 16 文档保留官方 MatMul 命令。完整的 G2 记录需要固定集成工具链、真实 Coral 分派、输出 Shape 和实际数值。[官方命令与验收](toolchain-research.md#5-官方-matmul-与-day-17-验收)
