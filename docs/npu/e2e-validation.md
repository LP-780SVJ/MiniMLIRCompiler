# 一键编译、执行与验证 ONNX

在 AI-Compiler 根目录使用已配置的 LLVM/MLIR 23.1.4 构建。E2E Python
环境需包含 ONNX、NumPy，默认 CPU 参考路径还需要 ONNX Runtime。

```bash
day28_project="$PWD/code/MiniMLIRCompiler"
day28_python="$PWD/.venv/bin/python"

"$day28_python" -m pip install onnxruntime==1.32.0
cmake -S "$day28_project" -B "$day28_project/build" \
  -DMLIR_DIR="$PWD/llvm-project/build/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/llvm-project/build/lib/cmake/llvm" \
  -DMINI_ONNX_PYTHON="$day28_python"
cmake --build "$day28_project/build" -j 4
day28_out=$(mktemp -d "$day28_project/build/day28.XXXXXX")

"$day28_python" -B "$day28_project/python/e2e/run_e2e.py" \
  --model "$day28_project/test/e2e/models/weighted_matmul.onnx" \
  --inputs "$day28_project/test/e2e/inputs/weighted_matmul.npz" \
  --out-dir "$day28_out/weighted"

"$day28_python" -B "$day28_project/python/e2e/run_e2e.py" \
  --model "$day28_project/test/e2e/models/matmul_multitile.onnx" \
  --inputs "$day28_project/test/e2e/inputs/matmul_multitile.npz" \
  --out-dir "$day28_out/multitile"

ctest --test-dir "$day28_project/build" --output-on-failure -j 4
```

将模型名和对应输入换成 `matmul` 或 `matmul_chain` 可直接运行其他固定
fixture。已有可信参考 NPZ
时可传 `--reference`；此路径只校验参考名称、Shape 和 FP32 DType，
参考数据与模型输入的对应关系由提供者保证。默认使用 `CPUExecutionProvider`
重新从原始 ONNX 与输入得到参考输出。

可用 `--target` 指定配置，`--opt` 和 `--codegen` 指向自定义构建。
默认 `rtol=1e-4`、`atol=1e-5`，保留比较器原有规则。CTest 只在
`MINI_ONNX_PYTHON` 可导入 ONNX Runtime 时注册 Day 28 的 9 项测试。
若以 `PYTHONPATH` 提供隔离安装目录，在 CMake 配置和直接执行 Driver
时带上它。配置会把该路径传给新增的 CTest。

## 输出目录

每次使用一个新目录，拒绝复用已存在路径。所有编译和运行输入都来自
`sources/` 内的快照。

| 产物 | 用途 |
| --- | --- |
| `model.metadata.json` | 根据真实 ONNX 的输入输出名称、顺序、Shape 生成 CodeGen ABI。 |
| `01-imported.mlir` 至 `06-scheduled.mlir` | 复现分阶段输入及输出；后五步通过 `mini-opt --verify-each`。 |
| `bundle/` | Runtime 加载的固定三文件 `manifest.json`、`program.json`、`weights.bin`。 |
| `compiler_stats.json` | CodeGen 可选统计，记录每组的 Operand/Result Tensor ID、Tile 数、计划峰值与高水位。文件位于 Bundle 外。 |
| `execution_trace.json`、`actual_outputs.npz` | 已执行指令、内存访问、K 段及按 ONNX 顺序命名的真实输出。 |
| `reference_outputs.npz`、`numerical_comparison.json` | 独立参考输出和现有比较器的完整结果。 |
| `logs/`、`validation_report.json` | 子进程的命令、工作目录、退出码和双流日志，以及汇总证据。 |

`mini-npu-codegen --stats-out <path>` 要求路径处在 Bundle 外，父目录已存在，
且目标文件尚不存在。统计来自验证后的 `ScheduleInput`，
`readScheduleInput()` 内部调用 `decodeMemoryPlan()`。

## 报告 v1 与退出行为

报告带 `schema_version: 1` 和 `status`。成功报告记录以下字段，
失败报告保留已到达阶段的字段。

| 字段 | 内容 |
| --- | --- |
| `compiler`、`versions` | 仓库 HEAD 及工作树状态、Pass 序列、工具版本和二进制 SHA-256、Python/NumPy/ONNX/ONNX Runtime 版本。 |
| `sources`、`artifacts` | 输入快照的来源、路径、摘要，以及已落盘产物的大小与 SHA-256。报告自身在摘要集合之外。 |
| `program`、`memory` | 编译产物的指令及字节量、I/O 绑定，和来自编译器的计划资源统计。 |
| `runtime`、`reference` | 实际执行条数与搬运量、参考来源与摘要。 |
| `numerical_validation` | 每项输出的误差、有限元素数量和不匹配数。 |
| `stages`、`failure_stage`、`diagnostic` | 子进程完整调用和日志索引、失败层与诊断。 |

退出码 0 要求加载、执行、Trace 检查及数值比较全部通过。错误返回非零，
新目录可写时保留失败报告。拒绝已有目录时保留原有证据。
定位问题时先查 `failure_stage`。导入、Lowering 或 SRAM 规划失败时查
对应 stderr 和上一阶段 IR。Loader 失败时看地址约束和文件摘要。
Runtime 失败时查 Trace 的首条失败指令及其 Buffer、Tensor ID。
Comparator 失败时查 `numerical_comparison.json` 的具体输出和误差。
功能结果与可观察的字节或条数不包含硬件延迟。
