# Virtual NPU ISA v1

`program.json` 是 Virtual NPU 的独立指令流。执行模型为一个矩阵单元、
同步 DMA、串行 M/N/K 调度与单缓冲。原生矩阵块上限为 `[32,32,32]`，
尾块使用有效的 M、N、K。所有 Tensor 使用紧凑 row-major FP32，
浮点数据在内存中按 Little-endian 存放。

## Program Schema

根对象固定为 `schema_version`（1）、`target`（`virtual-npu-v1`）、
`schedule_contract`（`serial-mnk-single-buffer-v1`）、`instruction_count`
和 `instructions`。JSON 使用 UTF-8，整数为 `0..2^53-1`，
布尔值使用 JSON 的 `true` / `false`。此范围可由常见 binary64
JSON 实现精确表示。见 [RFC 8259 §6](https://www.rfc-editor.org/rfc/rfc8259.html#section-6)。

每条指令恰好有如下公共字段及其 Opcode 的字段。

| 公共字段 | 契约 |
| --- | --- |
| `opcode` | 下表中的一种字符串。 |
| `sequence_index`、`command_id` | 全局执行位置与身份。v1 串行契约要求二者都等于数组下标。 |
| `matmul_index`、`local_event_index` | 原命令组编号与组内事件编号。两者从零连续编号，每个组的局部编号重新从零开始。 |
| `tile_index` | 组内原 Tile 的编号，从零开始并单调连续。 |
| `dependencies` | 升序且唯一的全局前驱 ID，前驱的 Sequence 必须更小。执行次序已经确定。 |

| Opcode | 额外字段 | 效果 |
| --- | --- | --- |
| `ZERO` | `buffer_id`, `dst_sram`, `rows`, `cols` | 将从 `dst_sram` 开始的 `rows × cols` 个元素写为 FP32 正零。 |
| `DMA_LOAD_2D` | `tensor_id`, `buffer_id`, `src_dram`, `dst_sram`, `rows`, `row_bytes`, `src_stride_bytes`, `dst_stride_bytes` | 逐行从 DRAM 向 SRAM 复制字节。 |
| `MATMUL_TILE` | `a_buffer_id`, `b_buffer_id`, `c_buffer_id`, `a_sram`, `b_sram`, `c_sram`, `m`, `n`, `k`, `accumulate` | 从 SRAM 读 `A[m,k]`、`B[k,n]` 和旧 `C[m,n]`，执行 `C += A @ B`。`accumulate` 固定为 `true`。 |
| `DMA_STORE_2D` | `tensor_id`, `buffer_id`, `src_sram`, `dst_dram`, `rows`, `row_bytes`, `src_stride_bytes`, `dst_stride_bytes` | 逐行从 SRAM 向 DRAM 的中间结果或输出 Tensor 复制字节。 |

Buffer ID 的作用域为 MatMul 组。Tensor ID 引用 Manifest 中的全局 DRAM
Tensor，地址以各自的内存空间起点为零，单位为字节。矩阵单元访问紧凑行，
A、B、C 的行宽分别为 `4*k`、`4*n`、`4*n` 字节。

二维 DMA 的第 `r` 行从 `src + r*src_stride_bytes` 复制
`row_bytes` 字节到 `dst + r*dst_stride_bytes`。行数与行宽为正，
行宽为 4 的倍数。SRAM 一侧的跨度等于行宽，DRAM 一侧的跨度等于所引用
完整 Tensor 的行跨度。DRAM 上每行必须落在同一个 Tensor 的单行内。
SRAM 起点须在保留区间之后并满足目标配置对齐。每条指令在下一条开始前完成。

未知 Opcode、非法字段、顺序、引用或地址范围导致加载失败。CodeGen 在
导出前复用 Day 25 的完整命令与状态验证。加载器检查外部指令格式、顺序、
依赖引用和内存访问边界。编译后端为同一输出块按 K 分块顺序使用 ZERO、
多次 LOAD/累加和 STORE。指令解释与 FP32 舍入规则见
[Simulator 契约](simulator-contract.md)。

## E2E 一致性

Day 28 逐条对比 Program 与 Trace 的公共字段及 Opcode 参数，再比对完成
的指令数和实际 DMA 复制字节。编译期的 `compiler_stats.json` 与运行
Trace 位于三文件 Bundle 之外。负例覆盖 DMA 地址越界、缺 LOAD、重复
ZERO、提前 STORE、逻辑 Buffer 错配，以及权重文件摘要不符。
见 [E2E 验证说明](e2e-validation.md)。
