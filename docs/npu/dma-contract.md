# Day 24 · DRAM 布局与 Command IR

`npu-insert-dma{config=…}` 消费已验证的 `npu.sram_planned_matmul`。它先给整个入口函数分配逻辑 DRAM 区间，再生成每条 MatMul 的同步命令。新 Op `npu.command_matmul` 保留原 Tensor Operand、Result、Tile Plan 和 SRAM Plan；模块增加 `npu.dram_layout`。命令描述读写设备存储，Op 保守地声明 MLIR MemRead/MemWrite 效果。

## DRAM 布局

v1 只支持单入口 `@main`、单块、无调用和控制流，值为静态正尺寸二维 FP32、无特殊 Tensor encoding。先按参数顺序、再按常量出现顺序、最后按 MatMul 产生结果的顺序给每个 SSA Value 分配一个 ID。按 64 字节对齐顺序分配，各区间不复用；这是 DRAM 布局 ABI，与配置中的 SRAM 对齐独立。

- `tensors` 中每条有 `id, kind, source_index, base, rows, cols, row_stride_bytes, size_bytes, dtype, layout`。源位置是参数、常量或 MatMul 的各自序号。对于也被后续计算使用的返回值，`kind="output"` 仍然使用原 MatMul 结果的唯一 ID。一项参数或权重被直接返回时则保持原来源类型。
- `input_ids`、`constant_ids`、`output_ids` 保存有序映射；输出列表允许重复 ID。外部 ONNX 名称由 Day 20 模型元数据按输入/输出位置补全，禁止根据 `%arg0`、`%0` 等打印名称猜测。
- `total_bytes` 是最后一个 Tensor 的排他结束地址，含对齐空洞。`row_stride_bytes = cols * 4`，`size_bytes = rows * row_stride_bytes`。每块范围是 `[base, base+size_bytes)`。两种地址空间独立，DRAM 地址 0 和 SRAM 地址 0 表示不同存储。

## 2D DMA

对 Tile `(m0,n0,k0,Mt,Nt,Kt)`，A/B/C 的 DRAM 起始字节由完整 Tensor 行宽决定。SRAM 局部块紧凑排列，行跨度等于有效行宽：

| 命令 | DRAM 起始地址 | SRAM 地址 | rows | row_bytes | DRAM stride | SRAM stride |
| --- | --- | --- | --- | --- | --- | --- |
| LOAD A | `A.base + 4*(m0*K+k0)` | A Binding | Mt | `4*Kt` | `4*K` | `4*Kt` |
| LOAD B | `B.base + 4*(k0*N+n0)` | B Binding | Kt | `4*Nt` | `4*N` | `4*Nt` |
| STORE C | `C.base + 4*(m0*N+n0)` | C Binding | Mt | `4*Nt` | `4*N` | `4*Nt` |

LOAD 的源侧是 DRAM，STORE 的源侧是 SRAM。每侧的排他末端是 `base + (rows-1)*stride + row_bytes`。命令生成和验证使用受检 i64 运算。Verifier 既检查实际访问是否落在绑定的 Tensor 或 SRAM **有效数据**区间内，也核对精确的 Tile 起点、行数和两个 stride。通过整个 Arena 的边界检查仍可能读错 Tensor、行或相邻 Buffer。

## 命令和 Day 25 交接

每条 `commands` 记录由强类型 C++ payload 编解码，持久化为 MLIR 字典：

| Opcode | 除 `event_index`、`tile_index` 之外的字段 |
| --- | --- |
| `ZERO` | `buffer_id, dst_sram, rows, cols` |
| `DMA_LOAD_2D` | `tensor_id, buffer_id, src_dram, dst_sram, rows, row_bytes, src_stride_bytes, dst_stride_bytes` |
| `MATMUL_TILE` | `a_buffer_id, b_buffer_id, c_buffer_id, a_sram, b_sram, c_sram, m, n, k, accumulate=true` |
| `DMA_STORE_2D` | `tensor_id, buffer_id, src_sram, dst_dram, rows, row_bytes, src_stride_bytes, dst_stride_bytes` |

模块验证从 SSA 重新核对整个布局、MatMul 的 `lhs_tensor_id/rhs_tensor_id/result_tensor_id` 和有序输出，Op 验证再解释每条命令。Buffer ID 只在单条 MatMul 内有效，必须按 Allocation 的 `id` 查找，即使数组重新排列也要解析到同一物理区间。

Day 25 可用 `decodeDRAMLayout()`、`decodeCommands()` 和 `decodeMemoryPlan()` 读取命令。每条 MatMul 的 `event_index` 从 0 开始，与 `serial-mnk-single-buffer-v1` 对齐。按入口块的 MatMul 顺序执行，每条必须完成其 STORE 才能开始下一条；每个输出块的顺序固定为 `ZERO → [LOAD A → LOAD B → MATMUL_TILE] × K → STORE`。Day 25 要检查跨命令、跨 MatMul 的读写依赖和资源活跃状态；若改变顺序，必须重新验证复用前提。本阶段没有权重打包、ISA 编码、实际 DMA 或数值执行。

全局顺序与独立状态检查已在 [Day 25 调度契约](scheduling-contract.md) 实现。
