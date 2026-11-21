# Day 22 · MatMul Tile IR 契约

`npu-tiling` 连接完整 Tensor 计算和后端资源规划。输入是 Day 21 的 `npu.matmul`，输出是 `npu.tiled_matmul`。每条输出 Op 保留位置、A/B Operand 和完整 C Result，仍有 `Pure` 语义。它附带一份可序列化的计算计划，后续 Pass 可以单独读取这份 MLIR。

## 计划字段

[ODS](../../include/NPU/NPUOps.td) 规定 `target`、三个 i64 原生上限 `tile_m/n/k`，以及装有 DictionaryAttr 的 `tiles`。当前 MLIR 通用格式把这几个固有属性打印在 `<{...}>` 中。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `m_offset,n_offset,k_offset` | i64 | 完整逻辑矩阵中的元素坐标 |
| `m_size,n_size,k_size` | i64 | 本次计算的有效尺寸，范围为 1 到对应原生上限 |
| `output_tile_id` | i64 | 当前 Op 内的编号，等于 `mIndex * nTiles + nIndex` |
| `is_first_k,is_last_k` | BoolAttr | 当前输出块归约的起点和终点；只有一段时两个同时为真 |

对于 `A[M,K] × B[K,N] → C[M,N]`，第 `(m0,n0,k0)` 条记录读取
`A[m0:m0+Mt, k0:k0+Kt]` 和 `B[k0:k0+Kt, n0:n0+Nt]`，
贡献到 `C[m0:m0+Mt, n0:n0+Nt]`。M/N 选择不同的结果区域；K 切片属于同一结果区域的归约。尾部有效尺寸按 `min(native, extent - offset)` 决定，无需补零。

## 构造与验证

[TilePlan](../../lib/Dialect/NPU/TilePlan.cpp) 对正维度按 `1 + (extent-1)/limit` 求块数，预检三个块数的乘积。最多生成 65,536 条静态记录（每个 Op），这是 IR 大小的工程上限。用除法界限判断乘积后再相乘；有效索引推导出的坐标始终小于原维度。按 M→N→K 顺序，K 最快变化。TilePlan 只保存参数，可按索引得到单条记录，不为 Verifier 再复制整张计划。

[属性编解码](../../lib/Dialect/NPU/TilePlanAttrs.cpp) 提供 `encodeTileDesc` 和 `decodeTileDesc`；后端及 [Op Verifier](../../lib/Dialect/NPU/NPUOps.cpp) 共用九个字段的定义。Verifier 先检查与 Compute IR 相同的 MatMul 类型关系，再按当前规范化次序逐条比较计划。精确比较同时排除缺失、重复、乱序、越界、K 空洞或重叠，以及错误归约标记。未来允许不同分块组合或执行重排时，需要重新定义覆盖与依赖验证。独立 Python 测试按 M/N 区域与连续 K 区间检查实际输出，不调用 C++ 生成公式。

[NPUTiling Pass](../../lib/Backend/NPUTiling.cpp) 显式从 JSON 读取 schema 1、目标名称、矩阵单元数、FP32 输入/累加类型、`accumulate` 和三个正整数原生上限。输入只接受模块、函数、返回、静态 FP32 Dense 常量及两种 NPU MatMul。它预检整个模块，随后用 IRRewriter 替换每条 Compute Op。已分块 Op 在同目标同上限下原样保留；上限不一致时报错，避免无意使用旧计划。Dialect 库拥有验证和数据契约，Backend 库依赖 Dialect，依赖方向保持单向。

## Day 23 的输入

实现和产物见 [SRAM Memory Plan 契约](sram-planning.md)。

1. 读取经过验证的 `npu.tiled_matmul`，检查实际 profile 的目标和原生上限与 Op 一致；`memory` 配置由 SRAM Planner 读取。
2. 从 `getTiles()` 取得记录，用 `decodeTileDesc` 解码。维持每个 Op 内部的 M→N→K 顺序；用生产者 Op 或稳定的计算标识加 `output_tile_id` 区分累加器。
3. 根据有效 `Mt,Nt,Kt` 用安全整数运算求原始 Buffer 字节数：A 为 `4*Mt*Kt`，B 为 `4*Kt*Nt`，C 为 `4*Mt*Nt`。一个满块三个 Buffer 各 4096 B。地址对齐、预留区和峰值容量由 Day 23 决定。
4. 从 first K 开始为 C 保留同一块累加存储，通过中间 K 块保留部分和，直到 last K 计算后的写回完成才可以回收。可在上一次读取结束后复用 A/B Buffer。DMA 后续使用元素坐标与原 Tensor 行跨度计算字节访问。

本阶段没有实际 SRAM 地址、DMA、ZERO、执行命令或数值结果。读取权重或前序 MatMul 的关系由完整 Tensor SSA 维护，输出块编号只在单条 MatMul 内有效。
