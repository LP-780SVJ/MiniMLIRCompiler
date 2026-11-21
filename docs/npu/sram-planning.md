# Day 23 · SRAM Memory Plan 契约

`npu-sram-plan` 消费已经验证的 `npu.tiled_matmul`。它给每次局部计算规划片上 SRAM，并把 Tile Plan 和资源分配一起保存为 `npu.sram_planned_matmul`。新 Op 仍然接收完整 A/B Tensor 并返回完整 C Tensor。它的 `Pure` 表示逻辑计算没有设备副作用；分配地址在本阶段只是编译产物中的数字。

## 目标配置

使用 `--npu-sram-plan="config=…"` 显式选择 [Virtual NPU profile](../../configs/virtual-npu-v1.json)。Tiling 和 SRAM Planning 共用版本、目标、原生上限和 FP32 累加模式的检查。SRAM Pass 另外要求紧凑 Row-major Tile、同步 DMA 和单缓冲，并从配置读取总容量、保留区和对齐。Tile IR 保存的目标和三个原生上限必须与配置相符。重复规划时，已经规划的 Op 只有在目标、原生上限和内存配置一致时才原样保留。

SRAM 字节地址从 0 起。保留区为 `[0,reserved_bytes)`，数据区从向上对齐后的保留区末尾开始。对齐必须为正的 2 的幂。分配和边界检查使用受检的 i64；字节、地址、事件和 ID 均为非负。IR 保存通过验证的配置参数；后续读取实际 profile 的 Pass 仍要比对这些值。

## 生命周期和分配

每个 MatMul 的事件索引从 0 开始，固定为 `serial-mnk-single-buffer-v1`：

```text
对于一个输出块：
ZERO C → [LOAD A → LOAD B → COMPUTE] × K 块数 → STORE C
```

`A/B` 是每条 K 计算的逻辑内容，分别从自己的 LOAD 起存活到 COMPUTE 完成。`C` 只为每个输出块创建一次，从 ZERO 起存活到 STORE 完成。所有区间半开 `[begin,end)`；旧 Buffer 在 `end <= next.begin` 时可以释放，因此允许相邻生命周期复用同一地址。事件编号表达先后和占用，没有硬件周期含义。不同 MatMul 必须串行完成全部写回；在此前提下它们可各自从同一 SRAM 起点分配。跨 MatMul 的完整中间 Tensor 仍需在后续阶段物化。

需求尺寸按有效 `Mt,Nt,Kt` 计算：A 为 `4*Mt*Kt`，B 为 `4*Kt*Nt`，C 为 `4*Mt*Nt`，容量按配置对齐。对齐填充只增加 SRAM 占用。First-Fit 按生命周期开始顺序，回收已结束的区间，优先选择能容纳需求的最低地址连续空间。容量不足与 First-Fit 连续空间不足分别诊断：前者可由活跃容量下界证明，后者受当前放置产生的碎片影响。

`peak_live_bytes` 是同一事件活跃 Buffer 的最大分配容量总和；`arena_high_water_bytes` 是从地址 0 计的最大排他结束地址，包含保留区造成的偏移和中间空洞。两项必须各自能从计划重新算出。

## IR 与 Day 24 接口

所有计划数据都是 `npu.sram_planned_matmul` 的固有属性，可随文本/字节码保存。顶层保留 Day 22 的 `target`、`tile_m/n/k` 与 `tiles`；新增 `memory_plan` 字典：

| 字段 | 含义 |
| --- | --- |
| `schedule_contract` | 固定为 `serial-mnk-single-buffer-v1` |
| `alignment_bytes`、`sram_capacity_bytes`、`reserved_bytes` | 当前内存配置 |
| `allocations` | `id`、`role`、`tile_index`、`output_tile_id`、`raw_bytes`、`allocated_bytes`、`alignment_bytes`、`address`、`live_begin/end` 和 `reused_address` |
| `tile_bindings` | 每条 Tile 的 `tile_index` 和 `a_buffer_id / b_buffer_id / c_buffer_id` |
| `peak_live_bytes`、`arena_high_water_bytes` | 校验与资源报告 |

Day 24 使用 `getTiles()` + `decodeTileDesc()` 读取元素坐标与有效尺寸，再用 `getMemoryPlan()` + `decodeMemoryPlan()` 根据 Binding 找到同一 Op 内的 Allocation。用 `id` 建索引，分配记录的数组下标没有执行含义。Buffer ID 仅在当前 Op 内有效，`reused_address` 表示该 Buffer 的起始地址曾分配给较早的 Buffer。按 Tile 顺序维护累加器，在 first K 前生成初始化，在 last K 后写回。调度若要改变事件顺序或让两次 MatMul 重叠，需要重新证明内存计划合法。

例如多块模型的第二条 Tile 读取 `A[0:32,32:64]`。对于完整 A 的 K=96，源行跨度为 384 字节，起始字节为 DRAM 基址加 128；对紧凑 SRAM Tile，目的行跨度和每行复制量都是 128 字节，共 32 行。目的基址从该 Tile 的 A Binding 读取。DRAM 基址和 DMA 命令仍由后续阶段生成。

## 验证与算法的分工

Op Verifier 先复用 Day 22 验证 Tile 覆盖，再读取计划字段。内存验证从 Tile 和固定事件契约推导尺寸、绑定和生存期，并独立核对地址对齐、保留区、范围、同时存活的地址冲突以及统计。它无需重跑 First-Fit，因此能检查手写或后续优化过的合法布局，也能定位地址、绑定或生命期被篡改的位置。Pass 在任何 IR 替换之前先为全部操作完成规划和验证。
