# Day 25 · 全局命令顺序与依赖契约

`npu-schedule` 消费 Day 24 的单入口 Command IR，验证 `npu.dram_layout` 和各个 `npu.command_matmul`，再为模块添加 `npu.execution_schedule`。命令仍引用原来的 Tensor ID、Buffer ID 和字节地址。调度记录只保存命令来源、执行位置与前驱，不存储第二份 payload。

## 三种编号

| 编号 | 含义 |
| --- | --- |
| `matmul_index, local_event_index` | 在 `@main` 的入口块中定位 MatMul，并定位该 Op 的 `commands` 项。每条 MatMul 的本地事件从 0 开始，与 Day 23 生命周期坐标一致。 |
| `command_id` | 全局唯一 ID，由 MatMul 命令数的前缀和加本地事件编号得到。 |
| `sequence_index` | 从 0 开始的连续执行位置。当前串行契约规定它和全局 ID 相等，两者的职责仍分开。 |
| `dependencies` | 必须先完成的全局 ID 列表，严格升序且无重复。 |

模块属性是字典，字段为 `schema_version = 1 : i64`、`contract = "serial-mnk-single-buffer-v1"`、`total_commands` 和 `commands` 数组。数组元素包含上述五个字段，所有索引为非负 i64。当前模块记录上限为 327,680 条命令、5,242,880 条依赖，属于静态 IR 预算。

## 依赖怎么来

[Command Effects](../../lib/Dialect/NPU/CommandEffects.cpp) 解释每条命令对 SRAM/DRAM 的真实读写。两个空间独立；访问范围为半开字节区间。二维 DMA 使用第一行起点到末行排他末端的包围区间，用于保守的冲突描述。

| 命令 | 读 | 写 |
| --- | --- | --- |
| ZERO | — | SRAM C |
| LOAD | DRAM Tensor | SRAM A 或 B |
| MATMUL_TILE | SRAM A、B、C | SRAM C |
| STORE | SRAM C | DRAM 结果 Tensor |

SRAM 用有序区间跟踪最近写入者和其后的读取者，建立 RAW（写后读）、WAR（读后写）和 WAW（写后写）。这能发现不同 Buffer ID 之间的同地址复用。A/B LOAD 为计算准备数据，原有计算完成读取后才能覆盖输入槽；同一 C 的清零、逐段累加、写回也通过 C 上的读写相连。K 分段的归属与先后还由 Tile/累加状态验证。

DRAM 布局给每个 Tensor 独占区间，输入和常量只读。各结果的 STORE 按顺序串起；消费计算结果的 LOAD 依赖该 Tensor 的最后一次 STORE，传递地依赖先前所有 STORE。每条 MatMul 的最后事件到下一条首事件另设串行边界。当前实现避免全量命令两两比较。依赖图描述必须遵守的约束；v1 另行规定完整串行顺序，即使某对命令之间没有边也不自动交换。

## 验证

[SchedulePlan](../../lib/Dialect/NPU/SchedulePlan.cpp) 检查记录覆盖、唯一 ID、MatMul/事件映射、连续 Sequence、依赖引用和无环性；还核对必需边以及每条边与 Sequence 的方向。现有合法调度原样保留，错误调度拒绝。新调度只有完全验证后才写属性。

[状态验证](../../lib/Dialect/NPU/ScheduleState.cpp) 按候选 Sequence 检查设备数据是否可用，不执行 FP32 计算。它检查物理 SRAM 区间里的内容来源，确认 A/B 装载和消费、C 初始化、K 归约次序、STORE 完成与 Buffer 实际生命周期。只有该 Tensor 的全部输出块都写回后，后续 LOAD 才能消费它。模块属性验证会在重读时运行这些检查，故篡改 Sequence 而保留正确的 Day 24 命令也会失败。

## Day 26 交接

通过 `decodeSchedule` 读取模块属性，按照 `sequence_index` 遍历，使用 `(matmul_index, local_event_index)` 定位原 `CommandRecord`。命令 payload 直接供 Virtual ISA 编码，不能重新推导地址或自行调换命令。通过 `decodeDRAMLayout` 读取 Tensor ID、区间和有序 I/O；从原 `arith.constant` 获取权重字节，入口名称按 ONNX 元数据映射。调度验证属于静态正确性检查；周期、权重打包与实际数值执行由后续阶段处理。
