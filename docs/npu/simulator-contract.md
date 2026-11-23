# Day 27 · Virtual NPU 功能级 Simulator

Simulator 消费 Day 26 的 [Program ABI](program-abi.md)，使用 Python 3.10+
和 NumPy。CLI 由仓库根目录执行 `python -m runtime.virtual_npu.run`。
运行时输入只有三文件 Bundle 与调用者的 NPZ。ONNX 导入、MLIR Pass
和 ONNX Runtime CPU 参考执行位于各自的流程。

## 加载与寻址

1. 复用 [Day 26 加载器](../../python/runtime/validate_package.py)，检查
   版本、目标、文件摘要、指令字段及顺序、地址范围、Tensor 表与 I/O。
2. 从 Manifest 读取容量，创建独立的字节 DRAM 和 SRAM。每个地址相对于
   所属空间的零地址，区间为半开 `[base, end)`。SRAM 地址检查保留区
   和对齐，DRAM 上的 DMA 窗口受所引用 Tensor 及其行边界约束。
3. 根据输入名称绑定 FP32 二维数组，检查精确 Shape，按 Little-endian
   row-major 写入。依据权重条目把原始 `weights.bin` 字节复制到 DRAM。
   Program 与权重的 SHA-256 用于检测内容错配，Profile 摘要用于溯源。

## 逻辑 Buffer 的生命周期

ByteMemory 分别保存数据字节与逐字节初始化标记。物理内存创建后，
每个字节都需要由合法的写入才可以读取。释放会失效标记。

Simulator 从指令事件识别 Buffer 的创建、消费和释放。键为
`(matmul_index, buffer_id)`。活跃项记录地址和有效形状，检查按
SRAM 对齐得到的占用区间；真正读写只覆盖有效字节。A、B 各自从
LOAD 存活到对应 MATMUL 完成。C 从 ZERO 存活到同一输出块 STORE 完成。
释放 A/B 后，下一 K 段可以在相同地址创建新的 ID。另一个 MatMul
组有独立的 Buffer ID 作用域。地址和有效尺寸由编译产物决定。

## 指令与检查

| 指令 | 功能效果及运行时要求 |
| --- | --- |
| `ZERO` | 创建 C，把有效区域的每个 FP32 元素写为正零。一个输出块执行一次。 |
| `DMA_LOAD_2D` | 从完整可用的 DRAM Tensor 逐行复制到新建的 A 或 B。源与目的分别使用自己的字节 Stride。 |
| `MATMUL_TILE` | 用指令中的 ID、地址和有效尺寸核对活跃 A/B/C，只从 SRAM 读取，将 FP32 部分和写回 C，再释放 A/B。 |
| `DMA_STORE_2D` | 检查全部 K 已累加且写回区域与计算一致，将有效 C 逐行复制到 DRAM，再释放 C。 |

DMA 在写目的地之前检查整个范围并读取全部源行，因此后续行的越界或
未初始化错误不会产生部分写入。传输按字节复制，跳过 Stride 中的行间空隙。

解释器按 Program 中的全局序号执行。它从 A 的 DRAM 列坐标与 B 的
DRAM 行坐标验证连续的 K 段，以 C 的输出坐标核对 STORE。一个 MatMul
的全部 STORE 必须按 M/N 顺序组成无重叠的完整覆盖，组结束后该 DRAM
Tensor 才能被下一组作为输入读取。程序结束还会验证声明的结果均已生成。
这些状态验证使用实际指令和张量表，不需要编译器的 Tile IR 或分配器。

## FP32 数值契约

对每个 C 元素，按当前指令的局部 K 从 0 到 `k-1` 依次计算
`product = round_f32(A[i,p] * B[p,j])`，再执行
`C[i,j] = round_f32(C[i,j] + product)`。NumPy 只在相互独立的
`i,j` 位置向量化。乘法与加法分别舍入，累加保持 FP32，K 段之间
保留写在 SRAM 的部分和。此契约允许用对照组单独验证 FMA 和归约次序
造成的差异，适用于本项目的 Virtual NPU v1。

## 输出、错误与 Trace

输出按 Manifest 顺序读取已经初始化的 DRAM，导出二维命名 FP32 NPZ。
一个 Tensor 可通过多个输出名引用。NPZ 使用命名的 NPY 成员，输入解析和
输出写入均禁用 pickle。NPZ 与可选 JSON Trace 分别原子写入目标路径。
非法 Bundle、输入或指令返回非零，已有输出在程序验证失败时保留。

Trace 的成功记录包含原指令、执行状态及可用的 K 区间和输出块坐标。
失败时记录阶段，指令错误还记录全局序号、MatMul、局部事件和 Tile。
DMA 字节统计只累计实际复制的有效字节。命令计数和搬运量用于检查功能
与数据路径。当前模型只支持同步单缓冲、单矩阵单元和静态二维 FP32
MatMul。周期、并行硬件、异步 DMA、量化、动态形状与实机性能属于后续
或其他目标的验证范围。
