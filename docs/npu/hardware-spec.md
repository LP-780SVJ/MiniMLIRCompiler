# Day 17 · Virtual NPU v1 硬件规格

[virtual-npu-v1.json](../../configs/virtual-npu-v1.json) 是方案 C 自研后端的机器可读规格。第三周的 NPU Compute IR 可以表达完整矩阵乘法。第四周的 Tiling、SRAM Planner、DMA 与 Simulator 使用同一目标配置解释这些计算如何执行。

## 计算与存储

| 资源 | v1 规定 | 对编译器的影响 |
| --- | --- | --- |
| 矩阵单元 | 1 个，原生 MatMul 块的 M/N/K 上限各为 32 | Tiling 要把超出上限的计算拆成多条命令。单条命令的形状上限与边界 Tile 策略分别检查。 |
| 数值 | A、B、C 累加器均用 FP32，每元素 4 B | 类型合法化与 Simulator 共用同一数据约定。具体舍入、是否使用融合乘加等由 Virtual ISA 设计落实。 |
| SRAM | 256 KiB，即 262144 B，地址从 0 起的字节偏移 | 规划器需要证明所有同时存活的 Buffer 落在可用范围。 |
| SRAM 保留区 | v1 为 0 B，预算为总量减保留量 | 容量检查器从配置读取可用预算。 |
| SRAM 对齐 | Buffer 起始地址按 64 B 对齐，容量模型按 64 B 取整每份分配 | Buffer 的数据尺寸和地址空间占用可以不同。 |
| DRAM / Layout | 简化线性字节地址空间，二维 Row-major | 权重、输入和结果的 Tile 地址由基址、坐标和行跨度求得。 |
| DMA | 同步 DRAM↔SRAM 二维 Strided DMA，单缓冲优先 | 命令执行完成后才继续下一条，便于检查装载、计算和写回依赖。 |

矩阵命令的语义是 `C_tile += A_tile @ B_tile`。对同一个输出 Tile，编译器发出一次 ZERO 或等价初始化，按 K 块次序累加，最后写回结果。累加器从初始化起一直存活到最后一次写回读取完成。A、B 可以在它们最后一次计算读取之后替换成下一批输入。把这两类生命周期混为一谈会导致覆盖尚未消费的数据，或者丢掉先前的部分和。

MAC 是一次乘加，矩阵单元以一个原生计算块为粒度组织多次乘加。SRAM 保存当前矩阵单元要读写的数据和部分和，DMA 负责将数据在 DRAM 和 SRAM 之间移动。这个职责划分使计算命令可以专注于本地计算，编译器可以分别检查计算块、地址和搬运依赖。v1 的规格描述可执行行为，周期和带宽要等后续性能模型确定。

## Macro Tile 与 Micro Tile

Macro Tile 是 Tiling 选出的数据组织单位，受片上预算和复用策略约束。Micro Tile 是一个矩阵命令处理的计算块，受原生 M/N/K 上限约束。

`64×64×64` 的 A、B、C 在当前单缓冲模型合计 **48 KiB**。若把这个 Macro Tile 分成 `32×32×32` 的 Micro Tile，M/N/K 各分成两块，得到 4 个输出小块。每个输出小块累加两个 K 块，合计 8 次原生块计算。这个计数描述分块关系，具体 Buffer 复用和搬运由后续 Pass 决定。

Day 22 先验证可整除形状，再定义非整除时的 Padding、Tail 或拒绝策略。容量脚本的原生判断只检查三个维度的上限，完整调度仍需检查矩阵边界、可用片上地址与 K 归约依赖。

## 为什么 DMA 要有两个行跨度

从 Row-major 矩阵 `A[M,K]` 的 `(m0,k0)` 搬 `Mt×Kt` Tile 时，第 `r` 行：

```text
src(r) = dram_base + 4 * ((m0 + r) * K + k0)
dst(r) = sram_base + 4 * r * Kt
row_bytes = 4 * Kt
src_stride_bytes = 4 * K
dst_stride_bytes = 4 * Kt
rows = Mt
```

例如 `K=96, Kt=32`，每行搬 128 B，下一行在源矩阵中相隔 384 B，在紧凑的 SRAM Tile 中相隔 128 B。将源跨度误填为 128 B 会读到同一长行里的别处。写回同理，应按输出矩阵的行跨度写回。v1 的同步命令按下列依赖执行：

```text
ZERO(C) → [LOAD(A/B) → MATMUL_ACC(C)] × K 块数 → STORE(C)
```

## 容量模型与测试入口

[check_sram.py](../../python/hardware/check_sram.py) 分别将 A 的 `4MtKt`、B 的 `4KtNt`、C 的 `4MtNt` 向上对齐，再比较总量与可用 SRAM。双缓冲估算各为 A、B 增加一份副本，仍只保留一个 C。实际重叠执行需要异步 DMA 和依赖机制，当前硬件契约使用同步 DMA。

```bash
.venv/bin/python code/MiniMLIRCompiler/python/hardware/check_sram.py \
  --config code/MiniMLIRCompiler/configs/virtual-npu-v1.json \
  --m 64 --n 64 --k 64
```

退出码 `0` 表示容量足够，`1` 表示输入或 profile 有误，`2` 表示容量不足。`Within native MatMul limits` 独立报告形状上限。脚本估算同时持有这些 Buffer 的开销，地址分配、活跃区间和实际 DMA 访问范围由后续规划器验证。实测记录见 [Day 17 日志](coral/build-log.md)。

## 后续接口

- Day 18 将配置、声明的支持条件、运算语义与同步 DMA 约定写入 NPU IR 和后端 Pass 契约。
- Day 22–23 读取原生维度、SRAM、保留量与对齐，先输出 Tile 需求，再分配满足生命周期的地址。
- Day 24–27 生成含字节地址、行数和跨度的命令，按相同配置执行并与 ONNX Runtime 对照结果。

Coral 另有 [研究用 Target Profile](../../configs/coral-target.json)。它记录固定 Compiler 的默认 DTCM 和 ISA Features，由官方工具链实验使用。Virtual NPU 的容量脚本只接受 Virtual NPU v1 配置。
