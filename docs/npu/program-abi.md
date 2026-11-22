# Virtual NPU v1 程序 ABI

程序包根目录固定包含三份普通文件。运行时使用程序包即可初始化，编译时
的 MLIR、ONNX、Metadata 和 Target Profile 可以留在编译环境。

| 文件 | 运行时用途 |
| --- | --- |
| `program.json` | 按数组顺序解释 [Virtual ISA v1](isa-spec.md)。 |
| `weights.bin` | 按权重条目复制字节，初始化 DRAM 常量区。 |
| `manifest.json` | 确定目标、校验文件、创建内存、绑定输入及读取输出。 |

## Manifest Schema

| 字段 | v1 契约 |
| --- | --- |
| `schema_version`、`program_schema_version` | 均为 1，分别标识包和指令格式。 |
| `target`、`target_profile_sha256` | `virtual-npu-v1` 与编译时 Profile 原始字节的 SHA-256。摘要用于构建溯源。 |
| `entry`、`dtype`、`byte_order`、`layout` | `main`、`float32`、`little`、`row_major`。 |
| `schedule_contract` | `serial-mnk-single-buffer-v1`，与 Program 一致。 |
| `hardware` | `native_tile`（M、N、K）、`sram_bytes`、`reserved_bytes`、`sram_alignment_bytes`、`dram_bytes`、`dram_alignment_bytes`（64）。 |
| `tensors` | 按 ID 升序的完整描述符。每项为 `id`、`kind`、`base`、`shape`（行、列）、`row_stride_bytes`、`size_bytes`。 |
| `inputs`、`outputs` | 按函数参数及返回值顺序的 `{name, tensor_id}`。同一方向名称必须唯一，一个 Tensor 可以通过多个输出名称返回。 |
| `weights` | `file`、`size_bytes`、`sha256` 和 `entries`。每个常量的条目为 `tensor_id`、`file_offset`、`size_bytes` 和 `dram_base`。 |
| `program` | `file`、`instruction_count`、`size_bytes`、`sha256`。 |

DRAM Tensor ID 从 0 连续。类别为 `input`、`constant`、`intermediate`
或 `output`。每个起点按 64 B 对齐，张量区间互不重叠。Rank 固定为 2，
`row_stride_bytes = 4 * cols`，`size_bytes = rows * row_stride_bytes`。
DRAM 总长为最后一个 Tensor 的末尾，输出列表允许重用输入或常量的 ID。

Metadata 可选。CodeGen 根据 `input_names`、`output_names` 的序号及
`inputs`、`outputs` 的形状绑定名称，同时检查版本、目标和 dtype。
仅当交换改变形状或接口结构时，该检查能检测错序。相同形状对应的原始名称
由 Metadata 提供。省略 Metadata 时使用 `input_0`、`output_0` 等。

## Weight Packing

按 `DRAMLayout.constantIds` 顺序写入。每个常量的第一个字节按 64 B
对齐，间隔补零，文件结束于最后一个有效字节。Splat 按逻辑元素数展开，
每元素使用原始 IEEE-754 FP32 位模式并固定 Little-endian。共享 Constant
SSA 值只打包一次。零常量仍包含零长度的 `weights.bin` 和空条目列表。

例如加权模型将 `weights.bin[0:48]` 复制到
`DRAM[64:112]`。文件偏移描述容器中的排列，DRAM 地址描述运行时
布局。文件保留常量，DRAM 同时包含输入、中间结果及输出空间。

## Day 27 加载流程

从 [独立加载器](../../python/runtime/validate_package.py) 调用
`load_package(path)`，得到已检查的 `manifest`、`program` 和原始
`weights`。加载失败抛出 `PackageError`。文件的 SHA-256 对精确落盘字节
计算，可发现损坏或错配。运行时按如下顺序使用数据。

1. 按 `hardware` 创建 DRAM 与 SRAM。
2. 按权重条目的 `file_offset`、`size_bytes`、`dram_base` 初始化 DRAM。
3. 按有序 `inputs` 绑定调用者提供的 FP32 张量并检查形状。
4. 按 Program 顺序解释指令，再按有序 `outputs` 读取并命名结果。

独立加载器检查格式和边界，FP32 执行与输出比较由后续阶段实现。
发布时 CodeGen 先写入同级临时目录，再调用随工具构建的同一加载器。
在 macOS 或支持 `renameat2` 的 Linux 上，用原子移动发布，新路径存在
时返回错误。导出工具要求 PATH 可找到 Python 3.7 或以上版本，目标父目录
需已存在。CodeGen 的 Python 回归测试使用 Python 3.10 或以上版本。
