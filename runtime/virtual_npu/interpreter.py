"""Execute the Day 26 program and check the lifetime of its logical buffers."""

from dataclasses import dataclass

from .matrix_unit import FP32_BYTES, accumulate_tile
from .memory import ByteMemory, ExecutionError, copy_2d
from .tensor_io import bind_dram, collect_outputs


def check(condition, message):
    if not condition:
        raise ExecutionError(message)


@dataclass(frozen=True)
class Buffer:
    key: tuple[int, int]
    address: int
    shape: tuple[int, int]

    @property
    def size(self):
        return FP32_BYTES * self.shape[0] * self.shape[1]


@dataclass(frozen=True)
class LoadedTile:
    buffer: Buffer
    tensor_id: int
    row: int
    col: int


class MatMulExecution:
    """State scoped to one serial command group, with at most three live buffers."""

    def __init__(self, index, manifest, dram, sram, available):
        self.index = index
        self.tensors = manifest["tensors"]
        self.alignment = manifest["hardware"]["sram_alignment_bytes"]
        self.dram, self.sram, self.available = dram, sram, available
        self.seen = set()
        self.c = None
        self.loads = []
        self.next_tile = 0
        self.next_k = 0
        self.origin = None
        self.operands = None
        self.result = None
        self.output_tile = 0
        self.store_row, self.store_col, self.band_rows = 0, 0, 0

    def _allocation_end(self, buffer):
        size = (buffer.size + self.alignment - 1) // self.alignment * self.alignment
        return buffer.address + size

    def _new_buffer(self, buffer_id, address, shape):
        buffer = Buffer((self.index, buffer_id), address, shape)
        check(buffer.key not in self.seen, f"reused logical Buffer ID {buffer.key}")
        end = self._allocation_end(buffer)
        self.sram.check_range(address, end - address)
        active = [self.c, *(tile.buffer for tile in self.loads)]
        for live in active:
            if live and address < self._allocation_end(live) and live.address < end:
                raise ExecutionError(f"SRAM write overlaps live Buffer ID {live.key}")
        return buffer

    def _release(self, buffer):
        self.sram.invalidate(buffer.address, buffer.size)

    def _dram_window(self, tensor_id, address):
        tensor = self.tensors[tensor_id]
        row, byte_col = divmod(address - tensor["base"], tensor["row_stride_bytes"])
        return row, byte_col // FP32_BYTES

    def _zero(self, cmd):
        check(self.c is None, "repeated ZERO before STORE of the current C")
        check(cmd["tile_index"] == self.next_tile, "ZERO does not start the next Tile")
        buffer = self._new_buffer(cmd["buffer_id"], cmd["dst_sram"], (cmd["rows"], cmd["cols"]))
        self.sram.write(buffer.address, bytes(buffer.size))
        self.seen.add(buffer.key)
        self.c = buffer
        self.next_k = 0
        self.origin = None
        return {}

    def _load(self, cmd):
        check(self.c is not None, "LOAD before ZERO")
        check(cmd["tile_index"] == self.next_tile and len(self.loads) < 2,
              "LOAD does not belong to the next A/B pair")
        tensor_id = cmd["tensor_id"]
        check(tensor_id in self.available,
              f"LOAD reads DRAM Tensor {tensor_id} before its complete writeback")
        rows, width = cmd["rows"], cmd["row_bytes"]
        buffer = self._new_buffer(cmd["buffer_id"], cmd["dst_sram"], (rows, width // FP32_BYTES))
        copy_2d(self.dram, self.sram, cmd["src_dram"], buffer.address,
                rows, width, cmd["src_stride_bytes"], cmd["dst_stride_bytes"])
        row, col = self._dram_window(tensor_id, cmd["src_dram"])
        self.seen.add(buffer.key)
        self.loads.append(LoadedTile(buffer, tensor_id, row, col))
        return {"role": "A" if len(self.loads) == 1 else "B", "dram_origin": [row, col]}

    def _check_buffer(self, buffer, cmd, role, shape):
        check(buffer.key == (self.index, cmd[f"{role}_buffer_id"]),
              f"{role.upper()} Buffer ID does not match the live buffer")
        check(buffer.address == cmd[f"{role}_sram"] and buffer.shape == shape,
              f"{role.upper()} SRAM address or shape does not match the live buffer")

    def _matmul(self, cmd):
        check(self.c is not None, "MATMUL requires ZERO for C")
        check(len(self.loads) == 2, "MATMUL requires both live A/B LOADs")
        check(cmd["tile_index"] == self.next_tile, "MATMUL has the wrong Tile index")
        a, b = self.loads
        m, n, k = cmd["m"], cmd["n"], cmd["k"]
        for role, buffer, shape in (("a", a.buffer, (m, k)), ("b", b.buffer, (k, n)),
                                    ("c", self.c, (m, n))):
            self._check_buffer(buffer, cmd, role, shape)

        operands = (a.tensor_id, b.tensor_id)
        check(self.operands is None or self.operands == operands, "LOAD tensor IDs changed within MatMul")
        a_shape, b_shape = (self.tensors[tensor_id]["shape"] for tensor_id in operands)
        end_k = self.next_k + k
        check(a_shape[1] == b_shape[0] and end_k <= a_shape[1] and a.col == b.row == self.next_k,
              f"K reduction requires offset {self.next_k} in both operands (A={a.col}, B={b.row})")
        origin = (a.row, b.col)
        check(self.origin is None or self.origin == origin, "K segments target different output regions")

        accumulate_tile(self.sram, cmd)
        for tile in self.loads:
            self._release(tile.buffer)
        self.loads.clear()
        self.origin, self.operands = origin, operands
        start_k, self.next_k = self.next_k, end_k
        self.next_tile += 1
        return {"k_offset": start_k, "k_end": end_k, "output_origin": list(origin)}

    def _store(self, cmd):
        check(self.c is not None and self.operands is not None and not self.loads and
              self.next_k == self.tensors[self.operands[0]]["shape"][1],
              "STORE before the K reduction is complete")
        check(cmd["tile_index"] == self.next_tile - 1, "STORE has the wrong Tile index")
        rows, width = cmd["rows"], cmd["row_bytes"]
        check(self.c.key == (self.index, cmd["buffer_id"]), "STORE Buffer ID does not match live C")
        check(cmd["src_sram"] == self.c.address and (rows, width // FP32_BYTES) == self.c.shape,
              "STORE SRAM address or shape does not match live C")
        tensor_id = cmd["tensor_id"]
        check(tensor_id not in self.available and (self.result is None or self.result == tensor_id),
              "STORE result Tensor ID changed or was already produced")
        output = self.tensors[tensor_id]
        expected_shape = [self.tensors[self.operands[0]]["shape"][0],
                          self.tensors[self.operands[1]]["shape"][1]]
        check(output["shape"] == expected_shape, "STORE tensor shape does not match MatMul")
        row, col = self._dram_window(tensor_id, cmd["dst_dram"])
        check((row, col) == self.origin, "STORE destination does not match the computed output region")
        check((row, col) == (self.store_row, self.store_col) and
              (col == 0 or rows == self.band_rows),
              "STORE regions must cover the result once in M/N order")

        copy_2d(self.sram, self.dram, self.c.address, cmd["dst_dram"],
                rows, width, cmd["src_stride_bytes"], cmd["dst_stride_bytes"])
        self._release(self.c)
        self.c, self.result = None, tensor_id
        self.band_rows = rows
        self.store_col += width // FP32_BYTES
        if self.store_col == output["shape"][1]:
            self.store_row += rows
            self.store_col = 0
        return {"output_origin": [row, col], "k_end": self.next_k}

    def step(self, cmd):
        handler = {
            "ZERO": self._zero,
            "DMA_LOAD_2D": self._load,
            "MATMUL_TILE": self._matmul,
            "DMA_STORE_2D": self._store,
        }[cmd["opcode"]]
        details = {"output_tile": self.output_tile, **handler(cmd)}
        if cmd["opcode"] == "DMA_STORE_2D":
            self.output_tile += 1
        return details

    def finish(self):
        check(self.c is None and not self.loads, f"MatMul {self.index} ended with live SRAM before STORE")
        check(self.result is not None and self.store_col == 0 and
              self.store_row == self.tensors[self.result]["shape"][0],
              f"MatMul {self.index} ended without complete DRAM writeback")
        self.available.add(self.result)


def execute(package, inputs, trace=None):
    """Execute an already loaded package and return named FP32 output arrays."""
    manifest = package.manifest
    dram = bind_dram(package, inputs)
    sram = ByteMemory(manifest["hardware"]["sram_bytes"], "SRAM")
    available = {tensor["id"] for tensor in manifest["tensors"]
                 if tensor["kind"] in {"input", "constant"}}
    group = None
    for cmd in package.program["instructions"]:
        index = cmd["matmul_index"]
        try:
            if group is None or group.index != index:
                if group is not None:
                    group.finish()
                group = MatMulExecution(index, manifest, dram, sram, available)
            details = group.step(cmd)
        except ExecutionError as error:
            if trace is not None:
                trace.append({**cmd, "status": "error", "error": str(error)})
            raise ExecutionError(
                f"instruction {cmd['sequence_index']} {cmd['opcode']} "
                f"(MatMul {index}, event {cmd['local_event_index']}, Tile {cmd['tile_index']}): {error}"
            ) from error
        if trace is not None:
            trace.append({**cmd, **details, "status": "ok"})
    try:
        group.finish()
        check(len(available) == len(manifest["tensors"]),
              "program ended without producing every declared result Tensor")
        return collect_outputs(manifest, dram)
    except ExecutionError as error:
        raise ExecutionError(f"after instruction {cmd['sequence_index']}: {error}") from error
