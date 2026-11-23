from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from python.runtime.validate_package import load_package
from runtime.virtual_npu.interpreter import execute
from runtime.virtual_npu.matrix_unit import accumulate_tile
from runtime.virtual_npu.memory import ByteMemory
from runtime.virtual_npu.tensor_io import load_inputs, save_outputs


PROJECT = Path(__file__).resolve().parents[2]


def manual_bundle(folder):
    payloads = [
        dict(opcode="ZERO", buffer_id=2, dst_sram=128, rows=2, cols=2),
        dict(opcode="DMA_LOAD_2D", buffer_id=0, tensor_id=0, src_dram=0, dst_sram=0,
             rows=2, row_bytes=12, src_stride_bytes=12, dst_stride_bytes=12),
        dict(opcode="DMA_LOAD_2D", buffer_id=1, tensor_id=1, src_dram=64, dst_sram=64,
             rows=3, row_bytes=8, src_stride_bytes=8, dst_stride_bytes=8),
        dict(opcode="MATMUL_TILE", a_buffer_id=0, b_buffer_id=1, c_buffer_id=2,
             a_sram=0, b_sram=64, c_sram=128, m=2, n=2, k=3, accumulate=True),
        dict(opcode="DMA_STORE_2D", buffer_id=2, tensor_id=2, src_sram=128, dst_dram=128,
             rows=2, row_bytes=8, src_stride_bytes=8, dst_stride_bytes=8),
    ]
    config = dict(schema_version=1, target="virtual-npu-v1",
                  schedule_contract="serial-mnk-single-buffer-v1")
    program = {**config, "instruction_count": 5, "instructions": [
        dict(cmd, sequence_index=i, command_id=i, matmul_index=0, local_event_index=i,
             tile_index=0, dependencies=deps)
        for i, (cmd, deps) in enumerate(zip(payloads, [[], [], [], [0, 1, 2], [3]], strict=True))
    ]}
    program_bytes = json.dumps(program).encode()
    manifest = dict(
        config, program_schema_version=1, entry="main", dtype="float32", layout="row_major",
        byte_order="little", target_profile_sha256=sha256(
            (PROJECT / "configs/virtual-npu-v1.json").read_bytes()).hexdigest(),
        hardware=dict(native_tile=[32, 32, 32], sram_bytes=262144, reserved_bytes=0,
                      sram_alignment_bytes=64, dram_bytes=144, dram_alignment_bytes=64),
        tensors=[
            dict(id=i, base=64 * i, shape=[rows, cols], kind=kind,
                 row_stride_bytes=cols * 4, size_bytes=rows * cols * 4)
            for i, (rows, cols, kind) in enumerate([(2, 3, "input"), (3, 2, "input"), (2, 2, "output")])
        ],
        inputs=[{"name": "A", "tensor_id": 0}, {"name": "B", "tensor_id": 1}],
        outputs=[{"name": "C", "tensor_id": 2}],
        program=dict(file="program.json", instruction_count=5, size_bytes=len(program_bytes),
                     sha256=sha256(program_bytes).hexdigest()),
        weights=dict(file="weights.bin", size_bytes=0, sha256=sha256(b"").hexdigest(), entries=[]),
    )
    (folder / "program.json").write_bytes(program_bytes)
    (folder / "weights.bin").write_bytes(b"")
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return load_package(folder)


class InterpreterTest(unittest.TestCase):
    def test_manual_five_instructions(self):
        with TemporaryDirectory() as directory:
            package = manual_bundle(Path(directory))
            inputs = {
                "A": np.array([[1, 2, 3], [4, 5, 6]], dtype=">f4"),
                "B": np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32, order="F"),
            }
            trace = []
            outputs = execute(package, inputs, trace)
        np.testing.assert_array_equal(outputs["C"], [[58, 64], [139, 154]])
        self.assertEqual(outputs["C"].dtype, np.dtype("float32"))
        self.assertEqual([cmd["sequence_index"] for cmd in trace], list(range(5)))
        self.assertEqual(trace[3]["k_end"], 3)
        self.assertEqual(trace[4]["output_origin"], [0, 0])

    def test_rounding_separates_multiply_and_add(self):
        sram = ByteMemory(192, "SRAM")
        next_after_one = np.float32(1 + 2**-23)
        negative_product = np.float32(-(1 + 2**-22))
        for address, value in [(0, next_after_one), (64, next_after_one), (128, negative_product)]:
            sram.write(address, np.asarray(value, dtype="<f4").tobytes())
        accumulate_tile(sram, dict(a_sram=0, b_sram=64, c_sram=128, m=1, n=1, k=1))
        self.assertEqual(sram.read(128, 4), bytes(4))  # A fused multiply-add leaves 2**-46.

    def test_accumulate_preserves_serial_k_order(self):
        sram = ByteMemory(192, "SRAM")
        values = [(0, [2**24, 1, -2**24]), (64, [1, 1, 1]), (128, [0])]
        for address, array in values:
            sram.write(address, np.asarray(array, dtype="<f4").tobytes())
        accumulate_tile(sram, dict(a_sram=0, b_sram=64, c_sram=128, m=1, n=1, k=3))
        self.assertEqual(sram.read(128, 4), bytes(4))

    def test_named_npz(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tensors.npz"
            outputs = {"file": np.array([[1]], dtype=np.float32),
                       "model/allow_pickle": np.array([[2]], dtype=np.float32)}
            save_outputs(path, outputs)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive.files, list(outputs))
                for name, value in outputs.items():
                    np.testing.assert_array_equal(archive[name], value)
            loaded = load_inputs(path, [{"name": name} for name in outputs])
            for name in outputs:
                np.testing.assert_array_equal(loaded[name], outputs[name])
