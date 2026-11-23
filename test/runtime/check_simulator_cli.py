"""Build and run real v1 programs and inject failures across the CLI boundary."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
FIXTURES = PROJECT / "test/e2e"
TARGET = PROJECT / "configs/virtual-npu-v1.json"
WEIGHTED = "weighted_matmul"
MULTI = "matmul_multitile"
PACKAGES = {WEIGHTED: "compiled_weighted", MULTI: "compiled_multitile"}


def invoke(*args):
    return subprocess.run(args, cwd=PROJECT, text=True, capture_output=True)


def success(result):
    assert result.returncode == 0, result.stderr
    return result.stdout


def compile_program(opt, codegen, source, root, name, metadata=None):
    scheduled = root / f"{name}.scheduled.mlir"
    pipeline = (f"npu-tiling{{config={TARGET}}},npu-sram-plan{{config={TARGET}}},"
                f"npu-insert-dma{{config={TARGET}}},npu-schedule")
    if metadata:
        pipeline = f"mini-npu-compile,{pipeline}"
    success(invoke(opt, source, f"--pass-pipeline=builtin.module({pipeline})",
                   "--verify-each", "-o", scheduled))
    bundle = root / f"{name}.bundle"
    args = ["--model-metadata", metadata] if metadata else []
    success(invoke(codegen, "--input", scheduled, "--target", TARGET, "--out-dir", bundle, *args))
    return bundle


def simulator(bundle, inputs, actual, trace):
    return invoke(sys.executable, "-B", "-m", "runtime.virtual_npu.run", "--bundle", bundle,
                  "--inputs", inputs, "--outputs", actual, "--trace", trace)


def check_model(opt, codegen, model, root):
    source = root / f"{model}.mini.mlir"
    success(invoke(sys.executable, "-B", "python/onnx_importer/importer.py", "--input",
                   FIXTURES / "models" / f"{model}.onnx", "--output", source))
    bundle = compile_program(opt, codegen, source, root, model,
                             FIXTURES / "models" / f"{model}.metadata.json")
    fixture = FIXTURES / "results" / PACKAGES[model]
    for path in bundle.iterdir():
        assert path.read_bytes() == (fixture / path.name).read_bytes(), path
    actual, trace_path = root / "actual.npz", root / "trace.json"
    success(simulator(bundle, FIXTURES / "inputs" / f"{model}.npz", actual, trace_path))
    report = json.loads(success(invoke(sys.executable, "-B", "python/e2e/compare_outputs.py",
                                       "--reference", FIXTURES / "reference" / f"{model}.npz",
                                       "--actual", actual)))
    assert report["passed"] and report["outputs"]["Y"]["mismatched_elements"] == 0, report
    trace = json.loads(trace_path.read_text())
    count, tiles, k_offsets, dma_bytes = (5, 1, [0], 104) if model == WEIGHTED else (
        44, 4, [0, 32, 64], 112 * 1024)
    assert trace["status"] == "ok" and trace["executed_instructions"] == count
    assert trace["dma_bytes"] == dma_bytes and trace["opcode_counts"] == {
        "ZERO": tiles, "DMA_LOAD_2D": 2 * tiles * len(k_offsets),
        "MATMUL_TILE": tiles * len(k_offsets), "DMA_STORE_2D": tiles,
    }
    instructions = trace["instructions"]
    assert [cmd["sequence_index"] for cmd in instructions] == list(range(count))
    expected_origins = [[0, 0]] if tiles == 1 else [[0, 0], [0, 32], [32, 0], [32, 32]]
    for index, origin in enumerate(expected_origins):
        group = [cmd for cmd in instructions if cmd["output_tile"] == index]
        assert [cmd["opcode"] for cmd in group] == (
            ["ZERO"] + ["DMA_LOAD_2D", "DMA_LOAD_2D", "MATMUL_TILE"] * len(k_offsets)
            + ["DMA_STORE_2D"])
        computes = [cmd for cmd in group if cmd["opcode"] == "MATMUL_TILE"]
        assert [cmd["k_offset"] for cmd in computes] == k_offsets
        assert all(cmd["output_origin"] == origin for cmd in computes)
        assert {cmd["c_buffer_id"] for cmd in computes} == {group[0]["buffer_id"], group[-1]["buffer_id"]}
        assert group[-1]["output_origin"] == origin
    print(json.dumps(report))


def mutate_bundle(root, model, name, edit):
    folder = root / name
    shutil.copytree(FIXTURES / "results" / PACKAGES[model], folder)
    path = folder / "program.json"
    program = json.loads(path.read_text())
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    edit(program, manifest, folder)
    contents = json.dumps(program).encode()
    manifest["program"].update(size_bytes=len(contents), sha256=sha256(contents).hexdigest(),
                               instruction_count=program["instruction_count"])
    path.write_bytes(contents)
    manifest_path.write_text(json.dumps(manifest))
    return folder


def edit_instruction(index, **fields):
    def edit(program, manifest, folder):
        program["instructions"][index].update(fields)
    return edit


def edit_order(select):
    def edit(program, manifest, folder):
        instructions = select(program["instructions"])
        # Keep structural numbering valid so the runtime reaches the injected
        # lifetime or reduction fault. These fixtures each have one MatMul group.
        for index, cmd in enumerate(instructions):
            cmd.update(sequence_index=index, command_id=index, local_event_index=index, dependencies=[])
        program.update(instructions=instructions, instruction_count=len(instructions))
    return edit


def duplicate_zero(commands):
    return commands[:4] + [dict(commands[0], tile_index=1)] + commands[5:]


def check_negative(root):
    cases = [
        (WEIGHTED, "opcode", edit_instruction(0, opcode="MATMUL_FULL"), "load", "unknown opcode"),
        (WEIGHTED, "sram", edit_instruction(0, dst_sram=262144), "load", "invalid SRAM range"),
        (WEIGHTED, "dram", edit_instruction(1, src_dram=160), "load", "invalid DMA DRAM range"),
        (WEIGHTED, "stride", edit_instruction(1, src_stride_bytes=16), "load", "invalid DMA DRAM stride"),
        (WEIGHTED, "tile", edit_instruction(3, n=33), "load", "integer outside"),
        (WEIGHTED, "tensor", edit_instruction(1, tensor_id=99), "load", "unknown DMA tensor ID"),
        (WEIGHTED, "weights", lambda p, m, f: (f / "weights.bin").write_bytes(b""),
         "load", "file length mismatch"),
        (WEIGHTED, "schema", lambda p, m, f: m.update(schema_version=2), "load", "unsupported schema"),
        (WEIGHTED, "no-zero", edit_order(lambda c: c[1:]), "execute", "LOAD before ZERO"),
        (WEIGHTED, "no-load", edit_order(lambda c: c[:1] + c[2:]), "execute", "requires both live A/B"),
        (WEIGHTED, "no-store", edit_order(lambda c: c[:-1]), "execute", "before STORE"),
        (WEIGHTED, "buffer", edit_instruction(3, a_buffer_id=91), "execute", "A Buffer ID"),
        (WEIGHTED, "buffer-address", edit_instruction(3, a_sram=192), "execute", "A SRAM address"),
        (WEIGHTED, "overlap", edit_instruction(1, dst_sram=0), "execute", "overlaps live Buffer ID"),
        (WEIGHTED, "unwritten-tensor", edit_instruction(1, tensor_id=2, src_dram=128, src_stride_bytes=16),
         "execute", "before its complete writeback"),
        (MULTI, "stale-load", edit_order(lambda c: c[:4] + c[5:]), "execute", "requires both live A/B"),
        (MULTI, "early-store", edit_order(lambda c: c[:4] + [dict(c[10], tile_index=0)]),
         "execute", "STORE before the K reduction is complete"),
        (MULTI, "duplicate-zero", edit_order(duplicate_zero), "execute", "repeated ZERO"),
        (MULTI, "wrong-k", edit_instruction(4, src_dram=0), "execute", "K reduction requires offset 32"),
        (MULTI, "stale-id", edit_instruction(4, buffer_id=1), "execute", "reused logical Buffer ID"),
        (MULTI, "wrong-store", edit_instruction(10, dst_dram=49280), "execute", "computed output region"),
        (MULTI, "missing-region", edit_order(lambda c: c[:11]), "execute", "without complete DRAM writeback"),
    ]
    for model, name, edit, phase, diagnostic in cases:
        bundle = mutate_bundle(root, model, name, edit)
        actual, trace = root / f"{name}.npz", root / f"{name}.trace.json"
        result = simulator(bundle, FIXTURES / "inputs" / f"{model}.npz", actual, trace)
        assert result.returncode == 1 and diagnostic in result.stderr, (name, result.stderr)
        assert not actual.exists()
        report = json.loads(trace.read_text())
        assert report["status"] == "error" and report["phase"] == phase
        assert diagnostic in report["error"]
        if name == "stale-load":
            failed = report["instructions"][-1]
            assert failed["status"] == "error" and failed["opcode"] == "MATMUL_TILE"
            assert failed["sequence_index"] == 5 and failed["tile_index"] == 1

    bundle = FIXTURES / "results" / PACKAGES[WEIGHTED]
    for name, arrays, diagnostic in (
        ("input-name", {"other": np.ones((2, 3), dtype=np.float32)}, "NPZ names"),
        ("input-dtype", {"X": np.ones((2, 3), dtype=np.float64)}, "must be an FP32 array"),
        ("input-shape", {"X": np.ones((3, 2), dtype=np.float32)}, "does not match"),
    ):
        inputs, actual, trace = (root / f"{name}{suffix}" for suffix in (".npz", ".out.npz", ".json"))
        np.savez(inputs, **arrays)
        actual.write_bytes(b"previous output")
        result = simulator(bundle, inputs, actual, trace)
        assert result.returncode == 1 and diagnostic in result.stderr, result.stderr
        assert actual.read_bytes() == b"previous output"
    print(f"PASS: {len(cases)} invalid packages or executions and 3 bad input bindings")


def check_tails(opt, codegen, root):
    for axis, (m, n, k) in zip("mnk", [(65, 9, 7), (7, 65, 9), (32, 32, 65)], strict=True):
        source = root / f"{axis}.mlir"
        at, bt, ct = f"tensor<{m}x{k}xf32>", f"tensor<{k}x{n}xf32>", f"tensor<{m}x{n}xf32>"
        source.write_text(
            f"func.func @main(%a: {at}, %b: {bt}) -> {ct} {{\n"
            f'  %c = "npu.matmul"(%a, %b) : ({at}, {bt}) -> {ct}\n'
            f"  return %c : {ct}\n}}\n")
        bundle = compile_program(opt, codegen, source, root, axis)
        inputs, actual, trace = (root / f"{axis}{suffix}" for suffix in (".npz", ".out.npz", ".json"))
        np.savez(inputs, input_0=np.ones((m, k), dtype=np.float32),
                 input_1=np.ones((k, n), dtype=np.float32))
        success(simulator(bundle, inputs, actual, trace))
        with np.load(actual, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["output_0"], np.full((m, n), k, dtype=np.float32))
        executed = json.loads(trace.read_text())["instructions"]
        assert [cmd[axis] for cmd in executed if cmd["opcode"] == "MATMUL_TILE"] == [32, 32, 1]
    print("PASS: M, N and K tails with a final active dimension of 1")


def check_chain(opt, codegen, root):
    source = root / "chain.mlir"
    source.write_text("""
    func.func @main(%x: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
      %w = arith.constant dense<1.0> : tensor<64x64xf32>
      %h = "npu.matmul"(%x, %w) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
      %y = "npu.matmul"(%h, %w) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
      return %y, %h, %y : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
    }
    """)
    bundle = compile_program(opt, codegen, source, root, "chain")
    inputs, actual, trace = root / "inputs.npz", root / "actual.npz", root / "trace.json"
    np.savez(inputs, input_0=np.ones((64, 64), dtype=np.float32))
    success(simulator(bundle, inputs, actual, trace))
    with np.load(actual, allow_pickle=False) as archive:
        assert archive.files == ["output_0", "output_1", "output_2"]
        for name, value in zip(archive.files, (4096, 64, 4096), strict=True):
            np.testing.assert_array_equal(archive[name], np.full((64, 64), value, dtype=np.float32))
    instructions = json.loads(trace.read_text())["instructions"]
    assert len(instructions) == 64
    assert instructions[31]["opcode"] == "DMA_STORE_2D"
    assert instructions[32]["matmul_index"] == 1
    assert instructions[32]["buffer_id"] == instructions[0]["buffer_id"]
    assert instructions[33]["tensor_id"] == instructions[31]["tensor_id"]
    print("PASS: producer writeback, reused SRAM IDs across MatMuls and three named outputs")


if __name__ == "__main__":
    opt, codegen, case = sys.argv[1:]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        if case == "negative":
            check_negative(root)
        elif case == "tails":
            check_tails(opt, codegen, root)
        elif case == "chain":
            check_chain(opt, codegen, root)
        else:
            assert case in PACKAGES, case
            check_model(opt, codegen, case, root)
