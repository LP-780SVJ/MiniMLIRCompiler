"""Check serialized Day 24 Command IR and its verified data movement."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


PROJECT = Path(__file__).resolve().parents[3]
PROFILE = PROJECT / "configs/virtual-npu-v1.json"
RESULTS = PROJECT / "test/e2e/results"
HEX = "--mlir-print-elementsattrs-with-hex-if-larger=0"
ATTR_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|-?\d+(?:\s*:\s*i64)?|[a-z_][a-z_0-9]*|[{}\[\],=]')


def run(mini, ir, *args):
    return subprocess.run([mini, "-", *args], input=ir, capture_output=True, text=True)


def success(result):
    assert result.returncode == 0, result.stderr
    return result.stdout


def reject(result, diagnostic):
    assert result.returncode == 1 and diagnostic in result.stderr, result
    assert not result.stdout, result.stdout


def dma(mini, ir, profile=PROFILE):
    return success(run(mini, ir, f"--npu-insert-dma=config={profile}", "--verify-each", HEX))


def make_matmul(m, n, k):
    lhs, rhs, out = f"tensor<{m}x{k}xf32>", f"tensor<{k}x{n}xf32>", f"tensor<{m}x{n}xf32>"
    return (
        f"func.func @main(%lhs: {lhs}, %rhs: {rhs}) -> {out} {{\n"
        f'  %out = "npu.matmul"(%lhs, %rhs) : ({lhs}, {rhs}) -> {out}\n'
        f"  return %out : {out}\n}}\n"
    )


def plan(mini, ir, profile=PROFILE):
    tiled = success(run(mini, ir, f"--npu-tiling=config={profile}", HEX))
    return success(run(mini, tiled, f"--npu-sram-plan=config={profile}", HEX))


def named_attr(text, name):
    match = re.search(rf"(?<![\w.]){re.escape(name)} = ([{{\[])", text)
    assert match, (name, text[:100])
    start = match.start(1)
    depth = 0
    for marker in re.finditer(r'"(?:[^"\\]|\\.)*"|[{}\[\]]', text[start:]):
        token = marker.group()
        if token in ("{", "["):
            depth += 1
        elif token in ("}", "]"):
            depth -= 1
            if not depth:
                return text[start : start + marker.end()]
    raise AssertionError(f"unterminated {name}")


def decode(text):
    # This reads only the compact DictionaryAttr subset emitted by MLIR. The
    # real parser and Verifier re-read every artifact before we inspect it.
    tokens = ATTR_TOKEN.findall(text)
    index = 0

    def take():
        nonlocal index
        value = tokens[index]
        index += 1
        return value

    def parse():
        value = take()
        if value == "{":
            fields = {}
            while tokens[index] != "}":
                key = take()
                assert key not in fields and take() == "="
                fields[key] = parse()
                if tokens[index] == ",":
                    take()
                else:
                    assert tokens[index] == "}"
            take()
            return fields
        if value == "[":
            entries = []
            while tokens[index] != "]":
                entries.append(parse())
                if tokens[index] == ",":
                    take()
                else:
                    assert tokens[index] == "]"
            take()
            return entries
        if value.endswith("i64"):
            return int(value.split(":")[0])
        return json.loads(value)

    result = parse()
    assert index == len(tokens)
    return result


def encode(value):
    if isinstance(value, (bool, str)):
        return json.dumps(value)
    if isinstance(value, int):
        return f"{value} : i64"
    if isinstance(value, list):
        return "[" + ", ".join(map(encode, value)) + "]"
    return "{" + ", ".join(f"{key} = {encode(item)}" for key, item in value.items()) + "}"


def strip_commands(ir):
    layout = named_attr(ir, "npu.dram_layout")
    ir = ir.replace(f" attributes {{npu.dram_layout = {layout}}}", "", 1)
    lines = []
    for line in ir.splitlines(keepends=True):
        if '"npu.command_matmul"' in line:
            commands = named_attr(line, "commands")
            line = line.replace(f"commands = {commands}, ", "", 1)
            line = re.sub(r"(?:lhs|rhs|result)_tensor_id = \d+ : i64, ", "", line)
            line = line.replace('"npu.command_matmul"', '"npu.sram_planned_matmul"')
        lines.append(line)
    return "".join(lines)


def check_layout(layout, expected):
    assert layout["schema_version"] == 1 and layout["tensor_alignment_bytes"] == 64
    assert len(layout["tensors"]) == len(expected)
    cursor = 0
    for ordinal, (record, spec) in enumerate(zip(layout["tensors"], expected, strict=True)):
        kind, source_index, rows, cols = spec
        base = (cursor + 63) // 64 * 64
        size = 4 * rows * cols
        assert record == {
            "id": ordinal, "kind": kind, "source_index": source_index,
            "base": base, "rows": rows, "cols": cols, "row_stride_bytes": 4 * cols,
            "size_bytes": size, "dtype": "float32", "layout": "row_major",
        }, record
        cursor = base + size
    assert layout["total_bytes"] == cursor
    for table, kind in (("input_ids", "input"), ("constant_ids", "constant")):
        assert layout[table] == [index for index, spec in enumerate(expected) if spec[0] == kind]


def check_commands(line, tensors, expected_ids):
    ids = tuple(int(re.search(rf"{role}_tensor_id = (\d+) : i64", line)[1])
                for role in ("lhs", "rhs", "result"))
    assert ids == expected_ids, ids
    tensor_a, tensor_b, tensor_c = (tensors[tensor_id] for tensor_id in ids)
    memory = decode(named_attr(line, "memory_plan"))
    allocations = {alloc["id"]: alloc for alloc in memory["allocations"]}
    tiles = decode(named_attr(line, "tiles"))
    commands = decode(named_attr(line, "commands"))
    expected = []

    for index, (tile, binding) in enumerate(zip(tiles, memory["tile_bindings"], strict=True)):
        assert binding["tile_index"] == index
        m, n, k = (tile[f"{axis}_size"] for axis in "mnk")
        m0, n0, k0 = (tile[f"{axis}_offset"] for axis in "mnk")
        alloc_a, alloc_b, alloc_c = (allocations[binding[f"{role}_buffer_id"]] for role in "abc")

        def expect(opcode, *live, **fields):
            event = len(expected)
            assert all(alloc["live_begin"] <= event < alloc["live_end"] for alloc in live)
            expected.append({"opcode": opcode, "event_index": event, "tile_index": index, **fields})

        def expect_dma(opcode, tensor, alloc, row, col, rows, cols, loading):
            assert 0 <= row < row + rows <= tensor["rows"]
            assert 0 <= col < col + cols <= tensor["cols"]
            row_bytes = 4 * cols
            stride = 4 * tensor["cols"]
            dram = tensor["base"] + 4 * (row * tensor["cols"] + col)
            assert dram + (rows - 1) * stride + row_bytes <= tensor["base"] + tensor["size_bytes"]
            assert rows * row_bytes == alloc["raw_bytes"]
            endpoints = (
                {"src_dram": dram, "dst_sram": alloc["address"],
                 "src_stride_bytes": stride, "dst_stride_bytes": row_bytes}
                if loading else
                {"src_sram": alloc["address"], "dst_dram": dram,
                 "src_stride_bytes": row_bytes, "dst_stride_bytes": stride}
            )
            expect(opcode, alloc, tensor_id=tensor["id"], buffer_id=alloc["id"],
                   rows=rows, row_bytes=row_bytes, **endpoints)

        if tile["is_first_k"]:
            expect("ZERO", alloc_c, buffer_id=alloc_c["id"], dst_sram=alloc_c["address"],
                   rows=m, cols=n)
        expect_dma("DMA_LOAD_2D", tensor_a, alloc_a, m0, k0, m, k, True)
        expect_dma("DMA_LOAD_2D", tensor_b, alloc_b, k0, n0, k, n, True)
        expect("MATMUL_TILE", alloc_a, alloc_b, alloc_c,
               a_buffer_id=alloc_a["id"], b_buffer_id=alloc_b["id"], c_buffer_id=alloc_c["id"],
               a_sram=alloc_a["address"], b_sram=alloc_b["address"], c_sram=alloc_c["address"],
               m=m, n=n, k=k, accumulate=True)
        if tile["is_last_k"]:
            expect_dma("DMA_STORE_2D", tensor_c, alloc_c, m0, n0, m, n, False)

    assert len(commands) == len(expected), (len(commands), len(expected))
    for emitted, required in zip(commands, expected, strict=True):
        assert emitted == required, (emitted, required)
    return commands


def inspect(mini, planned, command_ir, specs, outputs, input_triplets):
    assert success(run(mini, command_ir, HEX)) == command_ir
    assert strip_commands(command_ir) == success(run(mini, planned, HEX))
    layout = decode(named_attr(command_ir, "npu.dram_layout"))
    check_layout(layout, specs)
    assert layout["output_ids"] == outputs, layout
    lines = [line for line in command_ir.splitlines() if '"npu.command_matmul"' in line]
    return [check_commands(line, layout["tensors"], ids)
            for line, ids in zip(lines, input_triplets, strict=True)]


def counters(commands):
    histogram = Counter(entry["opcode"] for entry in commands)
    traffic = sum(entry["rows"] * entry["row_bytes"] for entry in commands
                  if entry["opcode"] in ("DMA_LOAD_2D", "DMA_STORE_2D"))
    return histogram, traffic


def check_positive(mini):
    for label, shape in (
        ("M tail", (65, 9, 7)), ("N tail", (7, 65, 9)), ("K tail", (32, 32, 65)),
    ):
        m, n, k = shape
        planned = plan(mini, make_matmul(m, n, k))
        command_ir = dma(mini, planned)
        commands, = inspect(mini, planned, command_ir,
                            [("input", 0, m, k), ("input", 1, k, n), ("output", 0, m, n)],
                            [2], [(0, 1, 2)])
        last_a = [cmd for cmd in commands if cmd["opcode"] == "DMA_LOAD_2D" and cmd["tensor_id"] == 0][-1]
        last_b = [cmd for cmd in commands if cmd["opcode"] == "DMA_LOAD_2D" and cmd["tensor_id"] == 1][-1]
        last_store = commands[-1]
        if label == "M tail":
            assert last_a["rows"] == last_store["rows"] == 1
        elif label == "N tail":
            assert last_b["row_bytes"] == last_store["row_bytes"] == 4
        else:
            assert (last_a["rows"], last_a["row_bytes"], last_a["src_stride_bytes"],
                    last_a["dst_stride_bytes"]) == (32, 4, 260, 4), last_a
            assert (last_b["rows"], last_b["row_bytes"]) == (1, 128), last_b
        assert dma(mini, command_ir) == command_ir
        print(f"PASS: {label}: exact addresses, active bytes, strides and repeat pass")

    planned = plan(mini, (RESULTS / "matmul_chain.npu.mlir").read_text())
    command_ir = dma(mini, planned)
    first, second = inspect(
        mini, planned, command_ir,
        [("input", 0, 2, 3), ("input", 1, 3, 4), ("input", 2, 4, 5),
         ("intermediate", 0, 2, 4), ("output", 1, 2, 5)],
        [4], [(0, 1, 3), (3, 2, 4)],
    )
    assert first[-1]["dst_dram"] == second[1]["src_dram"]
    print("PASS: serial MatMuls store and reload the same intermediate DRAM tensor")

    shared = """
func.func @main(%x: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
  %w = arith.constant dense<[[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [1.0, 0.0, 1.0]]> : tensor<3x3xf32>
  %h = "npu.matmul"(%x, %w) : (tensor<2x3xf32>, tensor<3x3xf32>) -> tensor<2x3xf32>
  %y = "npu.matmul"(%h, %w) : (tensor<2x3xf32>, tensor<3x3xf32>) -> tensor<2x3xf32>
  return %y, %h, %y : tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>
}
"""
    planned = plan(mini, shared)
    command_ir = dma(mini, planned)
    first, second = inspect(
        mini, planned, command_ir,
        [("input", 0, 2, 3), ("constant", 0, 3, 3), ("output", 0, 2, 3), ("output", 1, 2, 3)],
        [3, 2, 3], [(0, 1, 2), (2, 1, 3)],
    )
    assert first[2]["src_dram"] == second[2]["src_dram"] == 64
    assert first[-1]["dst_dram"] == second[1]["src_dram"] == 128
    print("PASS: shared weight, intermediate also returned, duplicate output IDs and ordered return")

    planned = (RESULTS / "weighted_matmul.sram.mlir").read_text()
    original = named_attr(planned, "memory_plan")
    memory = decode(original)
    memory["allocations"].reverse()
    reordered = planned.replace(original, encode(memory), 1)
    result = dma(mini, reordered)
    inspect(mini, reordered, result,
            [("input", 0, 2, 3), ("constant", 0, 3, 4), ("output", 0, 2, 4)],
            [2], [(0, 1, 2)])
    assert named_attr(result, "commands") == named_attr(dma(mini, planned), "commands")
    print("PASS: Buffer IDs resolve correctly even when the allocation array is reversed")


def check_onnx(mini):
    import numpy as np
    import onnx
    from onnx import numpy_helper

    for name, dims, groups, count, traffic in (
        ("weighted_matmul", (2, 4, 3), 1, 1, 104),
        ("matmul_multitile", (64, 64, 96), 4, 12, 112 * 1024),
    ):
        m, n, k = dims
        planned = (RESULTS / f"{name}.sram.mlir").read_text()
        command_ir = dma(mini, planned)
        commands, = inspect(mini, planned, command_ir,
                            [("input", 0, m, k), ("constant", 0, k, n), ("output", 0, m, n)],
                            [2], [(0, 1, 2)])
        histogram, byte_count = counters(commands)
        assert histogram == {"ZERO": groups, "DMA_LOAD_2D": count * 2,
                             "MATMUL_TILE": count, "DMA_STORE_2D": groups}
        assert byte_count == traffic
        payload, = re.findall(r'arith.constant dense<"0x([0-9a-fA-F]+)">', command_ir)
        model = onnx.load(PROJECT / "test/e2e/models" / f"{name}.onnx")
        weight, = model.graph.initializer
        np.testing.assert_array_equal(
            np.frombuffer(bytes.fromhex(payload), dtype="<u4"),
            numpy_helper.to_array(weight).view(np.uint32).ravel(),
        )
        print(f"PASS: {name}: {len(commands)} commands, {byte_count} DMA bytes; weight bits and SSA preserved")
        if count == 12:
            for index in (0, 1, 4, 16, 22, 43):
                print(f"  event {index}: {commands[index]}")


def check_negative(mini, directory):
    planned = (RESULTS / "matmul_multitile.sram.mlir").read_text()
    command_ir = dma(mini, planned)
    original = named_attr(command_ir, "commands")
    commands = decode(original)
    for label, event, patch, diagnostic in (
        ("zero rows", 1, {"rows": 0}, "invalid 2D DMA dimensions or stride"),
        ("zero row bytes", 1, {"row_bytes": 0}, "invalid 2D DMA dimensions or stride"),
        ("short source stride", 1, {"src_stride_bytes": 4}, "invalid 2D DMA dimensions or stride"),
        ("short destination stride", 1, {"dst_stride_bytes": 4}, "invalid 2D DMA dimensions or stride"),
        ("packed stride on the full tensor", 1, {"src_stride_bytes": 128}, "DMA geometry does not match tile region"),
        ("incorrect source tensor", 1, {"tensor_id": 1}, "DMA tensor_id does not match operand or result"),
        ("missing source tensor", 1, {"tensor_id": 99}, "DMA tensor_id does not match operand or result"),
        ("wrong SRAM binding", 1, {"buffer_id": 2}, "DMA buffer_id does not match tile binding"),
        ("missing SRAM binding", 1, {"buffer_id": 99}, "DMA buffer_id does not match tile binding"),
        ("DRAM outside tensor but in arena", 1, {"src_dram": 24576}, "DMA access exceeds bound DRAM tensor"),
        ("DMA final DRAM row out of bounds", 1, {"src_dram": 24576 - 128}, "DMA access exceeds bound DRAM tensor"),
        ("DMA outside arena", 1, {"src_dram": 262144}, "DMA access exceeds bound DRAM tensor"),
        ("SRAM access into adjacent buffer", 1, {"dst_sram": 8192}, "DMA access exceeds bound SRAM buffer"),
        ("SRAM access beyond raw bytes", 1, {"dst_sram": 4097}, "DMA access exceeds bound SRAM buffer"),
        ("DRAM byte extent overflow", 1, {"src_dram": (1 << 63) - 64}, "DMA byte range overflow"),
        ("wrong C address", 3, {"c_sram": 4096}, "MATMUL_TILE does not match tile dimensions or bindings"),
        ("full-size tile exceeds native limit", 3, {"m": 33}, "MATMUL_TILE does not match tile dimensions or bindings"),
        ("incorrect ZERO region", 0, {"cols": 33}, "ZERO does not match tile C binding"),
        ("wrong event index", 0, {"event_index": 1}, "event_index or tile_index does not match tile event"),
        ("unknown opcode", 0, {"opcode": "DMA_SWIZZLE"}, "unsupported command opcode"),
    ):
        changed = deepcopy(commands)
        changed[event].update(patch)
        reject(run(mini, command_ir.replace(original, encode(changed), 1), "-o", "/dev/null"), diagnostic)
        print(f"PASS: Command Verifier rejected {label}")

    missing_k = commands[:7] + commands[10:]
    reject(run(mini, command_ir.replace(original, encode(missing_k), 1), "-o", "/dev/null"),
           "command count does not match tile events")
    early_store = deepcopy(commands)
    early_store[9], early_store[10] = early_store[10], early_store[9]
    for event in (9, 10):
        early_store[event]["event_index"] = event
    reject(run(mini, command_ir.replace(original, encode(early_store), 1), "-o", "/dev/null"),
           "opcode does not match tile event")
    malformed = deepcopy(commands)
    del malformed[1]["rows"]
    reject(run(mini, command_ir.replace(original, encode(malformed), 1), "-o", "/dev/null"), "field 'rows'")
    print("PASS: missing K commands, early STORE, and missing schema field rejected")

    current_layout = named_attr(command_ir, "npu.dram_layout")
    reject(run(mini, command_ir.replace(f" attributes {{npu.dram_layout = {current_layout}}}", "", 1),
               "-o", "/dev/null"), "npu.dram_layout")
    wrong_id = command_ir.replace("lhs_tensor_id = 0", "lhs_tensor_id = 1", 1)
    reject(run(mini, wrong_id, "-o", "/dev/null"), "tensor")
    for mutation in ("base", "output"):
        layout = decode(current_layout)
        if mutation == "base":
            layout["tensors"][1]["base"] = 0
        else:
            layout["output_ids"] = [0]
        reject(run(mini, command_ir.replace(current_layout, encode(layout), 1), "-o", "/dev/null"), "DRAM")
    print("PASS: absent layout, wrong operand ID, overlapping DRAM descriptor, wrong output IDs rejected")

    reject(run(mini, planned, "--npu-insert-dma"), "config")
    reject(run(mini, planned, f"--npu-insert-dma=config={directory / 'absent.json'}"), "config")
    invalid = directory / "invalid.json"
    invalid.write_text("{not json")
    reject(run(mini, planned, f"--npu-insert-dma=config={invalid}"), "Expected object key")
    for group, field, value, diagnostic in (
        ("dma", "supports_2d_strided", False, "2D strided DMA"),
        ("memory", "dram_model", "unknown", "linear byte DRAM"),
        ("memory", "sram_bytes", 65536, "does not match target profile"),
        ("compute", "native_matmul_tile", dict.fromkeys("mnk", 16), "does not match target profile"),
    ):
        profile = json.loads(PROFILE.read_text())
        profile[group][field] = value
        invalid.write_text(json.dumps(profile))
        reject(run(mini, planned, f"--npu-insert-dma=config={invalid}"), diagnostic)
    args = (f"--npu-insert-dma=config={PROFILE}",)
    reject(run(mini, make_matmul(2, 4, 3), *args), "run SRAM planning before DMA insertion")
    tiled = (RESULTS / "weighted_matmul.tiled.mlir").read_text()
    reject(run(mini, tiled, *args), "run SRAM planning before DMA insertion")
    print("PASS: config incompatibility and earlier-stage IR rejected")

    planned = plan(mini, make_matmul(2, 4, 3).replace(
        "%lhs: tensor", "%unused: tensor<1x4611686018427387904xf32>, %lhs: tensor", 1
    ))
    reject(run(mini, planned, *args), "DRAM tensor address or byte size overflow")
    print("PASS: whole-tensor DRAM extent overflow rejected before command generation")

    chain = plan(mini, (RESULTS / "matmul_chain.npu.mlir").read_text())
    first, field, rest = chain.rpartition("sram_capacity_bytes = 262144 : i64")
    assert field
    mismatched = first + "sram_capacity_bytes = 262208 : i64" + rest
    failed = run(mini, mismatched, *args, "--mlir-print-ir-after-all", "--mlir-print-ir-after-failure")
    reject(failed, "SRAM plan does not match target profile")
    _, failure_dump = failed.stderr.rsplit("IR Dump After", maxsplit=1)
    assert failure_dump.count('"npu.sram_planned_matmul"') == 2
    assert '"npu.command_matmul"' not in failure_dump and "npu.dram_layout" not in failure_dump
    print("PASS: second MatMul failure leaves both ops and module layout untouched")


if __name__ == "__main__":
    mini, case = sys.argv[1:]
    if case == "positive":
        check_positive(mini)
    elif case == "negative":
        with tempfile.TemporaryDirectory(prefix="npu-dma-") as temporary:
            check_negative(mini, Path(temporary))
    elif case == "onnx":
        check_onnx(mini)
    else:
        raise SystemExit(f"unknown test group: {case}")
