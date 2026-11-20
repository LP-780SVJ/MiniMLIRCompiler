"""Verify actual Tile IR, including coverage independently of the C++ planner."""

from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


PROJECT = Path(__file__).resolve().parents[3]
PROFILE = PROJECT / "configs/virtual-npu-v1.json"
HEX = "--mlir-print-elementsattrs-with-hex-if-larger=0"
PLAN = re.compile(r"tiles = \[(.*?)\]\}>")


def run(mini, ir, *args):
    return subprocess.run([mini, "-", *args], input=ir, capture_output=True, text=True)


def success(result):
    assert result.returncode == 0, result.stderr
    return result.stdout


def reject(result, message):
    assert result.returncode == 1 and message in result.stderr, result


def tile(mini, ir, profile=PROFILE):
    return success(run(mini, ir, f"--npu-tiling=config={profile}", "--verify-each", HEX))


def make_matmul(m, n, k):
    lhs, rhs, result = f"tensor<{m}x{k}xf32>", f"tensor<{k}x{n}xf32>", f"tensor<{m}x{n}xf32>"
    return (
        f"func.func @main(%lhs: {lhs}, %rhs: {rhs}) -> {result} {{\n"
        f'  %result = "npu.matmul"(%lhs, %rhs) : ({lhs}, {rhs}) -> {result}\n'
        f"  return %result : {result}\n}}\n"
    )


def descriptors(ir):
    plans = []
    for text in PLAN.findall(ir):
        entries = []
        for payload in re.findall(r"\{([^{}]*)\}", text):
            desc = {}
            for field in payload.split(", "):
                name, value = field.split(" = ")
                desc[name] = value == "true" if value in ("true", "false") else int(value.removesuffix(" : i64"))
            entries.append(desc)
        plans.append(entries)
    return plans


def check_coverage(shape, limits, tiles):
    m, n, k = shape
    groups = defaultdict(list)
    for entry in tiles:
        for axis, extent, limit in zip("mnk", shape, limits, strict=True):
            offset, size = entry[f"{axis}_offset"], entry[f"{axis}_size"]
            assert 0 <= offset and 1 <= size <= limit and offset + size <= extent, entry
        groups[entry["output_tile_id"]].append(entry)
    assert list(groups) == list(range(len(groups))), groups
    assert [entry for group in groups.values() for entry in group] == tiles

    covered = set()
    positions = []
    for group in groups.values():
        head = group[0]
        rectangle = tuple(head[key] for key in ("m_offset", "n_offset", "m_size", "n_size"))
        row, col, rows, cols = rectangle
        positions.append((row, col))
        for element in ((i, j) for i in range(row, row + rows) for j in range(col, col + cols)):
            assert element not in covered, rectangle
            covered.add(element)
        next_k = 0
        for index, entry in enumerate(group):
            assert tuple(entry[key] for key in ("m_offset", "n_offset", "m_size", "n_size")) == rectangle
            assert entry["k_offset"] == next_k, entry
            assert entry["is_first_k"] == (index == 0), entry
            assert entry["is_last_k"] == (index == len(group) - 1), entry
            next_k += entry["k_size"]
        assert next_k == k, group
    assert positions == sorted(positions) and len(covered) == m * n
    return len(groups)


def check_positive(mini, directory):
    for label, shape, expected_counts, tail in (
        ("small", (2, 4, 3), (1, 1), None),
        ("full", (32, 32, 32), (1, 1), None),
        ("multi", (64, 64, 96), (12, 4), None),
        ("M tail", (65, 9, 7), (3, 3), "m"),
        ("N tail", (7, 65, 9), (3, 3), "n"),
        ("K tail", (32, 32, 65), (3, 1), "k"),
    ):
        source = make_matmul(*shape)
        result = tile(mini, source)
        entries, = descriptors(result)
        assert (len(entries), check_coverage(shape, (32, 32, 32), entries)) == expected_counts
        if tail:
            assert [entry[f"{tail}_offset"] for entry in entries] == [0, 32, 64]
            assert [entry[f"{tail}_size"] for entry in entries] == [32, 32, 1]
        assert preserve_ssa(result) == success(run(mini, source, HEX)), result
        assert tile(mini, result) == result
        print(f"PASS: {label}: {len(entries)} records, {expected_counts[1]} output blocks")

    source = (PROJECT / "test/npu/conversion/matmul-lowering.mlir").read_text()
    chain = success(run(mini, source, "--pass-pipeline=builtin.module(mini-to-npu)"))
    result = tile(mini, chain)
    first, second = descriptors(result)
    assert check_coverage((2, 4, 3), (32, 32, 32), first) == 1
    assert check_coverage((2, 5, 4), (32, 32, 32), second) == 1
    assert '"npu.tiled_matmul"(%0, %arg2)' in result and "return %1, %0" in result, result
    assert preserve_ssa(result) == success(run(mini, chain, HEX)), result
    print("PASS: chain, shared intermediate use and ordered return")

    profile = json.loads(PROFILE.read_text())
    profile["compute"]["native_matmul_tile"] = {"m": 2, "n": 3, "k": 4}
    alternate = directory / "alternate.json"
    alternate.write_text(json.dumps(profile))
    result = tile(mini, make_matmul(5, 7, 9), alternate)
    entries, = descriptors(result)
    assert len(entries) == 27 and check_coverage((5, 7, 9), (2, 3, 4), entries) == 9
    assert "tile_m = 2 : i64, tile_n = 3 : i64" in result and "tile_k = 4 : i64" in result
    reject(run(mini, result, f"--npu-tiling=config={PROFILE}"), "does not match target config limits")
    print("PASS: alternate configured limits, mismatching re-entry rejected")


def preserve_ssa(ir):
    return re.sub(r'"npu\.tiled_matmul"(\([^\n]*?\)) <\{[^\n]*\}>',
                  r'"npu.matmul"\1', ir)


def check_negative(mini, directory):
    ir = tile(mini, make_matmul(32, 32, 65))
    original = PLAN.search(ir).group(1)
    first, second, last = re.findall(r"\{[^{}]*\}", original)
    for label, entries, diagnostic in (
        ("lost record", [first, last], "expected 3 tile records, got 2"),
        ("duplicate", [first, first, last], "full M/N/K coverage"),
        ("K gap", [first, second.replace("k_offset = 32", "k_offset = 33"), last], "full M/N/K coverage"),
        ("K overlap", [first, second.replace("k_offset = 32", "k_offset = 31"), last], "full M/N/K coverage"),
        ("M out of bounds", [first.replace("m_offset = 0", "m_offset = 1"), second, last], "full M/N/K coverage"),
        ("N out of bounds", [first.replace("n_size = 32", "n_size = 33"), second, last], "full M/N/K coverage"),
        ("K out of bounds", [first, second, last.replace("k_size = 1", "k_size = 2")], "full M/N/K coverage"),
        ("overflowing offset", [first, second, last.replace("k_offset = 64", "k_offset = 9223372036854775807")], "full M/N/K coverage"),
        ("wrong output ID", [first.replace("output_tile_id = 0", "output_tile_id = 1"), second, last], "requires output_tile_id 0"),
        ("first flag", [first.replace("is_first_k = true", "is_first_k = false"), second, last], "incorrect is_first_k / is_last_k"),
        ("last flag", [first, second, last.replace("is_last_k = true", "is_last_k = false")], "incorrect is_first_k / is_last_k"),
        ("missing field", [first.replace("m_offset = 0 : i64, ", ""), second, last], "field 'm_offset' must be i64"),
        ("wrong field type", [first.replace("m_offset = 0 : i64", "m_offset = 0 : i32"), second, last], "field 'm_offset' must be i64"),
        ("not a dictionary", ["0 : i64", second, last], "tile #0 must be a dictionary"),
    ):
        reject(run(mini, ir.replace(original, ", ".join(entries))), diagnostic)
        print(f"PASS: rejected {label}")
    reject(run(mini, ir.replace("tile_m = 32", "tile_m = -32")), "native tile limits must be positive")
    reject(run(mini, ir.replace("virtual-npu-v1", "other")), "unsupported tile plan target")

    source = make_matmul(2, 4, 3)
    reject(run(mini, source, "--npu-tiling"), "requires a config path")
    reject(run(mini, source, f"--npu-tiling=config={directory / 'missing.json'}"), "cannot read target config")
    invalid = directory / "invalid.json"
    invalid.write_text("{bad json")
    reject(run(mini, source, f"--npu-tiling=config={invalid}"), "target config")
    original_profile = PROFILE.read_text()
    for key, replacement, message in (
        ('"schema_version": 1', '"schema_version": 2', "schema_version 1"),
        ('"target": "virtual-npu-v1"', '"target": "other"', "expected target virtual-npu-v1"),
        ('"matrix_units": 1', '"matrix_units": 2', "expected one matrix unit"),
        ('"input_dtype": "float32"', '"input_dtype": "bf16"', "expected float32"),
        ('"accumulator_dtype": "float32"', '"accumulator_dtype": "float16"', "expected float32"),
        ('"matmul_semantics": "accumulate"', '"matmul_semantics": "overwrite"', "expected accumulate"),
        ('"m": 32', '"m": 0', "must be a positive i64 integer"),
        ('"m": 32', '"m": -32', "must be a positive i64 integer"),
        ('"m": 32', '"m": 1.5', "must be a positive i64 integer"),
        ('"m": 32', '"m": 9223372036854775808', "must be a positive i64 integer"),
    ):
        invalid.write_text(original_profile.replace(key, replacement))
        reject(run(mini, source, f"--npu-tiling=config={invalid}"), message)
    args = (f"--npu-tiling=config={PROFILE}",)
    reject(run(mini, make_matmul(2097153, 1, 1), *args), "tile count exceeds limit of 65536")
    reject(run(mini, make_matmul(128, 128, 131073), *args), "tile count exceeds limit of 65536")
    reject(run(mini, make_matmul(9223372036854775807, 1, 1), *args), "tile count exceeds limit of 65536")
    reject(run(mini, source.replace('"npu.matmul"', '"mini.matmul"'), *args), "unsupported operation in npu-tiling input")
    print("PASS: malformed/invalid profiles, oversize and i64 limits, invalid stage input")


def check_onnx(mini, directory):
    import numpy as np
    import onnx
    from onnx import numpy_helper

    for name, prefix, shape, counts in (
        ("weighted_matmul", "day22", (2, 4, 3), (1, 1)),
        ("matmul_multitile", "day22-multitile", (64, 64, 96), (12, 4)),
    ):
        model = PROJECT / "test/e2e/models" / f"{name}.onnx"
        mini_path = directory / f"{prefix}.mini.mlir"
        success(subprocess.run(
            [sys.executable, PROJECT / "python/onnx_importer/importer.py",
             "--input", model, "--output", mini_path], capture_output=True, text=True,
        ))
        compute = success(run(mini, mini_path.read_text(), "--pass-pipeline=builtin.module(mini-npu-compile)", HEX))
        result = tile(mini, compute)
        assert success(run(mini, result, HEX)) == result
        entries, = descriptors(result)
        assert (len(entries), check_coverage(shape, (32, 32, 32), entries)) == counts
        assert preserve_ssa(result) == compute
        assert '"npu.tiled_matmul"(%arg0, %cst)' in result and "%arg1:" not in result
        payload, = re.findall(r'arith.constant dense<"0x([0-9a-fA-F]+)">', result)
        weight, = onnx.load(model).graph.initializer
        np.testing.assert_array_equal(
            np.frombuffer(bytes.fromhex(payload), dtype="<u4"),
            numpy_helper.to_array(weight).view(np.uint32).ravel(),
        )
        print(f"PASS: {name}: {len(entries)} records, {counts[1]} output groups; weights bit-identical")
        if name == "matmul_multitile":
            for entry in entries:
                print("  ", tuple(entry[key] for key in (
                    "output_tile_id", "m_offset", "n_offset", "k_offset",
                    "m_size", "n_size", "k_size", "is_first_k", "is_last_k",
                )))


if __name__ == "__main__":
    mini, case = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="npu-tiling-") as temporary:
        directory = Path(temporary)
        if case == "positive":
            check_positive(mini, directory)
        elif case == "negative":
            check_negative(mini, directory)
        elif case == "onnx":
            check_onnx(mini, directory)
        else:
            raise SystemExit(f"unknown test case: {case}")
