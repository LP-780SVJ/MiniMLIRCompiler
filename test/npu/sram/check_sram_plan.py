"""Exercise the SRAM planner and independently inspect the printed memory plans."""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


PROJECT = Path(__file__).resolve().parents[3]
PROFILE = PROJECT / "configs/virtual-npu-v1.json"
HEX = "--mlir-print-elementsattrs-with-hex-if-larger=0"
MEMORY_PLAN = re.compile(r"memory_plan = (\{(?:[^{}]|\{[^{}]*\})*\})")
TILE_PLAN = re.compile(r"tiles = \[(.*?)\]")
DICT_FIELDS = re.compile(r'(\w+) = (\[.*?\]|-?\d+ : i64|true|false|"[^"]*")')


def run(mini, ir, *args):
    return subprocess.run([mini, "-", *args], input=ir, capture_output=True, text=True)


def require_success(result):
    assert result.returncode == 0, result.stderr
    return result.stdout


def require_failure(result, diagnostic):
    assert result.returncode == 1 and diagnostic in result.stderr, result
    assert not result.stdout, result.stdout


def plan(mini, ir, profile=PROFILE):
    return require_success(run(mini, ir, f"--npu-sram-plan=config={profile}", "--verify-each", HEX))


def make_matmul(m, n, k):
    lhs = f"tensor<{m}x{k}xf32>"
    rhs = f"tensor<{k}x{n}xf32>"
    result = f"tensor<{m}x{n}xf32>"
    return (
        f"func.func @main(%lhs: {lhs}, %rhs: {rhs}) -> {result} {{\n"
        f'  %result = "npu.matmul"(%lhs, %rhs) : ({lhs}, {rhs}) -> {result}\n'
        f"  return %result : {result}\n}}\n"
    )


def tile(mini, ir, profile=PROFILE):
    return require_success(run(mini, ir, f"--npu-tiling=config={profile}", HEX))


def decode_fields(payload):
    fields = {}
    for name, text in DICT_FIELDS.findall(payload):
        if text.startswith("["):
            fields[name] = decode_records(text)
        elif text.endswith(" : i64"):
            fields[name] = int(text.removesuffix(" : i64"))
        else:
            fields[name] = json.loads(text)
    return fields


def decode_records(text):
    return [decode_fields(record) for record in re.findall(r"\{([^{}]*)\}", text)]


def encode_attr(value):
    if isinstance(value, bool) or isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return f"{value} : i64"
    if isinstance(value, list):
        return "[" + ", ".join(map(encode_attr, value)) + "]"
    return "{" + ", ".join(f"{key} = {encode_attr(entry)}" for key, entry in value.items()) + "}"


def without_memory_plan(ir):
    return re.sub(MEMORY_PLAN.pattern + ", ", "", ir).replace(
        '"npu.sram_planned_matmul"', '"npu.tiled_matmul"'
    )


def check_memory_plan(memory, tiles, capacity=262144, reserved=0, alignment=64):
    assert memory["schedule_contract"] == "serial-mnk-single-buffer-v1"
    assert memory["alignment_bytes"] == alignment
    assert memory["sram_capacity_bytes"] == capacity
    assert memory["reserved_bytes"] == reserved
    assert len(memory["tile_bindings"]) == len(tiles)

    by_id = {record["id"]: record for record in memory["allocations"]}
    assert len(by_id) == len(memory["allocations"])
    assert set(by_id) == set(range(len(by_id)))
    assert len(by_id) == 2 * len(tiles) + sum(entry["is_first_k"] for entry in tiles)
    outputs = {}
    used = set()
    event = 0
    for index, (desc, binding) in enumerate(zip(tiles, memory["tile_bindings"], strict=True)):
        assert binding["tile_index"] == index
        group = desc["output_tile_id"]
        alloc_a, alloc_b, alloc_c = (
            by_id[binding[f"{name}_buffer_id"]] for name in "abc"
        )
        for alloc, role, size in (
            (alloc_a, "A", desc["m_size"] * desc["k_size"] * 4),
            (alloc_b, "B", desc["k_size"] * desc["n_size"] * 4),
            (alloc_c, "C", desc["m_size"] * desc["n_size"] * 4),
        ):
            assert alloc["id"] not in used or role == "C", alloc
            used.add(alloc["id"])
            assert alloc["role"] == role and alloc["output_tile_id"] == group, alloc
            assert alloc["raw_bytes"] == size and alloc["alignment_bytes"] == alignment, alloc
            assert alloc["allocated_bytes"] == (size + alignment - 1) // alignment * alignment, alloc
        if desc["is_first_k"]:
            assert group not in outputs
            outputs[group] = alloc_c
            assert alloc_c["tile_index"] == index and alloc_c["live_begin"] == event
            event += 1  # ZERO
        assert alloc_c is outputs[group], (binding, alloc_c)
        for alloc, load in ((alloc_a, event), (alloc_b, event + 1)):
            assert alloc["tile_index"] == index
            assert (alloc["live_begin"], alloc["live_end"]) == (load, event + 3), alloc
        event += 3  # two LOADs and MATMUL_ACC
        if desc["is_last_k"]:
            event += 1  # STORE
            assert alloc_c["live_end"] == event, alloc_c
        else:
            assert alloc_c["live_end"] > event, alloc_c
    assert set(by_id) == used

    starting = defaultdict(list)
    for alloc in by_id.values():
        begin, end = alloc["live_begin"], alloc["live_end"]
        address, size = alloc["address"], alloc["allocated_bytes"]
        assert 0 <= begin < end <= event
        assert reserved <= address < address + size <= capacity
        assert address % alignment == 0
        starting[begin].append(alloc)
    seen_addresses = set()
    active = []
    peak = 0
    for moment, new_allocs in sorted(starting.items()):
        active = [alloc for alloc in active if alloc["live_end"] > moment]
        for alloc in new_allocs:
            assert alloc["reused_address"] == (alloc["address"] in seen_addresses)
            seen_addresses.add(alloc["address"])
            active.append(alloc)
        regions = sorted((alloc["address"], alloc["address"] + alloc["allocated_bytes"]) for alloc in active)
        assert all(left[1] <= right[0] for left, right in zip(regions, regions[1:])), regions
        peak = max(peak, sum(end - begin for begin, end in regions))
    high_water = max(alloc["address"] + alloc["allocated_bytes"] for alloc in by_id.values())
    assert (peak, high_water) == (memory["peak_live_bytes"], memory["arena_high_water_bytes"])
    return peak, high_water


def inspect(mini, source, result, profile=PROFILE):
    assert require_success(run(mini, result, HEX)) == result
    assert without_memory_plan(result) == require_success(run(mini, source, HEX))
    tiles = [decode_records(text) for text in TILE_PLAN.findall(result)]
    plans = [decode_fields(text) for text in MEMORY_PLAN.findall(result)]
    assert plans and len(tiles) == len(plans)
    config = json.loads(profile.read_text())["memory"]
    for memory, records in zip(plans, tiles, strict=True):
        check_memory_plan(memory, records, config["sram_bytes"],
                          config["reserved_bytes"], config["alignment_bytes"])
    return plans


def expect_fixed_slots(memory, expected):
    assert (memory["peak_live_bytes"], memory["arena_high_water_bytes"]) == expected
    start = memory["arena_high_water_bytes"] - memory["peak_live_bytes"]
    slot_bytes = memory["peak_live_bytes"] // 3
    for alloc in memory["allocations"]:
        assert alloc["address"] == start + slot_bytes * "CAB".index(alloc["role"]), alloc


def check_positive(mini, directory):
    for label, shape, count, outputs, peak in (
        ("small", (2, 4, 3), 1, 1, 192),
        ("native", (32, 32, 32), 1, 1, 12288),
        ("multi", (64, 64, 96), 12, 4, 12288),
        ("M tail", (65, 9, 7), 3, 3, 2304),
        ("N tail", (7, 65, 9), 3, 3, 2304),
        ("K tail", (32, 32, 65), 3, 1, 12288),
        ("streamed 256 cubed", (256, 256, 256), 512, 64, 12288),
    ):
        source = tile(mini, make_matmul(*shape))
        result = plan(mini, source)
        memory, = inspect(mini, source, result)
        assert len(memory["tile_bindings"]) == count
        assert sum(record["role"] == "C" for record in memory["allocations"]) == outputs
        assert memory["peak_live_bytes"] == peak, memory
        if label in {"small", "native", "multi", "streamed 256 cubed"}:
            expect_fixed_slots(memory, (peak, peak))
        assert plan(mini, result) == result
        print(f"PASS: {label}: {count} tiles, {outputs} outputs, peak {peak}, high-water {memory['arena_high_water_bytes']}")

    source = (PROJECT / "test/npu/conversion/matmul-lowering.mlir").read_text()
    source = require_success(run(mini, source, "--pass-pipeline=builtin.module(mini-to-npu)"))
    chain = tile(mini, source)
    result = plan(mini, chain)
    first, second = inspect(mini, chain, result)
    assert all(len(memory["tile_bindings"]) == 1 for memory in (first, second))
    assert first["allocations"][0]["address"] == second["allocations"][0]["address"] == 0
    assert second["allocations"][0]["live_begin"] == 0
    assert '"npu.sram_planned_matmul"(%0, %arg2)' in result and "return %1, %0" in result
    print("PASS: serial chain reuses SRAM and preserves shared SSA result")

    source = tile(mini, make_matmul(2, 4, 3))
    alternate = directory / "reserve.json"
    profile = json.loads(PROFILE.read_text())
    profile["memory"].update(reserved_bytes=1, sram_bytes=256)
    alternate.write_text(json.dumps(profile))
    result = plan(mini, source, alternate)
    memory, = inspect(mini, source, result, alternate)
    expect_fixed_slots(memory, (192, 256))
    require_failure(run(mini, result, f"--npu-sram-plan=config={PROFILE}"),
                    "existing SRAM plan does not match target config")

    result = plan(mini, source)
    original = MEMORY_PLAN.search(result).group(1)
    memory = decode_fields(original)
    for alloc in memory["allocations"]:
        alloc["address"] += 64
    memory["arena_high_water_bytes"] += 64
    require_success(run(mini, result.replace(original, encode_attr(memory)), "-o", "/dev/null"))
    print("PASS: reserved-byte alignment, mismatch on re-entry, and independent verifier accepts legal non-first-fit layout")

    profile = json.loads(PROFILE.read_text())
    profile["compute"]["native_matmul_tile"] = dict.fromkeys("mnk", 64)
    alternate.write_text(json.dumps(profile))
    source = tile(mini, make_matmul(64, 64, 64), alternate)
    memory, = inspect(mini, source, plan(mini, source, alternate), alternate)
    expect_fixed_slots(memory, (49152, 49152))
    print("PASS: custom 64 cubed profile reserves 48 KiB for one resident block")


def check_negative(mini, directory):
    source = tile(mini, make_matmul(32, 32, 65))
    result = plan(mini, source)
    original = MEMORY_PLAN.search(result).group(1)
    correct = decode_fields(original)
    require_failure(run(mini, result.replace(f"memory_plan = {original}, ", ""), "-o", "/dev/null"),
                    "requires attribute 'memory_plan'")
    require_failure(run(mini, result.replace("address = 0 : i64", "address = 0 : i32", 1), "-o", "/dev/null"),
                    "field 'address' must be a nonnegative i64")

    for label, section, index, updates, diagnostic in (
        ("unaligned address", "allocations", 1, {"address": 1}, "address is not aligned"),
        ("out of SRAM", "allocations", 1, {"address": 262144}, "outside usable SRAM"),
        ("overflowing address + size", "allocations", 1, {"address": (1 << 63) - 64}, "outside usable SRAM"),
        ("insufficient buffer size", "allocations", 1, {"allocated_bytes": 64}, "buffer size does not match"),
        ("simultaneous A/B alias", "allocations", 2, {"address": 4096}, "simultaneously live buffers overlap"),
        ("partial A/B overlap", "allocations", 2, {"address": 8192 - 64}, "simultaneously live buffers overlap"),
        ("A/B overwrites live C", "allocations", 3, {"address": 0}, "simultaneously live buffers overlap"),
        ("A prematurely released", "allocations", 1, {"live_end": 3}, "lifetime does not match required events"),
        ("C prematurely released", "allocations", 0, {"live_end": 10}, "lifetime does not match required events"),
        ("reversed lifetime", "allocations", 1, {"live_end": 0}, "invalid buffer lifetime"),
        ("duplicate ID", "allocations", 1, {"id": 0}, "duplicate buffer ID"),
        ("unknown binding", "tile_bindings", 0, {"a_buffer_id": 999}, "tile binding references unknown buffer"),
        ("wrong binding", "tile_bindings", 0, {"a_buffer_id": 2}, "tile binding does not match"),
        ("incorrect reuse flag", "allocations", 1, {"reused_address": True}, "incorrect reused_address"),
        ("incorrect peak", None, None, {"peak_live_bytes": 192}, "peak_live_bytes"),
        ("incorrect arena", None, None, {"arena_high_water_bytes": 192}, "arena_high_water_bytes"),
        ("unknown schedule", None, None, {"schedule_contract": "parallel"}, "schedule_contract"),
    ):
        changed = deepcopy(correct)
        record = changed[section][index] if section else changed
        record.update(updates)
        ir = result.replace(original, encode_attr(changed))
        require_failure(run(mini, ir, "-o", "/dev/null"), diagnostic)
        print(f"PASS: verifier rejected {label}")

    require_failure(run(mini, source, "--npu-sram-plan"), "requires a config path")
    require_failure(run(mini, source, f"--npu-sram-plan=config={directory / 'missing.json'}"),
                    "cannot read target config")
    invalid = directory / "invalid.json"
    invalid.write_text("{bad json")
    require_failure(run(mini, source, f"--npu-sram-plan=config={invalid}"), "target config")
    for section, key, value, diagnostic in (
        ("memory", "layout", "column_major", "expected row_major SRAM layout"),
        ("dma", "mode", "asynchronous", "expected synchronous DMA"),
        ("dma", "buffering", "double", "expected synchronous DMA"),
        ("memory", "sram_bytes", 8192, "insufficient SRAM capacity"),
        ("memory", "reserved_bytes", 262145, "reserved"),
        ("memory", "alignment_bytes", 0, "alignment"),
        ("memory", "sram_bytes", 1 << 63, "sram_bytes"),
        ("compute", "native_matmul_tile", {"m": 16, "n": 32, "k": 32}, "tile plan does not match target config"),
    ):
        profile = json.loads(PROFILE.read_text())
        profile[section][key] = value
        invalid.write_text(json.dumps(profile))
        require_failure(run(mini, source, f"--npu-sram-plan=config={invalid}"), diagnostic)
    require_failure(run(mini, make_matmul(2, 4, 3), f"--npu-sram-plan=config={PROFILE}"),
                    "requires npu.tiled_matmul before SRAM planning")

    profile = json.loads(PROFILE.read_text())
    profile["compute"]["native_matmul_tile"] = dict.fromkeys("mnk", 256)
    invalid.write_text(json.dumps(profile))
    resident = tile(mini, make_matmul(256, 256, 256), invalid)
    require_failure(run(mini, resident, f"--npu-sram-plan=config={invalid}"), "insufficient SRAM capacity")

    profile = json.loads(PROFILE.read_text())
    profile["compute"]["native_matmul_tile"] = {"m": 1 << 31, "n": 1, "k": 1 << 31}
    invalid.write_text(json.dumps(profile))
    huge = tile(mini, make_matmul(1 << 31, 1, 1 << 31), invalid)
    require_failure(run(mini, huge, f"--npu-sram-plan=config={invalid}"), "overflow")
    print("PASS: rejected malformed/incompatible profiles, capacity shortage, wrong stage and size overflow")

    profile = json.loads(PROFILE.read_text())
    profile["memory"]["sram_bytes"] = 8192
    invalid.write_text(json.dumps(profile))
    module = tile(mini, make_matmul(2, 4, 3) + make_matmul(32, 32, 32).replace("@main", "@large"))
    failed = run(mini, module, f"--npu-sram-plan=config={invalid}", "--mlir-print-ir-after-all",
                 "--mlir-print-ir-after-failure")
    require_failure(failed, "insufficient SRAM capacity")
    _, ir_dump = failed.stderr.split("IR Dump After", maxsplit=1)
    assert ir_dump.count('"npu.tiled_matmul"') == 2 and '"npu.sram_planned_matmul"' not in ir_dump, ir_dump
    print("PASS: failure of second MatMul does not rewrite the first")


def check_onnx(mini):
    import numpy as np
    import onnx
    from onnx import numpy_helper

    fixtures = PROJECT / "test/e2e"
    for name, count, peak in (("weighted_matmul", 1, 192), ("matmul_multitile", 12, 12288)):
        source = (fixtures / "results" / f"{name}.tiled.mlir").read_text()
        result = plan(mini, source)
        memory, = inspect(mini, source, result)
        assert len(memory["tile_bindings"]) == count
        expect_fixed_slots(memory, (peak, peak))
        assert '"npu.sram_planned_matmul"(%arg0, %cst)' in result and "%arg1:" not in result
        payload, = re.findall(r'arith.constant dense<"0x([0-9a-fA-F]+)">', result)
        weight, = onnx.load(fixtures / "models" / f"{name}.onnx").graph.initializer
        np.testing.assert_array_equal(
            np.frombuffer(bytes.fromhex(payload), dtype="<u4"),
            numpy_helper.to_array(weight).view(np.uint32).ravel(),
        )
        print(f"PASS: {name}: {count} bindings, peak and high-water {peak} B, SSA and weight bits preserved")


if __name__ == "__main__":
    mini, case = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="npu-sram-") as temporary:
        directory = Path(temporary)
        if case == "positive":
            check_positive(mini, directory)
        elif case == "negative":
            check_negative(mini, directory)
        elif case == "onnx":
            check_onnx(mini)
        else:
            raise SystemExit(f"unknown test case: {case}")
