"""Exercise real schedule IR and independently check its memory dependencies."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import runpy
import sys


DMA_TEST = runpy.run_path(str(Path(__file__).resolve().parents[1] / "dma/check_dma_insertion.py"))
PROJECT = DMA_TEST["PROJECT"]
RESULTS = DMA_TEST["RESULTS"]
HEX = DMA_TEST["HEX"]
run, success, named_attr, decode, encode, dma, tile_and_plan, make_matmul = (
    DMA_TEST[name] for name in ("run", "success", "named_attr", "decode", "encode", "dma", "plan", "make_matmul")
)
SCHEDULE = "npu.execution_schedule"
COMMAND_OP = '"npu.command_matmul"'
CONTRACT = "serial-mnk-single-buffer-v1"
SHARED_CHAIN = """
func.func @main(%x: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
  %w = arith.constant dense<1.0> : tensor<64x64xf32>
  %h = "npu.matmul"(%x, %w) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  %y = "npu.matmul"(%h, %w) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %y, %h, %y : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
}
"""


def schedule(mini, ir):
    return success(run(mini, ir, "--npu-schedule", "--verify-each", HEX))


def blocks(ir):
    return [line for line in ir.splitlines() if COMMAND_OP in line]


def extract(ir):
    return decode(named_attr(ir, SCHEDULE))


def byte_effects(command):
    def access(space, mode, address, rows, row_bytes, stride):
        return (space, mode, address, address + (rows - 1) * stride + row_bytes)

    opcode = command["opcode"]
    if opcode == "ZERO":
        address, rows, cols = (command[key] for key in ("dst_sram", "rows", "cols"))
        return [access("sram", "w", address, rows, 4 * cols, 4 * cols)]
    if opcode in {"DMA_LOAD_2D", "DMA_STORE_2D"}:
        rows, width, src_stride, dst_stride = (
            command[key] for key in ("rows", "row_bytes", "src_stride_bytes", "dst_stride_bytes")
        )
        if opcode == "DMA_LOAD_2D":
            return [
                access("dram", "r", command["src_dram"], rows, width, src_stride),
                access("sram", "w", command["dst_sram"], rows, width, dst_stride),
            ]
        return [
            access("sram", "r", command["src_sram"], rows, width, src_stride),
            access("dram", "w", command["dst_dram"], rows, width, dst_stride),
        ]
    assert opcode == "MATMUL_TILE", command
    m, n, k = (command[key] for key in "mnk")
    a, b, c = (command[f"{role}_sram"] for role in "abc")
    return [
        access("sram", "r", a, m, 4 * k, 4 * k),
        access("sram", "r", b, k, 4 * n, 4 * n),
        access("sram", "r", c, m, 4 * n, 4 * n),
        access("sram", "w", c, m, 4 * n, 4 * n),
    ]


def inspect(mini, original, scheduled):
    assert success(run(mini, scheduled, HEX)) == scheduled
    assert schedule(mini, scheduled) == scheduled
    serialized = named_attr(scheduled, SCHEDULE)
    assert scheduled.replace(f", {SCHEDULE} = {serialized}", "", 1) == success(run(mini, original, HEX))
    metadata = decode(serialized)
    assert metadata["schema_version"] == 1 and metadata["contract"] == CONTRACT
    entries = metadata["commands"]
    source_blocks = blocks(scheduled)
    commands = []
    origins = []
    for index, block in enumerate(source_blocks):
        local = decode(named_attr(block, "commands"))
        commands.extend(local)
        origins.extend((index, command["event_index"]) for command in local)
    assert metadata["total_commands"] == len(entries) == len(commands)
    ancestors = []
    effects = []
    for seq, (entry, command, origin) in enumerate(zip(entries, commands, origins, strict=True)):
        assert set(entry) == {
            "command_id", "matmul_index", "local_event_index", "sequence_index", "dependencies",
        }, entry
        assert entry["command_id"] == entry["sequence_index"] == seq, entry
        assert (entry["matmul_index"], entry["local_event_index"]) == origin, entry
        deps = entry["dependencies"]
        assert sorted(set(deps)) == deps and all(0 <= dep < seq for dep in deps), entry
        inherited = set(deps)
        for dep in deps:
            inherited.update(ancestors[dep])
        accesses = byte_effects(command)
        for prev, prior in enumerate(effects):
            conflict = any(
                space == other_space and mode + other_mode != "rr"
                and start < other_end and other_start < end
                for space, mode, start, end in prior
                for other_space, other_mode, other_start, other_end in accesses
            )
            assert not conflict or prev in inherited, (prev, seq, prior, accesses, inherited)
        effects.append(accesses)
        ancestors.append(inherited)
    return entries, commands, ancestors


def compile_commands(mini, source):
    return dma(mini, tile_and_plan(mini, source))


def check_positive(mini):
    for axis, shape, expected in (
        ("M", (65, 9, 7), 15), ("N", (7, 65, 9), 15), ("K", (32, 32, 65), 11),
    ):
        command_ir = compile_commands(mini, make_matmul(*shape))
        entries, commands, ancestors = inspect(mini, command_ir, schedule(mini, command_ir))
        assert len(entries) == expected
        tiles = decode(named_attr(blocks(command_ir)[0], "tiles"))
        assert tiles[-1][f"{axis.lower()}_size"] == 1
        if axis == "K":
            assert [cmd["k"] for cmd in commands if cmd["opcode"] == "MATMUL_TILE"] == [32, 32, 1]
            assert {3, 6, 9} <= ancestors[10]
        print(f"PASS: {axis} tail, {expected} commands and address dependencies")

    command_ir = compile_commands(mini, SHARED_CHAIN)
    entries, commands, ancestors = inspect(mini, command_ir, schedule(mini, command_ir))
    assert len(entries) == 64
    layout = decode(named_attr(command_ir, "npu.dram_layout"))
    assert layout["constant_ids"] == [1] and layout["output_ids"] == [3, 2, 3]
    assert entries[0]["local_event_index"] == entries[32]["local_event_index"] == 0
    assert entries[0]["command_id"] == 0 and entries[32]["command_id"] == 32
    producer_stores = {7, 15, 23, 31}
    assert all(commands[index]["tensor_id"] == 2 and commands[index]["opcode"] == "DMA_STORE_2D"
               for index in producer_stores)
    assert 31 in ancestors[32] and producer_stores <= ancestors[33]
    assert commands[33]["tensor_id"] == 2 and commands[33]["opcode"] == "DMA_LOAD_2D"
    assert 31 in entries[33]["dependencies"], entries[33]
    assert {cmd["tensor_id"] for cmd in commands if cmd["opcode"] == "DMA_LOAD_2D"} == {0, 1, 2}
    print("PASS: each of four H stores precedes consumer LOAD H; shared weight and ordered duplicate outputs")


def check_onnx(mini):
    import numpy as np
    import onnx
    from onnx import numpy_helper

    for model, count, group_size, offsets, peak in (
        ("weighted_matmul", 5, 5, [0], 192),
        ("matmul_multitile", 44, 11, [0, 32, 64], 12288),
    ):
        original = (RESULTS / f"{model}.command.mlir").read_text()
        scheduled = schedule(mini, original)
        entries, commands, ancestors = inspect(mini, original, scheduled)
        assert len(entries) == count
        block, = blocks(scheduled)
        memory = decode(named_attr(block, "memory_plan"))
        tiles = decode(named_attr(block, "tiles"))
        assert memory["peak_live_bytes"] == peak
        group_count = count // group_size
        assert Counter(cmd["opcode"] for cmd in commands) == {
            "ZERO": group_count, "DMA_LOAD_2D": 2 * len(tiles),
            "MATMUL_TILE": len(tiles), "DMA_STORE_2D": group_count,
        }
        regions = set()
        for begin in range(0, count, group_size):
            end = begin + group_size - 1
            first, store = commands[begin], commands[end]
            assert first["opcode"] == "ZERO" and store["opcode"] == "DMA_STORE_2D"
            assert first["buffer_id"] == store["buffer_id"]
            accumulates = list(range(begin + 3, end, 3))
            assert [tiles[commands[step]["tile_index"]]["k_offset"] for step in accumulates] == offsets
            assert {begin, begin + 1, begin + 2} <= set(entries[begin + 3]["dependencies"])
            assert set(accumulates) <= ancestors[end]
            for step in accumulates:
                assert commands[step]["c_buffer_id"] == first["buffer_id"]
                if step + 3 < end:
                    assert step in entries[step + 1]["dependencies"]
                    assert step in entries[step + 2]["dependencies"]
                    assert step in entries[step + 3]["dependencies"]
            desc = tiles[store["tile_index"]]
            region = (desc["m_offset"], desc["n_offset"])
            assert region not in regions
            regions.add(region)
        expected_regions = {(0, 0), (0, 32), (32, 0), (32, 32)} if count == 44 else {(0, 0)}
        assert regions == expected_regions
        weight, = onnx.load(PROJECT / "test/e2e/models" / f"{model}.onnx").graph.initializer
        payload = scheduled.split('arith.constant dense<"0x', 1)[1].split('">', 1)[0]
        np.testing.assert_array_equal(
            np.frombuffer(bytes.fromhex(payload), dtype="<u4"),
            numpy_helper.to_array(weight).view(np.uint32).ravel(),
        )
        edge_count = sum(len(entry["dependencies"]) for entry in entries)
        print(f"PASS: {model}: {count} records, {edge_count} dependency edges, "
              f"{group_count} complete ZERO/K/STORE groups; weights and SRAM peak unchanged")
        if count == 44:
            for index in (3, 4, 5, 6, 7, 8, 9, 10, 11, 43):
                print(f"  seq {index} {commands[index]['opcode']}: predecessors {entries[index]['dependencies']}")
        else:
            assert [entry["dependencies"] for entry in entries] == [[], [], [], [0, 1, 2], [3]]


def reject_schedule(mini, ir, fragments, *options):
    result = run(mini, ir, *options, "-o", "/dev/null")
    assert result.returncode == 1 and any(fragment in result.stderr.lower() for fragment in fragments), (
        result.returncode, fragments, result.stderr.splitlines()[:1],
    )
    assert not result.stdout
    return result


def check_negative(mini):
    original = (RESULTS / "matmul_multitile.command.mlir").read_text()
    scheduled = schedule(mini, original)
    serialized = named_attr(scheduled, SCHEDULE)
    baseline = decode(serialized)

    def rejected(label, mutate, fragments):
        altered = deepcopy(baseline)
        mutate(altered)
        broken = scheduled.replace(serialized, encode(altered), 1)
        assert broken != scheduled
        assert blocks(broken) == blocks(scheduled)
        result = reject_schedule(mini, broken, fragments)
        print(f"PASS: {label}: {result.stderr.splitlines()[0]}")

    def patch_command(index, **fields):
        return lambda plan: plan["commands"][index].update(fields)

    def reorder(first, second):
        def apply(plan):
            records = plan["commands"]
            records[first]["sequence_index"], records[second]["sequence_index"] = second, first
        return apply

    def order_prefix(*command_ids):
        def apply(plan):
            for sequence, command in enumerate(command_ids):
                plan["commands"][command]["sequence_index"] = sequence
        return apply

    for label, mutate, fragments in (
        ("duplicate ID", patch_command(1, command_id=0), ("command id", "command_id", "duplicate")),
        ("duplicate sequence", patch_command(1, sequence_index=0), ("sequence",)),
        ("sequence gap", patch_command(1, sequence_index=44), ("sequence",)),
        ("missing schedule row", lambda p: p["commands"].pop(), ("count", "total", "missing")),
        ("total count mismatch", lambda p: p.update(total_commands=0), ("count", "total")),
        ("unknown MatMul", patch_command(1, matmul_index=19), ("origin", "matmul", "reference")),
        ("unknown local event", patch_command(1, local_event_index=99), ("origin", "event", "reference")),
        ("duplicate origin", patch_command(1, local_event_index=0), ("does not match matmul and local event",)),
        ("unknown dependency", patch_command(3, dependencies=[0, 1, 999]), ("dependency", "predecessor")),
        ("duplicate dependency", patch_command(3, dependencies=[0, 0, 1, 2]), ("dependencies must be sorted and unique",)),
        ("cyclic dependencies", patch_command(0, dependencies=[3]), ("cycle in command dependencies",)),
        ("missing LOAD→COMPUTE dependency", patch_command(3, dependencies=[]), ("missing required dependency",)),
        ("COMPUTE before A load", reorder(3, 1), ("reads a/b before load",)),
        ("COMPUTE before B load", reorder(3, 2), ("reads a/b before load",)),
        ("COMPUTE before ZERO", order_prefix(1, 2, 3, 0), ("reads c before zero",)),
        ("early overwrite of A", reorder(3, 4), ("overlaps a still-live buffer",)),
        ("early overwrite of B", reorder(3, 5), ("overlaps a still-live buffer",)),
        ("early STORE", reorder(9, 10), ("store before the k reduction is complete",)),
        ("reversed K compute segments", order_prefix(0, 4, 5, 6, 1, 2, 3), ("violates k reduction order",)),
        ("compatible data but changed lifetime", reorder(1, 2), ("actual sram lifetime does not match",)),
        ("invalid contract", lambda p: p.update(contract="async"), ("contract",)),
        ("invalid schema version", lambda p: p.update(schema_version=3), ("schema",)),
        ("missing origin field", lambda p: p["commands"][0].pop("local_event_index"), ("schema", "local_event_index")),
    ):
        rejected(label, mutate, fragments)

    def reordered_without_declared_edges(plan):
        order_prefix(1, 2, 3, 0)(plan)
        for item in plan["commands"]:
            item["dependencies"] = []

    rejected("invalid order with formally acyclic empty DAG", reordered_without_declared_edges,
             ("reads c before zero",))
    reject_schedule(
        mini, scheduled.replace(serialized, encode({**baseline, "contract": "async"}), 1),
        ("unsupported execution schedule contract",), "--npu-schedule",
    )
    print("PASS: rerunning the pass refuses to replace an invalid saved schedule")
    chain = schedule(mini, compile_commands(mini, SHARED_CHAIN))
    initial = named_attr(chain, SCHEDULE)
    changed = decode(initial)
    reorder(31, 33)(changed)
    reject_schedule(mini, chain.replace(initial, encode(changed), 1),
                    ("reads a dram tensor before all output tiles are written",))
    print("PASS: schedule-only mutation attempts LOAD H after only three of four producer stores")

    unconverted = (RESULTS / "weighted_matmul.sram.mlir").read_text()
    reject_schedule(mini, unconverted, ("dram_layout", "command", "dma"), "--npu-schedule")
    print("PASS: SRAM-only input rejected at scheduling stage")


if __name__ == "__main__":
    mini, group = sys.argv[1:]
    {"positive": check_positive, "negative": check_negative, "onnx": check_onnx}[group](mini)
