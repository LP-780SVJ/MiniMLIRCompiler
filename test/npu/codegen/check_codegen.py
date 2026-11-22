"""Compile real scheduled IR and check the standalone package boundary."""

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve()
SCHEDULE_TEST = runpy.run_path(str(HERE.parents[1] / "schedule/check_npu_schedule.py"))
PROJECT, RESULTS, SCHEDULE, SHARED_CHAIN, blocks, named_attr, decode, encode = (
    SCHEDULE_TEST[name] for name in (
        "PROJECT", "RESULTS", "SCHEDULE", "SHARED_CHAIN", "blocks", "named_attr", "decode", "encode"
    )
)
PROFILE = PROJECT / "configs/virtual-npu-v1.json"
MODELS = PROJECT / "test/e2e/models"
VALIDATOR = PROJECT / "python/runtime/validate_package.py"
DRAM = "npu.dram_layout"
FILES = ("program.json", "weights.bin", "manifest.json")
WEIGHTED = "weighted_matmul"
MULTITILE = "matmul_multitile"


def invoke(args, env=None):
    return subprocess.run(args, env=env, capture_output=True, text=True)


def succeeded(result):
    assert result.returncode == 0, result.stderr
    return result.stdout


def rejected(result, message):
    assert result.returncode == 1 and message in result.stderr, result


def compile_ir(codegen, source, out, metadata=None, target=PROFILE, env=None):
    args = [codegen, "--input", source, "--target", target, "--out-dir", out]
    if metadata is not None:
        args.extend(["--model-metadata", metadata])
    return invoke(args, env)


def validator(package):
    return invoke([sys.executable, "-I", VALIDATOR, "--package", package])


def metadata(model):
    return MODELS / f"{model}.metadata.json"


def scheduled(model):
    return RESULTS / f"{model}.scheduled.mlir"


def lower(mini, text, destination):
    commands = SCHEDULE_TEST["compile_commands"](mini, text)
    result = SCHEDULE_TEST["schedule"](mini, commands)
    destination.write_text(result)
    return result


def package_json(package):
    return tuple(json.loads((package / name).read_bytes())
                 for name in ("program.json", "manifest.json"))


def compare_ir(ir, program, manifest):
    # The existing test-only attribute inspector can independently compare all
    # serialized payload fields. CodeGen itself always uses the MLIR parser.
    schedule = decode(named_attr(ir, SCHEDULE))
    local_commands = [decode(named_attr(line, "commands")) for line in blocks(ir)]
    for sequence_record, instruction in zip(
        sorted(schedule["commands"], key=lambda record: record["sequence_index"]),
        program["instructions"], strict=True,
    ):
        group, event = sequence_record["matmul_index"], sequence_record["local_event_index"]
        payload = local_commands[group][event].copy()
        assert payload.pop("event_index") == event
        assert instruction == {**sequence_record, **payload}, instruction

    layout = decode(named_attr(ir, DRAM))
    assert manifest["hardware"]["dram_bytes"] == layout["total_bytes"]
    for port, attribute in (("inputs", "input_ids"), ("outputs", "output_ids")):
        assert [entry["tensor_id"] for entry in manifest[port]] == layout[attribute]
    for tensor, source in zip(manifest["tensors"], layout["tensors"], strict=True):
        assert tensor == {
            **{key: source[key] for key in ("id", "kind", "base", "row_stride_bytes", "size_bytes")},
            "shape": [source["rows"], source["cols"]],
        }


def check_e2e(mini, codegen, root):
    for model, opcode_counts, dma_bytes, weight_bytes, dram_bytes in (
        (WEIGHTED, (1, 2, 1, 1), 104, 48, 160),
        (MULTITILE, (4, 24, 12, 4), 112 * 1024, 24576, 65536),
    ):
        out = root / model
        succeeded(compile_ir(codegen, scheduled(model), out, metadata(model)))
        assert f"{sum(opcode_counts)} instructions, {weight_bytes} weight bytes" in succeeded(validator(out))
        ir = scheduled(model).read_text()
        program, manifest = package_json(out)
        compare_ir(ir, program, manifest)
        names = ("ZERO", "DMA_LOAD_2D", "MATMUL_TILE", "DMA_STORE_2D")
        assert Counter(cmd["opcode"] for cmd in program["instructions"]) == dict(zip(names, opcode_counts))
        assert sum(cmd["rows"] * cmd["row_bytes"] for cmd in program["instructions"]
                   if cmd["opcode"].startswith("DMA_")) == dma_bytes
        assert manifest["inputs"] == [{"name": "X", "tensor_id": 0}]
        assert manifest["outputs"] == [{"name": "Y", "tensor_id": 2}]
        assert manifest["hardware"]["dram_bytes"] == dram_bytes
        assert manifest["target_profile_sha256"] == sha256(PROFILE.read_bytes()).hexdigest()
        k_offsets = []
        for instruction in program["instructions"]:
            if instruction["opcode"] == "DMA_LOAD_2D" and instruction["tensor_id"] == 0:
                tensor = manifest["tensors"][0]
                offset = instruction["src_dram"] - tensor["base"]
                k_offsets.append(offset % tensor["row_stride_bytes"] // 4)
        assert k_offsets == ([0] if model == WEIGHTED else [0, 32, 64] * 4)
        hex_ir = succeeded(invoke([mini, scheduled(model), SCHEDULE_TEST["HEX"]]))
        encoded = hex_ir.split('arith.constant dense<"0x', 1)[1].split('">', 1)[0]
        assert (out / "weights.bin").read_bytes() == bytes.fromhex(encoded)
        assert len(manifest["weights"]["entries"]) == 1
        print(f"PASS: {model} exact command/DRAM round-trip, lengths, names and weight bits")

    chain = root / "chain.mlir"
    ir = lower(mini, SHARED_CHAIN, chain)
    out = root / "shared"
    succeeded(compile_ir(codegen, chain, out))
    program, manifest = package_json(out)
    compare_ir(ir, program, manifest)
    assert [entry["name"] for entry in manifest["outputs"]] == ["output_0", "output_1", "output_2"]
    assert [entry["tensor_id"] for entry in manifest["outputs"]] == [3, 2, 3]
    assert len(manifest["weights"]["entries"]) == 1
    assert (out / "weights.bin").read_bytes() == struct.pack("<f", 1.0) * 4096
    assert program["instructions"][32]["matmul_index"] == 1
    assert program["instructions"][32]["local_event_index"] == 0
    assert 31 in program["instructions"][33]["dependencies"]
    succeeded(validator(out))
    print("PASS: two MatMuls, one shared splat, producer/consumer and ordered duplicate outputs")

    constants = root / "constants.mlir"
    words = [0, 0x80000000, 1, 0x80000001, 0x7FC12345, 0x7F800000, 0xFF800000, 0x7F812345]
    lower(mini, """
      func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %w0 = arith.constant dense<[[0x00000000, 0x80000000, 0x00000001, 0x80000001],
                                   [0x7FC12345, 0x7F800000, 0xFF800000, 0x7F812345]]> : tensor<2x4xf32>
        %w1 = arith.constant dense<0x80000000> : tensor<4x2xf32>
        %h = "npu.matmul"(%x, %w0) : (tensor<2x2xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
        %y = "npu.matmul"(%h, %w1) : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
        return %y : tensor<2x2xf32>
      }
    """, constants)
    out = root / "aligned"
    succeeded(compile_ir(codegen, constants, out))
    succeeded(validator(out))
    _, manifest = package_json(out)
    assert (out / "weights.bin").read_bytes() == (
        struct.pack("<8I", *words) + bytes(32) + struct.pack("<I", 0x80000000) * 8
    )
    assert manifest["tensors"][3]["kind"] == "intermediate"
    assert manifest["weights"]["entries"] == [
        {"tensor_id": 1, "file_offset": 0, "size_bytes": 32, "dram_base": 64},
        {"tensor_id": 2, "file_offset": 64, "size_bytes": 32, "dram_base": 128},
    ]
    print("PASS: exact signed zero, subnormals, NaNs, infinities, aligned offsets and zero padding")

    no_weights = root / "unweighted.mlir"
    lower(mini, SCHEDULE_TEST["make_matmul"](2, 4, 3), no_weights)
    out = root / "empty-weights"
    succeeded(compile_ir(codegen, no_weights, out))
    _, manifest = package_json(out)
    assert [port["name"] for port in manifest["inputs"]] == ["input_0", "input_1"]
    assert manifest["weights"] == {
        "entries": [], "file": "weights.bin", "sha256": sha256(b"").hexdigest(), "size_bytes": 0,
    }
    assert (out / "weights.bin").read_bytes() == b""
    succeeded(validator(out))
    print("PASS: two inputs and an empty, independently loadable weight file")


def check_reproducible(codegen, root):
    first, second, reordered = (root / name for name in ("first", "second", "reordered"))
    source = scheduled(MULTITILE)
    for out in (first, second):
        succeeded(compile_ir(codegen, source, out, metadata(MULTITILE)))
    ir = source.read_text()
    attribute = named_attr(ir, SCHEDULE)
    entries = decode(attribute)
    entries["commands"].reverse()
    alternate = root / "reordered.mlir"
    alternate.write_text(ir.replace(attribute, encode(entries), 1))
    succeeded(compile_ir(codegen, alternate, reordered, metadata(MULTITILE)))
    for name in FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes() == (reordered / name).read_bytes()
    print("PASS: all three files byte-identical across exports and reversed schedule storage")


def check_bad_source(mini, codegen, root):
    ir = scheduled(WEIGHTED).read_text()
    schedule = named_attr(ir, SCHEDULE)
    layout = named_attr(ir, DRAM)
    bad_schedule = decode(schedule)
    bad_schedule["commands"][1]["local_event_index"] = 7
    bad_layout = decode(layout)
    bad_layout["output_ids"] = [0]
    for name, text, diagnostic in (
        ("no-schedule", ir.replace(f", {SCHEDULE} = {schedule}", "", 1), "missing npu.execution_schedule"),
        ("bad-schedule", ir.replace(schedule, encode(bad_schedule), 1), "unknown local event index"),
        ("no-layout", ir.replace(f"{DRAM} = {layout}, ", "", 1), "npu.dram_layout must be a dictionary"),
        ("bad-layout", ir.replace(layout, encode(bad_layout), 1), "does not match"),
        ("compute-only", SCHEDULE_TEST["make_matmul"](2, 4, 3), "missing npu.execution_schedule"),
    ):
        source = root / f"{name}.mlir"
        out = root / name
        source.write_text(text)
        rejected(compile_ir(codegen, source, out), diagnostic)
        assert not out.exists() and not list(root.glob(f"{name}.tmp-*"))

    profile = json.loads(PROFILE.read_text())
    bad_target = root / "bad-target.json"
    profile["memory"]["sram_bytes"] //= 2
    bad_target.write_text(json.dumps(profile))
    rejected(compile_ir(codegen, scheduled(WEIGHTED), root / "target", target=bad_target),
             "does not match target profile")
    profile["compute"]["native_matmul_tile"]["n"] = 16
    bad_target.write_text(json.dumps(profile))
    rejected(compile_ir(codegen, scheduled(WEIGHTED), root / "native", target=bad_target),
             "requires native tile 32x32x32")

    bad_metadata = root / "bad-metadata.json"
    record = json.loads(metadata(WEIGHTED).read_text())
    record["outputs"]["Y"][0] = 4
    bad_metadata.write_text(json.dumps(record))
    rejected(compile_ir(codegen, scheduled(WEIGHTED), root / "shape", bad_metadata), "shape/order mismatch")
    two_inputs = root / "two-inputs.mlir"
    lower(mini, SCHEDULE_TEST["make_matmul"](2, 4, 3), two_inputs)
    record["inputs"] = {"lhs": [2, 3], "rhs": [3, 4]}
    record["input_names"] = ["rhs", "lhs"]
    record["outputs"]["Y"] = [2, 4]
    bad_metadata.write_text(json.dumps(record))
    rejected(compile_ir(codegen, two_inputs, root / "order", bad_metadata), "shape/order mismatch")
    record["dtype"] = "float16"
    bad_metadata.write_text(json.dumps(record))
    rejected(compile_ir(codegen, two_inputs, root / "dtype", bad_metadata), "schema, target or dtype")
    print("PASS: missing/mismatched schedule or layout, compute IR, bad target and metadata shape/order")


def check_publication(codegen, root, valid):
    before = {name: (valid / name).read_bytes() for name in FILES}
    source = scheduled(WEIGHTED)
    rejected(compile_ir(codegen, source, valid), "already exists")
    assert before == {name: (valid / name).read_bytes() for name in FILES}
    link = root / "output-link"
    link.symlink_to(root / "does-not-exist")
    rejected(compile_ir(codegen, source, link), "already exists")
    assert link.is_symlink()

    bin_dir = root / "test-bin"
    bin_dir.mkdir()
    shim = bin_dir / "python3"
    env = {**os.environ, "PATH": str(bin_dir)}
    shim.write_text(f"#!{sys.executable}\n"
                    "from pathlib import Path\nimport sys\n"
                    f"assert {{f.name for f in Path(sys.argv[4]).iterdir()}} == {set(FILES)!r}\n"
                    "sys.exit(19)\n")
    shim.chmod(0o755)
    failed_output = root / "failed-validation"
    rejected(compile_ir(codegen, source, failed_output, env=env), "validation failed (exit 19)")
    assert not failed_output.exists() and not list(root.glob("failed-validation.tmp-*"))

    raced = root / "race"
    shim.write_text(f"#!{sys.executable}\n"
                    "from pathlib import Path\nimport sys, os\n"
                    f"dest = Path({str(raced)!r})\n"
                    "dest.mkdir()\n"
                    "(dest / 'owner').write_text('other publisher')\n"
                    "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n")
    rejected(compile_ir(codegen, source, raced, env=env), "cannot publish package")
    assert list(raced.iterdir()) == [raced / "owner"]
    assert (raced / "owner").read_text() == "other publisher" and not list(root.glob("race.tmp-*"))
    print("PASS: existing output, dangling symlink, post-write validator rejection and concurrent publication")


def rewrite_file(package, filename, record, contents):
    (package / filename).write_bytes(contents)
    record["size_bytes"] = len(contents)
    record["sha256"] = sha256(contents).hexdigest()


def check_bad_package(root, good):
    def mutate(name, diagnostic, edit, repair_program=False):
        out = root / f"bad-{name}"
        shutil.copytree(good, out)
        program, manifest = package_json(out)
        edit(out, program, manifest)
        if repair_program:
            rewrite_file(out, "program.json", manifest["program"], json.dumps(program).encode())
        (out / "manifest.json").write_text(json.dumps(manifest))
        rejected(validator(out), diagnostic)

    for name, diagnostic, index, key, value in (
        ("opcode", "unknown opcode", 0, "opcode", "BRANCH"),
        ("id", "command ID", 1, "command_id", 0),
        ("sequence", "non-contiguous sequence", 1, "sequence_index", 5),
        ("origin", "event origin", 1, "local_event_index", 3),
        ("dependency", "dependency ID", 3, "dependencies", [0, 3]),
        ("tensor-id", "unknown DMA tensor ID", 1, "tensor_id", 14),
        ("sram-bound", "invalid SRAM range", 1, "dst_sram", 262144),
        ("dram-bound", "invalid DMA DRAM range", 1, "src_dram", 160),
        ("stride", "invalid DMA SRAM stride", 1, "dst_stride_bytes", 4),
        ("row-crossing", "invalid DMA DRAM range or row crossing", 2, "src_dram", 68),
        ("tile-size", "integer outside", 3, "m", 33),
        ("json-range", "integer outside", 1, "src_dram", 2**53),
        ("bool-as-int", "integer outside", 1, "rows", True),
        ("accumulate", "expected accumulate=true", 3, "accumulate", 1),
    ):
        mutate(name, diagnostic,
               lambda _, p, __: p["instructions"][index].update({key: value}), True)
    mutate("missing-field", "missing or unexpected fields",
           lambda _, p, __: p["instructions"][3].pop("n"), True)
    mutate("program-version", "unsupported schema version", lambda _, p, __: p.update(schema_version=2), True)
    mutate("manifest-version", "unsupported schema version", lambda _, __, m: m.update(schema_version=2))
    mutate("target", "incompatible target", lambda _, __, m: m.update(target="other"))
    mutate("no-input", "missing or misordered input", lambda _, __, m: m["inputs"].clear())
    mutate("io-reference", "unknown tensor ID", lambda _, __, m: m["outputs"][0].update(tensor_id=999))
    mutate("overlap", "invalid or overlapping DRAM range", lambda _, __, m: m["tensors"][1].update(base=0))
    mutate("count", "unexpected array or length",
           lambda _, p, m: (p.update(instruction_count=4), m["program"].update(instruction_count=4)), True)
    mutate("path", "unsafe package filename", lambda _, __, m: m["weights"].update(file="../weights.bin"))
    mutate("missing-file", "expected exactly", lambda out, _, __: (out / "weights.bin").unlink())
    mutate("symlink", "unsafe package file", lambda out, _, __: replace_with_symlink(out, good))
    mutate("truncated", "file length mismatch",
           lambda out, _, __: (out / "weights.bin").write_bytes(b""))
    mutate("bad-entry-size", "does not match tensor descriptor",
           lambda _, __, m: m["weights"]["entries"][0].update(size_bytes=4))
    mutate("bad-entry-offset", "invalid offset or truncated tensor",
           lambda _, __, m: m["weights"]["entries"][0].update(file_offset=64))
    mutate("weight-byte", "SHA-256 mismatch", lambda out, _, __: flip_byte(out / "weights.bin"))
    mutate("program-byte", "SHA-256 mismatch", lambda out, _, __: flip_byte(out / "program.json"))
    mutate("json-duplicate-key", "duplicate field",
           lambda out, _, m: rewrite_file(out, "program.json", m["program"], b'{"a":1,"a":2}'))
    print("PASS: structure, integer bounds, I/O, SRAM/DRAM, weights, digests and safe paths")


def flip_byte(path):
    contents = bytearray(path.read_bytes())
    contents[0] ^= 1
    path.write_bytes(contents)


def replace_with_symlink(out, good):
    file = out / "weights.bin"
    file.unlink()
    file.symlink_to(good / file.name)


def check_negative(mini, codegen, root):
    valid = root / "valid"
    succeeded(compile_ir(codegen, scheduled(WEIGHTED), valid, metadata(WEIGHTED)))
    succeeded(validator(valid))
    check_bad_source(mini, codegen, root)
    check_publication(codegen, root, valid)
    check_bad_package(root, valid)
    source = root / "splat.mlir"
    lower(mini, SHARED_CHAIN, source)
    splat = root / "splat"
    succeeded(compile_ir(codegen, source, splat))
    _, manifest = package_json(splat)
    rewrite_file(splat, "weights.bin", manifest["weights"], struct.pack("<f", 1.0))
    (splat / "manifest.json").write_text(json.dumps(manifest))
    rejected(validator(splat), "invalid offset or truncated tensor")
    print("PASS: truncated real splat rejected even with corrected file length and SHA-256")


if __name__ == "__main__":
    mini, codegen, case = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        if case == "e2e":
            check_e2e(mini, codegen, root)
        elif case == "negative":
            check_negative(mini, codegen, root)
        else:
            assert case == "reproducible", case
            check_reproducible(codegen, root)
