"""Run the Day 28 CLI with real ONNX models and check its evidence."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import TensorProto, helper


FIXTURES = Path(__file__).resolve().parent
PROJECT = FIXTURES.parents[1]
DRIVER = PROJECT / "python/e2e/run_e2e.py"
PROFILE = PROJECT / "configs/virtual-npu-v1.json"
FILES = {"program.json", "manifest.json", "weights.bin"}
MODELS = {"small": "matmul", "weighted": "weighted_matmul", "multitile": "matmul_multitile",
          "chain": "matmul_chain"}
ARTIFACTS = {
    "01-imported.mlir", "02-npu-compute.mlir", "03-tiled.mlir", "04-sram-planned.mlir",
    "05-command.mlir", "06-scheduled.mlir", "model.metadata.json", "compiler_stats.json",
    "actual_outputs.npz", "reference_outputs.npz", "execution_trace.json", "numerical_comparison.json",
}


def cli(options, model, inputs, out, *flags, codegen=None):
    return subprocess.run(
        [sys.executable, "-B", DRIVER, "--opt", options.opt, "--codegen", codegen or options.codegen,
         "--model", model, "--inputs", inputs, "--out-dir", out, *flags],
        cwd=PROJECT, capture_output=True, text=True,
    )


def fixture(name):
    return FIXTURES / "models" / f"{name}.onnx", FIXTURES / "inputs" / f"{name}.npz"


def read_json(path):
    return json.loads(path.read_text())


def passed(result, out, reference_source="onnxruntime"):
    assert result.returncode == 0, result.stderr
    report = read_json(out / "validation_report.json")
    assert report["status"] == "passed" and report["numerical_validation"]["passed"], report
    assert report["reference"]["source"] == reference_source
    if reference_source == "onnxruntime":
        assert report["reference"]["provider"] == "CPUExecutionProvider"
        assert report["versions"]["onnxruntime"]
    assert report["compiler"]["tools"]["mini-opt"]["version"]
    assert all(step["exit_code"] == 0 for step in report["stages"])
    assert {path.name for path in (out / "bundle").iterdir()} == FILES
    paths = ARTIFACTS | {f"bundle/{filename}" for filename in FILES}
    for name in paths:
        data = (out / name).read_bytes()
        assert report["artifacts"][name] == {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
    assert report["program"]["instruction_count"] == report["runtime"]["executed_instructions"]
    assert report["program"]["dma_transferred_bytes"] == report["runtime"]["dma_bytes"]
    return report


def failed(result, out, phase, diagnostic):
    assert result.returncode != 0 and diagnostic in result.stderr, (phase, result.stderr, result.stdout)
    report = read_json(out / "validation_report.json")
    assert report["status"] == "failed" and report["failure_stage"] == phase, report
    assert diagnostic in report["diagnostic"], report
    return report


def make_matmul(root, name, a_shape, b_shape):
    model_path, input_path = root / f"{name}.onnx", root / f"{name}.npz"
    graph = helper.make_graph([helper.make_node("MatMul", ["A", "B"], ["Y"])], name, [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape),
    ], [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (a_shape[0], b_shape[1]))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    onnx.checker.check_model(model)
    model_path.write_bytes(model.SerializeToString(deterministic=True))
    rng = np.random.default_rng(2026)
    np.savez(input_path, A=rng.standard_normal(a_shape, dtype=np.float32),
             B=rng.standard_normal(b_shape, dtype=np.float32))
    return model_path, input_path


def check_chain(out, multiple_outputs):
    manifest = read_json(out / "bundle/manifest.json")
    stats = read_json(out / "compiler_stats.json")
    trace = read_json(out / "execution_trace.json")["instructions"]
    first, second = stats["matmul_groups"]
    store, load = trace[4], trace[6]
    intermediate_id = first["result_tensor_id"]
    assert intermediate_id == second["operand_tensor_ids"][0] == store["tensor_id"] == load["tensor_id"]
    assert store["dst_dram"] == load["src_dram"] and store["command_id"] in load["dependencies"]
    assert store["opcode"] == "DMA_STORE_2D" and load["opcode"] == "DMA_LOAD_2D"
    assert store["matmul_index"] == 0 and load["matmul_index"] == 1
    assert manifest["tensors"][intermediate_id]["kind"] == ("output" if multiple_outputs else "intermediate")
    expected = [{"name": "Y", "tensor_id": second["result_tensor_id"]}]
    if multiple_outputs:
        expected.append({"name": "H", "tensor_id": intermediate_id})
    assert manifest["outputs"] == expected
    for name in ("actual_outputs.npz", "reference_outputs.npz"):
        with np.load(out / name, allow_pickle=False) as archive:
            assert archive.files == [port["name"] for port in expected]
    assert stats["memory"]["peak_live_bytes"] == max(first["peak_live_bytes"], second["peak_live_bytes"])


def check_positive(options, root, case):
    if case == "tails":
        model, inputs = make_matmul(root, "tail_m", (65, 7), (7, 9))
    elif case == "multioutput":
        chain, inputs = fixture("matmul_chain")
        model = root / "multioutput.onnx"
        graph = onnx.load(chain)
        # Output order is intentionally the reverse of production order.
        graph.graph.output.append(helper.make_tensor_value_info("H", TensorProto.FLOAT, [2, 4]))
        onnx.checker.check_model(graph, full_check=True)
        model.write_bytes(graph.SerializeToString(deterministic=True))
    else:
        model, inputs = fixture(MODELS[case])
    out = root / case
    report = passed(cli(options, model, inputs, out), out)
    program, memory = report["program"], report["memory"]
    measured = (program["output_tiles"], program["matmul_tiles"], program["instruction_count"],
                program["dma_transferred_bytes"], memory["peak_live_bytes"])
    expected = {
        "small": (1, 1, 5, 104, 192), "weighted": (1, 1, 5, 104, 192),
        "multitile": (4, 12, 44, 114688, 12288), "chain": (2, 2, 10, 256, 256),
        "tails": (3, 3, 15, 4916, 2304), "multioutput": (2, 2, 10, 256, 256),
    }[case]
    assert measured == expected, (case, measured, expected)
    assert memory["sram_capacity_bytes"] == 262144
    if case in ("chain", "multioutput"):
        check_chain(out, case == "multioutput")
    elif case == "tails":
        records = read_json(out / "execution_trace.json")["instructions"]
        assert [record["m"] for record in records if record["opcode"] == "MATMUL_TILE"] == [32, 32, 1]
    elif case in ("small", "weighted"):
        manifest = read_json(out / "bundle/manifest.json")
        assert [port["name"] for port in manifest["inputs"]] == (["A", "B"] if case == "small" else ["X"])
        assert manifest["weights"]["size_bytes"] == (0 if case == "small" else 48)
    print(case, measured, report["numerical_validation"]["outputs"])


def check_reproducible(options, root):
    model, inputs = fixture("weighted_matmul")
    profile = read_json(PROFILE)
    profile["memory"]["reserved_bytes"] = 1
    target = root / "target.json"
    target.write_text(json.dumps(profile))
    first, second = root / "first", root / "second with spaces"
    a = passed(cli(options, model, inputs, first, "--target", target), first)
    golden = FIXTURES / "reference/weighted_matmul.npz"
    b = passed(cli(options, model, inputs, second, "--target", target, "--reference", golden),
               second, reference_source="provided")
    for file in FILES:
        rel = f"bundle/{file}"
        assert a["artifacts"][rel] == b["artifacts"][rel]
        assert (first / rel).read_bytes() == (second / rel).read_bytes()
    assert a["artifacts"]["compiler_stats.json"] == b["artifacts"]["compiler_stats.json"]
    assert a["memory"]["peak_live_bytes"] == 192 and a["memory"]["arena_high_water_bytes"] == 256
    commit = (first / "validation_report.json").read_bytes()
    retry = cli(options, model, inputs, first)
    assert retry.returncode != 0 and "new output directory" in retry.stderr
    assert (first / "validation_report.json").read_bytes() == commit
    # The optional statistics output can never add a fourth file to the Bundle.
    bundle = root / "bad-bundle"
    result = subprocess.run([
        options.codegen, "--input", first / "06-scheduled.mlir", "--target", target,
        "--out-dir", bundle, "--stats-out", bundle / "stats.json",
    ], capture_output=True, text=True)
    assert result.returncode == 1 and "must be outside the program package" in result.stderr
    assert not bundle.exists()
    print("PASS: Bundle and compiler stats byte-identical, peak/high-water differ, existing run preserved")


def make_codegen_shim(root, codegen, fault):
    script = root / f"{fault}.py"
    script.write_text(f"""
from hashlib import sha256
from pathlib import Path
import json
import subprocess
import sys

args = sys.argv[1:]
result = subprocess.run([{str(codegen)!r}, *args])
if result.returncode or "--out-dir" not in args:
    sys.exit(result.returncode)
base = Path(args[args.index("--out-dir") + 1])
fault = {fault!r}
if fault == "weights":
    path = base / "weights.bin"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    sys.exit(0)

program = json.loads((base / "program.json").read_bytes())
manifest = json.loads((base / "manifest.json").read_bytes())
commands = program["instructions"]
if fault == "dma":
    commands[1]["src_dram"] = manifest["hardware"]["dram_bytes"]
elif fault == "load":
    del commands[1]
elif fault == "zero":
    commands[4] = dict(commands[0], tile_index=1)
elif fault == "store":
    commands[4] = dict(commands[10], tile_index=0)
    del commands[5:]
elif fault == "buffer":
    commands[3]["a_buffer_id"] = 99
else:
    raise AssertionError(fault)
for index, command in enumerate(commands):
    command.update(sequence_index=index, command_id=index, local_event_index=index, dependencies=[])
program["instruction_count"] = len(commands)
data = json.dumps(program).encode()
(base / "program.json").write_bytes(data)
manifest["program"].update(sha256=sha256(data).hexdigest(), size_bytes=len(data),
                           instruction_count=len(commands))
(base / "manifest.json").write_text(json.dumps(manifest))
""")
    executable = root / f"codegen-{fault}"
    executable.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))} \"$@\"\n")
    executable.chmod(0o755)
    return executable


def check_negative(options, root):
    invalid = make_matmul(root, "illegal_k", (2, 3), (4, 2))
    out = root / "invalid-k"
    failed(cli(options, *invalid, out), out, "importer", "E_SHAPE_MISMATCH")
    assert not (out / "01-imported.mlir").exists()
    out = root / "relu"
    failed(cli(options, *fixture("matmul_relu"), out), out, "mini-npu-compile", "mini.relu")
    assert (out / "01-imported.mlir").is_file() and not (out / "02-npu-compute.mlir").exists()
    out = root / "capacity"
    profile = read_json(PROFILE)
    profile["memory"]["sram_bytes"] = 8192
    target = root / "tiny-sram.json"
    target.write_text(json.dumps(profile))
    failed(cli(options, *fixture("matmul_multitile"), out, "--target", target),
           out, "npu-sram-plan", "insufficient SRAM capacity")
    assert (out / "03-tiled.mlir").exists() and not (out / "bundle").exists()

    for fault, model, phase, message in (
        ("dma", "weighted_matmul", "loader", "invalid DMA DRAM range"),
        ("weights", "weighted_matmul", "loader", "SHA-256 mismatch"),
        ("load", "weighted_matmul", "runtime", "requires both live A/B"),
        ("zero", "matmul_multitile", "runtime", "repeated ZERO"),
        ("store", "matmul_multitile", "runtime", "STORE before the K reduction is complete"),
        ("buffer", "weighted_matmul", "runtime", "A Buffer ID"),
    ):
        out = root / fault
        shim = make_codegen_shim(root, options.codegen, fault)
        report = failed(cli(options, *fixture(model), out, codegen=shim), out, phase, message)
        assert not (out / "actual_outputs.npz").exists()
        if phase == "runtime":
            trace = read_json(out / "execution_trace.json")
            assert trace["instructions"][-1]["status"] == "error"
            assert report["runtime"]["executed_instructions"] < report["program"]["instruction_count"]
        else:
            assert not (out / "execution_trace.json").exists()
    print("PASS: diagnosed importer, Lowering, SRAM Planner, Loader, and Runtime failures")


def check_comparator(options, root):
    model, inputs = fixture("weighted_matmul")
    clean = root / "clean"
    passed(cli(options, model, inputs, clean), clean)
    with np.load(clean / "actual_outputs.npz", allow_pickle=False) as outputs:
        corrupt = outputs["Y"].copy()
    corrupt.flat[0] += np.float32(1)
    changed = root / "perturbed.npz"
    np.savez(changed, Y=corrupt)
    result = subprocess.run([
        sys.executable, "-B", PROJECT / "python/e2e/compare_outputs.py", "--reference",
        clean / "reference_outputs.npz", "--actual", changed,
    ], capture_output=True, text=True)
    assert result.returncode == 1 and json.loads(result.stdout)["outputs"]["Y"]["mismatched_elements"] == 1
    out = root / "failed"
    report = failed(cli(options, model, inputs, out, "--reference", changed), out,
                    "comparator", "numerical comparison failed")
    assert not report["numerical_validation"]["passed"]
    assert report["numerical_validation"]["outputs"]["Y"]["mismatched_elements"] == 1
    print("PASS: one changed element produces a failed numerical and unified report")


def check_case(options, root):
    if options.case == "negative":
        check_negative(options, root)
    elif options.case == "comparator-negative":
        check_comparator(options, root)
    elif options.case == "reproducible":
        check_reproducible(options, root)
    else:
        check_positive(options, root, options.case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opt", type=Path)
    parser.add_argument("codegen", type=Path)
    parser.add_argument("case", choices=[*MODELS, "tails", "multioutput", "negative",
                                        "comparator-negative", "reproducible"])
    parser.add_argument("--out-root", type=Path, help="keep evidence in a new directory")
    options = parser.parse_args()
    options.opt, options.codegen = options.opt.absolute(), options.codegen.absolute()
    if options.out_root:
        options.out_root.mkdir(parents=True)
        check_case(options, options.out_root.absolute())
    else:
        with TemporaryDirectory() as directory:
            check_case(options, Path(directory))
