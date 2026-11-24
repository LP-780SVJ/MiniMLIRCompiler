"""Compile an ONNX model for Virtual NPU, execute it, and report validation."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import onnx

PROJECT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT))

from python.e2e.e2e_metadata import CPU_PROVIDER, build_metadata, check_arrays, run_reference
from python.runtime.validate_package import load_package
from runtime.virtual_npu.tensor_io import atomic_file, load_inputs, save_outputs


PIPELINE = (
    ("mini-npu-compile", "--pass-pipeline=builtin.module(mini-npu-compile)", "02-npu-compute.mlir"),
    ("npu-tiling", "--npu-tiling=config=sources/target.json", "03-tiled.mlir"),
    ("npu-sram-plan", "--npu-sram-plan=config=sources/target.json", "04-sram-planned.mlir"),
    ("npu-insert-dma", "--npu-insert-dma=config=sources/target.json", "05-command.mlir"),
    ("npu-schedule", "--npu-schedule", "06-scheduled.mlir"),
)


class StageError(ValueError):
    """A validation stage failed to satisfy its contract."""


def sha256_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    with atomic_file(path) as stream:
        stream.write((json.dumps(data, indent=2, allow_nan=False) + "\n").encode())


def git_state():
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                           text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        return revision, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return None, None


def checked_npz(path, shapes, role):
    arrays = load_inputs(path, [{"name": name} for name in shapes])
    check_arrays(arrays, shapes, role)
    return arrays


def program_counts(instructions):
    counts = dict(Counter(cmd["opcode"] for cmd in instructions))
    return {
        "instruction_count": len(instructions),
        "opcode_counts": counts,
        "output_tiles": counts.get("DMA_STORE_2D", 0),
        "matmul_tiles": counts.get("MATMUL_TILE", 0),
        "dma_transferred_bytes": sum(cmd["rows"] * cmd["row_bytes"] for cmd in instructions
                                     if cmd["opcode"].startswith("DMA_")),
    }


def verify_trace(trace, instructions, stats, manifest):
    executed = trace["instructions"]
    if trace["status"] != "ok" or trace["phase"] != "complete" or len(executed) != len(instructions):
        raise StageError("Trace did not complete every Program instruction")
    for cmd, event in zip(instructions, executed, strict=True):
        if event.get("status") != "ok" or any(event.get(key) != value for key, value in cmd.items()):
            raise StageError(f"Trace differs from Program at sequence {cmd['sequence_index']}")
    observed = program_counts(executed)
    if (trace["executed_instructions"] != observed["instruction_count"]
            or trace["opcode_counts"] != observed["opcode_counts"]
            or trace["dma_bytes"] != observed["dma_transferred_bytes"]):
        raise StageError("Trace summary differs from executed instructions")
    if any(stats[key] != observed[key] for key in ("instruction_count", "matmul_tiles", "output_tiles")):
        raise StageError("Compiler Stats counts differ from executed instructions")
    if (stats["target"] != manifest["target"]
            or stats["memory"]["sram_capacity_bytes"] != manifest["hardware"]["sram_bytes"]
            or stats["memory"]["dram_capacity_bytes"] != manifest["hardware"]["dram_bytes"]):
        raise StageError("Compiler Stats target or memory limits differ from the Manifest")


class ValidationRun:
    def __init__(self, args, root):
        self.args, self.root = args, root
        self.phase = "setup"
        self.report = {
            "schema_version": 1,
            "status": "failed",
            "model": args.model.name,
            "stages": [],
            "sources": {},
        }

    def invoke(self, stage, command, cwd=PROJECT):
        self.phase = stage
        argv = [str(arg) for arg in command]
        stdout = self.root / "logs" / f"{stage}.stdout.log"
        stderr = self.root / "logs" / f"{stage}.stderr.log"
        record = {"stage": stage, "argv": argv, "cwd": str(cwd),
                  "stdout": str(stdout.relative_to(self.root)), "stderr": str(stderr.relative_to(self.root))}
        self.report["stages"].append(record)
        with stdout.open("wb") as out, stderr.open("wb") as err:
            try:
                result = subprocess.run(argv, cwd=cwd, stdout=out, stderr=err)
            except OSError as error:
                err.write(str(error).encode())
                record["exit_code"] = None
                raise
        record["exit_code"] = result.returncode
        return result.returncode, stdout.read_text(errors="replace"), stderr.read_text(errors="replace")

    def checked(self, stage, command, cwd=PROJECT):
        code, stdout, stderr = self.invoke(stage, command, cwd)
        if code:
            raise StageError(stderr.strip() or f"{stage} exited with status {code}")
        return stdout

    def snapshot(self, source, name):
        path = self.root / "sources" / name
        shutil.copyfile(source, path)
        self.report["sources"][name] = {"original": str(source.resolve()),
                                        "snapshot": str(path.relative_to(self.root)),
                                        "sha256": sha256_file(path)}
        return path

    def run(self):
        args, root = self.args, self.root
        (root / "logs").mkdir()
        (root / "sources").mkdir()
        model = self.snapshot(args.model, "model.onnx")
        inputs = self.snapshot(args.inputs, "inputs.npz")
        target = self.snapshot(args.target, "target.json")
        reference = self.snapshot(args.reference, "reference.npz") if args.reference else None
        revision, dirty = git_state()
        self.report["compiler"] = {
            "git_commit": revision, "working_tree_dirty": dirty,
            "pipeline": [stage[0] for stage in PIPELINE] + ["mini-npu-codegen"],
            "tools": {},
        }
        self.report["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                                   "onnx": onnx.__version__}
        for name, path in (("mini-opt", args.opt), ("mini-npu-codegen", args.codegen)):
            version = self.checked(f"{name}-version", [path, "--version"])
            self.report["compiler"]["tools"][name] = {
                "path": str(path), "version": version.strip(), "sha256": sha256_file(path),
            }

        ir = root / "01-imported.mlir"
        self.checked("importer", [sys.executable, "-B", PROJECT / "python/onnx_importer/importer.py",
                                  "--input", model, "--output", ir])
        self.phase = "metadata"
        model_bytes = model.read_bytes()
        target_name = json.loads(target.read_text())["target"]
        self.report["target"] = target_name
        inferred = onnx.shape_inference.infer_shapes(
            onnx.load_model_from_string(model_bytes), check_type=True, strict_mode=True)
        metadata = build_metadata(inferred, model_bytes, inputs.read_bytes(), target_name)
        metadata.update(rtol=args.rtol, atol=args.atol)
        metadata_path = root / "model.metadata.json"
        write_json(metadata_path, metadata)
        self.phase = "input-binding"
        feeds = checked_npz(inputs, metadata["inputs"], "input")

        for stage, flag, name in PIPELINE:
            destination = root / name
            self.checked(stage, [args.opt, ir, flag, "--verify-each", "-o", destination], cwd=root)
            ir = destination
        bundle = root / "bundle"
        stats_path = root / "compiler_stats.json"
        self.checked("codegen", [args.codegen, "--input", ir, "--target", target,
                                 "--model-metadata", metadata_path, "--out-dir", bundle,
                                 "--stats-out", stats_path])

        self.phase = "loader"
        package = load_package(bundle)
        instructions = package.program["instructions"]
        self.report["program"] = program_counts(instructions)
        self.report["program"]["inputs"] = package.manifest["inputs"]
        self.report["program"]["outputs"] = package.manifest["outputs"]
        self.phase = "compiler-stats"
        stats = json.loads(stats_path.read_text())
        self.report["memory"] = stats["memory"]

        actual, trace_path = root / "actual_outputs.npz", root / "execution_trace.json"
        code, _, stderr = self.invoke("simulator", [
            sys.executable, "-B", "-m", "runtime.virtual_npu.run",
            "--bundle", bundle, "--inputs", inputs, "--outputs", actual, "--trace", trace_path,
        ])
        trace = json.loads(trace_path.read_text()) if trace_path.is_file() else None
        if trace is not None:
            self.report["runtime"] = {
                field: trace[field] for field in ("status", "phase", "executed_instructions",
                                                 "opcode_counts", "dma_bytes")
            }
        if code:
            if trace and trace["phase"] == "execute":
                self.phase = "runtime"
            elif trace and trace["phase"] == "load":
                self.phase = "loader"
            raise StageError(stderr.strip() or f"Simulator exited with status {code}")

        self.phase = "execution-validation"
        if trace is None:
            raise StageError("Simulator did not write an execution trace")
        verify_trace(trace, instructions, stats, package.manifest)
        expected_outputs = metadata["output_names"]
        with np.load(actual, allow_pickle=False) as archive:
            if archive.files != expected_outputs:
                raise StageError(f"output names/order {archive.files} differ from {expected_outputs}")
        if ([p["name"] for p in package.manifest["inputs"]] != metadata["input_names"]
                or [p["name"] for p in package.manifest["outputs"]] != expected_outputs):
            raise StageError("Manifest I/O names or order differ from the ONNX model")

        self.phase = "reference"
        reference_out = root / "reference_outputs.npz"
        if reference is not None:
            checked_npz(reference, metadata["outputs"], "reference")
            shutil.copyfile(reference, reference_out)
            self.report["reference"] = {"source": "provided", "sha256": sha256_file(reference_out)}
        else:
            outputs, ort_version = run_reference(model_bytes, feeds, metadata["inputs"], metadata["outputs"])
            save_outputs(reference_out, outputs)
            self.report["versions"]["onnxruntime"] = ort_version
            self.report["reference"] = {
                "source": "onnxruntime", "provider": CPU_PROVIDER, "sha256": sha256_file(reference_out),
            }

        code, stdout, stderr = self.invoke("comparator", [
            sys.executable, "-B", PROJECT / "python/e2e/compare_outputs.py",
            "--reference", reference_out, "--actual", actual, "--rtol", args.rtol, "--atol", args.atol,
        ])
        if code not in (0, 1):
            raise StageError(stderr.strip() or f"comparator exited with status {code}")
        comparison = json.loads(stdout)
        self.report["numerical_validation"] = comparison
        write_json(root / "numerical_comparison.json", comparison)
        if code or comparison.get("passed") is not True:
            raise StageError("numerical comparison failed; see numerical_comparison.json")

    def finish(self):
        self.report["artifacts"] = {
            path.relative_to(self.root).as_posix(): {
                "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            } for path in sorted(self.root.rglob("*")) if path.is_file()
        }
        write_json(self.root / "validation_report.json", self.report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=PROJECT / "configs/virtual-npu-v1.json")
    parser.add_argument("--opt", type=Path, default=PROJECT / "build/bin/mini-opt")
    parser.add_argument("--codegen", type=Path, default=PROJECT / "build/bin/mini-npu-codegen")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    for value in (args.rtol, args.atol):
        if not math.isfinite(value) or value < 0:
            parser.error("tolerances must be finite and nonnegative")
    args.opt, args.codegen = args.opt.absolute(), args.codegen.absolute()
    root = args.out_dir.absolute()
    try:
        root.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"e2e setup: cannot create new output directory: {error}", file=sys.stderr)
        return 1

    execution = ValidationRun(args, root)
    try:
        execution.run()
    except Exception as error:
        message = str(error) or type(error).__name__
        execution.report.update(failure_stage=execution.phase, diagnostic=message)
        print(f"e2e {execution.phase}: {message}", file=sys.stderr)
    else:
        execution.report["status"] = "passed"
    try:
        execution.finish()
    except OSError as error:
        print(f"e2e report: {error}", file=sys.stderr)
        return 1
    print(root / "validation_report.json")
    return 0 if execution.report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
