"""Execute a Virtual NPU bundle and export named FP32 output tensors."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

from python.runtime.validate_package import load_package

from .interpreter import execute
from .memory import ExecutionError
from .tensor_io import atomic_file, load_inputs, save_outputs


def write_trace(path, records, phase, error=None):
    executed = [record for record in records if record["status"] == "ok"]
    report = {
        "status": "ok" if error is None else "error",
        "phase": phase,
        "executed_instructions": len(executed),
        "opcode_counts": dict(Counter(record["opcode"] for record in executed)),
        "dma_bytes": sum(record["rows"] * record["row_bytes"] for record in executed
                         if record["opcode"].startswith("DMA_")),
        "instructions": records,
    }
    if error is not None:
        report["error"] = str(error)
    with atomic_file(path) as stream:
        stream.write((json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def check_paths(bundle, inputs, outputs, trace):
    root = bundle.resolve()
    targets = [path.resolve() for path in (outputs, trace) if path is not None]
    if len(set(targets)) != len(targets):
        raise ExecutionError("output NPZ and trace must use distinct paths")
    if any(path == inputs.resolve() or path.is_relative_to(root) for path in targets):
        raise ExecutionError("output paths must be separate from the bundle and input NPZ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()

    phase = "paths"
    records = [] if args.trace else None
    try:
        check_paths(args.bundle, args.inputs, args.outputs, args.trace)
        phase = "load"
        package = load_package(args.bundle)
        phase = "inputs"
        inputs = load_inputs(args.inputs, package.manifest["inputs"])
        phase = "execute"
        outputs = execute(package, inputs, records)
        phase = "export"
        save_outputs(args.outputs, outputs)
        if args.trace:
            phase = "trace"
            write_trace(args.trace, records, "complete")
    except (OSError, ValueError, BadZipFile, EOFError, OverflowError, MemoryError) as error:
        print(f"virtual-npu {phase}: {error}", file=sys.stderr)
        if args.trace and phase not in ("paths", "trace"):
            try:
                write_trace(args.trace, records, phase, error)
            except OSError as trace_error:
                print(f"cannot write trace: {trace_error}", file=sys.stderr)
        return 1
    print(f"executed {package.program['instruction_count']} instructions, exported {len(outputs)} outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
