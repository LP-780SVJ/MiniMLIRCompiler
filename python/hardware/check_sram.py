#!/usr/bin/env python3
"""Estimate the aligned A, B and accumulator storage for a Virtual NPU tile."""

import argparse
import json
import sys
from pathlib import Path


class InputParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def checked_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def align_up(size, alignment):
    return ((size + alignment - 1) // alignment) * alignment


def check_tile(config, m, n, k, double_buffer_inputs=False):
    if not isinstance(config, dict):
        raise ValueError("profile must be a JSON object")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("expected schema_version 1")
    if config.get("target") != "virtual-npu-v1":
        raise ValueError("expected a virtual-npu-v1 profile")

    dims = dict(zip(("m", "n", "k"), (m, n, k)))
    for axis, size in dims.items():
        checked_int(size, f"tile.{axis}")

    compute = config["compute"]
    memory = config["memory"]
    if not isinstance(compute, dict) or not isinstance(memory, dict):
        raise ValueError("compute and memory must be JSON objects")
    if compute["input_dtype"] != "float32" or compute["accumulator_dtype"] != "float32":
        raise ValueError("input and accumulator dtypes must be float32")

    native = compute["native_matmul_tile"]
    if not isinstance(native, dict):
        raise ValueError("compute.native_matmul_tile must be a JSON object")
    native_limits = {
        axis: checked_int(native[axis], f"compute.native_matmul_tile.{axis}")
        for axis in dims
    }
    sram = checked_int(memory["sram_bytes"], "memory.sram_bytes")
    reserved = checked_int(memory["reserved_bytes"], "memory.reserved_bytes", minimum=0)
    alignment = checked_int(memory["alignment_bytes"], "memory.alignment_bytes")
    if reserved >= sram:
        raise ValueError("memory.reserved_bytes must be smaller than memory.sram_bytes")

    input_copies = 2 if double_buffer_inputs else 1
    buffers = [
        ("A", 4 * m * k, input_copies),
        ("B", 4 * k * n, input_copies),
        ("C accumulator", 4 * m * n, 1),
    ]
    available = sram - reserved
    required = 0
    print(f"Tile (M, N, K): {m} x {n} x {k}")
    for name, size, copies in buffers:
        allocated = align_up(size, alignment)
        required += allocated * copies
        print(f"{name}: {size} raw B, {allocated} aligned B x {copies}")
    fits = required <= available
    is_native = all(size <= native_limits[axis] for axis, size in dims.items())
    print(f"SRAM required: {required} B ({required / 1024:g} KiB)")
    print(f"SRAM available: {available} B ({available / 1024:g} KiB)")
    print(f"Capacity fits: {fits}")
    print(f"Within native MatMul limits: {is_native}")
    return fits


def main():
    parser = InputParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    for axis in ("m", "n", "k"):
        parser.add_argument(f"--{axis}", type=int, required=True)
    parser.add_argument("--double-buffer-inputs", action="store_true")
    args = parser.parse_args()

    try:
        with args.config.open(encoding="utf-8") as source:
            config = json.load(source)
        fits = check_tile(config, args.m, args.n, args.k, args.double_buffer_inputs)
    except KeyError as error:
        print(f"error: missing profile field {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0 if fits else 2


if __name__ == "__main__":
    sys.exit(main())
