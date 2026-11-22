"""Load and validate a Virtual NPU v1 package using only the standard library."""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


TARGET = "virtual-npu-v1"
CONTRACT = "serial-mnk-single-buffer-v1"
NATIVE_TILE = [32, 32, 32]
MAX_INTEGER = (1 << 53) - 1
DRAM_ALIGNMENT = WEIGHT_ALIGNMENT = 64
PACKAGE_FILES = {"manifest.json", "program.json", "weights.bin"}
COMMON_FIELDS = (
    "opcode command_id sequence_index matmul_index local_event_index tile_index dependencies"
)
DMA_FIELDS = "buffer_id tensor_id rows row_bytes src_stride_bytes dst_stride_bytes"
OPCODE_FIELDS = {
    "ZERO": "buffer_id dst_sram rows cols",
    "DMA_LOAD_2D": f"{DMA_FIELDS} src_dram dst_sram",
    "MATMUL_TILE": "a_buffer_id b_buffer_id c_buffer_id a_sram b_sram c_sram m n k accumulate",
    "DMA_STORE_2D": f"{DMA_FIELDS} src_sram dst_dram",
}


class PackageError(ValueError):
    """The package cannot be loaded under the v1 ABI."""


@dataclass(frozen=True)
class ProgramPackage:
    manifest: dict
    program: dict
    weights: bytes


def require(condition, message):
    if not condition:
        raise PackageError(message)


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def object_fields(value, fields, where):
    require(type(value) is dict and value.keys() == set(fields.split()),
            f"{where}: missing or unexpected fields")


def array(value, where, length=None):
    require(type(value) is list and (length is None or len(value) == length),
            f"{where}: unexpected array or length")
    return value


def integer(value, where, minimum=0, maximum=MAX_INTEGER):
    require(type(value) is int and minimum <= value <= maximum,
            f"{where}: integer outside [{minimum}, {maximum}]")
    return value


def version(value, where):
    require(type(value) is int and value == 1, f"{where}: unsupported schema version")


def sha256_value(value, where):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value),
            f"{where}: expected lowercase SHA-256")


def parse_json(data, name):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{name}: duplicate field {key!r}")
            result[key] = value
        return result

    def reject_number(token):
        raise PackageError(f"{name}: expected an integer, got {token!r}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_float=reject_number, parse_constant=reject_number)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise PackageError(f"{name}: invalid JSON: {error}") from error


def validate_manifest(manifest):
    object_fields(
        manifest,
        "schema_version program_schema_version target target_profile_sha256 entry dtype "
        "byte_order layout schedule_contract hardware tensors inputs outputs weights program",
        "manifest",
    )
    version(manifest["schema_version"], "manifest")
    version(manifest["program_schema_version"], "manifest program")
    sha256_value(manifest["target_profile_sha256"], "target_profile_sha256")
    for key, value in (
        ("target", TARGET), ("entry", "main"), ("dtype", "float32"),
        ("byte_order", "little"), ("layout", "row_major"),
        ("schedule_contract", CONTRACT),
    ):
        require(manifest[key] == value, f"manifest: incompatible {key}")

    hardware = manifest["hardware"]
    object_fields(hardware, "native_tile sram_bytes reserved_bytes sram_alignment_bytes "
                  "dram_bytes dram_alignment_bytes", "hardware")
    for axis in array(hardware["native_tile"], "native_tile", 3):
        integer(axis, "native tile dimension", 1)
    require(hardware["native_tile"] == NATIVE_TILE, "unsupported native tile")
    sram_bytes = integer(hardware["sram_bytes"], "SRAM capacity", 1)
    reserved = integer(hardware["reserved_bytes"], "SRAM reserved bytes", 0, sram_bytes - 1)
    alignment = integer(hardware["sram_alignment_bytes"], "SRAM alignment", 1)
    require(alignment & (alignment - 1) == 0 and align(reserved, alignment) < sram_bytes,
            "invalid SRAM alignment or reserved range")
    dram_bytes = integer(hardware["dram_bytes"], "DRAM capacity", 1)
    integer(hardware["dram_alignment_bytes"], "DRAM alignment", DRAM_ALIGNMENT, DRAM_ALIGNMENT)

    tensors = {}
    end = 0
    for tensor_id, tensor in enumerate(array(manifest["tensors"], "tensors")):
        where = f"tensor {tensor_id}"
        object_fields(tensor, "id kind base shape row_stride_bytes size_bytes", where)
        integer(tensor["id"], f"{where} ID", tensor_id, tensor_id)
        require(tensor["kind"] in ("input", "constant", "intermediate", "output"),
                f"{where}: unknown kind")
        rows, cols = array(tensor["shape"], f"{where} shape", 2)
        integer(rows, f"{where} rows", 1)
        integer(cols, f"{where} columns", 1)
        stride = integer(tensor["row_stride_bytes"], f"{where} stride", 1)
        size = integer(tensor["size_bytes"], f"{where} size", 1)
        require(stride == 4 * cols and size == rows * stride,
                f"{where}: shape/stride/size mismatch")
        base = integer(tensor["base"], f"{where} DRAM base", 0, dram_bytes)
        require(base == align(end, DRAM_ALIGNMENT) and base + size <= dram_bytes,
                f"{where}: invalid or overlapping DRAM range")
        end = base + size
        tensors[tensor_id] = tensor
    require(dram_bytes == end, "DRAM capacity does not match tensor layout")

    for direction in ("inputs", "outputs"):
        names = set()
        port_ids = []
        for port in array(manifest[direction], direction):
            object_fields(port, "name tensor_id", direction)
            name = port["name"]
            require(type(name) is str and name and "\0" not in name and
                    all(not 0xD800 <= ord(char) <= 0xDFFF for char in name) and name not in names,
                    f"{direction}: invalid or duplicate name")
            names.add(name)
            tensor_id = integer(port["tensor_id"], f"{direction} tensor ID")
            require(tensor_id in tensors, f"{direction}: unknown tensor ID")
            port_ids.append(tensor_id)
        if direction == "inputs":
            expected = [t["id"] for t in tensors.values() if t["kind"] == "input"]
            require(port_ids == expected, "inputs: missing or misordered input descriptor")
        else:
            returned = set(port_ids)
            require(all(t["id"] in returned for t in tensors.values() if t["kind"] == "output") and
                    all(tensors[t]["kind"] != "intermediate" for t in returned),
                    "outputs: missing descriptor or intermediate tensor")
    return tensors


def validate_file(record, expected_name, contents):
    require(record["file"] == expected_name, f"unsafe package filename for {expected_name}")
    size = integer(record["size_bytes"], f"{expected_name} size")
    require(size == len(contents), f"{expected_name}: file length mismatch")
    sha256_value(record["sha256"], f"{expected_name} sha256")
    require(hashlib.sha256(contents).hexdigest() == record["sha256"],
            f"{expected_name}: SHA-256 mismatch")


def validate_weights(manifest, tensors, contents):
    weights = manifest["weights"]
    object_fields(weights, "file size_bytes sha256 entries", "weights")
    validate_file(weights, "weights.bin", contents)
    constants = [tensor for tensor in tensors.values() if tensor["kind"] == "constant"]
    entries = array(weights["entries"], "weight entries", len(constants))
    end = 0
    for entry, tensor in zip(entries, constants):
        object_fields(entry, "tensor_id file_offset size_bytes dram_base", "weight entry")
        for key, expected in (
            ("tensor_id", tensor["id"]), ("size_bytes", tensor["size_bytes"]),
            ("dram_base", tensor["base"]),
        ):
            require(integer(entry[key], f"weight {key}") == expected,
                    f"weight {key} does not match tensor descriptor")
        offset = integer(entry["file_offset"], "weight file offset")
        require(offset == align(end, WEIGHT_ALIGNMENT) and
                offset + entry["size_bytes"] <= len(contents),
                "weight entry: invalid offset or truncated tensor")
        require(not any(contents[end:offset]), "nonzero weight padding")
        end = offset + entry["size_bytes"]
    require(end == len(contents), "unexpected trailing weight data")


def validate_payload(cmd, tensors, hardware, where):
    m_max, n_max, k_max = hardware["native_tile"]

    def sram_range(base, size):
        alignment = hardware["sram_alignment_bytes"]
        require(base >= hardware["reserved_bytes"] and base % alignment == 0 and
                base + size <= hardware["sram_bytes"], f"{where}: invalid SRAM range")
        return base, base + size

    opcode = cmd["opcode"]
    if opcode == "ZERO":
        rows = integer(cmd["rows"], f"{where} rows", 1, m_max)
        cols = integer(cmd["cols"], f"{where} cols", 1, n_max)
        sram_range(cmd["dst_sram"], rows * cols * 4)
    elif opcode == "MATMUL_TILE":
        m = integer(cmd["m"], f"{where} m", 1, m_max)
        n = integer(cmd["n"], f"{where} n", 1, n_max)
        k = integer(cmd["k"], f"{where} k", 1, k_max)
        require(cmd["accumulate"] is True, f"{where}: expected accumulate=true")
        areas = sorted([sram_range(cmd["a_sram"], m * k * 4),
                        sram_range(cmd["b_sram"], k * n * 4),
                        sram_range(cmd["c_sram"], m * n * 4)])
        require(areas[0][1] <= areas[1][0] and areas[1][1] <= areas[2][0],
                f"{where}: overlapping SRAM operands")
        require(len({cmd["a_buffer_id"], cmd["b_buffer_id"], cmd["c_buffer_id"]}) == 3,
                f"{where}: overlapping buffer IDs")
    else:
        load = opcode == "DMA_LOAD_2D"
        rows = integer(cmd["rows"], f"{where} rows", 1, max(m_max, k_max) if load else m_max)
        row_bytes = integer(cmd["row_bytes"], f"{where} row bytes", 4,
                            4 * (max(k_max, n_max) if load else n_max))
        require(row_bytes % 4 == 0, f"{where}: DMA width must be in whole FP32 elements")
        buffer = cmd["dst_sram"] if load else cmd["src_sram"]
        dram = cmd["src_dram"] if load else cmd["dst_dram"]
        sram_stride = cmd["dst_stride_bytes"] if load else cmd["src_stride_bytes"]
        dram_stride = cmd["src_stride_bytes"] if load else cmd["dst_stride_bytes"]
        require(sram_stride == row_bytes, f"{where}: invalid DMA SRAM stride")
        sram_range(buffer, rows * row_bytes)
        tensor = tensors.get(cmd["tensor_id"])
        require(tensor is not None, f"{where}: unknown DMA tensor ID")
        require(load or tensor["kind"] in ("intermediate", "output"),
                f"{where}: STORE destination is read-only")
        require(dram_stride == tensor["row_stride_bytes"] and dram_stride >= row_bytes,
                f"{where}: invalid DMA DRAM stride")
        offset = dram - tensor["base"]
        require(offset >= 0 and offset % 4 == 0 and
                offset % dram_stride + row_bytes <= dram_stride and
                offset + (rows - 1) * dram_stride + row_bytes <= tensor["size_bytes"],
                f"{where}: invalid DMA DRAM range or row crossing")


def validate_program(program, manifest, tensors):
    object_fields(program, "schema_version target schedule_contract instruction_count instructions",
                  "program")
    version(program["schema_version"], "program")
    require(program["target"] == TARGET and program["schedule_contract"] == CONTRACT,
            "program: incompatible target or schedule contract")
    count = integer(program["instruction_count"], "instruction count", 1)
    integer(manifest["program"]["instruction_count"], "manifest instruction count", count, count)
    instructions = array(program["instructions"], "instructions", count)
    group, event, tile = -1, -1, -1
    for sequence, cmd in enumerate(instructions):
        where = f"instruction {sequence}"
        opcode = cmd.get("opcode") if type(cmd) is dict else None
        require(type(opcode) is str and opcode in OPCODE_FIELDS, f"{where}: unknown opcode")
        object_fields(cmd, f"{COMMON_FIELDS} {OPCODE_FIELDS[opcode]}", where)
        for key in cmd.keys() - {"opcode", "dependencies", "accumulate"}:
            integer(cmd[key], f"{where} {key}")
        require(cmd["sequence_index"] == sequence, f"{where}: non-contiguous sequence")
        require(cmd["command_id"] == sequence, f"{where}: duplicate or inconsistent command ID")
        current_group = cmd["matmul_index"]
        if current_group == group + 1:
            group, event, tile = current_group, -1, -1
        require(current_group == group and cmd["local_event_index"] == event + 1,
                f"{where}: invalid MatMul/local event origin")
        require(cmd["tile_index"] in (tile, tile + 1), f"{where}: invalid tile index")
        event, tile = cmd["local_event_index"], cmd["tile_index"]
        previous = -1
        for dep in array(cmd["dependencies"], f"{where} dependencies"):
            previous = integer(dep, f"{where} dependency ID", previous + 1, sequence - 1)
        validate_payload(cmd, tensors, manifest["hardware"], where)


def load_package(path):
    """Return checked manifest/program dictionaries and the original weight bytes."""
    try:
        root = Path(path)
        require(not root.is_symlink() and root.is_dir(), "expected a package directory")
        require({file.name for file in root.iterdir()} == PACKAGE_FILES,
                "expected exactly manifest.json, program.json and weights.bin")
        data = {}
        for name in sorted(PACKAGE_FILES):
            file = root / name
            require(not file.is_symlink() and file.is_file(), f"unsafe package file: {name}")
            data[name] = file.read_bytes()

        manifest = parse_json(data["manifest.json"], "manifest.json")
        tensors = validate_manifest(manifest)
        validate_weights(manifest, tensors, data["weights.bin"])
        object_fields(manifest["program"], "file size_bytes sha256 instruction_count",
                      "program record")
        validate_file(manifest["program"], "program.json", data["program.json"])
        program = parse_json(data["program.json"], "program.json")
        validate_program(program, manifest, tensors)
        return ProgramPackage(manifest, program, data["weights.bin"])
    except (OSError, ValueError) as error:
        if isinstance(error, PackageError):
            raise
        raise PackageError(str(error)) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        package = load_package(args.package)
    except PackageError as error:
        print(f"invalid package: {error}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"valid package: {package.program['instruction_count']} instructions, "
              f"{len(package.weights)} weight bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
