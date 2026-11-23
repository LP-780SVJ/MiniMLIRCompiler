"""Bind caller tensors and fixed weights to the program's DRAM ABI."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile
import os

import numpy as np

from .matrix_unit import FP32_LE
from .memory import ByteMemory, ExecutionError


def load_inputs(path, ports):
    expected = {f"{port['name']}.npy": port["name"] for port in ports}
    with ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)) or set(members) != expected.keys():
            raise ExecutionError("input NPZ names do not match manifest inputs")
        inputs = {}
        for member, name in expected.items():
            with archive.open(member) as stream:
                inputs[name] = np.lib.format.read_array(stream, allow_pickle=False)
    return inputs


def bind_dram(package, inputs):
    manifest = package.manifest
    tensors = manifest["tensors"]
    ports = manifest["inputs"]
    if inputs.keys() != {port["name"] for port in ports}:
        raise ExecutionError("input names do not match manifest inputs")
    dram = ByteMemory(manifest["hardware"]["dram_bytes"], "DRAM")
    for port in ports:
        name, tensor_id = port["name"], port["tensor_id"]
        value = inputs[name]
        tensor = tensors[tensor_id]
        if (not isinstance(value, np.ndarray) or value.dtype.kind != "f"
                or value.dtype.itemsize != FP32_LE.itemsize):
            raise ExecutionError(f"input {name!r} must be an FP32 array")
        if value.shape != tuple(tensor["shape"]):
            raise ExecutionError(f"input {name!r} shape {value.shape} does not match {tensor['shape']}")
        dram.write(tensor["base"], value.astype(FP32_LE, copy=False).tobytes(order="C"))
    for entry in manifest["weights"]["entries"]:
        start = entry["file_offset"]
        dram.write(entry["dram_base"], package.weights[start:start + entry["size_bytes"]])
    return dram


def collect_outputs(manifest, dram):
    outputs = {}
    for port in manifest["outputs"]:
        tensor = manifest["tensors"][port["tensor_id"]]
        raw = dram.read(tensor["base"], tensor["size_bytes"])
        outputs[port["name"]] = (
            np.frombuffer(raw, dtype=FP32_LE).reshape(tensor["shape"]).astype(np.float32)
        )
    return outputs


@contextmanager
def atomic_file(path):
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_outputs(path, outputs):
    # Use explicit archive member names, including names such as "file"
    # that collide with keyword arguments of numpy.savez.
    with atomic_file(path) as stream, ZipFile(stream, "w", allowZip64=True) as archive:
        for name, output in outputs.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, output, allow_pickle=False)
