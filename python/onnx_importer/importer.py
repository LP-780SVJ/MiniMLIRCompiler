"""Import the supported ONNX subset into generic Mini MLIR."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from math import prod

from google.protobuf.message import DecodeError
import numpy as np
import onnx
from onnx import TensorProto, checker, numpy_helper, shape_inference


OP_ARITY = {"MatMul": 2, "Add": 2, "Mul": 2, "Relu": 1}
MAX_INITIALIZER_BYTES = 64 * 1024


class ModelError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: str

    @property
    def mlir_type(self) -> str:
        return "tensor<" + "x".join([*map(str, self.shape), self.dtype]) + ">"


@dataclass(frozen=True)
class ValueRef:
    ssa: str
    type: TensorType


def node_label(index: int, node: onnx.NodeProto) -> str:
    return f"node {index} ({node.op_type} {node.name!r}, inputs {list(node.input)!r})"


def check_subset(model: onnx.ModelProto) -> None:
    for opset in model.opset_import:
        if opset.domain != "":
            raise ModelError("E_UNSUPPORTED_DOMAIN", f"opset domain {opset.domain!r}")
    versions = [opset.version for opset in model.opset_import]
    if versions != [18]:
        raise ModelError("E_UNSUPPORTED_OPSET", f"expected one standard opset 18, got {versions}")

    if not model.HasField("graph"):
        raise ModelError("E_INVALID_GRAPH", "model is missing a graph")
    graph = model.graph
    if graph.sparse_initializer:
        raise ModelError("E_INITIALIZER_UNSUPPORTED", "Sparse Initializer is unsupported")
    input_names = {info.name for info in graph.input}
    for tensor in graph.initializer:
        if tensor.name in input_names:
            raise ModelError("E_INITIALIZER_UNSUPPORTED", f"Initializer {tensor.name!r} is an overridable input")
        if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
            raise ModelError("E_INITIALIZER_UNSUPPORTED", f"Initializer {tensor.name!r}: External Data")
    if model.functions or model.training_info:
        raise ModelError("E_UNSUPPORTED_OP", "local functions and training graphs are unsupported")
    if not graph.node or not graph.output:
        raise ModelError("E_INVALID_GRAPH", "expected at least one node and one graph output")

    for index, node in enumerate(graph.node):
        where = node_label(index, node)
        if node.domain != "":
            raise ModelError("E_UNSUPPORTED_DOMAIN", f"{where}: domain {node.domain!r}")
        if node.op_type not in OP_ARITY or node.overload:
            raise ModelError("E_UNSUPPORTED_OP", f"{where}: unsupported operator")
        if node.attribute:
            raise ModelError("E_ATTRIBUTE_UNSUPPORTED", f"{where}: {node.op_type} takes no attributes")
        arity = OP_ARITY[node.op_type]
        if len(node.input) != arity or len(node.output) != 1 or not all((*node.input, *node.output)):
            raise ModelError("E_INVALID_GRAPH", f"{where}: expected {arity} named inputs and one named output")


def read_tensor_type(info: onnx.ValueInfoProto, where: str) -> TensorType:
    location = f"{where}: tensor {info.name!r}"
    if not info.type.HasField("tensor_type"):
        raise ModelError("E_UNSUPPORTED_DTYPE", f"{location}: expected a Tensor type")
    tensor = info.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT:
        raise ModelError("E_UNSUPPORTED_DTYPE", f"{location}: expected FLOAT (FP32), got ONNX type {tensor.elem_type}")
    if not tensor.HasField("shape"):
        raise ModelError("E_DYNAMIC_SHAPE", f"{location}: missing shape")
    dimensions = tensor.shape.dim
    if any(not dim.HasField("dim_value") for dim in dimensions):
        raise ModelError("E_DYNAMIC_SHAPE", f"{location}: expected static dimensions, got {tensor.shape}")
    shape = tuple(dim.dim_value for dim in dimensions)
    if any(dim <= 0 for dim in shape):
        raise ModelError("E_SHAPE_MISMATCH", f"{location}: expected positive dimensions, got {shape}")
    return TensorType(shape, "f32")


def collect_type_info(graph: onnx.GraphProto) -> dict[str, list[onnx.ValueInfoProto]]:
    types: dict[str, list[onnx.ValueInfoProto]] = {}
    for info in (*graph.input, *graph.value_info, *graph.output):
        types.setdefault(info.name, []).append(info)
    return types


def get_value_type(type_map: dict[str, list[onnx.ValueInfoProto]], name: str, where: str) -> TensorType:
    annotations = type_map.get(name)
    if not annotations:
        raise ModelError("E_DYNAMIC_SHAPE", f"{where}: missing type for {name!r}")
    types = {read_tensor_type(info, where) for info in annotations}
    if len(types) != 1:
        raise ModelError("E_SHAPE_MISMATCH", f"{where}: conflicting types for {name!r}")
    return types.pop()


def lookup_value(value_map: dict[str, ValueRef], name: str, where: str) -> ValueRef:
    if name not in value_map:
        raise ModelError("E_INVALID_GRAPH", f"{where}: use of undefined value {name!r}")
    return value_map[name]


def read_initializer(tensor: TensorProto) -> np.ndarray:
    where = f"Initializer {tensor.name!r}"
    if tensor.data_type != TensorProto.FLOAT:
        raise ModelError("E_UNSUPPORTED_DTYPE", f"{where}: expected FP32")
    shape = tuple(tensor.dims)
    if any(dim <= 0 for dim in shape):
        raise ModelError("E_SHAPE_MISMATCH", f"{where}: expected positive dimensions, got {shape}")
    count = prod(shape)
    byte_count = count * np.dtype(np.float32).itemsize
    if byte_count > MAX_INITIALIZER_BYTES:
        raise ModelError("E_INITIALIZER_UNSUPPORTED", f"{where}: exceeds {MAX_INITIALIZER_BYTES} bytes")
    if tensor.HasField("raw_data"):
        valid_size = len(tensor.raw_data) == byte_count and not tensor.float_data
    else:
        valid_size = len(tensor.float_data) == count
    if not valid_size:
        raise ModelError("E_INITIALIZER_UNSUPPORTED", f"{where}: payload does not match {count} FP32 elements")
    try:
        array = numpy_helper.to_array(tensor)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ModelError("E_INITIALIZER_UNSUPPORTED", f"{where}: {error}") from error
    if array.dtype != np.float32:
        raise ModelError("E_UNSUPPORTED_DTYPE", f"{where}: decoded {array.dtype}, expected FP32")
    if array.shape != shape or array.size != count:
        raise ModelError("E_SHAPE_MISMATCH", f"{where}: decoded data does not match {shape}")
    if not np.isfinite(array).all():
        raise ModelError("E_INITIALIZER_UNSUPPORTED", f"{where}: expected finite weights")
    return array


def format_dense(bits: int | list) -> str:
    if isinstance(bits, list):
        return "[" + ", ".join(map(format_dense, bits)) + "]"
    return f"0x{bits:08X}"


def emit_node(
    index: int,
    node: onnx.NodeProto,
    type_map: dict[str, list[onnx.ValueInfoProto]],
    value_map: dict[str, ValueRef],
) -> str:
    where = node_label(index, node)
    operands = [lookup_value(value_map, name, where) for name in node.input]
    output_name = node.output[0]
    if output_name in value_map:
        raise ModelError("E_INVALID_GRAPH", f"{where}: repeated definition of {output_name!r}")
    result_type = get_value_type(type_map, output_name, where)
    expected_shape = operands[0].type.shape
    if node.op_type == "MatMul":
        lhs_shape, rhs_shape = (value.type.shape for value in operands)
        if any(len(shape) != 2 for shape in (lhs_shape, rhs_shape, result_type.shape)):
            raise ModelError("E_UNSUPPORTED_RANK", f"{where}: expected rank 2, got {lhs_shape} and {rhs_shape}")
        rows, lhs_k = lhs_shape
        rhs_k, columns = rhs_shape
        if lhs_k != rhs_k:
            raise ModelError("E_SHAPE_MISMATCH", f"{where}: K mismatch for {lhs_shape} and {rhs_shape}")
        expected_shape = (rows, columns)
    elif len(operands) == 2 and operands[0].type != operands[1].type:
        raise ModelError("E_BROADCAST_UNSUPPORTED", f"{where}: requires identical shapes")
    if result_type.shape != expected_shape:
        raise ModelError("E_SHAPE_MISMATCH", f"{where}: expected {expected_shape}, got {result_type.shape}")
    result = ValueRef(f"%v{index}", result_type)
    value_map[output_name] = result
    ssa_operands = ", ".join(value.ssa for value in operands)
    operand_types = ", ".join(value.type.mlir_type for value in operands)
    return (
        f'    {result.ssa} = "mini.{node.op_type.lower()}"({ssa_operands})'
        f" : ({operand_types}) -> {result.type.mlir_type}"
    )


def import_model(model: onnx.ModelProto) -> str:
    check_subset(model)
    try:
        checker.check_model(model)
    except checker.ValidationError as error:
        raise ModelError("E_INVALID_GRAPH", f"ONNX Checker: {error}") from error
    constants = [(tensor.name, read_initializer(tensor)) for tensor in model.graph.initializer]
    try:
        inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    except shape_inference.InferenceError as error:
        raise ModelError("E_SHAPE_MISMATCH", f"ONNX Shape Inference: {error}") from error

    graph = model.graph
    type_map = collect_type_info(graph)
    for name, inferred_types in collect_type_info(inferred.graph).items():
        type_map.setdefault(name, []).extend(inferred_types)

    value_map: dict[str, ValueRef] = {}
    arguments = []
    for index, info in enumerate(graph.input):
        if info.name in value_map:
            raise ModelError("E_INVALID_GRAPH", f"duplicate graph input {info.name!r}")
        value = ValueRef(f"%arg{index}", get_value_type(type_map, info.name, "graph input"))
        value_map[info.name] = value
        arguments.append(f"{value.ssa}: {value.type.mlir_type}")

    operations = []
    for index, (name, array) in enumerate(constants):
        if name in value_map:
            raise ModelError("E_INVALID_GRAPH", f"duplicate definition of {name!r}")
        value_type = TensorType(array.shape, "f32")
        if name in type_map and get_value_type(type_map, name, "initializer") != value_type:
            raise ModelError("E_SHAPE_MISMATCH", f"Initializer {name!r}: conflicting type annotation")
        value = ValueRef(f"%c{index}", value_type)
        value_map[name] = value
        payload = format_dense(array.view(np.uint32).tolist())
        operations.append(f"    {value.ssa} = arith.constant dense<{payload}> : {value.type.mlir_type}")

    operations.extend(emit_node(index, node, type_map, value_map) for index, node in enumerate(graph.node))
    outputs = []
    for info in graph.output:
        value = lookup_value(value_map, info.name, "graph output")
        if value.type != get_value_type(type_map, info.name, "graph output"):
            raise ModelError("E_SHAPE_MISMATCH", f"graph output {info.name!r} has a conflicting type")
        outputs.append(value)

    output_types = ", ".join(value.type.mlir_type for value in outputs)
    function_result = output_types if len(outputs) == 1 else f"({output_types})"
    returned = ", ".join(value.ssa for value in outputs)
    return "\n".join([
        "module {",
        f"  func.func @main({', '.join(arguments)}) -> {function_result} {{",
        *operations,
        f"    func.return {returned} : {output_types}",
        "  }",
        "}",
        "",
    ])


def publish_ir(output: Path, ir: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(ir)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="binary ONNX ModelProto")
    parser.add_argument("--output", required=True, type=Path, help="destination Mini MLIR file")
    args = parser.parse_args()
    try:
        if args.input.resolve() == args.output.resolve():
            raise ModelError("E_OUTPUT", "input and output must be different paths")
        model = onnx.load(args.input, format="protobuf", load_external_data=False)
    except (DecodeError, OSError) as error:
        print(f"{args.input}: E_MODEL_READ: {error}", file=sys.stderr)
        return 1
    except ModelError as error:
        print(f"{args.input}: {error}", file=sys.stderr)
        return 1

    try:
        ir = import_model(model)
        publish_ir(args.output, ir)
    except ModelError as error:
        print(f"{args.input}: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"{args.output}: E_OUTPUT: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
