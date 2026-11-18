"""Import the Day 19 ONNX subset into generic Mini MLIR."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile

from google.protobuf.message import DecodeError
import onnx
from onnx import TensorProto, checker, shape_inference


class ModelError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, int]
    dtype: str

    @property
    def mlir_type(self) -> str:
        rows, columns = self.shape
        return f"tensor<{rows}x{columns}x{self.dtype}>"


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
    if graph.initializer or graph.sparse_initializer:
        names = [value.name for value in graph.initializer]
        names.extend(value.values.name for value in graph.sparse_initializer)
        raise ModelError("E_INITIALIZER_UNSUPPORTED", f"Day 19 does not import Initializers: {names!r}")
    if model.functions or model.training_info:
        raise ModelError("E_UNSUPPORTED_OP", "local functions and training graphs are unsupported")
    if not graph.node or not graph.output:
        raise ModelError("E_INVALID_GRAPH", "expected at least one MatMul node and one graph output")

    for index, node in enumerate(graph.node):
        where = node_label(index, node)
        if node.domain != "":
            raise ModelError("E_UNSUPPORTED_DOMAIN", f"{where}: domain {node.domain!r}")
        if node.op_type != "MatMul" or node.overload:
            raise ModelError("E_UNSUPPORTED_OP", f"{where}: only standard MatMul is supported")
        if node.attribute:
            raise ModelError("E_ATTRIBUTE_UNSUPPORTED", f"{where}: MatMul takes no attributes")
        if len(node.input) != 2 or len(node.output) != 1 or not all((*node.input, *node.output)):
            raise ModelError("E_INVALID_GRAPH", f"{where}: expected two named inputs and one named output")


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
    if len(dimensions) != 2:
        raise ModelError("E_UNSUPPORTED_RANK", f"{location}: expected rank 2, got {len(dimensions)}")
    if any(not dim.HasField("dim_value") for dim in dimensions):
        raise ModelError("E_DYNAMIC_SHAPE", f"{location}: expected static dimensions, got {tensor.shape}")
    rows, columns = (dim.dim_value for dim in dimensions)
    if rows <= 0 or columns <= 0:
        raise ModelError("E_SHAPE_MISMATCH", f"{location}: expected positive dimensions, got [{rows}, {columns}]")
    return TensorType((rows, columns), "f32")


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


def emit_matmul(
    index: int,
    node: onnx.NodeProto,
    type_map: dict[str, list[onnx.ValueInfoProto]],
    value_map: dict[str, ValueRef],
) -> str:
    where = node_label(index, node)
    lhs, rhs = (lookup_value(value_map, name, where) for name in node.input)
    output_name = node.output[0]
    if output_name in value_map:
        raise ModelError("E_INVALID_GRAPH", f"{where}: repeated definition of {output_name!r}")
    rows, lhs_k = lhs.type.shape
    rhs_k, columns = rhs.type.shape
    if lhs_k != rhs_k:
        raise ModelError("E_SHAPE_MISMATCH", f"{where}: K mismatch for {lhs.type.shape} and {rhs.type.shape}")
    result_type = get_value_type(type_map, output_name, where)
    if result_type.shape != (rows, columns):
        raise ModelError("E_SHAPE_MISMATCH", f"{where}: expected [{rows}, {columns}], got {result_type.shape}")
    result = ValueRef(f"%v{index}", result_type)
    value_map[output_name] = result
    return (
        f'    {result.ssa} = "mini.matmul"({lhs.ssa}, {rhs.ssa})'
        f" : ({lhs.type.mlir_type}, {rhs.type.mlir_type}) -> {result.type.mlir_type}"
    )


def import_model(model: onnx.ModelProto) -> str:
    check_subset(model)
    try:
        checker.check_model(model)
    except checker.ValidationError as error:
        raise ModelError("E_INVALID_GRAPH", f"ONNX Checker: {error}") from error
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

    operations = [emit_matmul(index, node, type_map, value_map) for index, node in enumerate(graph.node)]
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
