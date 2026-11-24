#!/usr/bin/env python3
"""Generate reproducible ONNX fixtures and optional CPU reference outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

if __package__:
    from .e2e_metadata import CPU_PROVIDER, build_metadata, run_reference
else:
    from e2e_metadata import CPU_PROVIDER, build_metadata, run_reference


FIXTURE = "matmul_relu"
OPSET = 18
IR_VERSION = 9
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "test" / "e2e"


def make_fixture():
    inputs = np.array([[-1.0, 2.0, 0.5], [3.0, -2.0, 1.0]], dtype=np.float32)
    weights = np.array(
        [[1.0, 0.0, -1.0, 2.0], [0.0, 1.0, 1.0, -1.0], [2.0, -1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "W"], ["H"], name="matmul"),
            helper.make_node("Relu", ["H"], ["Y"], name="relu"),
        ],
        FIXTURE,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
        initializer=[numpy_helper.from_array(weights, name="W")],
    )
    model = helper.make_model(
        graph,
        producer_name="MiniNPUCompiler",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model, {"X": inputs}


def make_weighted_fixture(name, inputs, weights, post_ops=()):
    output_shape = [inputs.shape[0], weights.shape[1]]
    initializers = [numpy_helper.from_array(weights, name="W")]
    operators = ["MatMul", *post_ops]
    nodes = []
    previous = "X"
    for index, op in enumerate(operators):
        output = "Y" if index == len(operators) - 1 else f"H{index}"
        operands = [previous]
        if op == "MatMul":
            operands.append("W")
        elif op == "Add":
            bias = np.arange(np.prod(output_shape), dtype=np.float32).reshape(output_shape) / 4
            initializers.append(numpy_helper.from_array(bias, name="B"))
            operands.append("B")
        nodes.append(helper.make_node(op, operands, [output], name=f"{op.lower()}_{index}"))
        previous = output
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, inputs.shape)],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
        initializer=initializers,
    )
    return helper.make_model(
        graph, producer_name="MiniNPUCompiler", ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )


def fixtures():
    relu, feeds = make_fixture()
    yield FIXTURE, relu, feeds, None
    weights = numpy_helper.to_array(relu.graph.initializer[0])
    for name, post_ops in [("weighted_matmul", ()), ("matmul_add_relu", ("Add", "Relu"))]:
        yield name, make_weighted_fixture(name, feeds["X"], weights, post_ops), feeds, None

    seed = 2026
    rng = np.random.default_rng(seed)
    for name in ("matmul", "matmul_chain"):
        model = onnx.load(DEFAULT_OUTPUT / "models" / f"{name}.onnx")
        feeds = {
            name: rng.standard_normal(shape, dtype=np.float32)
            for name, shape in shapes(model.graph.input).items()
        }
        yield name, model, feeds, seed
    inputs = rng.standard_normal((64, 96), dtype=np.float32)
    weights = rng.standard_normal((96, 64), dtype=np.float32)
    name = "matmul_multitile"
    yield name, make_weighted_fixture(name, inputs, weights), {"X": inputs}, seed


def shapes(value_infos):
    return {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in value_infos
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--with-reference", action="store_true", help="also run ONNX Runtime CPU"
    )
    parser.add_argument("--all", action="store_true", help="generate the complete Day 20 suite")
    args = parser.parse_args()
    selected = fixtures() if args.all else (next(fixtures()),)
    for name, model, feeds, seed in selected:
        write_fixture(name, model, feeds, seed, args.output_dir, args.with_reference)


def write_fixture(name, model, feeds, seed, output_dir, with_reference):
    onnx.checker.check_model(model, full_check=True)
    model_bytes = model.SerializeToString(deterministic=True)
    graph = model.graph
    input_shapes = shapes(graph.input)
    output_shapes = shapes(graph.output)
    inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    intermediate_shapes = shapes(inferred.graph.value_info)
    model_file = Path("models") / f"{name}.onnx"
    input_file = Path("inputs") / f"{name}.npz"
    reference_file = Path("reference") / f"{name}.npz"
    metadata_file = Path("models") / f"{name}.metadata.json"

    reference = None
    ort_version = None
    if with_reference:
        try:
            reference, ort_version = run_reference(
                model_bytes, feeds, input_shapes, output_shapes
            )
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        for tensor_name, array in reference.items():
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite reference output for {tensor_name}")

    for directory in ("models", "inputs"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    (output_dir / model_file).write_bytes(model_bytes)
    input_path = output_dir / input_file
    np.savez_compressed(input_path, **feeds)

    metadata_path = output_dir / metadata_file
    metadata = {
        **build_metadata(model, model_bytes, input_path.read_bytes(), "virtual-npu-v1"),
        "seed": seed,
        "intermediates": intermediate_shapes,
        "rtol": 1e-4,
        "atol": 1e-5,
        "model_file": model_file.as_posix(),
        "inputs_file": input_file.as_posix(),
    }
    if reference is not None:
        (output_dir / reference_file.parent).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / reference_file, **reference)
        metadata["reference"] = {
            "file": reference_file.as_posix(),
            "provider": CPU_PROVIDER,
            "onnxruntime_version": ort_version,
        }
    elif metadata_path.exists():
        old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        # Preserve provenance only when a model-only replay uses the same fixture.
        if (
            old_metadata.get("model_sha256") == metadata["model_sha256"]
            and old_metadata.get("inputs_sha256") == metadata["inputs_sha256"]
            and "reference" in old_metadata
            and (output_dir / reference_file).is_file()
        ):
            metadata["reference"] = old_metadata["reference"]
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"{name}: ONNX Checker passed (IR {IR_VERSION}, Opset {OPSET})")
    print(f"Inputs: {input_shapes}; initializers: {[w.name for w in graph.initializer]}")
    print(f"Nodes: {[node.op_type for node in graph.node]}")
    print(f"Inferred intermediate shapes: {intermediate_shapes}; outputs: {output_shapes}")
    if reference is not None:
        print(f"ONNX Runtime {ort_version} ({CPU_PROVIDER}):")
        for tensor_name, array in reference.items():
            values = array.tolist() if array.size <= 16 else f"min={array.min():.6g}, max={array.max():.6g}"
            print(f"{tensor_name} ({array.dtype}, {array.shape}): {values}")


if __name__ == "__main__":
    main()
