#!/usr/bin/env python3
"""Generate the Day 18 weighted MatMul → ReLU ONNX fixture."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


FIXTURE = "matmul_relu"
OPSET = 18
IR_VERSION = 9
CPU_PROVIDER = "CPUExecutionProvider"
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


def shapes(value_infos):
    return {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in value_infos
    }


def run_reference(model_bytes, feeds, input_shapes, output_shapes):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        if error.name != "onnxruntime":
            raise
        raise SystemExit("--with-reference requires the onnxruntime package") from error

    session = ort.InferenceSession(model_bytes, providers=[CPU_PROVIDER])
    if session.get_providers() != [CPU_PROVIDER]:
        raise RuntimeError(f"unexpected reference providers: {session.get_providers()}")
    for values, expected_shapes, arrays in (
        (session.get_inputs(), input_shapes, feeds),
        (session.get_outputs(), output_shapes, None),
    ):
        if [value.name for value in values] != list(expected_shapes):
            raise ValueError("ONNX Runtime tensor names or order differ from the model")
        for value in values:
            expected = tuple(expected_shapes[value.name])
            if value.type != "tensor(float)" or tuple(value.shape) != expected:
                raise ValueError(f"unexpected ONNX Runtime metadata for {value.name}")
            if arrays is not None:
                array = arrays[value.name]
                if array.dtype != np.float32 or array.shape != expected:
                    raise ValueError(f"invalid fixture input {value.name}")

    output_names = list(output_shapes)
    outputs = dict(zip(output_names, session.run(output_names, feeds), strict=True))
    for name, array in outputs.items():
        if array.dtype != np.float32 or array.shape != tuple(output_shapes[name]):
            raise ValueError(f"invalid reference output type or shape for {name}")
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite reference output for {name}")
    return outputs, ort.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--with-reference", action="store_true", help="also run ONNX Runtime CPU"
    )
    args = parser.parse_args()

    model, feeds = make_fixture()
    model_bytes = model.SerializeToString(deterministic=True)
    graph = model.graph
    input_shapes = shapes(graph.input)
    output_shapes = shapes(graph.output)
    intermediate_shapes = shapes(graph.value_info)
    output_dir = args.output_dir
    model_file = Path("models") / f"{FIXTURE}.onnx"
    input_file = Path("inputs") / f"{FIXTURE}.npz"
    reference_file = Path("reference") / f"{FIXTURE}.npz"
    metadata_file = Path("models") / f"{FIXTURE}.metadata.json"

    reference = None
    ort_version = None
    if args.with_reference:
        reference, ort_version = run_reference(
            model_bytes, feeds, input_shapes, output_shapes
        )

    for directory in ("models", "inputs"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    (output_dir / model_file).write_bytes(model_bytes)
    input_path = output_dir / input_file
    np.savez_compressed(input_path, **feeds)

    metadata_path = output_dir / metadata_file
    metadata = {
        "schema_version": 1,
        "opset": OPSET,
        "onnx_ir_version": IR_VERSION,
        "onnx_version": onnx.__version__,
        "numpy_version": np.__version__,
        "dtype": "float32",
        "seed": None,
        "input_names": list(input_shapes),
        "output_names": list(output_shapes),
        "inputs": input_shapes,
        "outputs": output_shapes,
        "intermediates": intermediate_shapes,
        "target": "virtual-npu-v1",
        "rtol": 1e-4,
        "atol": 1e-5,
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "inputs_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
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

    print(f"ONNX Checker: passed (IR {IR_VERSION}, Opset {OPSET})")
    print(f"Inputs: {input_shapes}; initializers: {[w.name for w in graph.initializer]}")
    print(f"Nodes: {[node.op_type for node in graph.node]}")
    print(f"Inferred intermediate shapes: {intermediate_shapes}; outputs: {output_shapes}")
    if reference is not None:
        print(f"ONNX Runtime {ort_version} ({CPU_PROVIDER}):")
        for name, array in reference.items():
            print(f"{name} ({array.dtype}): {array.tolist()}")


if __name__ == "__main__":
    main()
