"""Model interface shared by fixture generation and the E2E driver."""

from hashlib import sha256

import numpy as np
import onnx
from onnx import TensorProto


CPU_PROVIDER = "CPUExecutionProvider"


def port_shapes(values):
    shapes = {}
    for value in values:
        tensor = value.type.tensor_type
        dims = tensor.shape.dim
        if (tensor.elem_type != TensorProto.FLOAT or len(dims) != 2 or
                any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in dims)):
            raise ValueError(f"port {value.name!r} must be a static rank-2 FP32 tensor")
        if not value.name or value.name in shapes:
            raise ValueError(f"invalid or repeated ONNX port name {value.name!r}")
        shapes[value.name] = [dim.dim_value for dim in dims]
    return shapes


def build_metadata(model, model_bytes, inputs_bytes, target):
    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = port_shapes(value for value in model.graph.input if value.name not in initializers)
    outputs = port_shapes(model.graph.output)
    return {
        "schema_version": 1,
        "opset": next((op.version for op in model.opset_import if not op.domain), None),
        "onnx_ir_version": model.ir_version,
        "onnx_version": onnx.__version__,
        "numpy_version": np.__version__,
        "dtype": "float32",
        "input_names": list(inputs),
        "output_names": list(outputs),
        "inputs": inputs,
        "outputs": outputs,
        "target": target,
        "model_sha256": sha256(model_bytes).hexdigest(),
        "inputs_sha256": sha256(inputs_bytes).hexdigest(),
    }


def check_arrays(arrays, shapes, role):
    if arrays.keys() != shapes.keys():
        raise ValueError(f"{role} NPZ names {list(arrays)} do not match {list(shapes)}")
    for name, shape in shapes.items():
        array = arrays[name]
        if array.dtype != np.float32 or array.shape != tuple(shape):
            raise ValueError(f"{role} {name!r}: expected FP32 {shape}, got {array.dtype} {array.shape}")


def run_reference(model_bytes, feeds, input_shapes, output_shapes):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        if error.name != "onnxruntime":
            raise
        raise RuntimeError("ONNX Runtime is required to generate the CPU reference") from error

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_bytes, sess_options=options, providers=[CPU_PROVIDER])
    if session.get_providers() != [CPU_PROVIDER]:
        raise RuntimeError(f"unexpected reference providers: {session.get_providers()}")
    for values, shapes in ((session.get_inputs(), input_shapes), (session.get_outputs(), output_shapes)):
        if [value.name for value in values] != list(shapes):
            raise ValueError("ONNX Runtime tensor names or order differ from the model")
        for value in values:
            if value.type != "tensor(float)" or tuple(value.shape) != tuple(shapes[value.name]):
                raise ValueError(f"unexpected ONNX Runtime metadata for {value.name}")

    check_arrays(feeds, input_shapes, "reference input")
    output_names = list(output_shapes)
    outputs = dict(zip(output_names, session.run(output_names, feeds), strict=True))
    check_arrays(outputs, output_shapes, "reference output")
    return outputs, ort.__version__
