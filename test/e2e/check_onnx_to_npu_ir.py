"""Compile real ONNX graphs to NPU Compute IR and check the lowering boundary."""

from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


E2E = Path(__file__).resolve().parent
MODELS = E2E / "models"
IMPORTER = E2E.parents[1] / "python/onnx_importer/importer.py"
PIPELINE = "--pass-pipeline=builtin.module(mini-npu-compile)"
HEX_FORMAT = "--mlir-print-elementsattrs-with-hex-if-larger=0"


def run(*args):
    return subprocess.run(args, capture_output=True, text=True)


def require_success(result):
    assert result.returncode == 0, result.stderr
    return result.stdout


def import_onnx(source, output):
    return run(sys.executable, IMPORTER, "--input", source, "--output", output)


def check_success(mini, directory):
    for name, op_count in (
        ("matmul", 1), ("weighted_matmul", 1), ("matmul_chain", 2), ("matmul_multitile", 1),
    ):
        model_path = MODELS / f"{name}.onnx"
        graph = onnx.load(model_path).graph
        source = directory / f"{name}.mini.mlir"
        output = directory / f"{name}.npu.mlir"
        require_success(import_onnx(model_path, source))
        before = require_success(run(mini, source, HEX_FORMAT))
        require_success(run(mini, source, PIPELINE, HEX_FORMAT, "-o", output))
        after = require_success(run(mini, output, HEX_FORMAT))
        assert after.count('"npu.matmul"') == op_count and '"mini.' not in after, after
        assert after == before.replace('"mini.matmul"', '"npu.matmul"'), after
        if name == "matmul_chain":
            assert '"npu.matmul"(%0, %arg2)' in after and "return %1 : tensor<2x5xf32>" in after
        if graph.initializer:
            assert len(graph.initializer) == 1
            assert "func.func @main(%arg0: tensor<" in after and "%arg1:" not in after
            assert '"npu.matmul"(%arg0, %cst)' in after
            payloads = re.findall(r'arith.constant dense<"0x([0-9A-Fa-f]+)"> : tensor<([^>]+)>', after)
            assert len(payloads) == 1, after
            payload, tensor_type = payloads[0]
            weights = numpy_helper.to_array(graph.initializer[0])
            assert tensor_type == "x".join([*map(str, weights.shape), "f32"]), tensor_type
            np.testing.assert_array_equal(
                np.frombuffer(bytes.fromhex(payload), dtype="<u4"),
                weights.view(np.uint32).ravel(),
            )
        print(f"PASS: {name}: import, full lowering, SSA/type/weight checks and reparse")


def check_failures(mini, directory):
    model = onnx.load(MODELS / "matmul_add_relu.onnx")
    del model.graph.node[0]
    del model.graph.initializer[0]
    model.graph.node[0].input[0] = "X"
    model.graph.input[0].CopyFrom(helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4]))
    add_relu = directory / "add_relu.onnx"
    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, add_relu)

    for source, illegal in ((add_relu, "mini.add"), (MODELS / "matmul_relu.onnx", "mini.relu")):
        mini_ir = directory / f"{source.stem}.mini.mlir"
        npu_ir = directory / f"{source.stem}.npu.mlir"
        require_success(import_onnx(source, mini_ir))
        require_success(run(mini, mini_ir, "-o", "/dev/null"))
        result = run(mini, mini_ir, PIPELINE, "-o", npu_ir)
        assert result.returncode == 1, result
        assert f"failed to legalize operation '{illegal}'" in result.stderr, result.stderr
        assert not npu_ir.exists(), npu_ir
        print(f"PASS: {source.stem}: importer exit 0, NPU pipeline exit 1 ({illegal})")

    model.graph.node[-1].op_type = "Sigmoid"
    unsupported = directory / "unsupported.onnx"
    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, unsupported)
    mini_ir = directory / "unsupported.mini.mlir"
    result = import_onnx(unsupported, mini_ir)
    assert result.returncode == 1 and "E_UNSUPPORTED_OP" in result.stderr, result
    assert not mini_ir.exists()
    print("PASS: valid ONNX Sigmoid rejected by importer, exit 1")


if __name__ == "__main__":
    mini, case = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="mini-onnx-to-npu-") as temporary:
        directory = Path(temporary)
        if case == "positive":
            check_success(mini, directory)
        elif case == "negative":
            check_failures(mini, directory)
        else:
            raise SystemExit(f"unknown test case: {case}")
