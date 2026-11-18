"""Check real ONNX files through the importer CLI and the Mini IR verifier."""

from pathlib import Path
import subprocess
import sys
import tempfile

import onnx
from onnx import TensorProto, helper


MODELS = Path(__file__).resolve().parent / "models"
IMPORTER = MODELS.parents[2] / "python/onnx_importer/importer.py"


def import_file(model: Path, output: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(IMPORTER), "--input", str(model), "--output", str(output)],
        capture_output=True, text=True,
    )


def checked_ir(mini_opt: str, model: Path, output: Path) -> str:
    imported = import_file(model, output)
    assert imported.returncode == 0, imported.stderr
    parsed = subprocess.run([mini_opt, str(output)], capture_output=True, text=True)
    assert parsed.returncode == 0, parsed.stderr
    return parsed.stdout


def check_single(mini_opt: str, directory: Path) -> None:
    ir = checked_ir(mini_opt, MODELS / "matmul.onnx", directory / "matmul.mlir")
    assert "func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32>" in ir, ir
    assert ir.count('"mini.matmul"') == 1, ir
    assert '%0 = "mini.matmul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>' in ir, ir
    assert "return %0 : tensor<2x4xf32>" in ir, ir
    print("PASS: single MatMul signature, operands, result type and returned SSA")


def check_chain(mini_opt: str, directory: Path) -> None:
    source = MODELS / "matmul_chain.onnx"
    model = onnx.load(source)
    assert not model.graph.value_info, "fixture should require inference of H"
    ir = checked_ir(mini_opt, source, directory / "chain.mlir")
    assert ir.count('"mini.matmul"') == 2, ir
    assert '%0 = "mini.matmul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>' in ir, ir
    assert '%1 = "mini.matmul"(%0, %arg2) : (tensor<2x4xf32>, tensor<4x5xf32>) -> tensor<2x5xf32>' in ir, ir
    assert "return %1 : tensor<2x5xf32>" in ir, ir

    model.graph.output.append(helper.make_tensor_value_info("H", TensorProto.FLOAT, [2, 4]))
    multi_output = directory / "multi_output.onnx"
    onnx.save_model(model, multi_output)
    ir = checked_ir(mini_opt, multi_output, directory / "multi_output.mlir")
    assert "-> (tensor<2x5xf32>, tensor<2x4xf32>)" in ir, ir
    assert "return %1, %0 : tensor<2x5xf32>, tensor<2x4xf32>" in ir, ir
    print("PASS: inferred intermediate, chained SSA use and graph output order")


def matmul_model(lhs, rhs, result, dtype=TensorProto.FLOAT) -> onnx.ModelProto:
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("MatMul", ["A", "B"], ["Y"], name="matmul_0")],
            "negative_matmul",
            [helper.make_tensor_value_info("A", dtype, lhs),
             helper.make_tensor_value_info("B", dtype, rhs)],
            [helper.make_tensor_value_info("Y", dtype, result)],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )


def expect_rejection(model: Path, output: Path, diagnostic: str) -> None:
    failed = import_file(model, output)
    assert failed.returncode == 1, (failed.returncode, failed.stderr)
    assert f"{model}: {diagnostic}" in failed.stderr, failed.stderr
    assert not output.exists(), output
    print(f"PASS: {model.stem}, exit {failed.returncode}, {failed.stderr.splitlines()[0]}")


def check_rejections(directory: Path) -> None:
    cases = [
        ("fp16", matmul_model([2, 3], [3, 4], [2, 4], TensorProto.FLOAT16), "E_UNSUPPORTED_DTYPE", True),
        ("bf16", matmul_model([2, 3], [3, 4], [2, 4], TensorProto.BFLOAT16), "E_UNSUPPORTED_DTYPE", True),
        ("batch", matmul_model([1, 2, 3], [3, 4], [1, 2, 4]), "E_UNSUPPORTED_RANK", True),
        ("dynamic", matmul_model(["M", 3], [3, 4], ["M", 4]), "E_DYNAMIC_SHAPE", True),
        ("zero", matmul_model([2, 0], [0, 4], [2, 4]), "E_SHAPE_MISMATCH", True),
        ("k_mismatch", matmul_model([2, 5], [3, 4], [2, 4]), "E_SHAPE_MISMATCH", False),
        ("result_mismatch", matmul_model([2, 3], [3, 4], [2, 5]), "E_SHAPE_MISMATCH", False),
    ]
    sig = onnx.load(MODELS / "matmul_chain.onnx")
    del sig.graph.node[-1].input[-1]
    sig.graph.node[-1].op_type = "Sigmoid"
    sig.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 4
    cases.append(("second_node_sigmoid", sig, "E_UNSUPPORTED_OP", True))

    domain = onnx.load(MODELS / "matmul.onnx")
    domain.graph.node[0].domain = "com.example"
    cases.append(("custom_domain", domain, "E_UNSUPPORTED_DOMAIN", False))
    opset = onnx.load(MODELS / "matmul.onnx")
    opset.opset_import[0].version = 17
    cases.append(("opset17", opset, "E_UNSUPPORTED_OPSET", True))
    undefined = onnx.load(MODELS / "matmul.onnx")
    undefined.graph.output[0].name = "missing"
    cases.append(("undefined_output", undefined, "E_INVALID_GRAPH", False))
    duplicate = onnx.load(MODELS / "matmul.onnx")
    duplicate.graph.node.append(duplicate.graph.node[0])
    cases.append(("duplicate_definition", duplicate, "E_INVALID_GRAPH", False))
    annotations = onnx.load(MODELS / "matmul_chain.onnx")
    annotations.graph.value_info.append(helper.make_tensor_value_info("H", TensorProto.FLOAT, [2, 5]))
    cases.append(("conflicting_value_info", annotations, "E_SHAPE_MISMATCH", False))

    for name, model, diagnostic, valid_onnx in cases:
        if valid_onnx:
            onnx.checker.check_model(model, full_check=True)
        source = directory / f"{name}.onnx"
        onnx.save_model(model, source)
        expect_rejection(source, directory / f"{name}.mlir", diagnostic)

    existing = directory / "existing.mlir"
    existing.write_bytes(b"previous successful output\n")
    failed = import_file(directory / "second_node_sigmoid.onnx", existing)
    assert failed.returncode == 1 and "E_UNSUPPORTED_OP" in failed.stderr, failed.stderr
    assert existing.read_bytes() == b"previous successful output\n"
    print("PASS: failed import preserves existing output")

    expect_rejection(MODELS / "matmul_relu.onnx", directory / "weighted.mlir", "E_INITIALIZER_UNSUPPORTED")
    invalid_bytes = directory / "invalid.onnx"
    invalid_bytes.write_bytes(b"not a protobuf")
    expect_rejection(invalid_bytes, directory / "invalid.mlir", "E_MODEL_READ")


if __name__ == "__main__":
    mini_opt, case = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="mini-onnx-") as temporary:
        directory = Path(temporary)
        if case == "matmul-import":
            check_single(mini_opt, directory)
        elif case == "two-matmul-import":
            check_chain(mini_opt, directory)
        elif case == "reject-unsupported":
            check_rejections(directory)
        else:
            raise SystemExit(f"unknown test case: {case}")
