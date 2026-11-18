"""Check real ONNX files through the importer CLI and the Mini IR verifier."""

from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


MODELS = Path(__file__).resolve().parent / "models"
IMPORTER = MODELS.parents[2] / "python/onnx_importer/importer.py"


def import_file(model: Path, output: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(IMPORTER), "--input", str(model), "--output", str(output)],
        capture_output=True, text=True,
    )


def checked_ir(mini_opt: str, model: Path, output: Path, *options: str) -> str:
    imported = import_file(model, output)
    assert imported.returncode == 0, imported.stderr
    parsed = subprocess.run([mini_opt, str(output), *options], capture_output=True, text=True)
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


def check_weighted(mini_opt: str, directory: Path) -> None:
    for name, chain in (
        ("weighted_matmul", ("matmul",)),
        ("matmul_relu", ("matmul", "relu")),
        ("matmul_add_relu", ("matmul", "add", "relu")),
    ):
        ir = checked_ir(mini_opt, MODELS / f"{name}.onnx", directory / f"{name}.mlir")
        assert "func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x4xf32>" in ir, ir
        assert '%0 = "mini.matmul"(%arg0, %cst)' in ir, ir
        assert re.findall(r'"mini\.(\w+)"', ir) == list(chain), ir
        assert "arith.constant dense<" in ir, ir
        if "add" in chain:
            assert '%1 = "mini.add"(%0, %cst_0)' in ir, ir
        if "relu" in chain:
            last = len(chain) - 1
            assert f'%{last} = "mini.relu"(%{last - 1})' in ir, ir
        assert f"return %{len(chain) - 1} : tensor<2x4xf32>" in ir, ir
        print(f"PASS: {name} parameter, constant and SSA chain")

    model = onnx.load(MODELS / "weighted_matmul.onnx")
    bits = np.array(
        [0x3F800000, 0xBF000000, 0x80000000, 0x00000001, 0x3EAAAAAB, 0,
         0x40400000, 0xC0800000, 0x3DCCCCCD, 0xBF800000, 0x00800000, 0x7F7FFFFF],
        dtype=np.uint32,
    )
    model.graph.initializer[0].CopyFrom(numpy_helper.from_array(bits.view(np.float32).reshape(3, 4), name="W"))
    model.graph.node.append(helper.make_node("MatMul", ["X", "W"], ["Z"]))
    model.graph.output.insert(0, helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 4]))
    source = directory / "shared.onnx"
    for raw in (True, False):
        if not raw:
            values = numpy_helper.to_array(model.graph.initializer[0])
            model.graph.initializer[0].CopyFrom(
                helper.make_tensor("W", TensorProto.FLOAT, [3, 4], values.reshape(-1))
            )
        onnx.save_model(model, source)
        ir = checked_ir(
            mini_opt, source, directory / "shared.mlir", "--mlir-print-elementsattrs-with-hex-if-larger=0",
        )
        payloads = re.findall(r'arith.constant dense<"0x([0-9A-Fa-f]+)"> : tensor<3x4xf32>', ir)
        assert len(payloads) == 1, ir
        decoded = np.frombuffer(bytes.fromhex(payloads[0]), dtype="<u4")
        np.testing.assert_array_equal(decoded, bits)
        assert ir.count('"mini.matmul"(%arg0, %cst)') == 2, ir
        assert "return %1, %0 : tensor<2x4xf32>, tensor<2x4xf32>" in ir, ir
    print("PASS: shared initializer and reversed result order; raw_data and float_data retain every FP32 bit")

    model = onnx.load(MODELS / "matmul_add_relu.onnx")
    model.graph.node[1].op_type = "Mul"
    source = directory / "matmul_mul_relu.onnx"
    onnx.save_model(model, source)
    ir = checked_ir(mini_opt, source, directory / "matmul_mul_relu.mlir")
    assert '%1 = "mini.mul"(%0, %cst_0)' in ir and '%2 = "mini.relu"(%1)' in ir, ir
    multitile = checked_ir(mini_opt, MODELS / "matmul_multitile.onnx", directory / "multitile.mlir")
    assert '"mini.matmul"(%arg0, %cst) : (tensor<64x96xf32>, tensor<96x64xf32>) -> tensor<64x64xf32>' in multitile


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
    broadcast = onnx.load(MODELS / "matmul_add_relu.onnx")
    broadcast.graph.initializer[1].CopyFrom(
        numpy_helper.from_array(np.zeros(4, dtype=np.float32), name="B")
    )
    cases.append(("broadcast_add", broadcast, "E_BROADCAST_UNSUPPORTED", True))
    bad_attribute = onnx.load(MODELS / "matmul.onnx")
    bad_attribute.graph.node[0].attribute.append(helper.make_attribute("alpha", 1.0))
    cases.append(("attribute", bad_attribute, "E_ATTRIBUTE_UNSUPPORTED", False))

    def add_weight_case(name, change, diagnostic):
        model = onnx.load(MODELS / "weighted_matmul.onnx")
        change(model)
        cases.append((name, model, diagnostic, False))

    def externalize(model):
        tensor = model.graph.initializer[0]
        onnx.external_data_helper.set_external_data(tensor, location="missing.bin")
        tensor.ClearField("raw_data")

    add_weight_case("external", externalize, "E_INITIALIZER_UNSUPPORTED")
    add_weight_case("overridable", lambda model: model.graph.input.append(
        helper.make_tensor_value_info("W", TensorProto.FLOAT, [3, 4])
    ), "E_INITIALIZER_UNSUPPORTED")
    add_weight_case("sparse", lambda model: model.graph.sparse_initializer.add(), "E_INITIALIZER_UNSUPPORTED")
    add_weight_case("truncated_weight", lambda model: setattr(
        model.graph.initializer[0], "raw_data", b"123"
    ), "E_INVALID_GRAPH")
    add_weight_case("nonfinite_weight", lambda model: model.graph.initializer[0].CopyFrom(
        numpy_helper.from_array(np.full((3, 4), np.nan, dtype=np.float32), name="W")
    ), "E_INITIALIZER_UNSUPPORTED")
    add_weight_case("oversize_weight", lambda model: model.graph.initializer[0].CopyFrom(
        numpy_helper.from_array(np.zeros((3, 6000), dtype=np.float32), name="W")
    ), "E_INITIALIZER_UNSUPPORTED")
    add_weight_case("fp16_weight", lambda model: model.graph.initializer[0].CopyFrom(
        numpy_helper.from_array(np.zeros((3, 4), dtype=np.float16), name="W")
    ), "E_UNSUPPORTED_DTYPE")

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
        elif case == "weighted-import":
            check_weighted(mini_opt, directory)
        elif case == "reject-unsupported":
            check_rejections(directory)
        else:
            raise SystemExit(f"unknown test case: {case}")
