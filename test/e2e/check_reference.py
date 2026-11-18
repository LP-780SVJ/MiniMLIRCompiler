"""Validate saved CPU reference fixtures and exercise the comparator CLI."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx


E2E = Path(__file__).resolve().parent
COMPARATOR = E2E.parents[1] / "python/e2e/compare_outputs.py"
FIXTURES = ("matmul", "weighted_matmul", "matmul_relu", "matmul_add_relu", "matmul_multitile", "matmul_chain")


def check_artifacts():
    for name in FIXTURES:
        metadata = json.loads((E2E / "models" / f"{name}.metadata.json").read_text())
        model_path = E2E / metadata["model_file"]
        input_path = E2E / metadata["inputs_file"]
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == metadata["model_sha256"]
        assert hashlib.sha256(input_path.read_bytes()).hexdigest() == metadata["inputs_sha256"]
        assert metadata["reference"]["provider"] == "CPUExecutionProvider"
        model = onnx.load(model_path)
        onnx.checker.check_model(model, full_check=True)
        assert model.ir_version == metadata["onnx_ir_version"]
        assert [(opset.domain, opset.version) for opset in model.opset_import] == [("", metadata["opset"])]
        for path, key, infos in (
            (input_path, "input", model.graph.input),
            (E2E / metadata["reference"]["file"], "output", model.graph.output),
        ):
            with np.load(path, allow_pickle=False) as archive:
                assert archive.files == metadata[f"{key}_names"] == [value.name for value in infos]
                for info in infos:
                    tensor = info.type.tensor_type
                    array = archive[info.name]
                    assert array.dtype == np.float32 and tensor.elem_type == onnx.TensorProto.FLOAT
                    assert list(array.shape) == [dim.dim_value for dim in tensor.shape.dim]
                    assert list(array.shape) == metadata[f"{key}s"][info.name]
                    assert np.isfinite(array).all()
        print(f"PASS: {name} model and input hashes, tensor metadata and CPU reference")

    with np.load(E2E / "reference/matmul_add_relu.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["Y"], [[0, 1.75, 3.5, 0], [6, 0, 0, 10.75]])


def check_comparator(directory):
    reference = E2E / "reference/matmul_relu.npz"
    with np.load(reference, allow_pickle=False) as archive:
        expected = archive["Y"]
    actual = directory / "actual.npz"

    def check(label, outputs, exit_code, *options, reference_path=reference):
        np.savez(actual, **outputs)
        result = subprocess.run(
            [sys.executable, str(COMPARATOR), "--reference", str(reference_path),
             "--actual", str(actual), *options],
            capture_output=True, text=True,
        )
        assert result.returncode == exit_code, (label, result)
        print(f"PASS: {label}, comparator exit {result.returncode}")
        if exit_code == 2:
            assert "tolerances must be finite and nonnegative" in result.stderr
            return None
        report = json.loads(result.stdout)
        assert report["passed"] == (exit_code == 0)
        return report

    assert check("identical", {"Y": expected}, 0)["outputs"]["Y"]["max_absolute_error"] == 0
    changed = expected.copy()
    changed[0, :2] += [1e-6, 1e-4]
    check("within absolute and relative tolerance", {"Y": changed}, 0)
    changed = expected.copy()
    changed[0, 1] += np.float32(0.1)
    stats = check("numeric corruption", {"Y": changed}, 1)["outputs"]["Y"]
    assert stats["mismatched_elements"] == 1 and stats["nonfinite_mismatches"] == 0
    assert np.isclose(stats["max_absolute_error"], 0.1)
    assert np.isclose(stats["max_relative_error"], 0.1 / 1.5)
    for label, outputs, diagnostic in (
        ("transposed shape", {"Y": expected.T}, "shape mismatch"),
        ("wrong dtype", {"Y": expected.astype(np.float64)}, "expected FP32"),
        ("wrong output name", {"Z": expected}, "output tensor names mismatch"),
    ):
        report = check(label, outputs, 1)
        assert report.get("error", report["outputs"].get("Y", {}).get("error")) == diagnostic
    for special in (np.nan, np.inf, -np.inf):
        changed = expected.copy()
        changed[0, 0] = special
        stats = check(f"injected {special}", {"Y": changed}, 1)["outputs"]["Y"]
        assert stats["mismatched_elements"] == stats["nonfinite_mismatches"] == 1

    nonfinite_ref = directory / "nonfinite.npz"
    special_values = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    np.savez(nonfinite_ref, Y=special_values)
    check("matching nonfinite positions", {"Y": special_values}, 0, reference_path=nonfinite_ref)
    check("reversed infinity signs", {"Y": special_values[[0, 2, 1]]}, 1, reference_path=nonfinite_ref)
    check("invalid tolerance", {"Y": expected}, 2, "--rtol", "nan")


if __name__ == "__main__":
    if sys.argv[1:] == ["artifacts"]:
        check_artifacts()
    elif sys.argv[1:] == ["comparator"]:
        with tempfile.TemporaryDirectory(prefix="mini-compare-") as directory:
            check_comparator(Path(directory))
    else:
        raise SystemExit("expected artifacts or comparator")
