"""Generate the Day 19 ONNX MatMul fixtures with runtime inputs only."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def make_model(inputs, nodes, output_shape) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        "day19_matmul",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in inputs],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[2] / "test/e2e/models",
    )
    output_dir = parser.parse_args().output_dir
    single = make_model(
        [("A", [2, 3]), ("B", [3, 4])],
        [helper.make_node("MatMul", ["A", "B"], ["Y"], name="matmul_0")],
        [2, 4],
    )
    chain = make_model(
        [("X", [2, 3]), ("W1", [3, 4]), ("W2", [4, 5])],
        [helper.make_node("MatMul", ["X", "W1"], ["H"], name="matmul_0"),
         helper.make_node("MatMul", ["H", "W2"], ["Y"], name="matmul_1")],
        [2, 5],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in [("matmul.onnx", single), ("matmul_chain.onnx", chain)]:
        path = output_dir / name
        onnx.save_model(model, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
