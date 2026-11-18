"""Compare named FP32 tensors in two NPZ archives and report JSON."""

import argparse
import json
import math
from pathlib import Path
import sys
from zipfile import BadZipFile

import numpy as np


def compare(reference_path, actual_path, rtol=1e-4, atol=1e-5):
    if any(not math.isfinite(tol) or tol < 0 for tol in (rtol, atol)):
        raise ValueError("tolerances must be finite and nonnegative")
    report = {"passed": True, "rtol": rtol, "atol": atol, "outputs": {}}
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        actual_path, allow_pickle=False
    ) as actual:
        names = reference.files
        if (
            not names or len(names) != len(set(names))
            or len(actual.files) != len(set(actual.files))
            or set(names) != set(actual.files)
        ):
            return {
                **report, "passed": False, "error": "output tensor names mismatch",
                "reference_names": names, "actual_names": actual.files,
            }
        for name in names:
            expected, received = reference[name], actual[name]
            if expected.shape != received.shape:
                result = {
                    "passed": False, "error": "shape mismatch",
                    "reference_shape": expected.shape, "actual_shape": received.shape,
                }
            elif expected.dtype != np.float32 or received.dtype != np.float32:
                result = {
                    "passed": False, "error": "expected FP32",
                    "reference_dtype": str(expected.dtype), "actual_dtype": str(received.dtype),
                }
            else:
                result = compare_tensor(expected, received, rtol, atol)
            report["outputs"][name] = result
            report["passed"] &= result["passed"]
    return report


def compare_tensor(expected, received, rtol, atol):
    finite = np.isfinite(expected) & np.isfinite(received)
    special_match = (
        (np.isnan(expected) & np.isnan(received))
        | (np.isposinf(expected) & np.isposinf(received))
        | (np.isneginf(expected) & np.isneginf(received))
    )
    nonfinite_mismatches = int(np.count_nonzero(~finite & ~special_match))
    reference = expected[finite].astype(np.float64)
    actual = received[finite].astype(np.float64)
    absolute = np.abs(actual - reference)
    scale = np.abs(reference)
    with np.errstate(over="ignore"):
        finite_mismatches = int(np.count_nonzero(absolute > atol + rtol * scale))
    relative = absolute[scale > 0] / scale[scale > 0]
    mismatches = finite_mismatches + nonfinite_mismatches
    return {
        "passed": mismatches == 0,
        "max_absolute_error": float(absolute.max(initial=0)),
        "max_relative_error": float(relative.max(initial=0)),
        "mismatched_elements": mismatches,
        "nonfinite_mismatches": nonfinite_mismatches,
        "finite_pairs": int(absolute.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    try:
        report = compare(args.reference, args.actual, args.rtol, args.atol)
    except (OSError, ValueError, TypeError, EOFError, BadZipFile) as error:
        print(f"cannot compare NPZ outputs: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
