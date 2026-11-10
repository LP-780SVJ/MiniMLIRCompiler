#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MINI="${1:-$PROJECT/build/bin/mini-opt}"
TEST_DIR="$PROJECT/test/verifier"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ ! -x "$MINI" ]]; then
  echo "ERROR: mini-opt not found: $MINI" >&2
  exit 1
fi

echo "=== Mini Verifier Tests ==="
"$MINI" "$TEST_DIR/valid.mlir" -o /dev/null
"$MINI" "$PROJECT/test/dialect/basic.mlir" -o /dev/null
echo "PASS: valid.mlir and Day 9 basic.mlir accepted"

expect_failure() {
  local filename="$1"
  local message="$2"
  if "$MINI" "$TEST_DIR/$filename" -o /dev/null \
      >"$TMP_DIR/stdout.log" 2>"$TMP_DIR/stderr.log"; then
    echo "FAIL: $filename unexpectedly accepted" >&2
    exit 1
  fi
  if ! grep -Fq "$message" "$TMP_DIR/stderr.log"; then
    echo "FAIL: $filename has unexpected diagnostics" >&2
    cat "$TMP_DIR/stderr.log" >&2
    exit 1
  fi
  echo "PASS: $filename rejected correctly"
}

expect_failure invalid-shape.mlir "requires lhs and rhs to have identical shapes"
expect_failure invalid-result.mlir "requires result shape to match input shape"
expect_failure invalid-relu-result.mlir "requires result shape to match input shape"
expect_failure invalid-type.mlir "must be statically shaped tensor of 32-bit float values"
expect_failure invalid-rank.mlir "must be statically shaped tensor of 32-bit float values"

if "$MINI" "$PROJECT/test/dialect/shape-mismatch.mlir" -o /dev/null \
    >"$TMP_DIR/stdout.log" 2>"$TMP_DIR/stderr.log"; then
  echo "FAIL: Day 9 shape mismatch unexpectedly accepted" >&2
  exit 1
fi
if ! grep -Fq "requires lhs and rhs to have identical shapes" "$TMP_DIR/stderr.log"; then
  echo "FAIL: Day 9 shape mismatch has unexpected diagnostics" >&2
  cat "$TMP_DIR/stderr.log" >&2
  exit 1
fi
echo "PASS: Day 9 shape mismatch now rejected"
echo "All Mini Verifier tests passed."
