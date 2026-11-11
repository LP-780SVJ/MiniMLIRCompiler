#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MINI="${1:-$PROJECT/build/bin/mini-opt}"
TEST="$PROJECT/test/conversion"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PIPELINE='builtin.module(mini-to-linalg)'

contains() {
  if ! grep -Fq -- "$1" "$2"; then
    echo "FAIL: expected '$1' in $2" >&2
    exit 1
  fi
}

absent() {
  if grep -Fq -- "$1" "$2"; then
    echo "FAIL: unexpected '$1' in $2" >&2
    exit 1
  fi
}

expect_count() {
  local actual
  actual="$(grep -Fc -- "$1" "$3" || true)"
  if [[ "$actual" != "$2" ]]; then
    echo "FAIL: expected $2 occurrences of '$1' in $3, found $actual" >&2
    exit 1
  fi
}

lower() {
  "$MINI" "$TEST/$1" --pass-pipeline="$PIPELINE" \
    --mlir-print-op-generic -o "$2"
}

check_elementwise_structure() {
  local file="$1"
  local shape="$2"
  contains 'affine_map<(d0, d1) -> (d0, d1)>' "$file"
  contains 'indexing_maps = [#map, #map, #map]' "$file"
  contains 'iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]' "$file"
  contains "-> tensor<$shape>" "$file"
}

check_return_uses_generic() {
  local result
  result="$(sed -n 's/^[[:space:]]*\(%[^[:space:]]*\) = "linalg.generic".*/\1/p' "$1" | tail -n 1)"
  if [[ -z "$result" ]]; then
    echo "FAIL: no linalg.generic result in $1" >&2
    exit 1
  fi
  contains "\"func.return\"($result)" "$1"
}

echo '=== MiniToLinalg Tests ==='

lower add-lowering.mlir "$TMP/add.mlir"
expect_count '"tensor.empty"' 1 "$TMP/add.mlir"
expect_count '"linalg.generic"' 1 "$TMP/add.mlir"
expect_count '"arith.addf"' 1 "$TMP/add.mlir"
contains '"linalg.yield"' "$TMP/add.mlir"
contains 'tensor<2x4xf32>' "$TMP/add.mlir"
check_elementwise_structure "$TMP/add.mlir" '2x4xf32'
check_return_uses_generic "$TMP/add.mlir"
absent '"mini.add"' "$TMP/add.mlir"
"$MINI" "$TMP/add.mlir" -o /dev/null
echo 'PASS: add lowering and round-trip'

lower mul-lowering.mlir "$TMP/mul.mlir"
expect_count '"tensor.empty"' 1 "$TMP/mul.mlir"
expect_count '"linalg.generic"' 1 "$TMP/mul.mlir"
expect_count '"arith.mulf"' 1 "$TMP/mul.mlir"
contains 'tensor<3x2xf32>' "$TMP/mul.mlir"
check_elementwise_structure "$TMP/mul.mlir" '3x2xf32'
check_return_uses_generic "$TMP/mul.mlir"
absent '"mini.mul"' "$TMP/mul.mlir"
"$MINI" "$TMP/mul.mlir" -o /dev/null
echo 'PASS: mul lowering and round-trip'

lower mixed-lowering.mlir "$TMP/mixed.mlir"
expect_count '"linalg.generic"' 2 "$TMP/mixed.mlir"
check_elementwise_structure "$TMP/mixed.mlir" '2x4xf32'
contains '"mini.relu"' "$TMP/mixed.mlir"
contains '"mini.add_relu"' "$TMP/mixed.mlir"
contains '"arith.addf"' "$TMP/mixed.mlir"
contains '"arith.mulf"' "$TMP/mixed.mlir"
absent '"mini.add"' "$TMP/mixed.mlir"
absent '"mini.mul"' "$TMP/mixed.mlir"
"$MINI" "$TMP/mixed.mlir" -o /dev/null
echo 'PASS: partial conversion and mixed IR round-trip'

"$MINI" "$TEST/unsupported.mlir" -o /dev/null
if "$MINI" "$TEST/unsupported.mlir" --pass-pipeline="$PIPELINE" \
    -o /dev/null >"$TMP/unsupported.log" 2>&1; then
  echo 'FAIL: unsupported mini.ping was accepted' >&2
  exit 1
fi
contains 'mini.ping' "$TMP/unsupported.log"
echo 'PASS: unsupported operation rejected by conversion'

if "$MINI" "$PROJECT/test/verifier/invalid-shape.mlir" \
    --pass-pipeline="$PIPELINE" -o /dev/null >"$TMP/shape.log" 2>&1; then
  echo 'FAIL: invalid input shape was accepted' >&2
  exit 1
fi
contains 'requires lhs and rhs to have identical shapes' "$TMP/shape.log"
echo 'PASS: invalid shape rejected by input verifier'

echo 'All MiniToLinalg tests passed.'
