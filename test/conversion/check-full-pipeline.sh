#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MINI="${1:-$PROJECT/build/bin/mini-opt}"
TEST="$PROJECT/test/conversion"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PIPELINE='builtin.module(mini-compile)'

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
  "$MINI" "$2" -o /dev/null
  absent '"mini.' "$2"
}

# The second tensor computation must consume the first one's SSA result, and
# the function must return the second result. Presence-only checks miss this.
check_two_stage_chain() {
  local first second
  first="$(sed -n 's/^[[:space:]]*\(%[^[:space:]]*\) = "linalg.generic".*/\1/p' "$1" | head -n 1)"
  second="$(sed -n 's/^[[:space:]]*\(%[^[:space:]]*\) = "linalg.generic".*/\1/p' "$1" | tail -n 1)"
  if [[ -z "$first" || -z "$second" || "$first" == "$second" ]]; then
    echo "FAIL: expected two distinct linalg.generic results in $1" >&2
    exit 1
  fi
  contains "\"linalg.generic\"($first," "$1"
  contains "\"func.return\"($second)" "$1"
}

echo '=== Mini Full Pipeline Tests ==='

# Each new pattern must also work without the fusion pipeline.
"$MINI" "$TEST/relu-lowering.mlir" \
  --pass-pipeline='builtin.module(mini-to-linalg)' \
  --mlir-print-op-generic -o "$TMP/relu.mlir"
expect_count '"linalg.generic"' 1 "$TMP/relu.mlir"
contains 'indexing_maps = [#map, #map]' "$TMP/relu.mlir"
contains '"arith.cmpf"' "$TMP/relu.mlir"
# In generic assembly on this MLIR build, OGT is serialized as predicate 2.
contains 'predicate = 2 : i64' "$TMP/relu.mlir"
contains '"arith.select"' "$TMP/relu.mlir"
absent '"mini.' "$TMP/relu.mlir"
"$MINI" "$TMP/relu.mlir" -o /dev/null

"$MINI" "$TEST/add-relu-lowering.mlir" \
  --pass-pipeline='builtin.module(mini-to-linalg)' \
  --mlir-print-op-generic -o "$TMP/add-relu.mlir"
expect_count '"linalg.generic"' 1 "$TMP/add-relu.mlir"
contains 'indexing_maps = [#map, #map, #map]' "$TMP/add-relu.mlir"
contains '"arith.addf"' "$TMP/add-relu.mlir"
contains '"arith.cmpf"' "$TMP/add-relu.mlir"
contains '"arith.select"' "$TMP/add-relu.mlir"
absent '"mini.' "$TMP/add-relu.mlir"
"$MINI" "$TMP/add-relu.mlir" -o /dev/null
echo 'PASS: standalone Relu and AddRelu lowering'

lower add-mul.mlir "$TMP/add-mul.mlir"
expect_count '"linalg.generic"' 2 "$TMP/add-mul.mlir"
check_two_stage_chain "$TMP/add-mul.mlir"
contains '"arith.addf"' "$TMP/add-mul.mlir"
contains '"arith.mulf"' "$TMP/add-mul.mlir"
echo 'PASS: Add/Mul full lowering'

"$MINI" "$TEST/full-pipeline.mlir" \
  --pass-pipeline='builtin.module(func.func(mini-fuse-add-relu))' \
  --mlir-print-op-generic -o "$TMP/after-fusion.mlir"
contains '"mini.add_relu"' "$TMP/after-fusion.mlir"
contains '"mini.mul"' "$TMP/after-fusion.mlir"
absent '"mini.add"' "$TMP/after-fusion.mlir"
absent '"mini.relu"' "$TMP/after-fusion.mlir"

lower full-pipeline.mlir "$TMP/full.mlir"
expect_count '"linalg.generic"' 2 "$TMP/full.mlir"
check_two_stage_chain "$TMP/full.mlir"
contains '"arith.addf"' "$TMP/full.mlir"
contains '"arith.cmpf"' "$TMP/full.mlir"
contains '"arith.select"' "$TMP/full.mlir"
contains '"arith.mulf"' "$TMP/full.mlir"
echo 'PASS: fusion followed by full lowering'

"$MINI" "$TEST/no-fusion.mlir" \
  --pass-pipeline='builtin.module(func.func(mini-fuse-add-relu))' \
  --mlir-print-op-generic -o "$TMP/not-fused.mlir"
contains '"mini.mul"' "$TMP/not-fused.mlir"
contains '"mini.relu"' "$TMP/not-fused.mlir"
absent '"mini.add_relu"' "$TMP/not-fused.mlir"

lower no-fusion.mlir "$TMP/no-fusion.mlir"
expect_count '"linalg.generic"' 2 "$TMP/no-fusion.mlir"
check_two_stage_chain "$TMP/no-fusion.mlir"
contains '"arith.mulf"' "$TMP/no-fusion.mlir"
contains '"arith.cmpf"' "$TMP/no-fusion.mlir"
contains '"arith.select"' "$TMP/no-fusion.mlir"
echo 'PASS: non-fused graph fully lowered'

"$MINI" "$TEST/unsupported.mlir" -o /dev/null
if "$MINI" "$TEST/unsupported.mlir" --pass-pipeline="$PIPELINE" \
    -o /dev/null >"$TMP/unsupported.log" 2>&1; then
  echo 'FAIL: unsupported mini.ping was accepted' >&2
  exit 1
fi
contains 'mini.ping' "$TMP/unsupported.log"
echo 'PASS: unsupported Mini operation rejected'

if "$MINI" "$PROJECT/test/verifier/invalid-shape.mlir" \
    --pass-pipeline="$PIPELINE" -o /dev/null >"$TMP/shape.log" 2>&1; then
  echo 'FAIL: invalid input shape was accepted' >&2
  exit 1
fi
contains 'requires lhs and rhs to have identical shapes' "$TMP/shape.log"
echo 'PASS: input IR verifier preserved'

"$MINI" "$TMP/full.mlir" --pass-pipeline="$PIPELINE" \
  --mlir-print-op-generic -o "$TMP/twice.mlir"
absent '"mini.' "$TMP/twice.mlir"
"$MINI" "$TMP/twice.mlir" -o /dev/null
echo 'PASS: final IR accepts a second pipeline run'

echo 'All full pipeline tests passed.'
