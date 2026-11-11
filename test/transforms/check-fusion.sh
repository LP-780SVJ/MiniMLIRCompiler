#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MINI="${1:-$PROJECT/build/bin/mini-opt}"
TEST="$PROJECT/test/transforms"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PIPELINE='builtin.module(func.func(mini-fuse-add-relu))'

contains() {
  if ! grep -Fq "$1" "$2"; then
    echo "FAIL: expected '$1' in $2" >&2
    exit 1
  fi
}

absent() {
  if grep -Fq "$1" "$2"; then
    echo "FAIL: unexpected '$1' in $2" >&2
    exit 1
  fi
}

expect_count() {
  local actual
  actual="$(grep -Fc "$1" "$3" || true)"
  if [[ "$actual" != "$2" ]]; then
    echo "FAIL: expected $2 occurrences of '$1' in $3, found $actual" >&2
    exit 1
  fi
}

echo "=== AddReluFusion Tests ==="

"$MINI" "$TEST/add-relu-direct.mlir" -o /dev/null
"$MINI" "$TEST/add-relu-direct-invalid.mlir" --verify-diagnostics -o /dev/null
echo "PASS: fused operation parses and verifies independently"

"$MINI" "$TEST/add-relu-positive.mlir" -o "$TMP/before.mlir"
"$MINI" "$TEST/add-relu-positive.mlir" --pass-pipeline="$PIPELINE" \
  -o "$TMP/once.mlir"
contains '"mini.add"' "$TMP/before.mlir"
contains '"mini.relu"' "$TMP/before.mlir"
absent '"mini.add_relu"' "$TMP/before.mlir"
contains '"mini.add_relu"' "$TMP/once.mlir"
expect_count '"mini.add_relu"' 1 "$TMP/once.mlir"
absent '"mini.add"' "$TMP/once.mlir"
absent '"mini.relu"' "$TMP/once.mlir"
echo "PASS: positive fusion"

"$MINI" "$TEST/add-relu-negative.mlir" --pass-pipeline="$PIPELINE" \
  -o "$TMP/negative.mlir"
contains '"mini.mul"' "$TMP/negative.mlir"
contains '"mini.relu"' "$TMP/negative.mlir"
expect_count '"mini.mul"' 1 "$TMP/negative.mlir"
expect_count '"mini.relu"' 2 "$TMP/negative.mlir"
absent '"mini.add_relu"' "$TMP/negative.mlir"
echo "PASS: negative cases"

"$MINI" "$TEST/add-relu-multi-use.mlir" --pass-pipeline="$PIPELINE" \
  -o "$TMP/multi.mlir"
contains '"mini.add"' "$TMP/multi.mlir"
contains '"mini.relu"' "$TMP/multi.mlir"
contains '"mini.mul"' "$TMP/multi.mlir"
expect_count '"mini.add"' 1 "$TMP/multi.mlir"
expect_count '"mini.relu"' 1 "$TMP/multi.mlir"
expect_count '"mini.mul"' 1 "$TMP/multi.mlir"
absent '"mini.add_relu"' "$TMP/multi.mlir"
echo "PASS: multi-use protection"

"$MINI" "$TEST/add-relu-positive.mlir" \
  --pass-pipeline='builtin.module(func.func(mini-fuse-add-relu,mini-fuse-add-relu))' \
  -o "$TMP/twice.mlir"
diff -u "$TMP/once.mlir" "$TMP/twice.mlir"
echo "PASS: repeated pass is stable"

"$MINI" "$TEST/add-relu-positive.mlir" \
  --pass-pipeline='builtin.module(func.func(canonicalize,mini-fuse-add-relu,canonicalize))' \
  -o "$TMP/full.mlir"
contains '"mini.add_relu"' "$TMP/full.mlir"
absent '"mini.add"' "$TMP/full.mlir"
echo "PASS: canonicalization and fusion pipeline"

"$MINI" "$TEST/add-relu-invalid.mlir" --pass-pipeline="$PIPELINE" \
  --verify-diagnostics -o /dev/null
echo "PASS: invalid input remains rejected"

echo "All AddReluFusion tests passed."
