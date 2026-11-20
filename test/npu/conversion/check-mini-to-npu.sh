#!/usr/bin/env bash
set -euo pipefail

tests="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mini="$1"
pipeline='builtin.module(mini-to-npu)'
result="$("$mini" "$tests/matmul-lowering.mlir" --pass-pipeline="$pipeline")"
printf '%s\n' "$result" | "$mini" - --pass-pipeline="$pipeline" -o /dev/null
[[ "$result" != *'"mini.'* ]]
[[ "$result" == *'%0 = "npu.matmul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>'* ]]
[[ "$result" == *'%1 = "npu.matmul"(%0, %arg2) : (tensor<2x4xf32>, tensor<4x5xf32>) -> tensor<2x5xf32>'* ]]
[[ "$result" == *'return %1, %0 : tensor<2x5xf32>, tensor<2x4xf32>'* ]]
printf '%s\n' 'PASS: chain, multiple uses, ordered returns and legal target re-entry'

for input in invalid-shape unsupported-op; do
  "$mini" "$tests/$input.mlir" --pass-pipeline="$pipeline" \
    --split-input-file --verify-diagnostics -o /dev/null
  printf 'PASS: %s diagnostics\n' "$input"
done
