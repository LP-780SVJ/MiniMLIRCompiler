func.func @bad_k(%lhs: tensor<2x3xf32>, %rhs: tensor<5x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{requires matching K dimensions}}
  %result = "mini.matmul"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<5x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
// -----
func.func @bad_dtype(%lhs: tensor<2x3xbf16>, %rhs: tensor<3x4xbf16>) -> tensor<2x4xbf16> {
  // expected-error@+1 {{must be statically shaped tensor of 32-bit float values}}
  %result = "mini.matmul"(%lhs, %rhs) : (tensor<2x3xbf16>, tensor<3x4xbf16>) -> tensor<2x4xbf16>
  return %result : tensor<2x4xbf16>
}
// -----
func.func @bad_dynamic(%lhs: tensor<?x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<?x4xf32> {
  // expected-error@+1 {{must be statically shaped tensor of 32-bit float values}}
  %result = "mini.matmul"(%lhs, %rhs) : (tensor<?x3xf32>, tensor<3x4xf32>) -> tensor<?x4xf32>
  return %result : tensor<?x4xf32>
}
