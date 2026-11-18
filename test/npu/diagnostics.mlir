func.func @bad_k(%lhs: tensor<2x3xf32>, %rhs: tensor<5x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{requires matching K dimensions, got lhs K = 3 and rhs K = 5}}
  %result = "npu.matmul"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<5x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
// -----
func.func @bad_rank(%lhs: tensor<1x2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{requires rank-2 tensors}}
  %result = "npu.matmul"(%lhs, %rhs) : (tensor<1x2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
// -----
func.func @bad_dtype(%lhs: tensor<2x3xf16>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{must be statically shaped tensor of 32-bit float values}}
  %result = "npu.matmul"(%lhs, %rhs) : (tensor<2x3xf16>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
// -----
func.func @bad_result(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x5xf32> {
  // expected-error@+1 {{requires result shape [2, 4]}}
  %result = "npu.matmul"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x5xf32>
  return %result : tensor<2x5xf32>
}
// -----
func.func @zero(%lhs: tensor<2x0xf32>, %rhs: tensor<0x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{requires positive dimensions}}
  %result = "npu.matmul"(%lhs, %rhs) : (tensor<2x0xf32>, tensor<0x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
// -----
func.func @dynamic(%lhs: tensor<?x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{must be statically shaped tensor of 32-bit float values}}
  %result = "npu.matmul"(%lhs, %rhs) : (tensor<?x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
