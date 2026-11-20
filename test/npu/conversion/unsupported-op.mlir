func.func @mixed(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %hidden = "mini.matmul"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  // expected-error@+1 {{failed to legalize operation 'mini.relu'}}
  %result = "mini.relu"(%hidden) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
// -----
func.func @unsupported_arith(%lhs: tensor<2x4xf32>, %rhs: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{failed to legalize operation 'arith.addf'}}
  %result = arith.addf %lhs, %rhs : tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
// -----
func.func @unsupported_constant() -> f32 {
  // expected-error@+1 {{failed to legalize operation 'arith.constant'}}
  %result = arith.constant 1.0 : f32
  return %result : f32
}
// -----
func.func @unsupported_call(%input: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{failed to legalize operation 'func.call'}}
  %result = func.call @unsupported_call(%input) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
