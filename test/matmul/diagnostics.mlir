func.func @bad_k(%lhs: tensor<2x3xf32>, %rhs: tensor<5x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{requires matching K dimensions, got lhs K = 3 and rhs K = 5}}
  %result = "mini.matmul"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<5x4xf32>) -> tensor<2x4xf32>
  func.return %result : tensor<2x4xf32>
}

// -----

func.func @bad_result(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x5xf32> {
  // expected-error@+1 {{requires result shape [2, 4], got 'tensor<2x5xf32>'}}
  %result = "mini.matmul"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x5xf32>
  func.return %result : tensor<2x5xf32>
}

// -----

func.func @bad_rank(%lhs: tensor<1x2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<1x2x4xf32> {
  // expected-error@+1 {{requires rank-2 lhs, rhs and result tensors}}
  %result = "mini.matmul"(%lhs, %rhs) : (tensor<1x2x3xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  func.return %result : tensor<1x2x4xf32>
}

// -----

func.func @bad_dtype(%lhs: tensor<2x3xf16>, %rhs: tensor<3x4xf16>) -> tensor<2x4xf16> {
  // expected-error@+1 {{must be statically shaped tensor of 32-bit float values}}
  %result = "mini.matmul"(%lhs, %rhs) : (tensor<2x3xf16>, tensor<3x4xf16>) -> tensor<2x4xf16>
  func.return %result : tensor<2x4xf16>
}

// -----

func.func @bad_dynamic(%lhs: tensor<?x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{must be statically shaped tensor of 32-bit float values}}
  %result = "mini.matmul"(%lhs, %rhs) : (tensor<?x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  func.return %result : tensor<2x4xf32>
}

// -----

func.func @bad_arity(%lhs: tensor<2x3xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{expected 2 operands, but found 1}}
  %result = "mini.matmul"(%lhs) : (tensor<2x3xf32>) -> tensor<2x4xf32>
  func.return %result : tensor<2x4xf32>
}

// -----

func.func @bad_zero(%lhs: tensor<2x0xf32>, %rhs: tensor<0x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{requires positive dimensions}}
  %result = "mini.matmul"(%lhs, %rhs) : (tensor<2x0xf32>, tensor<0x4xf32>) -> tensor<2x4xf32>
  func.return %result : tensor<2x4xf32>
}
