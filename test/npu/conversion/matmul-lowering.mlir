module {
  func.func @chain(%input: tensor<2x3xf32>, %first: tensor<3x4xf32>,
                   %second: tensor<4x5xf32>) -> (tensor<2x5xf32>, tensor<2x4xf32>) {
    %hidden = "mini.matmul"(%input, %first) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %result = "mini.matmul"(%hidden, %second) : (tensor<2x4xf32>, tensor<4x5xf32>) -> tensor<2x5xf32>
    return %result, %hidden : tensor<2x5xf32>, tensor<2x4xf32>
  }
}
