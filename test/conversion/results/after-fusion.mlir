module {
  func.func @forward(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "mini.add_relu"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %1 = "mini.mul"(%0, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}

