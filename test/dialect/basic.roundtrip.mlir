module {
  func.func @forward(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "mini.add"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %1 = "mini.relu"(%0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "mini.mul"(%1, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %2 : tensor<2x4xf32>
  }
}

