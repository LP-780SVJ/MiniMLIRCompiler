module {
  func.func @no_fusion(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "mini.mul"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %1 = "mini.relu"(%0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}

