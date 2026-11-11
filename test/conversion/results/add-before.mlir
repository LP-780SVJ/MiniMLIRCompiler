module {
  func.func @add(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "mini.add"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}

