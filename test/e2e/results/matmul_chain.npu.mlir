module {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4x5xf32>) -> tensor<2x5xf32> {
    %0 = "npu.matmul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %1 = "npu.matmul"(%0, %arg2) : (tensor<2x4xf32>, tensor<4x5xf32>) -> tensor<2x5xf32>
    return %1 : tensor<2x5xf32>
  }
}

