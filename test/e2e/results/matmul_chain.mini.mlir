module {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4x5xf32>) -> tensor<2x5xf32> {
    %v0 = "mini.matmul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %v1 = "mini.matmul"(%v0, %arg2) : (tensor<2x4xf32>, tensor<4x5xf32>) -> tensor<2x5xf32>
    func.return %v1 : tensor<2x5xf32>
  }
}
