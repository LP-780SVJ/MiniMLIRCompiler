module {
  func.func @invalid_mul_result(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %0 = "mini.mul"(%a, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
    func.return %0 : tensor<2x8xf32>
  }
}
