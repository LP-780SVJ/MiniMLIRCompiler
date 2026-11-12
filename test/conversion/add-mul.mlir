module {
  func.func @add_mul(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %sum = "mini.add"(%a, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %product = "mini.mul"(%sum, %a) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %product : tensor<2x4xf32>
  }
}
