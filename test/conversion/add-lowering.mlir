module {
  func.func @add(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = "mini.add"(%a, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %result : tensor<2x4xf32>
  }
}
