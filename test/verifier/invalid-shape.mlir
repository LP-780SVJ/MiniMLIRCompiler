module {
  func.func @invalid_add(%a: tensor<2x4xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %0 = "mini.add"(%a, %b) : (tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
}
