module {
  func.func @add_relu(%lhs: tensor<2x4xf32>, %rhs: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = "mini.add_relu"(%lhs, %rhs) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %result : tensor<2x4xf32>
  }
}
