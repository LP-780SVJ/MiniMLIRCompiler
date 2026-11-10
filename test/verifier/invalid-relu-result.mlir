module {
  func.func @invalid_relu_result(%a: tensor<2x4xf32>) -> tensor<3x4xf32> {
    %0 = "mini.relu"(%a) : (tensor<2x4xf32>) -> tensor<3x4xf32>
    func.return %0 : tensor<3x4xf32>
  }
}
