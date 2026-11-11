module {
  func.func @mul(%a: tensor<3x2xf32>, %b: tensor<3x2xf32>) -> tensor<3x2xf32> {
    %result = "mini.mul"(%a, %b) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
    func.return %result : tensor<3x2xf32>
  }
}
