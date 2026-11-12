module {
  func.func @no_fusion(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %product = "mini.mul"(%a, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %result = "mini.relu"(%product) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %result : tensor<2x4xf32>
  }
}
