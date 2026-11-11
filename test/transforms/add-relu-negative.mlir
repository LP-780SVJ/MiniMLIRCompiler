module {
  func.func @mul_then_relu(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %mul = "mini.mul"(%a, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %out = "mini.relu"(%mul) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
  func.func @relu_only(%a: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %out = "mini.relu"(%a) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
