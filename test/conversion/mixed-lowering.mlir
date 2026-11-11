module {
  func.func @mixed(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %sum = "mini.add"(%a, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %activated = "mini.relu"(%sum) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %product = "mini.mul"(%activated, %a) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %out = "mini.add_relu"(%product, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
