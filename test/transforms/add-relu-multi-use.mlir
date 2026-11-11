module {
  func.func @multi_use(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
    %sum = "mini.add"(%a, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %relu = "mini.relu"(%sum) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %mul = "mini.mul"(%sum, %a) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %relu, %mul : tensor<2x4xf32>, tensor<2x4xf32>
  }
}
