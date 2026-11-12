module {
  func.func @forward(%x: tensor<2x4xf32>, %bias: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %sum = "mini.add"(%x, %bias) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %activated = "mini.relu"(%sum) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %product = "mini.mul"(%activated, %x) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %product : tensor<2x4xf32>
  }
}
