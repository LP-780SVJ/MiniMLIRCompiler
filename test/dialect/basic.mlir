
module {
  func.func @forward(
      %x: tensor<2x4xf32>,
      %bias: tensor<2x4xf32>
  ) -> tensor<2x4xf32> {

    // 1. 逐元素加法
    %sum = "mini.add"(%x, %bias)
      : (tensor<2x4xf32>, tensor<2x4xf32>)
        -> tensor<2x4xf32>

    // 2. ReLU
    %activated = "mini.relu"(%sum)
      : (tensor<2x4xf32>)
        -> tensor<2x4xf32>

    // 3. 逐元素乘法
    %result = "mini.mul"(%activated, %x)
      : (tensor<2x4xf32>, tensor<2x4xf32>)
        -> tensor<2x4xf32>

    func.return %result : tensor<2x4xf32>
  }
}
