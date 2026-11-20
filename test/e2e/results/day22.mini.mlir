module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %c0 = arith.constant dense<[[0x3F800000, 0x00000000, 0xBF800000, 0x40000000], [0x00000000, 0x3F800000, 0x3F800000, 0xBF800000], [0x40000000, 0xBF800000, 0x00000000, 0x3F800000]]> : tensor<3x4xf32>
    %v0 = "mini.matmul"(%arg0, %c0) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    func.return %v0 : tensor<2x4xf32>
  }
}
