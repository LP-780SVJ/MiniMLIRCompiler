module {
  func.func @matmul(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %result = "mini.matmul"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    func.return %result : tensor<2x4xf32>
  }

  func.func @matmul_tail(%lhs: tensor<33x37xf32>, %rhs: tensor<37x35xf32>) -> tensor<33x35xf32> {
    %result = "mini.matmul"(%lhs, %rhs) : (tensor<33x37xf32>, tensor<37x35xf32>) -> tensor<33x35xf32>
    func.return %result : tensor<33x35xf32>
  }
}
