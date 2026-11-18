module {
  func.func @small(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %result = "npu.matmul"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
  func.func @whole_tensor(%lhs: tensor<64x96xf32>, %rhs: tensor<96x64xf32>) -> tensor<64x64xf32> {
    %result = "npu.matmul"(%lhs, %rhs) : (tensor<64x96xf32>, tensor<96x64xf32>) -> tensor<64x64xf32>
    return %result : tensor<64x64xf32>
  }
  func.func @tail(%lhs: tensor<33x37xf32>, %rhs: tensor<37x35xf32>) -> tensor<33x35xf32> {
    %result = "npu.matmul"(%lhs, %rhs) : (tensor<33x37xf32>, tensor<37x35xf32>) -> tensor<33x35xf32>
    return %result : tensor<33x35xf32>
  }
}
