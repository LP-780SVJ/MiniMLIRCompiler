module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %cst = arith.constant dense<[[1.000000e+00, 0.000000e+00, -1.000000e+00, 2.000000e+00], [0.000000e+00, 1.000000e+00, 1.000000e+00, -1.000000e+00], [2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x4xf32>
    %0 = "npu.matmul"(%arg0, %cst) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}

