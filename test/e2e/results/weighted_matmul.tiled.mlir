module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %cst = arith.constant dense<[[1.000000e+00, 0.000000e+00, -1.000000e+00, 2.000000e+00], [0.000000e+00, 1.000000e+00, 1.000000e+00, -1.000000e+00], [2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x4xf32>
    %0 = "npu.tiled_matmul"(%arg0, %cst) <{target = "virtual-npu-v1", tile_k = 32 : i64, tile_m = 32 : i64, tile_n = 32 : i64, tiles = [{is_first_k = true, is_last_k = true, k_offset = 0 : i64, k_size = 3 : i64, m_offset = 0 : i64, m_size = 2 : i64, n_offset = 0 : i64, n_size = 4 : i64, output_tile_id = 0 : i64}]}> : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}

