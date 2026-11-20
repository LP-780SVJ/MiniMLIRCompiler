func.func @tail(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %result = "npu.tiled_matmul"(%lhs, %rhs) <{
    target = "virtual-npu-v1", tile_m = 32 : i64, tile_n = 32 : i64, tile_k = 32 : i64,
    tiles = [{m_offset = 0 : i64, n_offset = 0 : i64, k_offset = 0 : i64,
              m_size = 2 : i64, n_size = 4 : i64, k_size = 3 : i64,
              output_tile_id = 0 : i64, is_first_k = true, is_last_k = true}]
  }> : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
