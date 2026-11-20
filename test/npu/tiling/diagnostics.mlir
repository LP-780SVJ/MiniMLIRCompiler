func.func @bad_k(%lhs: tensor<2x3xf32>, %rhs: tensor<5x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{requires matching K dimensions}}
  %result = "npu.tiled_matmul"(%lhs, %rhs) <{
    target = "virtual-npu-v1", tile_m = 32 : i64, tile_n = 32 : i64, tile_k = 32 : i64, tiles = []
  }> : (tensor<2x3xf32>, tensor<5x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
// -----
func.func @missing_tiles(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{requires attribute 'tiles'}}
  %result = "npu.tiled_matmul"(%lhs, %rhs) <{
    target = "virtual-npu-v1", tile_m = 32 : i64, tile_n = 32 : i64, tile_k = 32 : i64
  }> : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
