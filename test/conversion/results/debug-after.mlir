#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @mixed(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = tensor.empty() : tensor<2x4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x4xf32>, tensor<2x4xf32>) outs(%0 : tensor<2x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.addf %in, %in_0 : f32
      linalg.yield %6 : f32
    } -> tensor<2x4xf32>
    %2 = "mini.relu"(%1) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %3 = tensor.empty() : tensor<2x4xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg0 : tensor<2x4xf32>, tensor<2x4xf32>) outs(%3 : tensor<2x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.mulf %in, %in_0 : f32
      linalg.yield %6 : f32
    } -> tensor<2x4xf32>
    %5 = "mini.add_relu"(%4, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %5 : tensor<2x4xf32>
  }
}

