module {
  func.func @invalid_type(%a: tensor<2x4xi32>, %b: tensor<2x4xi32>) -> tensor<2x4xi32> {
    %0 = "mini.add"(%a, %b) : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
    func.return %0 : tensor<2x4xi32>
  }
}
