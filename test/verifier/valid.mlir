module {
  func.func @valid_add(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "mini.add"(%a, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
  func.func @valid_mul(%a: tensor<3x2xf32>, %b: tensor<3x2xf32>) -> tensor<3x2xf32> {
    %0 = "mini.mul"(%a, %b) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
    func.return %0 : tensor<3x2xf32>
  }
  func.func @valid_relu(%a: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = "mini.relu"(%a) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    func.return %0 : tensor<4x8xf32>
  }
}
