module {
  func.func @invalid(%a: tensor<2x4xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
    // expected-error@+1 {{requires lhs and rhs to have identical shapes}}
    %result = "mini.add_relu"(%a, %b) : (tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    func.return %result : tensor<2x4xf32>
  }
}
