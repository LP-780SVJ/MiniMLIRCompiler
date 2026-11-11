module {
  func.func @invalid(%a: tensor<2x4xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
    // expected-error@+1 {{requires lhs and rhs to have identical shapes}}
    %sum = "mini.add"(%a, %b) : (tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %out = "mini.relu"(%sum) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
