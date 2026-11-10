// RUN: mini-opt %s --verify-diagnostics
module {
  func.func @bad_add(%a: tensor<2x4xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
    // expected-error@+1 {{requires lhs and rhs to have identical shapes}}
    %0 = "mini.add"(%a, %b) : (tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
}
