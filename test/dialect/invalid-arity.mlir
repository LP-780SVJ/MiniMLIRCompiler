module {
  func.func @test(
      %x: tensor<2x4xf32>,
      %y: tensor<2x4xf32>
  ) -> tensor<2x4xf32> {
    // expected-error@+1 {{'mini.relu' op requires a single operand}}
    %0 = "mini.relu"(%x, %y)
      : (tensor<2x4xf32>, tensor<2x4xf32>)
        -> tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
}
