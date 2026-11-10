module {
  func.func @test(
      %x: tensor<2x4xi32>,
      %y: tensor<2x4xi32>
  ) -> tensor<2x4xi32> {
    // expected-error@+1 {{'mini.add' op operand #0 must be statically shaped tensor of 32-bit float values}}
    %0 = "mini.add"(%x, %y)
      : (tensor<2x4xi32>, tensor<2x4xi32>)
        -> tensor<2x4xi32>
    func.return %0 : tensor<2x4xi32>
  }
}
