module {
  func.func @test(
      %x: tensor<?x4xf32>
  ) -> tensor<?x4xf32> {
    // expected-error@+1 {{'mini.relu' op operand #0 must be statically shaped tensor of 32-bit float values}}
    %0 = "mini.relu"(%x)
      : (tensor<?x4xf32>)
        -> tensor<?x4xf32>
    func.return %0 : tensor<?x4xf32>
  }
}
