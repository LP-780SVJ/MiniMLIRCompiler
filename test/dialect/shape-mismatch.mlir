module {
  func.func @test(
      %x: tensor<2x4xf32>,
      %y: tensor<3x4xf32>
  ) -> tensor<2x4xf32> {
    %0 = "mini.add"(%x, %y)
      : (tensor<2x4xf32>, tensor<3x4xf32>)
        -> tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
}
