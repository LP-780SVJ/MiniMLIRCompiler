module {
  func.func @relu(%input: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = "mini.relu"(%input) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %result : tensor<2x4xf32>
  }
}
