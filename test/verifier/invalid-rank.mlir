module {
  func.func @invalid_rank(%a: tensor<*xf32>) -> tensor<*xf32> {
    %0 = "mini.relu"(%a) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %0 : tensor<*xf32>
  }
}
